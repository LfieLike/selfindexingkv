#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>
#include <cuda_fp16.h>  // 引入半精度浮点数的支持
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 128
#define EMB_DIM 128
#define NUM_PER_THREAD 8
template <typename T>
__device__ float convert_to_float(T value) {
    // Return 0 by default, indicating misuse if not specialized correctly.
    return 0.0f;
}

template <>
__device__ float convert_to_float<c10::Half>(c10::Half value) {
    return __half2float(value);
}

template <>
__device__ float convert_to_float<float>(float value) {
    return value;
}

template <>
__device__ float convert_to_float<at::BFloat16>(at::BFloat16 value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ T convert_from_float(float value) {
    // Return 0 by default, indicating misuse if not specialized correctly.
    return static_cast<T>(0);
}
template <>
__device__ uint8_t convert_from_float<uint8_t>(float value) {
    return static_cast<uint8_t>(value);
}
template <>
__device__ c10::Half convert_from_float<c10::Half>(float value) {
    return __float2half(value);
}

template <>
__device__ float convert_from_float<float>(float value) {
    return value;
}

template <>
__device__ at::BFloat16 convert_from_float<at::BFloat16>(float value) {
    return static_cast<at::BFloat16>(value);
}

// warp内规约最大值
__inline__ __device__ __half2 warpReduceMax(__half2 val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = __hmax2(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp内规约最小值
__inline__ __device__ __half2 warpReduceMin(__half2 val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = __hmin2(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp内规约最小值
__inline__ __device__ __half2* warpReduceMAX_and_Min(__half2* val) {
    // __half2* convert_val =  reinterpret_cast<__half2*>(val)
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val[0] = __hmax2(val[0], __shfl_down_sync(0xFFFFFFFF, val[0], offset));
        val[1] = __hmin2(val[1], __shfl_down_sync(0xFFFFFFFF, val[1], offset));
    }
    return val;
}

// 每个线程读取各自的数，然后通过warp内的通讯的方式获取最大最小值，然后再对各自取得数进行量化并回填

template<typename T>
__global__ void quantize(
    T*  key_states,
    T*  value_states,
    float*  quant_param,
    // 需要压缩的keystates
    uint8_t* kv_quant,
    int batch_size, int head_size, int len, int emb_dim
    ) {

    size_t batch_id = blockIdx.x;
    size_t head_id = blockIdx.y;
    size_t pro_id = blockIdx.z;
// batch_id和head_id不能*NUM_PER_THREAD*WARPS_PER_BLOCK 来表示位移，因为可能会有越界的线程。
    int base_index = (batch_id * head_size * len * EMB_DIM) 
                   + (head_id * len * EMB_DIM) 
                   + (pro_id * 128*4);
    // warp排序是列主序,获取当前线程在block中的id.
    // int th_id = threadIdx.y*4+threadIdx.x;
    int sub_th_id = threadIdx.x;
    // 首先需要知道当前线程处理哪一个向量, 向量长度除每个线程处理的元素数量，得到当前处理的向量的id
    int vec_id = threadIdx.y;
    // 获取当前线程处理的vec的起始位置
    int base_index_key = base_index+vec_id*128;
    // 判断边界
    if(pro_id * 128*4 + vec_id*128>=len*EMB_DIM){
        return;
    }
    int base_index_quant = base_index_key;
    int extra_offset = 4 * sub_th_id;

    // 使用指针访问key和value的状态
    half key_4[4], value_4[4];
    int2* key_4half = reinterpret_cast<int2*>(key_states + base_index_key + extra_offset);
    int2* value_4half = reinterpret_cast<int2*>(value_states + base_index_key + extra_offset);

    // 直接一次性读取4个元素，减少内存访问
    reinterpret_cast<int2*>(key_4)[0] = key_4half[0];
    reinterpret_cast<int2*>(value_4)[0] = value_4half[0];

    // 使用Warp内最大最小值操作
    __half key_local_max = __hmax(__hmax(key_4[0], key_4[1]), __hmax(key_4[2], key_4[3]));
    __half key_local_min = __hmin(__hmin(key_4[0], key_4[1]), __hmin(key_4[2], key_4[3]));
    __half value_local_max = __hmax(__hmax(value_4[0], value_4[1]), __hmax(value_4[2], value_4[3]));
    __half value_local_min = __hmin(__hmin(value_4[0], value_4[1]), __hmin(value_4[2], value_4[3]));

    // 在Warp内找到全局最大值和最小值
    // __half2 kv_local_max = __halves2half2(key_local_max, value_local_max);
    // __half2 kv_local_min = __halves2half2(key_local_min, value_local_min);
    __half2 kv_local_max_and_min[2];
    kv_local_max_and_min[0] = __halves2half2(key_local_max, value_local_max);
    kv_local_max_and_min[1] = __halves2half2(key_local_min, value_local_min);
    warpReduceMAX_and_Min(kv_local_max_and_min);
    // __half2 kv_warp_max = warpReduceMax(kv_local_max_and_min[0]);
    // __half2 kv_warp_min = warpReduceMin(kv_local_max_and_min[1]);
    __half2 kv_warp_max = kv_local_max_and_min[0];
    __half2 kv_warp_min =kv_local_max_and_min[1];
    // 广播最大值和最小值到所有线程
    kv_warp_max = __shfl_sync(0xFFFFFFFF, kv_warp_max, 0);
    kv_warp_min = __shfl_sync(0xFFFFFFFF, kv_warp_min, 0);

    // 提取key和value的Warp级别最大最小值
    __half key_warp_max = __low2half(kv_warp_max);
    __half value_warp_max = __high2half(kv_warp_max);
    __half key_warp_min = __low2half(kv_warp_min);
    __half value_warp_min = __high2half(kv_warp_min);

    // 计算量化尺度和零点
    float key_scale = (__half2float(key_warp_max) - __half2float(key_warp_min)) / 15.0f;
    float key_zp = roundf(-__half2float(key_warp_min) / key_scale);
    float value_scale = (__half2float(value_warp_max) - __half2float(value_warp_min)) / 15.0f;
    float value_zp = roundf(-__half2float(value_warp_min) / value_scale);

    uint8_t quant_result[4];

    // 展开循环，提高效率
    #pragma unroll
    for(int i = 0; i < 4; i++){
        float quantized_key = (__half2float(key_4[i]) / key_scale) + key_zp;
        quantized_key = fminf(fmaxf(roundf(quantized_key), 0.0f), 15.0f);
        float quantized_value = (__half2float(value_4[i]) / value_scale) + value_zp;
        quantized_value = fminf(fmaxf(roundf(quantized_value), 0.0f), 15.0f);
        quant_result[i] = (convert_from_float<uint8_t>(quantized_key) << 4) |
                        (convert_from_float<uint8_t>(roundf(quantized_value)));
    }

    // 将结果存储到全局内存中
    int* output_point = reinterpret_cast<int*>(kv_quant + base_index_key + extra_offset);
    output_point[0] = reinterpret_cast<int*>(quant_result)[0];

    // 仅在主线程更新量化参数
    if(sub_th_id == 0){
        float4 quant_param_local;
        quant_param_local.x = key_zp;
        quant_param_local.y = key_scale;
        quant_param_local.z = value_zp;
        quant_param_local.w = value_scale;
        float4* quant_param_point = reinterpret_cast<float4*>(quant_param + base_index_key / 32);
        quant_param_point[0] = quant_param_local;
    }
    return;
}


torch::TensorOptions getOptionsForType(const std::type_info& typeInfo) {
    if (typeInfo == typeid(c10::Half)) {
        return torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kHalf);
    } else if (typeInfo == typeid(float)) {
        return torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    } else if (typeInfo == typeid(at::BFloat16)) {
        return torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBFloat16);
    } else {
        // Default case for unexpected types
        throw std::runtime_error("Unsupported type for tensor options.");
    }
}

template <typename T>
std::tuple<torch::Tensor, torch::Tensor> MyResduialQuantCudaTemplate(
    torch::Tensor key,
    torch::Tensor value,
    int bit
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_outlier_norm = getOptionsForType(typeid(float));

    int batch = key.size(0);
    int head = key.size(1);
    int len = key.size(2);

    // token的emb_dim一般为128
    int emb_dim = key.size(3);

    // 获取 key_states 的设备信息
    auto device = key.device();

    auto key_value_quant = torch::zeros({batch, head, len, emb_dim}, options.device(device)).contiguous();
    // auto outlier_quant = torch::zeros({batch, head, len, outlier_num*4}, options.device(device)).contiguous();
    auto quant_param = torch::zeros({batch, head,len,4}, options_outlier_norm.device(device)).contiguous();
    int numProjBlocks = (len*emb_dim+(128*4)-1) / (128*4);
    dim3 numBlocks(batch , head, numProjBlocks);
    dim3 threadsPerBlockDim(32,4);



//     Compiler hints for using L2 Persistent Cache
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);                                                                  // Create CUDA stream
    // int device_id{0};
    // cudaGetDevice(&device_id);                                                                  // Device ID


    quantize<<<numBlocks, threadsPerBlockDim, 0>>>(
    key.data_ptr<T>(),
    value.data_ptr<T>(),
    quant_param.data_ptr<float>(),
    key_value_quant.data_ptr<uint8_t>(),
    batch,head,len,
    emb_dim);
                                                         // Remove any persistent lines in L2

    return std::make_tuple(quant_param,key_value_quant);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("My_quant_resduial_half_half", &MyResduialQuantCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("key"),
    py::arg("value"),
    py::arg("bit"));

}
