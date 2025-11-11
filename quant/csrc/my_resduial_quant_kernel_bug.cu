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
    // 采取uint8作为量化格式，kv采用4bit量化，接在一起保存于一个uint8中，因此偏移量一样。
    int base_index_quant = base_index_key;
    // 每个线程处理4个元素，添加额外的偏移量
    int extra_offset = 4*sub_th_id;
    // T* key = key_states + base_index_key + extra_offset;
    half key_4[4];
    int2* key_4half = reinterpret_cast<int2*>(key_states + base_index_key + extra_offset);
    reinterpret_cast<int2*>(key_4)[0] = key_4half[0];

    __half key_local_max = __hmax(__hmax(key_4[0], key_4[1]), __hmax(key_4[2], key_4[3]));
    __half Key_local_min = __hmin(__hmin(key_4[0], key_4[1]), __hmin(key_4[2], key_4[3]));
    // T* value = value_states + base_index_key + extra_offset;
    int2* value_4half = reinterpret_cast<int2*>(value_states + base_index_key + extra_offset);
    half value_4[4];
    reinterpret_cast<int2*>(value_4)[0] = value_4half[0];
    __half value_local_max = __hmax(__hmax(value_4[0], value_4[1]), __hmax(value_4[2], value_4[3]));
    __half value_local_min = __hmin(__hmin(value_4[0], value_4[1]), __hmin(value_4[2], value_4[3]));
    // 在warp内找到全局最大值和最小值
    __half2 kv_local_max = __halves2half2(key_local_max, value_local_max);
    __half2 kv_local_min = __halves2half2(key_local_min, value_local_min);
    __half2 kv_warp_max = warpReduceMax(kv_local_max);
    __half2 kv_warp_min = warpReduceMin(kv_local_min);
    // half key_warp_max = warpReduceMax(key_local_max);
    // half key_warp_min = warpReduceMin(Key_local_min);
    // half value_warp_max = warpReduceMax(value_local_max);
    // half value_warp_min = warpReduceMin(value_local_min);
    // 广播最大值和最小值到所有线程
    kv_warp_max = __shfl_sync(0xFFFFFFFF, kv_warp_max, 0);
    kv_warp_min = __shfl_sync(0xFFFFFFFF, kv_warp_min, 0);
    // __half low = __low2half(h2_val);   // 1.0
    // __half high = __high2half(h2_val); // 2.0
    __half key_warp_max = __low2half(kv_warp_max);
    __half value_warp_max = __high2half(kv_warp_max);
    __half key_warp_min = __low2half(kv_warp_min);
    __half value_warp_min = __high2half(kv_warp_min);
    // value_warp_max = __shfl_sync(0xFFFFFFFF, value_warp_max, 0);
    // value_warp_min = __shfl_sync(0xFFFFFFFF, value_warp_min, 0);
    printf("base_index_quant:%d\n",base_index_quant);
    // scales = (max_vals - min_vals) / (qmax - qmin)
    // zero_points = qmin - min_vals / scales
    // zero_points = zero_points.round()
    // # 进行量化
    // quantized_tensor = ((tensor / scales) + zero_points).round().clamp(qmin, qmax)
    float key_scale =  (__half2float(key_warp_max) - __half2float(key_warp_min)) /15.0f;
    float key_zp = roundf(-__half2float(key_warp_min)/key_scale);
    float value_scale =  (__half2float(value_warp_max) - __half2float(value_warp_min)) /15.0f;
    float value_zp = roundf(-__half2float(value_warp_min)/value_scale);
    // zero_points_slip_K.view(-1,1),scales_slip_K.view(-1,1),zero_points_slip_V.view(-1,1),scales_slip_V
    if(sub_th_id==0){
        float4 quant_param_loacl;
        quant_param_loacl.x = key_zp;
        quant_param_loacl.y = key_scale;
        quant_param_loacl.z = value_zp;
        quant_param_loacl.w = value_scale;
        // float key_scale =  (__half2float(key_warp_max) - __half2float(key_warp_min)) /15.0f;
        printf("%f,%f,%f,%f,key_warp_min:%f,key_warp_max:%f\n",key_zp,key_scale,value_zp,value_scale,__half2float(key_warp_min),__half2float(key_warp_max));
        // 每个vec有一套量化参数
        float4* quant_param_point = reinterpret_cast<float4*>(quant_param + base_index_key/32);
        // printf("%f,%f,%f,%f,key_warp_min:%f,key_warp_max:%f\n,base_index_key:%d",key_zp,key_scale,value_zp,value_scale,__half2float(key_warp_min),__half2float(key_warp_max),base_index_key);
        quant_param_point[0] = quant_param_loacl;
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
    auto quant_param = torch::zeros({batch, head,len*4}, options_outlier_norm.device(device)).contiguous();
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
