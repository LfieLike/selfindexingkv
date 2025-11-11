#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>

#define WARP_SIZE 16
#define WARPS_PER_BLOCK 256
#define EMB_DIM 128
#define NUM_PER_THREAD 8
#define THREAD_NUM 1024
// warp内规约最大值
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 8; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset,WARP_SIZE));
    }
    return val;
}

// warp内规约最小值
__inline__ __device__ float warpReduceMin(float val) {
    for (int offset = 8; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset,WARP_SIZE));
    }
    return val;
}
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



template<typename T>
__global__ void quantize_with_outliers_kernel(
    T*  key_states,
    T*  value_states,
    // 需要压缩的keystates
    int32_t* __restrict__ key_value_quant,
    uint8_t* __restrict__ key_1bit_quant,
    T* zp_key,
    T* scale_key,
    T* zp_value,
    T* scale_value,
    int64_t* select_index,
    int batch_size, int head_size, int key_len,int len
    ) {
    // key_states.data_ptr<T>(),
    // value_states.data_ptr<T>(),
    // key_value_quant.data_ptr<uint8_t>(),
    // key_1bit_quant.data_ptr<uint8_t>(),
    // maxvalue.data_ptr<T>(),
    // zp_key.data_ptr<T>(),
    // scale_key.data_ptr<T>(),
    // zp_value.data_ptr<T>(),
    // scale_value.data_ptr<T>(),
    // batch,head,len
    size_t batch_id = blockIdx.x;
    size_t head_id = blockIdx.y;
    size_t pro_id = blockIdx.z;
// batch_id和head_id不能*NUM_PER_THREAD*WARPS_PER_BLOCK 来表示位移，因为可能会有越界的线程。
    int base_index = (batch_id * head_size * len * EMB_DIM) 
                   + (head_id * len * EMB_DIM) 
                   + (pro_id * NUM_PER_THREAD*THREAD_NUM);
    size_t th_id = threadIdx.x;
    // 首先需要知道当前线程处理的数据的偏移量，所以先要读取对应的index
    int index_offset = batch_id* head_size * len+head_id * len+pro_id*64+threadIdx.x/16;
    int64_t local_index = select_index[index_offset];
    int vec_id = th_id*NUM_PER_THREAD;
    // 获取当前线程处理的vec的起始位置,*16 表示 128/8
    int base_index_key = (batch_id * head_size * key_len + head_id * key_len  + local_index)*16 + threadIdx.x%128;
    // 判断边界
    if(pro_id * NUM_PER_THREAD*THREAD_NUM + vec_id>=len*EMB_DIM){
        return;
    }
    // 采取uint8作为量化格式，所以量化指针除8
    int quant_offset_1bit = base_index_key;
    int quant_offset_2bit = base_index_key;
    // 每个线程处理NUM_PER_THREAD个元素，添加额外的偏移量
    // int max_value_base_index = (batch_id * head_size * EMB_DIM) + (head_id * EMB_DIM);
    // T* value = value_states + base_index_key/4;
    int32_t local_key_value_2bit = key_value_quant[quant_offset_2bit];
    int32_t local_key_1bit = key_1bit_quant[quant_offset_1bit];
    int quant_param_offset = (base_index_key)/4;
    float local_zp_key = __half2float(zp_key[quant_param_offset]);
    float local_scale_key = __half2float(scale_key[quant_param_offset]);
    float local_zp_value = __half2float(zp_value[quant_param_offset]);
    float local_scale_value = __half2float(scale_value[quant_param_offset]);
    half value_half_data[8];
    half key_half_data[8];

    #pragma unroll
    for(int i=0;i<8;i++){
        float quant_res = static_cast<float>(((local_key_value_2bit)>>(i*2))&(0b11));
        quant_res = ((local_key_1bit>>i)&1) ? (quant_res*local_scale_key+local_zp_key) : -(quant_res*local_scale_key+local_zp_key);
        key_half_data[i] = __float2half(quant_res);


        float value_quant_res = static_cast<float>(((local_key_value_2bit)>>((i+8)*2))&(0b11));
        value_quant_res = (value_quant_res*local_scale_value+local_zp_value);
        value_half_data[i] = __float2half(value_quant_res);
        float now_value = __half2float(value_half_data[i]);
        float quantized_value = (now_value - local_zp_value)/ local_scale_value;
    }
    int write_offset = base_index + threadIdx.x*8;
    reinterpret_cast<int4*>(value_states+base_index_key*8)[0] = reinterpret_cast<int4*>(value_half_data)[0];
    reinterpret_cast<int4*>(key_states+base_index_key*8)[0] = reinterpret_cast<int4*>(key_half_data)[0];
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
std::tuple<torch::Tensor, torch::Tensor> MyQuantCudaTemplate(    
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor key_value_quant,
    torch::Tensor key_1bit_quant,
    torch::Tensor zp_key,
    torch::Tensor scale_key,
    torch::Tensor zp_value,
    torch::Tensor scale_value,
    torch::Tensor select_index
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_outlier_norm = getOptionsForType(typeid(T));
    auto options_int = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt32);
    int batch = key_value_quant.size(0);
    int head = key_value_quant.size(1);
    int len = select_index.size(2);
    int key_len = key_value_quant.size(2);
    // token的emb_dim一般为128

    int dim_1bit = key_1bit_quant.size(3);
    int dim_2bit = key_1bit_quant.size(3);
    // 获取 value_states 的设备信息
    auto device = value_states.device();
    // auto key_1bit_quant = torch::zeros({batch, head, len, dim_1bit}, options.device(device)).contiguous();
    // auto key_value_quant = torch::zeros({batch, head, len, dim_2bit}, options_int.device(device)).contiguous();
    // warp_size 表示一个warp中的线程数，一般为32  WARPS_PER_BLOCK 表示一个block中warp的数量， 一个block最大的线程数是1024，因此WARPS_PER_BLOCK的最大值为32
    // 每个线程处理NUM_PER_THREAD个元素
    int numProjBlocks = (len*EMB_DIM+(NUM_PER_THREAD*THREAD_NUM)-1) / (NUM_PER_THREAD*THREAD_NUM);
    dim3 numBlocks(batch , head, numProjBlocks);
    dim3 threadsPerBlockDim(THREAD_NUM);


    quantize_with_outliers_kernel<<<numBlocks, threadsPerBlockDim, 0>>>(
    key_states.data_ptr<T>(),
    value_states.data_ptr<T>(),
    key_value_quant.data_ptr<int32_t>(),
    key_1bit_quant.data_ptr<uint8_t>(),
    zp_key.data_ptr<T>(),
    scale_key.data_ptr<T>(),
    zp_value.data_ptr<T>(),
    scale_value.data_ptr<T>(),
    select_index.data_ptr<int64_t>(),
    batch,head,key_len,len);
                                                         // Remove any persistent lines in L2

    return std::make_tuple(key_value_quant,key_1bit_quant);
}

    // torch::Tensor key_states,
    // torch::Tensor value_states,
    // torch::Tensor zp_key,
    // torch::Tensor scale_key,
    // torch::Tensor zp_value,
    // torch::Tensor scale_value
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("My_quant_half_half", &MyQuantCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("key_states"),
    py::arg("value_states"),
    py::arg("key_value_quant"),
    py::arg("key_1bit_quant"),
    py::arg("zp_key"),
    py::arg("scale_key"),
    py::arg("zp_value"),
    py::arg("scale_value"),
    py::arg("select_index"));

}
