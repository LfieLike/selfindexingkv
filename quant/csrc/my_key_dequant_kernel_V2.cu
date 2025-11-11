#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>

#define WARP_SIZE 32
#define EMB_DIM 128
#define THREAD_NUM 512
#define NUMBER_PER_THREAD 8
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
    uint8_t*  key_states_1bit,
    // 压缩好的1bit key
    uint8_t*  key_states_2bit,
    T* dequant_dst,
    T* zp,
    T* scale,
    int64_t* index,int batch_size, int head_size, int len,int len_1bit, int group_num
    ) {

    size_t batch_id = blockIdx.x;
    size_t head_id = blockIdx.y;
    size_t group_id = blockIdx.z;
// batch_id和head_id不能*NUM_PER_THREAD*WARPS_PER_BLOCK 来表示位移，因为可能会有越界的线程。
    int base_index = (batch_id * head_size * len * EMB_DIM) 
                   + (head_id * len * EMB_DIM) 
                   + (group_id * THREAD_NUM*NUMBER_PER_THREAD);
                //    每个group负责128个 128维的向量的压缩
    // warp排序是列主序,获取当前线程在block中的id.
    size_t th_id = threadIdx.x;
    // 首先需要知道当前线程处理哪一个向量, 每16个线程处理一个向量
    int vec_id = th_id/16;
    // 获取当前线程处理的vec的起始位置
    int base_index_key = base_index+th_id*NUMBER_PER_THREAD;
    // 判断边界,因为是写死的，所以这里不需要判断
    if(group_id * THREAD_NUM*NUMBER_PER_THREAD + vec_id*128>=len*EMB_DIM){
        return;
    }
    // 获取量化参数，首先需要读取index
    int64_t now_index = index[batch_id* head_size * len+head_id * len+group_id*32+vec_id];
    // printf("offset:%lld\n",now_index);
    // 再根据index读取 1bitmask，还有zp scale
    int offset_1bit = ((th_id*NUMBER_PER_THREAD)%EMB_DIM+(batch_id * head_size * len_1bit * EMB_DIM+head_id * len_1bit * EMB_DIM + now_index*EMB_DIM))/8;
    uint8_t* key_states_1bit_ptr = key_states_1bit + offset_1bit;
    uint8_t now_key_states_1bit = key_states_1bit_ptr[0];
    // zp 和 scale 是 每32个向量共享的
    int len_zp = len_1bit/32;
    int index_zp = now_index/32;
    int quant_param_offset = (batch_id * head_size * len_zp * EMB_DIM+ head_id * len_zp * EMB_DIM) + index_zp*EMB_DIM+(th_id*NUMBER_PER_THREAD)%EMB_DIM;
    // printf("quant_param_offset:%d,head_size:%d,len_zp:%d,index_zp:%d\n",quant_param_offset,head_size,len_zp,index_zp);
    T* zp_ptr = zp + quant_param_offset;
    T* scale_ptr = scale + quant_param_offset;
    // 最后是2bit的偏移量 
    uint8_t* key_states_2bit_ptr = key_states_2bit +  (base_index_key)/4;
    uint8_t now_key_states_2bit[2];
    now_key_states_2bit[0] = key_states_2bit_ptr[0];
    now_key_states_2bit[1] = key_states_2bit_ptr[1];
    float dequant_result[8];
    #pragma unroll
    for(int i=0;i<4;++i){
        float now_zp = __half2float(zp_ptr[i]);
        float now_scale = __half2float(scale_ptr[i]);
        float quant_res = static_cast<float>((now_key_states_2bit[0]>>(i*2))&(0b11));
        dequant_result[i] = ((now_key_states_1bit>>i)&1) ? (quant_res*now_scale+now_zp) : -(quant_res*now_scale+now_zp);
    }

    #pragma unroll
    for(int i=0;i<4;++i){
        float now_zp = __half2float(zp_ptr[i+4]);
        float now_scale = __half2float(scale_ptr[i+4]);
        float quant_res = static_cast<float>((now_key_states_2bit[1]>>(i*2))&(0b11));
        dequant_result[i+4] = ((now_key_states_1bit>>(i+4))&1) ? (quant_res*now_scale+now_zp) : -(quant_res*now_scale+now_zp);
    }
    #pragma unroll
    for(int i=0;i<8;++i){
        dequant_dst[base_index_key+i] = __float2half(dequant_result[i]);
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
// 根据提取出来的2bit进行解压
template <typename T>
void MyKeyDeQuantCudaTemplate(
    torch::Tensor key_states_1bit,
    torch::Tensor key_states_2bit,
    torch::Tensor dequant_dst,
    torch::Tensor zp,
    torch::Tensor scale,
    torch::Tensor index,
    int group_size
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_outlier_norm = getOptionsForType(typeid(T));

    int batch = key_states_2bit.size(0);
    int head = key_states_2bit.size(1);
    int len = key_states_2bit.size(2);
    int len_1bit = key_states_1bit.size(2);
    int group_num = (len+31)/32;
    // 每个block 最多处理32个向量
    // 获取 key_states 的设备信息
    auto device = key_states_2bit.device();
    dim3 numBlocks(batch , head, group_num);
    dim3 threadsPerBlockDim(512);


    quantize_with_outliers_kernel<<<numBlocks, threadsPerBlockDim, 0>>>(
    key_states_1bit.data_ptr<uint8_t>(),
    key_states_2bit.data_ptr<uint8_t>(),
    dequant_dst.data_ptr<T>(),
    zp.data_ptr<T>(),
    scale.data_ptr<T>(),
    index.data_ptr<int64_t>(),
    batch,head,len,len_1bit,group_num);

    return;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("My_key_dequant_V2", &MyKeyDeQuantCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("key_states_1bit"),
    py::arg("key_states_2bit"),
    py::arg("dequant_dst"),
    py::arg("zp"),
    py::arg("scale"),
    py::arg("index"),
    py::arg("group_size"));
}
