#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>
#define NUM_PER_THREAD 16
#define WARP_SIZE 256
#define WARPS_PER_BLOCK 128
#define EMB_DIM 128
__inline__ __device__
float reduce16(float sum) {
    // mask全设置为0x0000FFFF 结果也对，不知道为啥，反正结果是对的，那就不管了。
    unsigned int mask = ((threadIdx.y*16+threadIdx.x) % 32 < 16) ? 0x0000FFFF : 0xFFFF0000;

    // 使用 __shfl_down_sync 在 16 个线程范围内进行归约
    sum += __shfl_down_sync(mask, sum, 8);  // 将前 8 个线程的值加到后 8 个线程上
    sum += __shfl_down_sync(mask, sum, 4);  // 将前 4 个线程的值加到后 4 个线程上
    sum += __shfl_down_sync(mask, sum, 2);  // 将前 2 个线程的值加到后 2 个线程上
    sum += __shfl_down_sync(mask, sum, 1);  // 将前 1 个线程的值加到后 1 个线程上

    return sum;  // 最终每组的第 0 号线程保存归约结果
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
    // query向量
    c10::Half* query_states,
    // 最终结果
    float* dst,
    // outlier channel的数目
    int batch_size, int head_size, int len
    ) {
    size_t batch_id = blockIdx.x;
    int base_index = batch_id*256*4;
    // warp排序是列主序,获取当前线程在block中的id.
    int th_id = threadIdx.x*4;
    // 读取query 4个数
    if(base_index+th_id>=batch_size*head_size*128){
        return;
    }
    c10::Half query_half[4];
    reinterpret_cast<float2*>(query_half)[0]=reinterpret_cast<float2*>(query_states+base_index+th_id)[0];
    float query_float[4];
    #pragma unroll
    for(int i=0;i<4;++i){
        query_float[i]= __half2float(query_half[i]);
        // printf("thid:%d,i:%d,query_float:%f\n",th_id,i,query_float[i]);
    }
    float res[16];
    #pragma unroll
    for(int i=0;i<16;++i){
        res[i]= (i&1?query_float[0]:-query_float[0]) +
        (i&2?query_float[1]:-query_float[1]) +
        (i&4?query_float[2]:-query_float[2]) +
        (i&8?query_float[3]:-query_float[3]);
    }
    float4* res_float4 = reinterpret_cast<float4*>(res);
    float4* dst_float4 = reinterpret_cast<float4*>(dst +(base_index+th_id)*4);
    dst_float4[0] = res_float4[0];
    dst_float4[1] = res_float4[1];
    dst_float4[2] = res_float4[2];
    dst_float4[3] = res_float4[3];
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
torch::Tensor MyScoreCudaTemplate(
    torch::Tensor query_states
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_uint32 = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt32);
    auto options_outlier_norm = getOptionsForType(typeid(float));

    int batch = query_states.size(0);
    int head = query_states.size(1);
    int len = query_states.size(2);
    

    auto device = query_states.device();

    // 使用 key_states 的设备信息来设置 res 的设备
    auto res = torch::zeros({batch, head, 512}, options_outlier_norm.device(device)).contiguous();
    // warp_size 表示一个warp中的线程数，一般为32  WARPS_PER_BLOCK 表示一个block中warp的数量， 一个block最大的线程数是1024，因此WARPS_PER_BLOCK的最大值为32
    // 每个线程处理8个元素，每16个线程处理一个向量的点积,一个warp处理两个向量的点积
    int numProjBlocks = (batch*head*len*EMB_DIM+255) / (256);
    dim3 numBlocks(numProjBlocks);
    // 
    dim3 threadsPerBlockDim(256);



//     Compiler hints for using L2 Persistent Cache
    quantize_with_outliers_kernel<c10::Half><<<numBlocks, threadsPerBlockDim, 0>>>(
    query_states.data_ptr<c10::Half>(),
    res.data_ptr<float>(),
    batch,head,len);
                                                         // Remove any persistent lines in L2

    return res;
}
    // torch::Tensor key_states,
    // torch::Tensor outlier_quant,
    // torch::Tensor outlier_idx,
    // torch::Tensor outlier_zp,
    // torch::Tensor outlier_scale,
    // torch::Tensor query_states,
    // int outlier_num
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("My_get_lut_half_float", &MyScoreCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("query_states"));
}
