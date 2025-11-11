#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>
#define NUM_PER_THREAD 8
#define THREAD_NUM 1024
#define WARPS_PER_BLOCK 128
#define EMB_DIM 8
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
    // key cache
    c10::Half* key_states,
    // query向量
    c10::Half* query_states,
    // 最终结果
    c10::Half* res,
    // outlier channel的数目
    int batch_size, int head_size, int len
    ) {
    size_t batch_id = blockIdx.x;
    size_t head_id = blockIdx.y;
    size_t pro_id = blockIdx.z;

    int base_index = (batch_id * head_size * len * EMB_DIM) 
                   + (head_id * len * EMB_DIM) 
                   + (pro_id * 8*THREAD_NUM);
    // warp排序是列主序,获取当前线程在block中的id. 每个线程计算8个数
    int th_id = threadIdx.y*16+threadIdx.x;
    // 获取当前线程处理的vec的起始位置
    int base_index_key = base_index+th_id*8;
    // 判断当前线程是否越界
    if(pro_id * 8*THREAD_NUM + th_id*8>=len*EMB_DIM){
        return;
    }
    // 获取当前线程计算的偏移量，每个线程计算NUM_PER_THREAD个元素，每thread_num_per_emb个线程计算计算一个
    // 每个线程处理NUM_PER_THREAD个元素，也就是每个线程读取一个uint8_t
    // 获取当前线程处理的query位置，每个head有一个对应的128维的query, 还要加上偏移量，每个线程需要读取8个元素
    
    // printf("vec_id:%d,query_index:%d,base_index:%d\n",vec_id,query_index,base_index);
    __shared__ c10::Half query_data[8];
    c10::Half key_data[8];
    if(th_id==0){
      int query_index = batch_id*head_size*EMB_DIM + head_id*EMB_DIM;
      //将query_data 转成int4指针，读取8个16位浮点
      reinterpret_cast<int4*>(&query_data)[0] = reinterpret_cast<int4*>(query_states+query_index)[0];
    }
    __syncthreads();
    reinterpret_cast<int4*>(&key_data)[0] = reinterpret_cast<int4*>(key_states+base_index_key)[0];

    float sub_res = 0;
    #pragma unroll
    for(int shift_i=0;shift_i<8;++shift_i){
        float query = __half2float(query_data[shift_i]);
        float key = __half2float(key_data[shift_i]);
        sub_res +=  query*key;
    }
    // 
    int base_index_res = (batch_id*head_size+head_id)*len+pro_id*THREAD_NUM+th_id;
    res[base_index_res] = __float2half(sub_res);
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
    torch::Tensor key_states,
    torch::Tensor query_states
    ) {

    auto res_norm = getOptionsForType(typeid(T));

    int batch = key_states.size(0);
    int head = key_states.size(1);
    int len = key_states.size(2);
    

    auto device = key_states.device();

    // 使用 key_states 的设备信息来设置 res 的设备
    auto res = torch::zeros({batch, head, len}, res_norm.device(device)).contiguous();
    // warp_size 表示一个warp中的线程数，一般为32  WARPS_PER_BLOCK 表示一个block中warp的数量， 一个block最大的线程数是1024，因此WARPS_PER_BLOCK的最大值为32
    // 每个线程处理8个元素，每16个线程处理一个向量的点积,一个warp处理两个向量的点积
    int numProjBlocks = (len*EMB_DIM+THREAD_NUM-1) / (THREAD_NUM);
    dim3 numBlocks(batch , head, numProjBlocks);
    dim3 threadsPerBlockDim(THREAD_NUM);



//     Compiler hints for using L2 Persistent Cache
    quantize_with_outliers_kernel<c10::Half><<<numBlocks, threadsPerBlockDim, 0>>>(
    key_states.data_ptr<c10::Half>(),
    query_states.data_ptr<c10::Half>(),
    res.data_ptr<c10::Half>(),
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
    m.def("My_score_8channle_half_half", &MyScoreCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("key_states"),
    py::arg("value_states"));
}

