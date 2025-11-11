#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>
#define NUM_PER_THREAD 512
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
    // 1bit 压缩的keystates，8个一组
    uint8_t* key_states,
    // query向量
    const float* query_lut,
    // 最终结果
    c10::Half* res,
    // outlier channel的数目
    int lut_size,int batch_size,int head_size,int len
    ) {
    size_t batch_id = blockIdx.x;
    size_t head_id = blockIdx.y;
    size_t pro_id = blockIdx.z;

    // batch_id和head_id不能*NUM_PER_THREAD*WARPS_PER_BLOCK 来表示位移，因为可能会有越界的线程。
    int base_index = (batch_id * head_size * len * EMB_DIM) 
                   + (head_id * len * EMB_DIM) 
                   + (pro_id * NUM_PER_THREAD*EMB_DIM);
    // warp排序是列主序,获取当前线程在block中的id.
    size_t th_id = threadIdx.x;
    int sub_th_id = threadIdx.x;
    int vec_id = threadIdx.x;
    // 获取当前线程处理的vec的起始位置
    int base_index_key = base_index+vec_id*128;
    // 判断边界 numProjBlocks个block要处理len*EMBDIM个元素

    int lut_offset = batch_id*head_size*lut_size + head_id*lut_size;
    __shared__ float sharedlut[512];
    sharedlut[th_id]=__ldg(&query_lut[lut_offset+th_id]);
    __syncthreads();
    // printf("th_id:%d sharedlut:%f\n",vec_id,sharedlut[th_id]);
    // if(th_id<)
    // 获取当前线程计算的偏移量，每个线程计算NUM_PER_THREAD个元素，每thread_num_per_emb个线程计算计算一个
    // 每个线程处理NUM_PER_THREAD个元素，也就是每个线程读取一个uint8_t
    // printf("sharedlut:%f\n",sharedlut[0]);
    float sub_res = 0;
    if(pro_id * NUM_PER_THREAD + vec_id>=len){
        return;
    }
    uint8_t onebit_key_set[16];
    reinterpret_cast<int4 *>(&onebit_key_set[0])[0] = reinterpret_cast<int4 *>(&key_states[base_index_key/8])[0];
    #pragma unroll
    for(int i=0;i<16;i++){
        // int pos = (th_id+i)%16;
        uint8_t onebit_key = onebit_key_set[i];
        sub_res += sharedlut[32*i+(onebit_key&15)]+sharedlut[32*i+16+(onebit_key>>4)];
        // printf("res = %f, sharedlut[%d],sharedlut[%d]:%f %f\n",sub_res,int(onebit_key&15),int(onebit_key>>4),sharedlut[32*i+(onebit_key&15)],sharedlut[32*i+16+(onebit_key>>4)]);
    }
    res[(batch_id*head_size+head_id)*len+pro_id*512+vec_id] = __float2half(sub_res);
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

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_uint32 = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt32);
    auto options_outlier_norm = getOptionsForType(typeid(T));

    int batch = key_states.size(0);
    int head = key_states.size(1);
    int len = key_states.size(2);
    int lut_size = query_states.size(-1);

    auto device = key_states.device();

    // 使用 key_states 的设备信息来设置 res 的设备
    auto res = torch::zeros({batch, head, len}, options_outlier_norm.device(device)).contiguous();
    // warp_size 表示一个warp中的线程数，一般为32  WARPS_PER_BLOCK 表示一个block中warp的数量， 一个block最大的线程数是1024，因此WARPS_PER_BLOCK的最大值为32
    // 每个线程处理8个元素，每16个线程处理一个向量的点积,一个warp处理两个向量的点积
    int numProjBlocks = (len+511) / (512);
    dim3 numBlocks(batch , head, numProjBlocks);
    dim3 threadsPerBlockDim(512);



//     Compiler hints for using L2 Persistent Cache
    quantize_with_outliers_kernel<c10::Half><<<numBlocks, threadsPerBlockDim, 0>>>(
    key_states.data_ptr<uint8_t>(),
    query_states.data_ptr<float>(),
    res.data_ptr<c10::Half>(),
    lut_size,batch,head,len);
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
    m.def("lut_gemv_half_half", &MyScoreCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("key_states"),
    py::arg("query_states"));
}
