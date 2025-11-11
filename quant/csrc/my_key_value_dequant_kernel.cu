#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>

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



template<typename T>
__global__ void quantize_with_outliers_kernel(
    uint8_t*  compressed_key,
    // 压缩好的1bit key
    uint8_t*  key_outlier_quant,
    // 压缩好的outlier
    T*  dequant_dst,
    // channel_wise的平均值
    T* channel_mean,
    T* quant_outlier_zp,
    // outlier量化所需要的zeropoint的
    T* quant_outlier_scale, 
    // outlier的量化的scale
    uint8_t* outlier_idx,
    // outlier的索引
    int outlier_num,
    // outlier channel的数目
    int batch_size, int head_size, int len,int buffer_len
    ) {
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int pro_id = blockIdx.z;
// batch_id和head_id不能*NUM_PER_THREAD*WARPS_PER_BLOCK 来表示位移，因为可能会有越界的线程。
    int base_index = (batch_id * head_size * len * EMB_DIM) 
                   + (head_id * len * EMB_DIM) 
                   + (pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK);
    // warp排序是列主序,获取当前线程在block中的id.
    __shared__ half share_dequant_key[8*EMB_DIM];
    size_t th_id = threadIdx.y*16+threadIdx.x;
    int sub_th_id = threadIdx.x;
    // 首先需要知道当前线程处理哪一个向量, 向量长度除每个线程处理的元素数量，得到当前处理的向量的id
    int thread_num_per_emb = EMB_DIM/NUM_PER_THREAD;
    int vec_id = threadIdx.y;
    // 获取当前线程处理的vec的起始位置
    int base_index_key = base_index+vec_id*128;
    // 判断边界 numProjBlocks个block要处理len*EMBDIM个元素
    int dequant_shfit = (batch_id * head_size * buffer_len * EMB_DIM) 
                + (head_id * buffer_len * EMB_DIM) 
                + (pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK) + th_id*8;
    int cnt_head_id = 1*head_id;
    int cnt_proid = 1*pro_id;
    if(pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK + vec_id*128>=len*EMB_DIM){
        return;
    }
    // printf("head:%d,pro_id:%d,dequant_shfit:%d\n",cnt_head_id,cnt_proid,dequant_shfit);
    // 获取当前线程计算的偏移量，每个线程计算NUM_PER_THREAD个元素，每thread_num_per_emb个线程计算计算一个
    // 每个线程处理NUM_PER_THREAD个元素，也就是每个线程读取一个uint8_t

    uint8_t onebit_key = compressed_key[base_index/8+th_id];
    T* sub_dequant_dst = dequant_dst+dequant_shfit;
    int4* output_as_int4 = reinterpret_cast<int4*>(sub_dequant_dst);
    // 采用向量化的写入，一个int4可以存8个half
    int4 key_data;
    half* key_half_data = reinterpret_cast<half*>(&key_data);
    // 每个head有emb_dim 个量化参数，之后会选取其中的outlier_num个
    int base_index_quant_param = batch_id * head_size *EMB_DIM + head_id*EMB_DIM;
    int extra_offset = NUM_PER_THREAD*sub_th_id;
    T* channel_mean_point = channel_mean+base_index_quant_param+extra_offset;
    const int4* mean_input_as_int4 = reinterpret_cast<const int4*>(channel_mean_point);
    int4 mean_data = mean_input_as_int4[0];
    half* mean_half_data = reinterpret_cast<half*>(&mean_data);
    // printf("base_index:%d,vec_id:%d,dequant_shfit:%d\n",base_index,vec_id,dequant_shfit);
    #pragma unroll
    for(int i=0;i<8;++i){
        float sub_mean = __half2float(mean_half_data[i]);
        share_dequant_key[th_id*8+i] =  __float2half((onebit_key>> i)&1 ? sub_mean:-sub_mean);
    }
    // output_as_int4[0] = key_data;
// ---------------------------------
    __syncthreads();
    // 处理outlier，每个线程处理一个outlier，首先需要计算读取的数据的位置
    if(sub_th_id<outlier_num){
        // 首先获取outlier key的指针位置， 每个vec有outlier个outlier base_index/128*outlier_num  +  vec_id*outlier_num
        int shift1 = base_index/128*outlier_num  +  vec_id*outlier_num + sub_th_id;
        // printf("vec_id:%d,sub_th_id:%d,shift1:%d\n",vec_id,sub_th_id,shift1);
        float sub_outlier_key = (float)(key_outlier_quant[shift1]);
        // 获取量化参数的指针,每个head有一个对应的outlier_num维的量化参数
        int shift2 = batch_id*gridDim.y*outlier_num + head_id*outlier_num +sub_th_id;
        int sub_outlier_idx = static_cast<int>(outlier_idx[shift2]);
        float zp = __half2float(quant_outlier_zp[shift2]);
        float scale = __half2float(quant_outlier_scale[shift2]);
        int query_shfit = batch_id*gridDim.y*EMB_DIM + head_id*EMB_DIM;
        float value = (sub_outlier_key-zp)*scale;
        share_dequant_key[vec_id*EMB_DIM+sub_outlier_idx] = __float2half(value);
    }
    __syncthreads();
    int4* true_output = reinterpret_cast<int4*>(share_dequant_key+th_id*8);
    output_as_int4[0] = true_output[0];
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
    // uint8_t*  compressed_key,
    // // 压缩好的1bit key
    // uint8_t*  key_outlier_quant,
    // // 压缩好的outlier
    // T*  dequant_dst,
    // // channel_wise的平均值
    // T* channel_mean,
    // T* quant_outlier_zp,
    // // outlier量化所需要的zeropoint的
    // T* quant_outlier_scale, 
    // // outlier的量化的scale
    // uint8_t* outlier_idx,
    // // outlier的索引
    // int outlier_num,
    // // outlier channel的数目
    // int batch_size, int head_size, int len
template <typename T>
torch::Tensor MyKeyDeQuantCudaTemplate(
    torch::Tensor compressed_key,
    torch::Tensor key_outlier_quant,
    torch::Tensor dequant_dst,
    torch::Tensor channel_mean,
    torch::Tensor outlier_idx,
    torch::Tensor quant_outlier_zp,
    torch::Tensor quant_outlier_scale,
    int outlier_num
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_outlier_norm = getOptionsForType(typeid(T));

    int batch = compressed_key.size(0);
    int head = compressed_key.size(1);
    int len = compressed_key.size(2);
    int buffer_len = dequant_dst.size(2);
    // warp_size 表示一个warp中的线程数，一般为32  WARPS_PER_BLOCK 表示一个block中warp的数量， 一个block最大的线程数是1024，因此WARPS_PER_BLOCK的最大值为32
    // 每个线程处理NUM_PER_THREAD个元素
    int numProjBlocks = (len*EMB_DIM+(NUM_PER_THREAD*WARPS_PER_BLOCK)-1) / (NUM_PER_THREAD*WARPS_PER_BLOCK);
    dim3 numBlocks(batch , head, numProjBlocks);
    dim3 threadsPerBlockDim(16,8);

    // auto key_states_ptr = key_states.data_ptr<T>();


//     Compiler hints for using L2 Persistent Cache
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);                                                                  // Create CUDA stream
    // int device_id{0};
    // cudaGetDevice(&device_id);                                                                  // Device ID


    quantize_with_outliers_kernel<<<numBlocks, threadsPerBlockDim, 0>>>(
    compressed_key.data_ptr<uint8_t>(),
    key_outlier_quant.data_ptr<uint8_t>(),
    dequant_dst.data_ptr<T>(),
    channel_mean.data_ptr<T>(),
    quant_outlier_zp.data_ptr<T>(),
    quant_outlier_scale.data_ptr<T>(),
    outlier_idx.data_ptr<uint8_t>(),
    outlier_num,batch,head,len,buffer_len);
                                                         // Remove any persistent lines in L2

    return dequant_dst;
}

    // torch::Tensor compressed_key,
    // torch::Tensor key_outlier_quant,
    // torch::Tensor dequant_dst,
    // torch::Tensor channel_mean,
    // torch::Tensor outlier_idx,
    // torch::Tensor quant_outlier_zp,
    // torch::Tensor quant_outlier_scale,
    // int outlier_num
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("My_key_dequant_half_half", &MyKeyDeQuantCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("compressed_key"),
    py::arg("key_outlier_quant"),
    py::arg("dequant_dst"),
    py::arg("channel_mean"),
    py::arg("outlier_idx"),
    py::arg("quant_outlier_zp"),
    py::arg("quant_outlier_scale"),
    py::arg("outlier_num"));

}
