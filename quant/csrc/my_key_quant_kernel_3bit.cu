#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>

#define WARP_SIZE 32
#define EMB_DIM 128
#define THREAD_NUM 1024
#define NUMBER_PER_THREAD 16
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
    T*  zp,
    T*  scale,
    // 需要压缩的keystates
    uint8_t* __restrict__ key_quant_1bit,
    // 普通的结果
    uint8_t* __restrict__ key_quant_3bit,
    int batch_size, int head_size, int len, int group_num,int emb_dim
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
    // 首先需要知道当前线程处理哪一个向量, 向量长度除每个线程处理的元素数量，得到当前处理的向量的id
    int vec_id = th_id*NUMBER_PER_THREAD;
    int local_shfit = vec_id%EMB_DIM;
    // 获取当前线程处理的vec的起始位置
    int base_index_key = base_index+vec_id;
    // 判断边界,因为是写死的，所以这里不需要判断
    // if(group_id * NUM_PER_THREAD*WARPS_PER_BLOCK + vec_id*128>=len*EMB_DIM){
    //     return;
    // }
    __shared__ float sharedzp[128];
    __shared__ float sharedscale[128];
    // 首先需要知道量化参数的偏移量
    int quant_param_index = (batch_id * head_size * group_num * EMB_DIM) 
                   + (head_id * group_num * EMB_DIM)
                   + group_id * EMB_DIM;
    if (th_id < 128) {
        sharedzp[th_id] = __half2float(zp[quant_param_index+th_id]);  // 将输入数据拷贝到共享内存
        sharedscale[th_id] = __half2float(scale[quant_param_index+th_id]);
    }
    __syncthreads(); 
    // 采取uint8作为量化格式，所以量化指针除8
    int base_index_quant = base_index_key/8;
    // 每个线程处理NUM_PER_THREAD个元素，添加额外的偏移量
    T* key = key_states + base_index_key;

    uint8_t* quant_1bit = key_quant_1bit + base_index_quant;
    uint8_t* quant_3bit = key_quant_3bit + int(base_index_key/8*3);
// 1bit量化

    uint8_t hashed_key = 0;
    uint8_t quant_3bit_key[6]={0,0,0,0,0,0};
    const int4* key_input_as_int4 = reinterpret_cast<const int4*>(key);
    int4 key_data = key_input_as_int4[0];
    half* key_half_data = reinterpret_cast<half*>(&key_data);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float now_key = __half2float(key_half_data[i]);
        hashed_key |= (now_key > 0 ? (1 << i) : 0);

        float quantized_value = ((now_key > 0 ? now_key : -now_key)/ sharedscale[local_shfit+i]) + sharedzp[local_shfit+i];
        quantized_value = fminf(fmaxf(quantized_value, 0.0f), 7.0f);
        if(threadIdx.x==0 && head_id==0){
            printf("quantized_value%f\n",quantized_value);
        }
        uint8_t quanted = convert_from_float<uint8_t>(roundf(quantized_value));
        quant_3bit_key[0] |= ((quanted & 0x01) << i);        // Extract the 1st bit
        quant_3bit_key[1] |= (((quanted >> 1) & 0x01) << i); // Extract the 2nd bit
        quant_3bit_key[2] |= (((quanted >> 2) & 0x01) << i); // Extract the 3rd bit
    }
    quant_1bit[0] = hashed_key;

    hashed_key = 0;
    key_data = key_input_as_int4[1];
    key_half_data = reinterpret_cast<half*>(&key_data);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float now_key = __half2float(key_half_data[i]);
        hashed_key |= (now_key > 0 ? (1 << i) : 0);
        float quantized_value = ((now_key > 0 ? now_key : -now_key)/ sharedscale[local_shfit+i+8]) + sharedzp[local_shfit+i+8];
        quantized_value = fminf(fmaxf(quantized_value, 0.0f), 7.0f);
        uint8_t quanted = convert_from_float<uint8_t>(roundf(quantized_value));
        quant_3bit_key[3] |= ((quanted & 0x01) << i);        // Extract the 1st bit
        quant_3bit_key[4] |= (((quanted >> 1) & 0x01) << i); // Extract the 2nd bit
        quant_3bit_key[5] |= (((quanted >> 2) & 0x01) << i); // Extract the 3rd bit
    }
    quant_1bit[1] = hashed_key;
    #pragma unroll
    for(int i=0;i<6;++i){
        quant_3bit[i]=quant_3bit_key[i];
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
// key 的维度
template <typename T>
std::tuple<torch::Tensor, torch::Tensor> MyQuantCudaTemplate(
    torch::Tensor key_states,
    torch::Tensor zp,
    torch::Tensor scale,
    int group_size
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_outlier_norm = getOptionsForType(typeid(T));

    int batch = key_states.size(0);
    int head = key_states.size(1);
    // 每group_size行在一个block中进行计算
    int len = key_states.size(2);
    int group_num = len/group_size;

    // token的emb_dim一般为128
    int emb_dim = key_states.size(3);

    int hash_dim = emb_dim / 8;
    // 8个数分别压缩成3bit，一起存成 3个8bit的数，使用共享内存
    int compress_dim = emb_dim/8 * 3;
    // 获取 key_states 的设备信息
    auto device = key_states.device();

    auto key_sign = torch::zeros({batch, head, len, hash_dim}, options.device(device)).contiguous();
    auto key_3bit = torch::zeros({batch, head, len, compress_dim}, options.device(device)).contiguous();
    dim3 numBlocks(batch , head, group_num);
    dim3 threadsPerBlockDim(1024);

    auto key_states_ptr = key_states.data_ptr<T>();

    quantize_with_outliers_kernel<<<numBlocks, threadsPerBlockDim, 0>>>(
    key_states_ptr,
    zp.data_ptr<T>(),
    scale.data_ptr<T>(),
    key_sign.data_ptr<uint8_t>(),
    key_3bit.data_ptr<uint8_t>(),
    batch,head,len,group_num,
    emb_dim);
                                                         // Remove any persistent lines in L2

    return std::make_tuple(key_sign,key_3bit);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("My_key_quant_V2", &MyQuantCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("key_states"),
    py::arg("zp"),
    py::arg("scale"),
    py::arg("group_size"));

}
