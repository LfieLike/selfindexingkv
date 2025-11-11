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

// dst half 存储最终结果。 输入数据需要包括，压缩后的数值，zp scale，以及他们对应的位置（每个位置需要搬运128个数）

template<typename T>
__global__ void quantize_with_outliers_kernel(
    uint8_t*  key_value_quantized_data,
    // 压缩好的数据
    T*  key_dequant_dst,
    T*  value_dequant_dst,
    
    T* key_quant_zp,
    T* key_quant_scale, 
    T* value_quant_zp,
    T* value_quant_scale, 
    int* quant_idx,
    // outlier的索引
    int num,
    // num 是要解压多少个
    int batch_size, int head_size, int len 
    // len是总长度
    ) {
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int pro_id = blockIdx.z;
    // 表示dequant_dst的基础位置
    int dequant_base_index = (batch_id * head_size * len * EMB_DIM) 
                   + (head_id * len * EMB_DIM);
    // 获取当前是解压哪一个
    int quant_base_index = (batch_id * head_size * num * EMB_DIM)+ (head_id * num * EMB_DIM);
    int idx_base_index = (batch_id * head_size * num)+ (head_id * num);
    // warp排序是列主序,获取当前线程在block中的id.
    int th_id = threadIdx.y*16+threadIdx.x;
    int sub_th_id = threadIdx.x;
    // 首先需要知道当前线程处理哪一个向量, 向量长度除每个线程处理的元素数量，得到当前处理的向量的id
    int vec_id = threadIdx.y;
    // 获取当前线程需要解压的的vec的起始位置，4bit压缩，使用uint8存储，每uint8存4个元素
    int quant_shift = (quant_base_index+pro_id*8*128+vec_id*128+sub_th_id*8);
    // 判断边界 numProjBlocks个block要处理num*EMBDIM个元素
    if(pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK + vec_id*128>=num*EMB_DIM){
        return;
    }
    // uint16_t onebit_key = *reinterpret_cast<const uint16_t*>(dequant_dst+quant_shift)[0];
    uint8_t key_value_quant_cache[8];
    half key_dequant_cache[8];
    half value_dequant_cache[8];
    int idx;
    // printf("pro_id:%d,th_id:%d,sub_th_id:%d\n",pro_id,th_id,sub_th_id);
    idx = quant_idx[idx_base_index+pro_id*8+vec_id];
    int true_shift_dequant = dequant_base_index+idx*EMB_DIM+sub_th_id*8;
    int2* int4_key_value_quant_cache = reinterpret_cast<int2*>(key_value_quant_cache);
    int2* int4_key_value_quantized_data = reinterpret_cast<int2*>(key_value_quantized_data+quant_shift);
    int4_key_value_quant_cache[0] = int4_key_value_quantized_data[0];
    reinterpret_cast<int4*>(key_dequant_cache)[0] = reinterpret_cast<int4*>(key_dequant_dst+true_shift_dequant)[0];
    reinterpret_cast<int4*>(value_dequant_cache)[0] = reinterpret_cast<int4*>(value_dequant_dst+true_shift_dequant)[0];
    // uint8_t value_quant_cache[4];

    // idx 记录了当前是原始cache中的第几行, 还要加上当前子线程的shift

    // printf("pro_id:%d,idx:%d,true_shift_dequant:%d,quant_shift:%d\n",pro_id,idx,true_shift_dequant,quant_shift);
    for(int i=0;i<8;i++){
        float temp = __half2float(key_dequant_cache[i]);  // 将 c10::Half 转换为 float
        temp += static_cast<float>(key_value_quant_cache[i] & 0x0F);  // 执行加法
        key_dequant_cache[i] = __float2half(temp);  // 将结果转换回 c10::Half 并赋值回去
        temp = __half2float(value_dequant_cache[i]);  // 将 c10::Half 转换为 float
        temp += static_cast<float>((key_value_quant_cache[i]>>4)&0x0F);  // 执行加法
        value_dequant_cache[i] = __float2half(temp);
    }
    reinterpret_cast<int4*>(key_dequant_dst+true_shift_dequant)[0] = reinterpret_cast<int4*>(key_dequant_cache)[0];
    reinterpret_cast<int4*>(value_dequant_dst+true_shift_dequant)[0] = reinterpret_cast<int4*>(value_dequant_cache)[0];
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
void MyScatterAddCudaTemplate(
    torch::Tensor  key_value_quantized_data,
    torch::Tensor  key_dequant_dst,
    torch::Tensor  value_dequant_dst,
    torch::Tensor key_quant_zp,
    torch::Tensor key_quant_scale, 
    torch::Tensor value_quant_zp,
    torch::Tensor value_quant_scale, 
    torch::Tensor quant_idx,
    int num,
    int len
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_outlier_norm = getOptionsForType(typeid(T));

    int batch = key_dequant_dst.size(0);
    int head = key_dequant_dst.size(1);
    // warp_size 表示一个warp中的线程数，一般为32  WARPS_PER_BLOCK 表示一个block中warp的数量， 一个block最大的线程数是1024，因此WARPS_PER_BLOCK的最大值为32
    // 每个线程处理NUM_PER_THREAD个元素
    int numProjBlocks = (len*EMB_DIM+(NUM_PER_THREAD*WARPS_PER_BLOCK)-1) / (NUM_PER_THREAD*WARPS_PER_BLOCK);
    dim3 numBlocks(batch , head, numProjBlocks);
    dim3 threadsPerBlockDim(16,8);

    // auto key_states_ptr = key_states.data_ptr<T>();


//     Compiler hints for using L2 Persistent Cache

    quantize_with_outliers_kernel<<<numBlocks, threadsPerBlockDim, 0>>>(
    key_value_quantized_data.data_ptr<uint8_t>(),
    key_dequant_dst.data_ptr<T>(),
    value_dequant_dst.data_ptr<T>(),
    key_quant_zp.data_ptr<T>(),
    key_quant_scale.data_ptr<T>(),
    value_quant_zp.data_ptr<T>(),
    value_quant_scale.data_ptr<T>(),
    quant_idx.data_ptr<int>(),
    num,batch,head,len);
                                                         // Remove any persistent lines in L2

    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("My_scatter_add_half_half", &MyScatterAddCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("key_value_quantized_data"),
    py::arg("key_dequant_dst"),
    py::arg("value_dequant_dst"),
    py::arg("key_quant_zp"),
    py::arg("key_quant_scale"),
    py::arg("value_quant_zp"),
    py::arg("value_quant_scale"),
    py::arg("quant_idx"),
    py::arg("num"),
    py::arg("len"));

}
