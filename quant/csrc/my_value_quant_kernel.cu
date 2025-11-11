#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>

#define WARP_SIZE 16
#define WARPS_PER_BLOCK 256
#define EMB_DIM 128
#define NUM_PER_THREAD 8

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
    T*  value_states,
    // 需要压缩的keystates
    uint8_t* __restrict__ value_quant,
    // 输出的残差zp,scale
    T* maxvalue,
    T* quant_param,
    T* value_resduial,
    int batch_size, int head_size, int len, int emb_dim
    ) {

    size_t batch_id = blockIdx.x;
    size_t head_id = blockIdx.y;
    size_t pro_id = blockIdx.z;
// batch_id和head_id不能*NUM_PER_THREAD*WARPS_PER_BLOCK 来表示位移，因为可能会有越界的线程。
    int base_index = (batch_id * head_size * len * EMB_DIM) 
                   + (head_id * len * EMB_DIM) 
                   + (pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK);
    size_t th_id = threadIdx.x;
    // 首先需要知道当前线程处理哪一个向量, 向量长度除每个线程处理的元素数量，得到当前处理的向量的id
    int thread_num_per_emb = emb_dim/NUM_PER_THREAD;
    int vec_id = th_id / thread_num_per_emb;
    // 获取当前线程处理的vec的起始位置
    int base_index_key = base_index+vec_id*128;
    // 判断边界
    if(pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK + vec_id*128>=len*EMB_DIM){
        return;
    }
    // 采取uint8作为量化格式，所以量化指针除8
    int base_index_quant = base_index_key/8;
    // 每个线程处理NUM_PER_THREAD个元素，添加额外的偏移量
    int extra_offset = NUM_PER_THREAD*(th_id%thread_num_per_emb);
    int max_value_base_index = (batch_id * head_size * EMB_DIM) + (head_id * EMB_DIM);
    T* value = value_states + base_index_key + extra_offset;
    uint8_t* quant = value_quant + base_index_quant + extra_offset/8;
    const int4* value_input_as_int4 = reinterpret_cast<const int4*>(value);
    int4 value_data = value_input_as_int4[0];
    half* value_half_data = reinterpret_cast<half*>(&value_data);

    float value_float_data[8];
    float value_float_data_org[8];
    half max_value_half[8];
    reinterpret_cast<int4*>(max_value_half)[0] = reinterpret_cast<int4*>(maxvalue+max_value_base_index+extra_offset)[0];
    #pragma unroll
    for(int i=0;i<8;i++){
        value_float_data_org[i] = __half2float(value_half_data[i]);
        value_float_data[i] = __half2float(value_half_data[i])/__half2float(max_value_half[i]);
    }
    float value_local_max = value_float_data[0];
    float value_local_min = value_float_data[0];
    for(int i=0;i<8;i++){
        // value_float_data[i] = __half2float(value_half_data[i])/__half2float(max_value_half[i]);
        value_local_max = max(value_local_max,value_float_data[i]);
        value_local_min = min(value_local_min,value_float_data[i]);
    }
    float warp_max = warpReduceMax(value_local_max);
    float warp_min = warpReduceMin(value_local_min);
    warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0,16);
    warp_min = __shfl_sync(0xFFFFFFFF, warp_min, 0,16);
    // printf("max:%f,min:%f\n",warp_max,warp_min);
    float value_scale = (warp_max - warp_min);
    // roundf 误差很大
    float value_zp = rintf(-warp_min / value_scale);


    uint8_t quant_output=0;
    int print_quant = 0;
    half resduial[8];
    for(int i=0;i<8;i++){
        float quantized_value = (value_float_data[i]/ value_scale) + value_zp;
        quantized_value = fminf(fmaxf(rintf(quantized_value), 0.0f), 1.0f);
        uint8_t quanted = convert_from_float<uint8_t>(roundf(quantized_value));
        quant_output |= ((quanted << i));
        print_quant|=((quanted << i));
        // printf("quantized_value%f,print_quant:%d",quantized_value,print_quant);
        // key_half_data[i] = __float2half(now_key - (now_key > 0 ? 1 : -1)*now_mean);
        float dequant = (((((quanted&1) ? (1) : 0)-value_zp)*value_scale))*__half2float(max_value_half[i]);
        resduial[i] = __float2half(value_float_data_org[i] -  dequant);
    }
    // printf("quant:%d\n",print_quant);
    quant[0] = quant_output;
    // const int4* resduial_output_as_int4 = ;
    reinterpret_cast<int4*>(value_resduial + base_index_key + extra_offset)[0] = reinterpret_cast<int4*>(resduial)[0];
    if(th_id%16==0){
        half local_quant_param[2];
        local_quant_param[0]=__float2half(value_zp);
        local_quant_param[1]=__float2half(value_scale);
        reinterpret_cast<int*>(quant_param+ base_index/64 +vec_id*2)[0] = reinterpret_cast<int*>(local_quant_param)[0];
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MyQuantCudaTemplate(
    torch::Tensor value_states,
    torch::Tensor maxvalue
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_outlier_norm = getOptionsForType(typeid(T));

    int batch = value_states.size(0);
    int head = value_states.size(1);
    int len = value_states.size(2);

    // token的emb_dim一般为128
    int emb_dim = value_states.size(3);

    int hash_dim = emb_dim / 8;

    // 获取 value_states 的设备信息
    auto device = value_states.device();

    auto value_quant = torch::zeros({batch, head, len, hash_dim}, options.device(device)).contiguous();
    auto quant_param = torch::zeros({batch, head, len,2}, options_outlier_norm.device(device)).contiguous();
    // auto scale = torch::zeros({batch, head, len}, options_outlier_norm.device(device)).contiguous();
    auto residual = torch::zeros({batch, head, len, emb_dim}, options_outlier_norm.device(device)).contiguous();
    // warp_size 表示一个warp中的线程数，一般为32  WARPS_PER_BLOCK 表示一个block中warp的数量， 一个block最大的线程数是1024，因此WARPS_PER_BLOCK的最大值为32
    // 每个线程处理NUM_PER_THREAD个元素
    int numProjBlocks = (len*emb_dim+(NUM_PER_THREAD*WARPS_PER_BLOCK)-1) / (NUM_PER_THREAD*WARPS_PER_BLOCK);
    dim3 numBlocks(batch , head, numProjBlocks);
    dim3 threadsPerBlockDim(WARPS_PER_BLOCK);

    auto value_states_ptr = value_states.data_ptr<T>();

    // T*  value_states,
    // // 需要压缩的keystates
    // uint8_t* __restrict__ value_quant,
    // // 输出的残差zp,scale
    // T* maxvalue,
    // T* quant_param,
    // T* value_resduial,

    quantize_with_outliers_kernel<<<numBlocks, threadsPerBlockDim, 0>>>(
    value_states_ptr,
    value_quant.data_ptr<uint8_t>(),
    maxvalue.data_ptr<T>(),
    quant_param.data_ptr<T>(),
    residual.data_ptr<T>(),
    batch,head,len,
    emb_dim);
                                                         // Remove any persistent lines in L2

    return std::make_tuple(quant_param,value_quant,residual);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("My_quant_half_half", &MyQuantCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("key_states"),
    py::arg("maxvalue"));

}
