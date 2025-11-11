#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 256
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
    T*  key_states,
    // 需要压缩的keystates
    uint8_t* __restrict__ key_quant,
    // 普通的结果
    uint8_t* __restrict__ outlier_quant,
    // 输出的残差
    T* resduial,
    // channel_wise的平均值
    T* channel_mean,
    T* quant_outlier_zp,
    // outlier量化所需要的zeropoint的
    T* quant_outlier_scale, 
    // outlier的量化的scale
    uint8_t* outlier_idx,
    // outlier的索引
    T* full_outlier_zp,
    // 每一个维度的zeropoint
    T* full_outlier_scale, 
    // 每一个维度scale
    int outlier_num,
    // outlier channel的数目
    int batch_size, int head_size, int len, int emb_dim
    ) {

    size_t batch_id = blockIdx.x;
    size_t head_id = blockIdx.y;
    size_t pro_id = blockIdx.z;
// batch_id和head_id不能*NUM_PER_THREAD*WARPS_PER_BLOCK 来表示位移，因为可能会有越界的线程。
    int base_index = (batch_id * head_size * len * EMB_DIM) 
                   + (head_id * len * EMB_DIM) 
                   + (pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK);
    // int base_index = (batch_id * gridDim.y * gridDim.z * NUM_PER_THREAD*WARPS_PER_BLOCK) 
    //                + (head_id * gridDim.z * NUM_PER_THREAD*WARPS_PER_BLOCK) 
    //                + (pro_id * NUM_PER_THREAD*WARPS_PER_BLOCK);
    // warp排序是列主序,获取当前线程在block中的id.
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
    T* key = key_states + base_index_key + extra_offset;

    uint8_t* quant = key_quant + base_index_quant + extra_offset/8;

    // 每个head有outlier_num outlier channel
    int base_index_outlier = batch_id * head_size * outlier_num + head_id*outlier_num;
    // 每个head有emb_dim 个量化参数，之后会选取其中的outlier_num个
    int base_index_quant_param = batch_id * head_size *emb_dim + head_id*emb_dim;
    T* channel_mean_point = channel_mean+base_index_quant_param+extra_offset;
    T* resduial_point = resduial+ base_index_key + extra_offset;
// 1bit量化

    uint8_t hashed_key = 0;
    // #pragma unroll
    // for (int shift_j = 0; shift_j < 8; shift_j++){
    //     float now_key = convert_to_float(key[shift_i + shift_j]);
    //     hashed_key |= (now_key > 0 ? (1 << shift_j) : 0);
    //     float cnt = channel_mean_point[shift_i+shift_j];
    //     // printf("extra_offset:%d,shift:%d,now_key:%f,means:%f\n",extra_offset,shift_j,now_key,cnt);
    //     resduial_point[shift_i+shift_j] = __float2half(now_key - (now_key > 0 ? 1 : -1) * channel_mean_point[shift_i+shift_j]);
    // }
    // const float4* half_data_as_float4_key = ;
    // const float4* half_data_as_float4_mean = ;
//     float4 key_val1 = reinterpret_cast<const float4*>(key)[0];     // 前 4 个 half
//     float4 key_val2 = reinterpret_cast<const float4*>(key+4)[0];       // 后 4 个 half
//     float4 mean_val1 = reinterpret_cast<const float4*>(channel_mean_point)[0];     // 前 4 个 half
//     float4 mean_val2 = reinterpret_cast<const float4*>(channel_mean_point+4)[0];       // 后 4 个 half
//     half temp_output[4];
//     // 转换 float4 为 float 数组，以便后续使用
//     float now_key = 0;
//     float now_mean = 0;
//     now_key = __half2float(reinterpret_cast<const half*>(&key_val1)[0]);
//     now_mean = __half2float(reinterpret_cast<const half*>(&mean_val1)[0]);
//     hashed_key |= (now_key > 0 ? (1 << 0) : 0);
//     temp_output[0] = __float2half(now_key - (now_key > 0 ? 1 : -1)*now_mean);

//     now_key = __half2float(reinterpret_cast<const half*>(&key_val1)[1]);
//     now_mean = __half2float(reinterpret_cast<const half*>(&mean_val1)[1]);
//     hashed_key |= (now_key > 0 ? (1 << 1) : 0);
//     temp_output[1] = __float2half(now_key - (now_key > 0 ? 1 : -1)*now_mean);

//     now_key = __half2float(reinterpret_cast<const half*>(&key_val1)[2]);
//     now_mean = __half2float(reinterpret_cast<const half*>(&mean_val1)[2]);
//     hashed_key |= (now_key > 0 ? (1 << 2) : 0);
//     temp_output[2] = __float2half(now_key - (now_key > 0 ? 1 : -1)*now_mean);

//     now_key = __half2float(reinterpret_cast<const half*>(&key_val1)[3]);
//     now_mean = __half2float(reinterpret_cast<const half*>(&mean_val1)[3]);
//     hashed_key |= (now_key > 0 ? (1 << 3) : 0);
//     temp_output[3] = __float2half(now_key - (now_key > 0 ? 1 : -1)*now_mean);
//     float4* output_as_float4 = reinterpret_cast<float4*>(resduial_point);
//     output_as_float4[0] = *reinterpret_cast<float4*>(&temp_output);
// // --------------------------------------------

//     now_key = __half2float(reinterpret_cast<const half*>(&key_val2)[0]);
//     now_mean = __half2float(reinterpret_cast<const half*>(&mean_val2)[0]);
//     hashed_key |= (now_key > 0 ? (1 << 4) : 0);
//     temp_output[0] = __float2half(now_key - (now_key > 0 ? 1 : -1)*now_mean);

//     now_key = __half2float(reinterpret_cast<const half*>(&key_val2)[1]);
//     now_mean = __half2float(reinterpret_cast<const half*>(&mean_val2)[1]);
//     hashed_key |= (now_key > 0 ? (1 << 5) : 0);
//     temp_output[1] = __float2half(now_key - (now_key > 0 ? 1 : -1)*now_mean);

//     now_key = __half2float(reinterpret_cast<const half*>(&key_val2)[2]);
//     now_mean = __half2float(reinterpret_cast<const half*>(&mean_val2)[2]);
//     hashed_key |= (now_key > 0 ? (1 << 6) : 0);
//     temp_output[2] = __float2half(now_key - (now_key > 0 ? 1 : -1)*now_mean);

//     now_key = __half2float(reinterpret_cast<const half*>(&key_val2)[3]);
//     now_mean = __half2float(reinterpret_cast<const half*>(&mean_val2)[3]);
//     hashed_key |= (now_key > 0 ? (1 << 7) : 0);
//     temp_output[3] = __float2half(now_key - (now_key > 0 ? 1 : -1)*now_mean);
//     output_as_float4 = reinterpret_cast<float4*>(resduial_point+4);
//     output_as_float4[0] = *reinterpret_cast<float4*>(&temp_output);




    const int4* key_input_as_int4 = reinterpret_cast<const int4*>(key);
    const int4* mean_input_as_int4 = reinterpret_cast<const int4*>(channel_mean_point);
    int4 key_data = key_input_as_int4[0];
    half* key_half_data = reinterpret_cast<half*>(&key_data);
    int4 mean_data = mean_input_as_int4[0];
    half* mean_half_data = reinterpret_cast<half*>(&mean_data);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float now_key = __half2float(key_half_data[i]);
        float now_mean = __half2float(mean_half_data[i]);
        hashed_key |= (now_key > 0 ? (1 << i) : 0);
        key_half_data[i] = __float2half(now_key - (now_key > 0 ? 1 : -1)*now_mean);  // 每个 half 减 1
    }

    int4* output_as_int4 = reinterpret_cast<int4*>(resduial_point);
    output_as_int4[0] = key_data;
    quant[0] = hashed_key;


    if (th_id%thread_num_per_emb<outlier_num){
        // outlier idx的指针,每个embedding都有outlier_num个，然后每个线程最多处理3个
        uint8_t* outlier_idx_point = outlier_idx + base_index_outlier;
        // outlier_zp的指针 ;
        T* outlier_zp_point = full_outlier_zp + base_index_quant_param;
        // outlier_scale
        T* outlier_scale_point = full_outlier_scale + base_index_quant_param;
        // outlier_quant,每emb，有outlier num个异常值,获取当前线程要写入的位置
        // int quant_offset = base_index+;
        uint8_t* outlier_quant_point = outlier_quant + base_index_key/128*outlier_num  ;
        // base_index/128*outlier_num  +   + sub_th_id;
        T* quant_outlier_zp_point = quant_outlier_zp + base_index_outlier ;
        T* quant_outlier_scale_point = quant_outlier_scale + base_index_outlier ;
        int shfit_cnt = base_index_key/128*outlier_num;
        // printf("%d\n",th_id%thread_num_per_emb);
        T* key_start = key_states + base_index_key;
        T* resduial_start = resduial+ base_index_key;
        // #pragma unroll
        int fianl_shift = th_id%thread_num_per_emb;
        int idx = outlier_idx_point[fianl_shift];
        float zp = convert_to_float(outlier_zp_point[idx]);    // 将模板类型转换为float
        float scale = convert_to_float(outlier_scale_point[idx]);  // 将scale转换为float
        float value = convert_to_float(key_start[idx]);  // 将key_emb中的值转换为float
        // 进行浮点运算
        // printf("vecid:%d,shift:%d,id:%d,value:%f\n",vec_id,shfit_cnt,idx,value);
        float quantized_value = (value / scale) + zp ;
        // 将结果转换回模板类型T，并存储到outlier_quant_point中
        quantized_value = fminf(fmaxf(quantized_value, 0.000f), 255.0f);
        outlier_quant_point[fianl_shift] = convert_from_float<uint8_t>(roundf(quantized_value));
        resduial_start[idx] = 0;
        // outlier_quant_point[fianl_shift] = __float2half(value);
        if (pro_id==0){
            quant_outlier_zp_point[fianl_shift] = convert_from_float<T>(zp);
            quant_outlier_scale_point[fianl_shift] = convert_from_float<T>(scale);
        }
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MyQuantCudaTemplate(
    torch::Tensor key_states,
    torch::Tensor key_states_means,
    torch::Tensor outlier_idx,
    torch::Tensor outlier_zp,
    torch::Tensor outlier_scale,
    int outlier_num
    ) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kUInt8);
    auto options_outlier_norm = getOptionsForType(typeid(T));

    int batch = key_states.size(0);
    int head = key_states.size(1);
    int len = key_states.size(2);

    // token的emb_dim一般为128
    int emb_dim = key_states.size(3);

    int hash_dim = emb_dim / 8;

    // 获取 key_states 的设备信息
    auto device = key_states.device();

    auto key_quant = torch::zeros({batch, head, len, hash_dim}, options.device(device)).contiguous();
    auto outlier_quant = torch::zeros({batch, head, len, outlier_num}, options.device(device)).contiguous();
    auto outlier_zp_quant = torch::zeros({batch, head, outlier_num}, options_outlier_norm.device(device)).contiguous();
    auto outlier_scale_quant = torch::zeros({batch, head, outlier_num}, options_outlier_norm.device(device)).contiguous();
    auto residual = torch::zeros({batch, head, len, emb_dim}, options_outlier_norm.device(device)).contiguous();
    // warp_size 表示一个warp中的线程数，一般为32  WARPS_PER_BLOCK 表示一个block中warp的数量， 一个block最大的线程数是1024，因此WARPS_PER_BLOCK的最大值为32
    // 每个线程处理NUM_PER_THREAD个元素
    int numProjBlocks = (len*emb_dim+(NUM_PER_THREAD*WARPS_PER_BLOCK)-1) / (NUM_PER_THREAD*WARPS_PER_BLOCK);
    dim3 numBlocks(batch , head, numProjBlocks);
    dim3 threadsPerBlockDim(WARPS_PER_BLOCK);

    auto key_states_ptr = key_states.data_ptr<T>();


//     Compiler hints for using L2 Persistent Cache
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);                                                                  // Create CUDA stream
    // int device_id{0};
    // cudaGetDevice(&device_id);                                                                  // Device ID


    quantize_with_outliers_kernel<<<numBlocks, threadsPerBlockDim, 0>>>(
    key_states_ptr,
    key_quant.data_ptr<uint8_t>(),
    outlier_quant.data_ptr<uint8_t>(),
    residual.data_ptr<T>(),
    key_states_means.data_ptr<T>(),
    outlier_zp_quant.data_ptr<T>(),
    outlier_scale_quant.data_ptr<T>(),
    outlier_idx.data_ptr<uint8_t>(),
    outlier_zp.data_ptr<T>(),
    outlier_scale.data_ptr<T>(),
    outlier_num,batch,head,len,
    emb_dim);
                                                         // Remove any persistent lines in L2

    return std::make_tuple(key_quant,outlier_quant,outlier_zp_quant,outlier_scale_quant,residual);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("My_quant_half_half", &MyQuantCudaTemplate<c10::Half>, "Quantize using Half precision",
    py::arg("key_states"),
    py::arg("key_states_means"),
    py::arg("outlier_idx"),
    py::arg("outlier_zp"),
    py::arg("outlier_scale"),
    py::arg("outlier_num"));

}
