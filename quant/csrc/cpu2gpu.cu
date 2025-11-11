#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>
#include <c10/cuda/CUDAStream.h>
#define NUM_PER_THREAD 8
#define THREAD_NUM 1024
#define WARPS_PER_BLOCK 128
#define EMB_DIM 16

__global__ void cpu2gpu(
    // key cache
    int32_t* key_value_quant_gpu,
    // query向量
    int32_t* key_value_quant_cpu,
    // 最终结果
    c10::Half* param_gpu,
    c10::Half* param_cpu,
    int64_t* select_index,
    // outlier channel的数目
    int batch_size, int head_size, int len,int head_dim,int indexlen
    ) {

    size_t batch_id = blockIdx.x;
    size_t head_id = blockIdx.y;
    size_t pro_id = blockIdx.z;
    if ((pro_id * blockDim.x) + threadIdx.x >= indexlen){
      return;
    }
    int index_offset = (batch_id * head_size * indexlen) 
                   + (head_id * indexlen) 
                   + (pro_id * blockDim.x) + threadIdx.x;
    // 先获取select_index
    int64_t now_index = select_index[index_offset];
    //现在要获取读取的偏移量和写入的偏移量
    int cpu_offset = (batch_id * head_size * len * head_dim) 
                   + (head_id * len * head_dim) 
                   + now_index * head_dim;
    int now_index1 = now_index;
    // printf("offset:%d,now_index:%d\n",cpu_offset,now_index1);
    int gpu_offset = index_offset* head_dim;
    int4* gpu_ptr = reinterpret_cast<int4*>(key_value_quant_gpu+gpu_offset);
    int4* cpu_ptr = reinterpret_cast<int4*>(key_value_quant_cpu+cpu_offset);
    int4* param_gpu_ptr = reinterpret_cast<int4*>(param_gpu+gpu_offset);
    int4* param_cpu_ptr = reinterpret_cast<int4*>(param_cpu+cpu_offset);
    // 读取gpu和cpu指针所指向的4个int32_t并将其存入int4结构
    gpu_ptr[0] = cpu_ptr[0];
    gpu_ptr[1] = cpu_ptr[1];
    gpu_ptr[2] = cpu_ptr[2];
    gpu_ptr[3] = cpu_ptr[3];
    param_gpu_ptr[0] = param_cpu_ptr[0];
    param_gpu_ptr[1] = param_cpu_ptr[1];
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

torch::Tensor MyScoreCudaTemplate(
    torch::Tensor key_value_quant_gpu,
    torch::Tensor key_value_quant_cpu,
    torch::Tensor param_gpu,
    torch::Tensor param_cpu,
    torch::Tensor select_index
    ) {

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int batch = key_value_quant_cpu.size(0);
    int head = key_value_quant_cpu.size(1);
    int len = key_value_quant_cpu.size(2);
    int head_dim = key_value_quant_cpu.size(3);
    int indexlen = select_index.size(2);


    // warp_size 表示一个warp中的线程数，一般为32  WARPS_PER_BLOCK 表示一个block中warp的数量， 一个block最大的线程数是1024，因此WARPS_PER_BLOCK的最大值为32
    // 每个线程处理8个元素，每16个线程处理一个向量的点积,一个warp处理两个向量的点积
    int numProjBlocks = (indexlen+1023) / (1024);
    dim3 numBlocks(batch , head, numProjBlocks);
    dim3 threadsPerBlockDim(1024);



//     Compiler hints for using L2 Persistent Cache
    cpu2gpu<<<numBlocks, threadsPerBlockDim, 0,stream>>>(
    key_value_quant_gpu.data_ptr<int32_t>(),
    key_value_quant_cpu.data_ptr<int32_t>(),
    param_gpu.data_ptr<c10::Half>(),
    param_cpu.data_ptr<c10::Half>(),
    select_index.data_ptr<int64_t>(),
    batch,head,len,head_dim,indexlen);
                                                         // Remove any persistent lines in L2
    return key_value_quant_gpu;
}
    // int32_t* key_value_quant_gpu,
    // // query向量
    // int32_t* key_value_quant_cpu,
    // // 最终结果
    // int64_t* select_index,
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu2gpu", &MyScoreCudaTemplate, "Quantize using Half precision",py::call_guard<py::gil_scoped_release>(),
    py::arg("key_value_quant_gpu"),
    py::arg("key_value_quant_cpu"),
    py::arg("param_gpu"),
    py::arg("param_cpu"),
    py::arg("select_index"));
}