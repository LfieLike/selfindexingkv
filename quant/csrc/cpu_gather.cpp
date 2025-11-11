#include <torch/extension.h>
#include <vector>
#include <thread>
#include <immintrin.h>
inline void gather_cpu(
    int32_t* __restrict__ quant_kv,
    c10::Half* __restrict__ quant_param,
    const int64_t* __restrict__ index,
    int32_t* __restrict__ output_kv,
    c10::Half* __restrict__ output_param,
    int index_batch, int index_head, int index_len,
    int kv_batch, int kv_head, int kv_len) 
{
    constexpr int len = 16;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < index_batch; ++i) {
        for(int j = 0; j < index_head; ++j) {
            const int64_t ij_index = i * index_head * index_len + j * index_len;
            const int64_t ij_kv = i * kv_head * kv_len + j * kv_len;
            
            for(int k=0; k<index_len; k+=4) {
                #pragma unroll
                for(int uk=0; uk<4; ++uk) {
                    const int tk = k + uk;
                    if(tk >= index_len) break;
                    
                    const int offset = index[ij_index + tk];
                    int32_t* src_kv = quant_kv + (ij_kv + offset)*len;
                    c10::Half* src_param = quant_param + (ij_kv + offset)*len;
                    int32_t* dst_kv = output_kv + (ij_index + tk)*len;
                    c10::Half* dst_param = output_param + (ij_index + tk)*len;

                    // 向量化加载存储
                    __m256i v0 = _mm256_loadu_si256((const __m256i*)src_kv);
                    __m256i v1 = _mm256_loadu_si256((const __m256i*)(src_kv + 8));
                    _mm256_stream_si256((__m256i*)dst_kv, v0);
                    _mm256_stream_si256((__m256i*)(dst_kv + 8), v1);
                    
                    __m256i vp = _mm256_loadu_si256((const __m256i*)src_param);
                    _mm256_stream_si256((__m256i*)dst_param, vp);
                }
            }
        }
    }
}


void my_cpu_gather(const torch::Tensor quant_kv,const torch::Tensor quant_param, const torch::Tensor& index,torch::Tensor output_kv,torch::Tensor output_param) {
    // auto output = torch::empty(index.sizes(), input.options());
    int batch = index.size(0);
    int head = index.size(1);
    int len = index.size(2);
    int kv_batch = quant_kv.size(0);
    int kv_head = quant_kv.size(1);
    int kv_len = quant_kv.size(2);
    gather_cpu(quant_kv.data_ptr<int32_t>(),quant_param.data_ptr<c10::Half>(),
        index.data_ptr<int64_t>(),output_kv.data_ptr<int32_t>(),output_param.data_ptr<c10::Half>(),
        batch,head,len,kv_batch,kv_head,kv_len
    );
    return;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_cpu_gather", &my_cpu_gather, "Multi-threaded gather operation",
    py::call_guard<py::gil_scoped_release>(),
    py::arg("quant_kv"),py::arg("quant_param"), py::arg("index"),py::arg("output_kv"),py::arg("output_param"));
}