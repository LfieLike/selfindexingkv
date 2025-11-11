#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <time.h> 
#include <torch/extension.h>
// #include <torch/extension.h>
// 摘抄自https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally 感谢大佬
inline float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}
inline void comput_one_row(float* table,uint8_t* data,float* res){
    __m128i vec = _mm_loadu_si128((__m128i*)data);
    __m256i low = _mm256_cvtepu8_epi32(vec); // 低8位
    __m256i high = _mm256_cvtepu8_epi32(_mm_bsrli_si128(vec, 8)); // 高8位
    __m256i increment_low = _mm256_setr_epi32(0, 256, 512, 768, 1024, 1280, 1536, 1792);
    __m256i increment_high = _mm256_setr_epi32(2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840);

    // 将low和high分别加上对应的值
    __m256i result_low = _mm256_add_epi32(low, increment_low);
    __m256i result_high = _mm256_add_epi32(high, increment_high);
    __m256 gather_low = _mm256_i32gather_ps(table, result_low, 4);  // 使用低 8 位索引查表
    __m256 gather_high = _mm256_i32gather_ps(table, result_high, 4);
    __m256 sum = _mm256_add_ps(gather_low, gather_high);
    res[0]=sum8(sum);
}
void lut_gemv(float* lut,uint8_t* key_states,float* res,int true_len,int batch,int head,int len,int dim){
    #pragma omp parallel for collapse(2) schedule(guided)
    for(int batch_id=0;batch_id<batch;batch_id++){ 
        for(int head_id=0;head_id<head;head_id++){
            int offset = batch_id*head*true_len+head_id*true_len;
            int lut_offset = (batch_id*head+head_id)*4096;
            int key_offset =  (batch_id*head*len+head_id*len)*dim;
            for(int len_id=0;len_id<true_len;len_id++){
                // printf("before:%f,offset+len_id:%d\n",res[offset+len_id],offset+len_id);
                // _mm_prefetch((char*)&key_states[key_offset + (len_id+4)*dim], _MM_HINT_T0);
                comput_one_row(lut+lut_offset,key_states+key_offset+len_id*dim,res+offset+len_id);
            }
        }
    }
    return;
}
void lut_gemv_naive(float* lut,uint8_t* key_states,float* res,int batch,int head,int len,int dim){
    for(int batch_id=0;batch_id<batch;batch_id++){ 
        for(int head_id=0;head_id<head;head_id++){
            int offset = batch_id*head*len+head_id*len;
            int lut_offset = (batch_id*head+head_id)*4096;
            int key_offset =  offset*16;
            for(int len_id=0;len_id<len;len_id++){
                float sum =0;
                for(int i=0;i<16;i++){
                    int index = key_states[key_offset+len_id*dim+i];
                    sum+=lut[lut_offset+256*i+index];
                }
                res[offset+len_id]=sum;
                // if(res[offset+len_id]!=sum){
                //     printf("%f %f\n",res[offset+len_id],sum);
                // }
            }
        }
    }
    return;
}
torch::Tensor cpu_lut_gemv(const torch::Tensor lut, const torch::Tensor key_states,torch::Tensor res,const int true_len) {
    int batch = key_states.size(0);
    int head = key_states.size(1);
    int len = key_states.size(2);
    int dim = key_states.size(3);
    lut_gemv(
        lut.data_ptr<float>(),
        key_states.data_ptr<uint8_t>(),
        res.data_ptr<float>(),
        true_len,batch,head,len,dim
    );
    return res;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_lut_gemv", &cpu_lut_gemv, "Quantize using Half precision",py::call_guard<py::gil_scoped_release>(),
    py::arg("lut"),
    py::arg("key_states"),
    py::arg("res"),
    py::arg("truelen"));
}

// float table[256];

// int main() {
//     const int batch = 2;
//     const int head = 32;
//     const int len = 4096;
//     const int dim = 16;

//     // Prepare mock LUT and key states data
//     float lut[batch * head * 4096];  // Just for testing, you would initialize this with real data
//     uint8_t key_states[batch * head * len * dim];  // Initialize with test values
//     float res[batch * head * len];  // To store the results

//     // Initialize LUT with dummy data
//     for (int i = 0; i < batch * head * 4096; ++i) {
//         lut[i] = (float)(i % 100);  // Fill with some values for testing
//     }

//     // Initialize key states with dummy data
//     for (int i = 0; i < batch * head * len * dim; ++i) {
//         key_states[i] = (uint8_t)(i % 256);  // Random byte values
//     }

//     // Call the main function
//     // Measure time for lut_gemv
//     clock_t start_gemv = clock();
//     lut_gemv(lut, key_states, res, batch, head, len, dim);
//     clock_t end_gemv = clock();
//     double duration_gemv = (double)(end_gemv - start_gemv) ;
//     printf("Time taken for lut_gemv: %f seconds\n", duration_gemv);
//     // Measure time for lut_gemv_naive
//     // clock_t start_naive = clock();
//     lut_gemv_naive(lut, key_states, res, batch, head, len, dim);
//     // clock_t end_naive = clock();
//     // double duration_naive = (double)(end_naive - start_naive);

//     // // Print the results

//     // printf("Time taken for lut_gemv_naive: %f seconds\n", duration_naive);
//     // printf("%d,%d\n",duration_gemv.count(),duration_naive.count());
//     // Print the results
//     // for (int i = 0; i < batch * head * len; ++i) {
//     //     // std::cout << "res[" << i << "] = " << res[i] << std::endl;
//     //     printf("%f ",res[i]);
//     // }

//     return 0;
// }