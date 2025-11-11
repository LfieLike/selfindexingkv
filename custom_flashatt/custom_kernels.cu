#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include "dtype_bfloat16.cuh"
#include "reduce_kernels.cuh"

inline int ceilDiv(int N, int D) {
    return (N + D - 1) / D;
}

template <typename Precision, int BLOCK_SIZE, int NUM_WARPS, int THREAD_GROUP_SIZE, int HEAD_SIZE, int STEP_SIZE, int QKV_HEAD_RATIO>
__global__ void custom_flash_attention_splitkv_kernel(int n_step_shift,int n_seq, int n_step, int q_num_heads, int kv_num_heads, int q_stride, int kv_batch_stride, int kv_head_stride, int kv_tok_stride, float scale, Precision* q, Precision* k, Precision* v, float* tmp_out, float* score_maxes, float* softmax_sums){
    const int tid = threadIdx.x;
    const int batch = blockIdx.x; 
    const int head = blockIdx.y;
    const int step = blockIdx.z;

    // store queries, keys, and values using default dtype (blfloat16 for llama3) to reduce smem usage
    __shared__ Precision sq[HEAD_SIZE * QKV_HEAD_RATIO];
    __shared__ Precision sk[HEAD_SIZE * STEP_SIZE];
    __shared__ Precision sv[HEAD_SIZE * STEP_SIZE];
    // and other flash attention intermidiate results using fp32 for better accuracy
    __shared__ float reduce_smem[NUM_WARPS];
    __shared__ float softmaxes[STEP_SIZE * QKV_HEAD_RATIO];
    __shared__ float scores[STEP_SIZE * QKV_HEAD_RATIO];

    int skv_size = HEAD_SIZE * STEP_SIZE;
    int n_tok = STEP_SIZE;

    // for each thread block, we execute in parallel flash attention 2 algorithm for STEP_SIZE tokens
    if (step == n_step - 1){
        n_tok = n_seq - STEP_SIZE * step;
        skv_size = HEAD_SIZE * n_tok;
    }

    // divide threads into groups of size MEM_THREAD_GROUP_SIZE, where each thread group loads HEAD_SIZE values from gmem to smem for k and v
    int MEM_THREAD_GROUP_SIZE = HEAD_SIZE / VEC_SIZE;
    int mem_group = tid / MEM_THREAD_GROUP_SIZE;
    int mem_lane = tid % MEM_THREAD_GROUP_SIZE;
    // if each thread read vecsize element, each round can read (BLOCK_SIZE * VEC_SIZE) / HEAD_SIZE element,
    for (int idx = 0; idx < n_tok; idx += (BLOCK_SIZE * VEC_SIZE) / HEAD_SIZE){
        // tid biao shi di tid ge thread
        // BLOCK_SIZE=128 biao shi BLOCK_SIZE ge xiancheng  meige block  each block have BLOCK_SIZE thread
        int sidx = mem_lane * VEC_SIZE + (mem_group + idx) * HEAD_SIZE;
        int gidx = batch * kv_batch_stride + head * kv_head_stride + (step * STEP_SIZE + idx + mem_group) * kv_tok_stride + mem_lane * VEC_SIZE;
        if (sidx < skv_size){
            reinterpret_cast<Precision_Vec *>(&sk[sidx])[0]  = reinterpret_cast<Precision_Vec *>(&k[gidx])[0];
            reinterpret_cast<Precision_Vec *>(&sv[sidx])[0]  = reinterpret_cast<Precision_Vec *>(&v[gidx])[0];        
        }
    }
    // load queries into smem
    // for each key and value pair load QKV_HEAD_RATIO queries (grouped attention) to reuse kv cache, and avoid reloading it multiple times 
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        sq[tid + query_idx * HEAD_SIZE] = q[tid + batch * q_stride + (head * QKV_HEAD_RATIO + query_idx) * HEAD_SIZE];
    }

    __syncthreads();

    // if(tid==0){
    //     for(int j=0;j<3;j++){
    //         for(int i=0;i<128;i++){
    //             float key = __bfloat162float(sk[i]);
    //             printf("%f ",key);
    //         }
    //         printf("\n");
    //     }
    // }

    // divide threads into groups of size THREAD_GROUP_SIZE, and compute query * key dot product (scores) from the flash attention 2 algorithm
    // Q @ K = S
    // THREAD_GROUP_SIZE = 8
    const int THREAD_VEC_SIZE = HEAD_SIZE / THREAD_GROUP_SIZE;
    int group = tid / THREAD_GROUP_SIZE;
    int lane = tid % THREAD_GROUP_SIZE;

    float rq[THREAD_VEC_SIZE * QKV_HEAD_RATIO];
    #pragma unroll
    for (int idx = 0; idx < THREAD_VEC_SIZE; idx++){
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            rq[idx + query_idx * THREAD_VEC_SIZE] =  p2float(sq[lane + idx * THREAD_GROUP_SIZE + query_idx * HEAD_SIZE]);
        }
    }

    float rk[THREAD_VEC_SIZE];
    for (int idx = 0; idx < n_tok; idx += (BLOCK_SIZE * THREAD_VEC_SIZE) / HEAD_SIZE){
        #pragma unroll
        for (int jdx = 0; jdx < THREAD_VEC_SIZE; jdx++){
            if ((group + idx) < n_tok){
                rk[jdx] =  p2float(sk[lane + jdx * THREAD_GROUP_SIZE + (group + idx) * HEAD_SIZE]);
            }
            else
            {
                rk[jdx] = 0.0;
            }
        }

        float qk_dot[QKV_HEAD_RATIO];
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            qk_dot[query_idx] = 0.0;
            #pragma unroll
            for (int jdx = 0; jdx < THREAD_VEC_SIZE; jdx++){
                qk_dot[query_idx] += rq[jdx + query_idx * THREAD_VEC_SIZE] * rk[jdx];
            }
        }
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            #pragma unroll
            for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
                qk_dot[query_idx] += __shfl_xor_sync(0xFFFFFFFF, qk_dot[query_idx], mask);
            }
            if (lane == 0){
                scores[group + idx + query_idx * STEP_SIZE] = scale * qk_dot[query_idx];
            }
        }

    }
    __syncthreads();

    // continue the flash attention 2 algorithm:
    // 1) get the max of scores:
    // m = max(S)
    // 2) compute the exp of the diff between the max and the scores:
    // P = exp(S - m)
    // 3) compute the "online softmax":
    // l = sum(P)
    float curr_score_max[QKV_HEAD_RATIO];
    float softmax_sum[QKV_HEAD_RATIO];
    float score;
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        if (tid < n_tok){
            score = scores[tid + query_idx * STEP_SIZE];
        }
        else{
            score = -FLT_MAX;
        }

        curr_score_max[query_idx] = block_max<NUM_WARPS>(reduce_smem, score);

        float softmax;
        if (tid < n_tok){
            softmax = __expf(score - curr_score_max[query_idx]);
        }
        else{
            softmax = 0.0;
        }
        // if(n_tok<64){
        //     printf("tid:%d,curr_score_max:%f,n_tok:%d,softmax:%f\n",tid,curr_score_max[query_idx],n_tok,softmax);
        // }
        softmaxes[tid + query_idx * STEP_SIZE] = softmax;
        softmax_sum[query_idx] = block_sum<NUM_WARPS>(reduce_smem, softmax);
    }


    // compute the softmax * value dot product from the flash attention 2 algorithm:
    // O = P @ V
    float sv_dot[QKV_HEAD_RATIO];
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            sv_dot[query_idx] = 0.0;
        }
    for (int jdx = 0; jdx < n_tok; jdx++){
        float sv_val = p2float(sv[tid + jdx * HEAD_SIZE]);
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            sv_dot[query_idx] += softmaxes[jdx + STEP_SIZE * query_idx] * sv_val;
        }
    }
    __syncthreads();
    int true_n_step = n_step+n_step_shift;
    #pragma unroll
    // multiply the output by the "online softmax"
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        // printf("step:%d,pos:%d,orgpos:%d,batch:%d,head:%d\n",step,
        // tid + batch * q_num_heads * true_n_step * HEAD_SIZE + (head * QKV_HEAD_RATIO + query_idx) * true_n_step * HEAD_SIZE + (step) * HEAD_SIZE,
        // tid + batch * q_num_heads * n_step * HEAD_SIZE + (head * QKV_HEAD_RATIO + query_idx) * n_step * HEAD_SIZE + step * HEAD_SIZE
        // ,batch,head);
        tmp_out[tid + batch * q_num_heads * true_n_step * HEAD_SIZE + (head * QKV_HEAD_RATIO + query_idx) * true_n_step * HEAD_SIZE + (step) * HEAD_SIZE] = __fdividef(1.0f, softmax_sum[query_idx] + 1e-6f) * sv_dot[query_idx];

    }

    // store the maxes and softmaxes for each step (thread block), to be used by the combine kernel
    if (tid == 0){
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            score_maxes[batch * q_num_heads * true_n_step + (head * QKV_HEAD_RATIO + query_idx) * true_n_step + (step)] = curr_score_max[query_idx];
            softmax_sums[batch * q_num_heads * true_n_step + (head * QKV_HEAD_RATIO + query_idx) * true_n_step + (step)] = softmax_sum[query_idx];
            // printf("pos:%d,score_maxes:%f\n",batch * q_num_heads * n_step_shift + (head * QKV_HEAD_RATIO + query_idx) * n_step_shift + step,curr_score_max[query_idx]);
        }
    }
}

template <typename Precision, int BLOCK_SIZE, int NUM_WARPS, int THREAD_GROUP_SIZE, int HEAD_SIZE, int STEP_SIZE, int QKV_HEAD_RATIO>
__global__ void quant_flash_attention_splitkv_kernel(int n_seq, int n_step, int q_num_heads, int kv_num_heads, int q_stride, int kv_batch_stride, int kv_head_stride, int kv_tok_stride, float scale, 
Precision* q, int32_t* key_value_quant,uint8_t* key_1bit,Precision* quant_param,float* tmp_out, float* score_maxes, float* softmax_sums){
    const int tid = threadIdx.x;
    const int batch = blockIdx.x; 
    const int head = blockIdx.y;
    const int step = blockIdx.z;

    // store queries, keys, and values using default dtype (blfloat16 for llama3) to reduce smem usage
    __shared__ Precision sq[HEAD_SIZE * QKV_HEAD_RATIO];
    __shared__ Precision sk[HEAD_SIZE * STEP_SIZE];
    __shared__ Precision sv[HEAD_SIZE * STEP_SIZE];
    // and other flash attention intermidiate results using fp32 for better accuracy
    __shared__ float reduce_smem[NUM_WARPS];
    __shared__ float softmaxes[STEP_SIZE * QKV_HEAD_RATIO];
    __shared__ float scores[STEP_SIZE * QKV_HEAD_RATIO];

    int skv_size = HEAD_SIZE * STEP_SIZE;
    int n_tok = STEP_SIZE;

    // for each thread block, we execute in parallel flash attention 2 algorithm for STEP_SIZE tokens
    if (step == n_step - 1){
        n_tok = n_seq - STEP_SIZE * step;
        skv_size = HEAD_SIZE * n_tok;
    }
    int32_t kv_quant_2bit[8];
    uint8_t local_key_1bit[8];
    Precision local_quant_param[8];
    int sidx = tid*64;
    int gidx = batch * kv_batch_stride + head * kv_head_stride + (step * STEP_SIZE + (tid/2))* kv_tok_stride + 8*(tid%2);
        
    if (n_tok>tid/2){
        reinterpret_cast<uint256 *>(&kv_quant_2bit[0])[0] = reinterpret_cast<uint256 *>(&key_value_quant[gidx])[0];
        
        reinterpret_cast<int2 *>(&local_key_1bit[0])[0] = reinterpret_cast<int2 *>(&key_1bit[gidx])[0];

        reinterpret_cast<int4 *>(&local_quant_param[0])[0] = reinterpret_cast<int4 *>(&quant_param[gidx])[0];
        #pragma unroll
        for (int k = 0; k < 64; ++k) {
            int j = k / 32;
            int x = (k % 32) / 8;
            int i = k % 8;
            int32_t quant_word = kv_quant_2bit[j * 4 + x];
            uint8_t sign_byte = local_key_1bit[j * 4 + x];

            float scale_k = p2float(local_quant_param[j * 4 + 0]);
            float zp_k    = p2float(local_quant_param[j * 4 + 1]);
            float scale_v = p2float(local_quant_param[j * 4 + 2]);
            float zp_v    = p2float(local_quant_param[j * 4 + 3]);

            float qk = float((quant_word >> (i * 2)) & 0x3);
            bool sign = (sign_byte >> i) & 0x1;
            qk = sign ? (qk * scale_k + zp_k) : -(qk * scale_k + zp_k);
            sk[sidx+k] = float2p(qk);

            float qv = float((quant_word >> ((i + 8) * 2)) & 0x3);
            qv = qv * scale_v + zp_v;
            sv[sidx+k] = float2p(qv);
        }
    }
    
    // for (int idx = 0; idx < n_tok; idx += (BLOCK_SIZE * VEC_SIZE) / HEAD_SIZE){
    //     int stride_cnt = (BLOCK_SIZE * VEC_SIZE) / HEAD_SIZE;
    //     print("idx:%d,%d\n",idx,stride_cnt)
    //     int sidx = mem_lane * VEC_SIZE + (mem_group + idx) * HEAD_SIZE;
    //     int gidx = batch * kv_batch_stride + head * kv_head_stride + (step * STEP_SIZE + idx + mem_group) * kv_tok_stride + mem_lane * VEC_SIZE;
    //     if (sidx < skv_size){
    //         reinterpret_cast<Precision_Vec *>(&sk[sidx])[0]  = reinterpret_cast<Precision_Vec *>(&k[gidx])[0];
    //         reinterpret_cast<Precision_Vec *>(&sv[sidx])[0]  = reinterpret_cast<Precision_Vec *>(&v[gidx])[0];        
    //     }
    // }
    // load queries into smem
    // for each key and value pair load QKV_HEAD_RATIO queries (grouped attention) to reuse kv cache, and avoid reloading it multiple times 
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        sq[tid + query_idx * HEAD_SIZE] = q[tid + batch * q_stride + (head * QKV_HEAD_RATIO + query_idx) * HEAD_SIZE];
    }

    __syncthreads();
    // if(tid==0){
    //     for(int j=0;j<3;j++){
    //         for(int i=0;i<128;i++){
    //             float key = __bfloat162float(sk[i]);
    //             printf("%f ",key);
    //         }
    //         printf("\n");
    //     }
    // }

    // divide threads into groups of size THREAD_GROUP_SIZE, and compute query * key dot product (scores) from the flash attention 2 algorithm
    // Q @ K = S
    const int THREAD_VEC_SIZE = HEAD_SIZE / THREAD_GROUP_SIZE;
    int group = tid / THREAD_GROUP_SIZE;
    int lane = tid % THREAD_GROUP_SIZE;

    float rq[THREAD_VEC_SIZE * QKV_HEAD_RATIO];
    #pragma unroll
    for (int idx = 0; idx < THREAD_VEC_SIZE; idx++){
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            rq[idx + query_idx * THREAD_VEC_SIZE] =  p2float(sq[lane + idx * THREAD_GROUP_SIZE + query_idx * HEAD_SIZE]);
        }
    }

    float rk[THREAD_VEC_SIZE];
    for (int idx = 0; idx < n_tok; idx += (BLOCK_SIZE * THREAD_VEC_SIZE) / HEAD_SIZE){
        #pragma unroll
        for (int jdx = 0; jdx < THREAD_VEC_SIZE; jdx++){
            if ((group + idx) < n_tok){
                rk[jdx] =  p2float(sk[lane + jdx * THREAD_GROUP_SIZE + (group + idx) * HEAD_SIZE]);
            }
            else
            {
                rk[jdx] = 0.0;
            }
        }

        float qk_dot[QKV_HEAD_RATIO];
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            qk_dot[query_idx] = 0.0;
            #pragma unroll
            for (int jdx = 0; jdx < THREAD_VEC_SIZE; jdx++){
                qk_dot[query_idx] += rq[jdx + query_idx * THREAD_VEC_SIZE] * rk[jdx];
            }
        }
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            #pragma unroll
            for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
                qk_dot[query_idx] += __shfl_xor_sync(0xFFFFFFFF, qk_dot[query_idx], mask);
            }
            if (lane == 0){
                scores[group + idx + query_idx * STEP_SIZE] = scale * qk_dot[query_idx];
            }
        }

    }
    __syncthreads();

    // continue the flash attention 2 algorithm:
    // 1) get the max of scores:
    // m = max(S)
    // 2) compute the exp of the diff between the max and the scores:
    // P = exp(S - m)
    // 3) compute the "online softmax":
    // l = sum(P)
    float curr_score_max[QKV_HEAD_RATIO];
    float softmax_sum[QKV_HEAD_RATIO];
    float score;
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        if (tid < n_tok){
            score = scores[tid + query_idx * STEP_SIZE];
        }
        else{
            score = -FLT_MAX;
        }

        curr_score_max[query_idx] = block_max<NUM_WARPS>(reduce_smem, score);

        float softmax;
        if (tid < n_tok){
            softmax = __expf(score - curr_score_max[query_idx]);
            softmaxes[tid + query_idx * STEP_SIZE] = softmax;
        }
        else{
            softmax = 0.0;
            softmaxes[tid + query_idx * STEP_SIZE] = softmax;
        }
        softmax_sum[query_idx] = block_sum<NUM_WARPS>(reduce_smem, softmax);
    }

    // compute the softmax * value dot product from the flash attention 2 algorithm:
    // O = P @ V
    float sv_dot[QKV_HEAD_RATIO];
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            sv_dot[query_idx] = 0.0;
        }
    for (int jdx = 0; jdx < n_tok; jdx++){
        float sv_val = p2float(sv[tid + jdx * HEAD_SIZE]);
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            sv_dot[query_idx] += softmaxes[jdx + STEP_SIZE * query_idx] * sv_val;
        }
    }
    __syncthreads();

    #pragma unroll
    // multiply the output by the "online softmax"
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        tmp_out[tid + batch * q_num_heads * n_step * HEAD_SIZE + (head * QKV_HEAD_RATIO + query_idx) * n_step * HEAD_SIZE + step * HEAD_SIZE] = __fdividef(1.0f, softmax_sum[query_idx] + 1e-6f) * sv_dot[query_idx];

    }

    // store the maxes and softmaxes for each step (thread block), to be used by the combine kernel
    if (tid == 0){
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            score_maxes[batch * q_num_heads * n_step + (head * QKV_HEAD_RATIO + query_idx) * n_step + step] = curr_score_max[query_idx];
            softmax_sums[batch * q_num_heads * n_step + (head * QKV_HEAD_RATIO + query_idx) * n_step + step] = softmax_sum[query_idx];
        }
    }
}

template <typename Precision, int BLOCK_SIZE, int NUM_WARPS, int THREAD_GROUP_SIZE, int HEAD_SIZE, int STEP_SIZE, int QKV_HEAD_RATIO>
__global__ void select_quant_flash_attention_splitkv_kernel(int step_shift,int n_seq, int n_step, int q_num_heads, int kv_num_heads, int q_stride, int kv_batch_stride, int kv_head_stride, int kv_tok_stride,int idx_batch_stride, int idx_head_stride, float scale, 
Precision* q,int64_t* topk_index, int32_t* key_value_quant,uint8_t* key_1bit,Precision* quant_param,float* tmp_out, float* score_maxes, float* softmax_sums){
    const int tid = threadIdx.x;
    const int batch = blockIdx.x; 
    const int head = blockIdx.y;
    const int step = blockIdx.z;

    // store queries, keys, and values using default dtype (blfloat16 for llama3) to reduce smem usage
    __shared__ Precision sq[HEAD_SIZE * QKV_HEAD_RATIO];
    __shared__ Precision sk[HEAD_SIZE * STEP_SIZE];
    __shared__ Precision sv[HEAD_SIZE * STEP_SIZE];
    // and other flash attention intermidiate results using fp32 for better accuracy
    __shared__ float reduce_smem[NUM_WARPS];
    __shared__ float softmaxes[STEP_SIZE * QKV_HEAD_RATIO];
    __shared__ float scores[STEP_SIZE * QKV_HEAD_RATIO];

    int skv_size = HEAD_SIZE * STEP_SIZE;
    int n_tok = STEP_SIZE;

    // for each thread block, we execute in parallel flash attention 2 algorithm for STEP_SIZE tokens
    if (step == n_step - 1){
        n_tok = n_seq - STEP_SIZE * step;
        skv_size = HEAD_SIZE * n_tok;
    }
    int idx_idx =  int(topk_index[batch * idx_batch_stride + head * idx_head_stride + step * STEP_SIZE + tid/2]);

    int32_t kv_quant_2bit[8];
    uint8_t local_key_1bit[8];
    Precision local_quant_param[8];
    int sidx = tid*64;
    int gidx = batch * kv_batch_stride + head * kv_head_stride + idx_idx* kv_tok_stride + 8*(tid%2);   
    if (n_tok>tid/2){
        // printf("tid:%d,n_tok:%d\n",tid,n_tok);
    // BLOCK_SIZE=128 Head_SIZE=128 setp_size=64  so  we want each 2 thread read 1 kv dim

        reinterpret_cast<uint256 *>(&kv_quant_2bit[0])[0] = reinterpret_cast<uint256 *>(&key_value_quant[gidx])[0];
        
        reinterpret_cast<int2 *>(&local_key_1bit[0])[0] = reinterpret_cast<int2 *>(&key_1bit[gidx])[0];

        reinterpret_cast<int4 *>(&local_quant_param[0])[0] = reinterpret_cast<int4 *>(&quant_param[gidx])[0];
        #pragma unroll
        for (int k = 0; k < 64; ++k) {
            int j = k / 32;
            int x = (k % 32) / 8;
            int i = k % 8;
            int32_t quant_word = kv_quant_2bit[j * 4 + x];
            uint8_t sign_byte = local_key_1bit[j * 4 + x];

            float scale_k = p2float(local_quant_param[j * 4 + 0]);
            float zp_k    = p2float(local_quant_param[j * 4 + 1]);
            float scale_v = p2float(local_quant_param[j * 4 + 2]);
            float zp_v    = p2float(local_quant_param[j * 4 + 3]);

            float qk = float((quant_word >> (i * 2)) & 0x3);
            bool sign = (sign_byte >> i) & 0x1;
            qk = sign ? (qk * scale_k + zp_k) : -(qk * scale_k + zp_k);
            sk[sidx+k] = float2p(qk);

            float qv = float((quant_word >> ((i + 8) * 2)) & 0x3);
            qv = qv * scale_v + zp_v;
            sv[sidx+k] = float2p(qv);
        }
    }
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        sq[tid + query_idx * HEAD_SIZE] = q[tid + batch * q_stride + (head * QKV_HEAD_RATIO + query_idx) * HEAD_SIZE];
    }

    __syncthreads();
    const int THREAD_VEC_SIZE = HEAD_SIZE / THREAD_GROUP_SIZE;
    int group = tid / THREAD_GROUP_SIZE;
    int lane = tid % THREAD_GROUP_SIZE;

    float rq[THREAD_VEC_SIZE * QKV_HEAD_RATIO];
    #pragma unroll
    for (int idx = 0; idx < THREAD_VEC_SIZE; idx++){
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            rq[idx + query_idx * THREAD_VEC_SIZE] =  p2float(sq[lane + idx * THREAD_GROUP_SIZE + query_idx * HEAD_SIZE]);
        }
    }

    float rk[THREAD_VEC_SIZE];
    for (int idx = 0; idx < n_tok; idx += (BLOCK_SIZE * THREAD_VEC_SIZE) / HEAD_SIZE){
        #pragma unroll
        for (int jdx = 0; jdx < THREAD_VEC_SIZE; jdx++){
            if ((group + idx) < n_tok){
                rk[jdx] =  p2float(sk[lane + jdx * THREAD_GROUP_SIZE + (group + idx) * HEAD_SIZE]);
            }
            else
            {
                rk[jdx] = 0.0;
            }
        }

        float qk_dot[QKV_HEAD_RATIO];
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            qk_dot[query_idx] = 0.0;
            #pragma unroll
            for (int jdx = 0; jdx < THREAD_VEC_SIZE; jdx++){
                qk_dot[query_idx] += rq[jdx + query_idx * THREAD_VEC_SIZE] * rk[jdx];
            }
        }
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            #pragma unroll
            for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
                qk_dot[query_idx] += __shfl_xor_sync(0xFFFFFFFF, qk_dot[query_idx], mask);
            }
            if (lane == 0){
                scores[group + idx + query_idx * STEP_SIZE] = scale * qk_dot[query_idx];
            }
        }

    }
    __syncthreads();

    // continue the flash attention 2 algorithm:
    // 1) get the max of scores:
    // m = max(S)
    // 2) compute the exp of the diff between the max and the scores:
    // P = exp(S - m)
    // 3) compute the "online softmax":
    // l = sum(P)
    float curr_score_max[QKV_HEAD_RATIO];
    float softmax_sum[QKV_HEAD_RATIO];
    float score;
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        if (tid < n_tok){
            score = scores[tid + query_idx * STEP_SIZE];
        }
        else{
            score = -FLT_MAX;
        }

        curr_score_max[query_idx] = block_max<NUM_WARPS>(reduce_smem, score);

        float softmax;
        if (tid < n_tok){
            softmax = __expf(score - curr_score_max[query_idx]);
            softmaxes[tid + query_idx * STEP_SIZE] = softmax;
        }
        else{
            softmax = 0.0;
            softmaxes[tid + query_idx * STEP_SIZE] = softmax;
        }
        softmax_sum[query_idx] = block_sum<NUM_WARPS>(reduce_smem, softmax);
    }

    // compute the softmax * value dot product from the flash attention 2 algorithm:
    // O = P @ V
    float sv_dot[QKV_HEAD_RATIO];
    #pragma unroll
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            sv_dot[query_idx] = 0.0;
        }
    for (int jdx = 0; jdx < n_tok; jdx++){
        float sv_val = p2float(sv[tid + jdx * HEAD_SIZE]);
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            sv_dot[query_idx] += softmaxes[jdx + STEP_SIZE * query_idx] * sv_val;
        }
    }
    __syncthreads();
    #pragma unroll
    // multiply the output by the "online softmax"
    for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
        // printf("step:%d,pos:%d,%f,%f\n",step,tid + batch * q_num_heads * (n_step+step_shift) * HEAD_SIZE + (head * QKV_HEAD_RATIO + query_idx) * (n_step+step_shift) * HEAD_SIZE + (step+step_shift) * HEAD_SIZE,softmax_sum[query_idx],sv_dot[query_idx]);
        tmp_out[tid + batch * q_num_heads * (n_step+step_shift) * HEAD_SIZE + (head * QKV_HEAD_RATIO + query_idx) * (n_step+step_shift) * HEAD_SIZE + (step+step_shift) * HEAD_SIZE] = __fdividef(1.0f, softmax_sum[query_idx] + 1e-6f) * sv_dot[query_idx];

    }

    // store the maxes and softmaxes for each step (thread block), to be used by the combine kernel
    if (tid == 0){
        #pragma unroll
        for (int query_idx = 0; query_idx < QKV_HEAD_RATIO; query_idx++){
            
            score_maxes[batch * q_num_heads * (n_step+step_shift) + (head * QKV_HEAD_RATIO + query_idx) * (n_step+step_shift) + (step+step_shift) ] = curr_score_max[query_idx];
            softmax_sums[batch * q_num_heads * (n_step+step_shift) + (head * QKV_HEAD_RATIO + query_idx) * (n_step+step_shift) + (step+step_shift) ] = softmax_sum[query_idx];
        }
    }
}







template <typename Precision, int BLOCK_SIZE, int NUM_WARPS, int THREAD_GROUP_SIZE, int HEAD_SIZE, int STEP_SIZE>
__global__ void custom_flash_attention_combine_kernel(int n_seq, int n_step, int q_stride, int kv_batch_stride, int kv_head_stride, int kv_tok_stride, Precision* out, float* tmp_out, float* score_maxes, float* softmax_sums){
    const int tid = threadIdx.x;
    const int batch = blockIdx.x; 
    const int head = blockIdx.y;
    const int step = blockIdx.z;
    const int kv_num_heads = gridDim.y;

    __shared__ float reduce_smem[NUM_WARPS];
    extern __shared__ float shared_softmax_sums[];
    // if total number of tokens < step size, just return the output from the splitkv kernel 
    if (n_step == 1){
        if (tid < HEAD_SIZE){
            out[tid + batch * q_stride + head * HEAD_SIZE] = float2p(tmp_out[tid + batch * kv_num_heads * HEAD_SIZE + head * HEAD_SIZE]);
            return;
        }
    }

    // get the global scores max across all steps
    float score_max = -FLT_MAX;
    if (tid < n_step){
        score_max = score_maxes[batch * kv_num_heads * n_step + head * n_step + tid];
    }
    float global_score_max = block_max<NUM_WARPS>(reduce_smem, score_max);
    __syncthreads();

    // compute the global softmax across all steps
    float softmax_sum = 0.0;
    if (tid < n_step){
        softmax_sum = softmax_sums[batch * kv_num_heads * n_step + head * n_step + tid];
        softmax_sum *=  expf(score_max - global_score_max);
        shared_softmax_sums[tid] = softmax_sum;
    }
    float global_softmax_sum = block_sum<NUM_WARPS>(reduce_smem, softmax_sum);
    __syncthreads();
    
    // complete the flash attention 2 algorithm and return final result
    float ro = 0.0;
    for (int idx = 0; idx < n_step; idx++){
        float float_tmp_out = tmp_out[tid + batch * kv_num_heads * n_step * HEAD_SIZE + head * n_step * HEAD_SIZE + idx * HEAD_SIZE];
        ro += float_tmp_out * shared_softmax_sums[idx] * __fdividef(1.0f, global_softmax_sum + 1e-6f);
        // printf("head:%d,tid:%d,idx%d,float_tmp_out:%f\n",head,tid,idx,float_tmp_out);
    }
    out[tid + batch * q_stride + head * HEAD_SIZE] = float2p(ro);
}








void cudaCheck(cudaError_t error, const char* file, int line){
    if (error != cudaSuccess){
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

std::tuple<torch::Tensor, torch::Tensor> flash_attention_decode(torch::Tensor query, torch::Tensor keys, torch::Tensor values) {
    // TORCH_CHECK(false, "FUCK");
    // kernel parameters
    const int HEAD_SIZE = query.size(3);
    const int BLOCK_SIZE = HEAD_SIZE;
    // tunable parameter, where 8 val provided the best performance
    const int THREAD_GROUP_SIZE = 8;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int BATCH_SIZE = keys.size(0);
    // parameter for kv cache to fit into 48kb static smem for each thread block, and avoid using dynamic smem
    const int STEP_SIZE = 64;
    const int kv_num_heads = keys.size(1);
    const int q_num_heads = query.size(1);
    const int n_seq = keys.size(2);
    const int q_stride = query.stride(0);
    const int kv_batch_stride = keys.stride(0);
    const int kv_head_stride = keys.stride(1);
    const int kv_tok_stride = keys.stride(2);
    const int n_step = ceilDiv(n_seq, STEP_SIZE);
    const int QKV_HEAD_RATIO = q_num_heads / kv_num_heads;
    float scale = 1.0 / sqrt(HEAD_SIZE);

    // splitkv kernel grid
    dim3 split_grid;
    split_grid.x = BATCH_SIZE;
    split_grid.y = kv_num_heads;
    split_grid.z = n_step;

    // combine kernel grid
    dim3 combine_grid;
    combine_grid.x = BATCH_SIZE;
    combine_grid.y = q_num_heads;
    combine_grid.z = 1;

    // tensors for storing intermediates and results
    auto result = torch::empty_like(query);
    auto tmp_result = torch::empty({BATCH_SIZE, q_num_heads, n_step, HEAD_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto score_maxes = torch::empty({BATCH_SIZE, q_num_heads, n_step}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto softmax_sums = torch::empty({BATCH_SIZE, q_num_heads, n_step}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // macros for launching kernels
    #define FLASH_ATTENTION_SPLITKV(HEAD_SIZE, QKV_HEAD_RATIO) custom_flash_attention_splitkv_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE, QKV_HEAD_RATIO><<<split_grid, BLOCK_SIZE>>>(0,n_seq, n_step, q_num_heads, kv_num_heads, q_stride, kv_batch_stride, kv_head_stride, kv_tok_stride, scale, (Precision*)(query.data_ptr<P2Torch>()), (Precision*)(keys.data_ptr<P2Torch>()), (Precision*)(values.data_ptr<P2Torch>()), tmp_result.data_ptr<float>(), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());
    #define FLASH_ATTENTION_COMBINE(HEAD_SIZE) custom_flash_attention_combine_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE><<<combine_grid, BLOCK_SIZE, n_step * sizeof(float)>>>(n_seq, n_step, q_stride, kv_batch_stride, kv_head_stride, kv_tok_stride, (Precision*)(result.data_ptr<P2Torch>()), tmp_result.data_ptr<float>(), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());

    // launch kernel according to head size, and num query / keys, value head ratio
    switch (HEAD_SIZE){
        case 64:
            switch (QKV_HEAD_RATIO){
                case 1:
                    // general non grouped attention
                    FLASH_ATTENTION_SPLITKV(64, 1);
                    break;
                case 4:
                    // llama 3.2 1b
                    FLASH_ATTENTION_SPLITKV(64, 4);
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported QKV head ratio for splitkv kernel: ", QKV_HEAD_RATIO);
                    break;
            }
            break;
        case 128:
            switch (QKV_HEAD_RATIO){
                case 1:
                    // general non grouped attention
                    FLASH_ATTENTION_SPLITKV(128, 1);
                    break;
                case 3:
                    // llama 3.2 3b
                    FLASH_ATTENTION_SPLITKV(128, 3);
                    break;
                case 4:
                    // llama 3.1 8b
                    FLASH_ATTENTION_SPLITKV(128, 4);
                    break;
                case 8:
                    // llama 3.3 70b
                    FLASH_ATTENTION_SPLITKV(128, 8);
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported QKV head ratio for splitkv kernel: ", QKV_HEAD_RATIO);
                    break;
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported head size for splitkv kernel: ", HEAD_SIZE);
            break;
    }
    

    switch (HEAD_SIZE){
        case 64:
            // head size 64
            FLASH_ATTENTION_COMBINE(64);
            break;
        case 128:
            // head size 128
            FLASH_ATTENTION_COMBINE(128);
            break;
        default:
            TORCH_CHECK(false, "Unsupported head size for combine kernel: ", HEAD_SIZE);
            break;
    }
    
    return std::make_tuple(result,tmp_result);
}




std::tuple<torch::Tensor, torch::Tensor> quant_flash_attention_decode(torch::Tensor query,    torch::Tensor key_value_quant,
    torch::Tensor key_1bit_quant,
    torch::Tensor quant_param) {

    // kernel parameters
    const int HEAD_SIZE = query.size(3);
    const int BLOCK_SIZE = HEAD_SIZE;
    // tunable parameter, where 8 val provided the best performance
    const int THREAD_GROUP_SIZE = 8;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int BATCH_SIZE = key_value_quant.size(0);
    // parameter for kv cache to fit into 48kb static smem for each thread block, and avoid using dynamic smem
    const int STEP_SIZE = 64;
    const int kv_num_heads = key_value_quant.size(1);
    const int q_num_heads = query.size(1);
    const int n_seq = key_value_quant.size(2);
    const int q_stride = query.stride(0);
    const int kv_batch_stride = key_value_quant.stride(0);
    const int kv_head_stride = key_value_quant.stride(1);
    const int kv_tok_stride = key_value_quant.stride(2);
    const int n_step = ceilDiv(n_seq, STEP_SIZE);
    const int QKV_HEAD_RATIO = q_num_heads / kv_num_heads;
    float scale = 1.0 / sqrt(HEAD_SIZE);

    // splitkv kernel grid
    dim3 split_grid;
    split_grid.x = BATCH_SIZE;
    split_grid.y = kv_num_heads;
    split_grid.z = n_step;

    // combine kernel grid
    dim3 combine_grid;
    combine_grid.x = BATCH_SIZE;
    combine_grid.y = q_num_heads;
    combine_grid.z = 1;

    // tensors for storing intermediates and results
    auto result = torch::empty_like(query);
    auto tmp_result = torch::empty({BATCH_SIZE, q_num_heads, n_step, HEAD_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto score_maxes = torch::empty({BATCH_SIZE, q_num_heads, n_step}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto softmax_sums = torch::empty({BATCH_SIZE, q_num_heads, n_step}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // macros for launching kernels
    #define QUANT_FLASH_ATTENTION_SPLITKV(HEAD_SIZE, QKV_HEAD_RATIO) quant_flash_attention_splitkv_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE, QKV_HEAD_RATIO><<<split_grid, BLOCK_SIZE>>>(n_seq, n_step, q_num_heads, kv_num_heads, q_stride, kv_batch_stride, kv_head_stride, kv_tok_stride, scale, (Precision*)(query.data_ptr<P2Torch>()), key_value_quant.data_ptr<int32_t>(),  key_1bit_quant.data_ptr<uint8_t>(),(Precision*)(quant_param.data_ptr<P2Torch>()), tmp_result.data_ptr<float>(), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());
    #define QUANT_FLASH_ATTENTION_COMBINE(HEAD_SIZE) custom_flash_attention_combine_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE><<<combine_grid, BLOCK_SIZE, n_step * sizeof(float)>>>(n_seq, n_step, q_stride, kv_batch_stride, kv_head_stride, kv_tok_stride, (Precision*)(result.data_ptr<P2Torch>()), tmp_result.data_ptr<float>(), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());

    // launch kernel according to head size, and num query / keys, value head ratio
    switch (HEAD_SIZE){
        case 64:
            switch (QKV_HEAD_RATIO){
                case 1:
                    // general non grouped attention
                    QUANT_FLASH_ATTENTION_SPLITKV(64, 1);
                    break;
                case 4:
                    // llama 3.2 1b
                    QUANT_FLASH_ATTENTION_SPLITKV(64, 4);
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported QKV head ratio for splitkv kernel: ", QKV_HEAD_RATIO);
                    break;
            }
            break;
        case 128:
            switch (QKV_HEAD_RATIO){
                case 1:
                    // general non grouped attention
                    QUANT_FLASH_ATTENTION_SPLITKV(128, 1);
                    break;
                case 3:
                    // llama 3.2 3b
                    QUANT_FLASH_ATTENTION_SPLITKV(128, 3);
                    break;
                case 4:
                    // llama 3.1 8b
                    QUANT_FLASH_ATTENTION_SPLITKV(128, 4);
                    break;
                case 8:
                    // llama 3.3 70b
                    QUANT_FLASH_ATTENTION_SPLITKV(128, 8);
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported QKV head ratio for splitkv kernel: ", QKV_HEAD_RATIO);
                    break;
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported head size for splitkv kernel: ", HEAD_SIZE);
            break;
    }
    

    switch (HEAD_SIZE){
        case 64:
            // head size 64
            FLASH_ATTENTION_COMBINE(64);
            break;
        case 128:
            // head size 128
            FLASH_ATTENTION_COMBINE(128);
            break;
        default:
            TORCH_CHECK(false, "Unsupported head size for combine kernel: ", HEAD_SIZE);
            break;
    }
    
    return std::make_tuple(result,tmp_result);
}



std::tuple<torch::Tensor, torch::Tensor> select_quant_flash_attention_decode(torch::Tensor query,    torch::Tensor key_value_quant,
    torch::Tensor key_1bit_quant,
    torch::Tensor quant_param,
    torch::Tensor topk_index) {

    // kernel parameters
    const int HEAD_SIZE = query.size(3);
    const int BLOCK_SIZE = HEAD_SIZE;
    // tunable parameter, where 8 val provided the best performance
    const int THREAD_GROUP_SIZE = 8;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int BATCH_SIZE = key_value_quant.size(0);
    // parameter for kv cache to fit into 48kb static smem for each thread block, and avoid using dynamic smem
    const int STEP_SIZE = 64;
    const int kv_num_heads = key_value_quant.size(1);
    const int q_num_heads = query.size(1);
    const int n_seq = topk_index.size(2);
    const int q_stride = query.stride(0);
    const int kv_batch_stride = key_value_quant.stride(0);
    const int kv_head_stride = key_value_quant.stride(1);
    const int kv_tok_stride = key_value_quant.stride(2);
    const int index_batch_stride = topk_index.stride(0);
    const int index_head_stride = topk_index.stride(1);
    const int n_step = ceilDiv(n_seq, STEP_SIZE);
    const int QKV_HEAD_RATIO = q_num_heads / kv_num_heads;
    float scale = 1.0 / sqrt(HEAD_SIZE);

    // splitkv kernel grid
    dim3 split_grid;
    split_grid.x = BATCH_SIZE;
    split_grid.y = kv_num_heads;
    split_grid.z = n_step;

    // combine kernel grid
    dim3 combine_grid;
    combine_grid.x = BATCH_SIZE;
    combine_grid.y = q_num_heads;
    combine_grid.z = 1;

    // tensors for storing intermediates and results
    auto result = torch::empty_like(query);
    auto tmp_result = torch::empty({BATCH_SIZE, q_num_heads, n_step, HEAD_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto score_maxes = torch::empty({BATCH_SIZE, q_num_heads, n_step}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto softmax_sums = torch::empty({BATCH_SIZE, q_num_heads, n_step}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // macros for launching kernels
    #define QUANT_FLASH_ATTENTION_SPLITKV(HEAD_SIZE, QKV_HEAD_RATIO) select_quant_flash_attention_splitkv_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE, QKV_HEAD_RATIO><<<split_grid, BLOCK_SIZE>>>(0,n_seq, n_step, q_num_heads, kv_num_heads, q_stride, kv_batch_stride, kv_head_stride, kv_tok_stride,index_batch_stride,index_head_stride, scale, (Precision*)(query.data_ptr<P2Torch>()),topk_index.data_ptr<int64_t>(), key_value_quant.data_ptr<int32_t>(),  key_1bit_quant.data_ptr<uint8_t>(),(Precision*)(quant_param.data_ptr<P2Torch>()), tmp_result.data_ptr<float>(), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());
    #define QUANT_FLASH_ATTENTION_COMBINE(HEAD_SIZE) custom_flash_attention_combine_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE><<<combine_grid, BLOCK_SIZE, n_step * sizeof(float)>>>(n_seq, n_step, q_stride, kv_batch_stride, kv_head_stride, kv_tok_stride, (Precision*)(result.data_ptr<P2Torch>()), tmp_result.data_ptr<float>(), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());

    // launch kernel according to head size, and num query / keys, value head ratio
    switch (HEAD_SIZE){
        case 64:
            switch (QKV_HEAD_RATIO){
                case 1:
                    // general non grouped attention
                    QUANT_FLASH_ATTENTION_SPLITKV(64, 1);
                    break;
                case 4:
                    // llama 3.2 1b
                    QUANT_FLASH_ATTENTION_SPLITKV(64, 4);
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported QKV head ratio for splitkv kernel: ", QKV_HEAD_RATIO);
                    break;
            }
            break;
        case 128:
            switch (QKV_HEAD_RATIO){
                case 1:
                    // general non grouped attention
                    QUANT_FLASH_ATTENTION_SPLITKV(128, 1);
                    break;
                case 3:
                    // llama 3.2 3b
                    QUANT_FLASH_ATTENTION_SPLITKV(128, 3);
                    break;
                case 4:
                    // llama 3.1 8b
                    QUANT_FLASH_ATTENTION_SPLITKV(128, 4);
                    break;
                case 8:
                    // llama 3.3 70b
                    QUANT_FLASH_ATTENTION_SPLITKV(128, 8);
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported QKV head ratio for splitkv kernel: ", QKV_HEAD_RATIO);
                    break;
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported head size for splitkv kernel: ", HEAD_SIZE);
            break;
    }
    

    switch (HEAD_SIZE){
        case 64:
            // head size 64
            FLASH_ATTENTION_COMBINE(64);
            break;
        case 128:
            // head size 128
            FLASH_ATTENTION_COMBINE(128);
            break;
        default:
            TORCH_CHECK(false, "Unsupported head size for combine kernel: ", HEAD_SIZE);
            break;
    }
    
    return std::make_tuple(result,tmp_result);
}


std::tuple<torch::Tensor, torch::Tensor> mix_select_quant_flash_attention_decode(torch::Tensor query, 
    torch::Tensor full_key,
    torch::Tensor full_value,   
    torch::Tensor key_value_quant,
    torch::Tensor key_1bit_quant,
    torch::Tensor quant_param,
    torch::Tensor topk_index) {

    // kernel parameters
    const int HEAD_SIZE = query.size(3);
    const int BLOCK_SIZE = HEAD_SIZE;
    // tunable parameter, where 8 val provided the best performance
    const int THREAD_GROUP_SIZE = 8;
    const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int BATCH_SIZE = key_value_quant.size(0);
    // parameter for kv cache to fit into 48kb static smem for each thread block, and avoid using dynamic smem
    const int STEP_SIZE = 64;
    const int kv_num_heads = key_value_quant.size(1);
    const int q_num_heads = query.size(1);
    const int n_seq = topk_index.size(2);
    const int q_stride = query.stride(0);
    const int kv_batch_stride = key_value_quant.stride(0);
    const int kv_head_stride = key_value_quant.stride(1);
    const int kv_tok_stride = key_value_quant.stride(2);
    const int index_batch_stride = topk_index.stride(0);
    const int index_head_stride = topk_index.stride(1);
    const int n_step = ceilDiv(n_seq, STEP_SIZE);
    const int QKV_HEAD_RATIO = q_num_heads / kv_num_heads;
    float scale = 1.0 / sqrt(HEAD_SIZE);
    // printf("fuck:%d\n",n_step);
    // splitkv kernel grid
    dim3 split_grid;
    split_grid.x = BATCH_SIZE;
    split_grid.y = kv_num_heads;
    split_grid.z = n_step;

    // combine kernel grid
    dim3 combine_grid;
    combine_grid.x = BATCH_SIZE;
    combine_grid.y = q_num_heads;
    combine_grid.z = 1;



    const int full_batch_stride = full_key.stride(0);
    const int full_head_stride = full_key.stride(1);
    const int full_tok_stride = full_key.stride(2);
    dim3 full_split_grid;
    full_split_grid.x = BATCH_SIZE;
    full_split_grid.y = kv_num_heads;
    full_split_grid.z = 1;
    // tensors for storing intermediates and results
    auto result = torch::empty_like(query);
    auto tmp_result = torch::empty({BATCH_SIZE, q_num_heads, n_step+1, HEAD_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto score_maxes = torch::empty({BATCH_SIZE, q_num_heads, n_step+1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto softmax_sums = torch::empty({BATCH_SIZE, q_num_heads, n_step+1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    const int true_n_step = ceilDiv(n_seq, STEP_SIZE)+1;
    // printf("ffff:%d\n",true_n_step);
    // macros for launching kernels
    
    #define MIX_QUANT_FLASH_ATTENTION_SPLITKV(HEAD_SIZE, QKV_HEAD_RATIO) select_quant_flash_attention_splitkv_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE, QKV_HEAD_RATIO><<<split_grid, BLOCK_SIZE>>>(1,n_seq, n_step, q_num_heads, kv_num_heads, q_stride, kv_batch_stride, kv_head_stride, kv_tok_stride,index_batch_stride,index_head_stride, scale, (Precision*)(query.data_ptr<P2Torch>()),topk_index.data_ptr<int64_t>(), key_value_quant.data_ptr<int32_t>(),  key_1bit_quant.data_ptr<uint8_t>(),(Precision*)(quant_param.data_ptr<P2Torch>()), tmp_result.data_ptr<float>(), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());
    #define MIX_FLASH_ATTENTION_SPLITKV(HEAD_SIZE, QKV_HEAD_RATIO) custom_flash_attention_splitkv_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE, QKV_HEAD_RATIO><<<full_split_grid, BLOCK_SIZE>>>(1,n_seq, n_step, q_num_heads, kv_num_heads, q_stride, full_batch_stride, full_head_stride, full_tok_stride, scale, (Precision*)(query.data_ptr<P2Torch>()), (Precision*)(full_key.data_ptr<P2Torch>()), (Precision*)(full_value.data_ptr<P2Torch>()), tmp_result.data_ptr<float>(), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());
    #define MIX_QUANT_FLASH_ATTENTION_COMBINE(HEAD_SIZE) custom_flash_attention_combine_kernel<Precision, HEAD_SIZE, (HEAD_SIZE / WARP_SIZE), THREAD_GROUP_SIZE, HEAD_SIZE, STEP_SIZE><<<combine_grid, BLOCK_SIZE, (n_step+1) * sizeof(float)>>>(n_seq, true_n_step, q_stride, kv_batch_stride, kv_head_stride, kv_tok_stride, (Precision*)(result.data_ptr<P2Torch>()), (tmp_result.data_ptr<float>()), score_maxes.data_ptr<float>(), softmax_sums.data_ptr<float>());
    // launch kernel according to head size, and num query / keys, value head ratio
    switch (HEAD_SIZE){
        // case 64:
        //     switch (QKV_HEAD_RATIO){
        //         case 1:
        //             // general non grouped attention
        //             QUANT_FLASH_ATTENTION_SPLITKV(64, 1);
        //             break;
        //         case 4:
        //             // llama 3.2 1b
        //             QUANT_FLASH_ATTENTION_SPLITKV(64, 4);
        //             break;
        //         default:
        //             TORCH_CHECK(false, "Unsupported QKV head ratio for splitkv kernel: ", QKV_HEAD_RATIO);
        //             break;
        //     }
        //     break;
        case 128:
            switch (QKV_HEAD_RATIO){
                case 1:
                    // general non grouped attention
                    MIX_QUANT_FLASH_ATTENTION_SPLITKV(128, 1);
                    MIX_FLASH_ATTENTION_SPLITKV(128, 1);
                    break;
                case 3:
                    // llama 3.2 3b
                    MIX_QUANT_FLASH_ATTENTION_SPLITKV(128, 3);
                    MIX_FLASH_ATTENTION_SPLITKV(128, 3);
                    break;
                case 4:
                    // llama 3.1 8b
                    MIX_QUANT_FLASH_ATTENTION_SPLITKV(128, 4);
                    MIX_FLASH_ATTENTION_SPLITKV(128, 4);
                    break;
                case 8:
                    // llama 3.3 70b
                    MIX_QUANT_FLASH_ATTENTION_SPLITKV(128, 8);
                    MIX_FLASH_ATTENTION_SPLITKV(128, 8);
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported QKV head ratio for splitkv kernel: ", QKV_HEAD_RATIO);
                    break;
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported head size for splitkv kernel: ", HEAD_SIZE);
            break;
    }
    

    switch (HEAD_SIZE){
        case 64:
            // head size 64
            MIX_QUANT_FLASH_ATTENTION_COMBINE(64);
            break;
        case 128:
            // head size 128
            MIX_QUANT_FLASH_ATTENTION_COMBINE(128);
            break;
        default:
            TORCH_CHECK(false, "Unsupported head size for combine kernel: ", HEAD_SIZE);
            break;
    }
    
    return std::make_tuple(result,tmp_result);
}