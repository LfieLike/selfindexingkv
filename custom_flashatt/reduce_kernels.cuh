#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

const int WARP_SIZE = 32;

// reduce kernel that sums inputs across a thread block
template <int NUM_WARPS>
inline __device__ float block_sum(float* sdata, float sum) {
    // get warp idx et intra warp lane idx for each thread idx
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // reduce across a warp
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask);
    }

    // a thread from each warp stores local sum into smem 
    if (lane == 0) {
        sdata[warp] = sum;
    }
    __syncthreads();

    // recompute sum again across warps to get final result
    if (lane < NUM_WARPS) {
        sum = sdata[lane];
    }

    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask);
    }

    // broadcast result
    return __shfl_sync(0xFFFFFFFF, sum, 0);
}

// same reduce kernel as block_sum, but uses max op instead
template <int NUM_WARPS>
inline __device__ float block_max(float* sdata, float score) {
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        score = fmaxf(score, __shfl_xor_sync(0xFFFFFFFF, score, mask));
    }
    
    if (lane == 0) {
        sdata[warp] = score;
    }
    __syncthreads();

    score = lane < NUM_WARPS ? sdata[lane] : -FLT_MAX;

    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        score = fmaxf(score, __shfl_xor_sync(0xFFFFFFFF, score, mask));
    }
    
    return __shfl_sync(0xFFFFFFFF, score, 0);
}