#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// set precision dtype, only blfoat16 is supported for now (used by llama 3 models)
typedef __half Precision;

// vector size for loading kv cache from global mem
const int VEC_SIZE = 16 / sizeof(Precision);

template <typename T, int VEC_SIZE>
struct Vec {};
template <>
struct Vec<__half, 2> {
  using Type = __half2;
};
struct bf16_4_t {
  __half2 x;
  __half2 y;
};
template <>
struct Vec<__half, 4> {
  using Type = bf16_4_t;
};
struct bf16_8_t {
  __half2 x;
  __half2 y;
  __half2 z;
  __half2 w;
};
template <>
struct Vec<__half, 8> {
  using Type = bf16_8_t;
};
using Precision_Vec = typename Vec<Precision, VEC_SIZE>::Type;

// convert kernel dtype to torch dtype
template <typename T>
struct Precision2Torch {};
template <>
struct Precision2Torch<__half> {
  using Type = at::Half;
};
using P2Torch = typename Precision2Torch<Precision>::Type;

// convert dtype from and to float32

inline __device__ Precision float2p(const float a){
    return  __float2half(a);
}

inline __device__ float p2float(const Precision a){
    return   __half2float(a);
}
struct __align__(32) uint256{
    uint64_t val[4];
};