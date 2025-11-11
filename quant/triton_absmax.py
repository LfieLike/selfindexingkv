import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl
from test_kernel import gpu_timer


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 32}, num_warps=8),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def abs_max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    offset_index = m_offset * K + pid_k

    # set mask
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset

    # 计算输入张量的绝对值
    inp_vals = tl.load(inp_ptrs, mask=mask)
    inp_vals = tl.abs(inp_vals)

    result_value, result_index = tl.max(inp_vals, axis=1, return_indices=True)

    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)

@gpu_timer
def abs_max_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS ABS MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.cuda.device(inp.device):
        abs_max_kernel[grid](inp, out_value, out_index, M, N, K)
    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value, indices=out_index)
    return out
@gpu_timer
def get_max(inp):
    
    # expected_values, expected_indices = torch.max(torch.abs(inp), dim=-2)
    expected_values, expected_indices = inp.abs().max(dim=-2)
    return expected_values, expected_indices
def test_abs_max_dim():
    # 创建测试张量
    inp = torch.randn((1,32,4096,128), device='cuda')

    # 在维度0上计算绝对值的最大值
    result = abs_max_dim(inp, dim=-2)

    # # 打印测试结果
    # print(f"输入张量: \n{inp}")
    # print(f"绝对值的最大值（维度0）: {result.values}")
    # print(f"绝对值的最大值索引（维度0）: {result.indices}")

    # 验证结果是否正确
    expected_values, expected_indices = get_max(inp)
    
    assert torch.all(result.values == expected_values), "测试失败：值结果不匹配"
    assert torch.all(result.indices == expected_indices), "测试失败：索引结果不匹配"
    print("测试通过：维度上的绝对值的最大值计算正确")


# 运行测试用例
test_abs_max_dim()
test_abs_max_dim()
test_abs_max_dim()
