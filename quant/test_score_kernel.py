import torch
import cuda_my_score  # 假设编译后的模块名称为 cuda_my_quant
import cuda_my_score_half
from functools import wraps
def gpu_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 创建 CUDA 事件以记录时间
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 同步以确保之前的操作完成
        torch.cuda.synchronize()

        # 记录起始时间
        start_event.record()

        # 调用实际的函数
        result = func(*args, **kwargs)

        # 记录结束时间
        end_event.record()

        # 同步确保计算完成
        torch.cuda.synchronize()

        # 计算经过的时间（以毫秒为单位）
        elapsed_time_ms = start_event.elapsed_time(end_event)

        print(f"{func.__name__} executed in: {elapsed_time_ms:.3f} ms")
        return result

    return wrapper


def extract_sign_and_compress(tensor):
    """
    提取Tensor中每个元素的符号位，并将每8个符号位压缩成一个uint8数。同时返回未压缩的符号张量，
    正数为1，负数为-1。
    :param tensor: 输入Tensor，数据类型为浮点类型
    :return: (符号位压缩后的uint8 Tensor, 没有压缩的符号位 Tensor)
    """
    # 提取符号位：正数为1，负数为-1
    shape = tensor.shape
    # sign_tensor = torch.where(tensor >= 0, torch.tensor(1, device=tensor.device), torch.tensor(-1, device=tensor.device))

    # 将符号位转换为0/1的形式，方便压缩
    sign_bits = (tensor >= 0).to(torch.uint8)
    sign_tensor = sign_bits.to(torch.float32)
    sign_tensor = sign_tensor*2 -1
    # 计算每8个符号位如何组合成一个uint8
    # 需要保证输入的长度是8的倍数，若不是8的倍数则进行填充
    num_elements = sign_bits.numel()
    padded_size = (num_elements + 7) // 8 * 8  # 向上取整为8的倍数
    padding = padded_size - num_elements
    
    # 对Tensor进行填充，保证符号位总数是8的倍数
    if padding > 0:
        sign_bits = torch.cat([sign_bits, torch.zeros(padding, dtype=torch.uint8, device=tensor.device)], dim=0)
    
    # 将符号位 reshape 成 (n, 8)，每一行代表8个符号位
    sign_bits = sign_bits.view(-1, 8)
    
    # 计算每行对应的uint8值，使用位移操作将8个符号位压缩为一个uint8
    compressed_signs = torch.zeros(sign_bits.size(0), dtype=torch.uint8, device=tensor.device)
    for i in range(8):
        compressed_signs |= (sign_bits[:, i] << i)
    
    return compressed_signs.view(*shape[:-1], -1), sign_tensor

@gpu_timer
def my_test(
        key_states,
        outlier_quant,
        outlier_idx,
        outlier_zp,
        outlier_scale,
        query_states,
        outlier_num
    ):
    # return cuda_my_score_half.My_score_half_half(
    #     key_states,
    #     query_states
    # )
    return cuda_my_score.My_score_half_half(
        key_states,
        outlier_quant,
        outlier_idx,
        outlier_zp,
        outlier_scale,
        query_states,
        outlier_num
    )
@gpu_timer
def my_test2(
        key_states,
        outlier_quant,
        outlier_idx,
        outlier_zp,
        outlier_scale,
        query_states,
        outlier_num
    ):
    return cuda_my_score_half.My_score_half_half(
        key_states,
        query_states
    )
@gpu_timer
def simple_test(query_states,org_key):
    return torch.matmul(query_states.unsqueeze(-2),org_key.transpose(-1,-2))
def mockoutlier(batch_size, head_size, seq_len, outlier_num):
    key_states = torch.randn((batch_size, head_size, seq_len, outlier_num), device='cuda')
    min_val, _ = torch.min(key_states, dim=-2)
    max_val, _ = torch.max(key_states, dim=-2)
    # 计算 scale 和 zp (量化范围是 [0, 255])
    outlier_scale = (max_val - min_val) / 255.0
    outlier_zp = -min_val / outlier_scale
    outlier_zp = torch.clamp(outlier_zp, 0, 255).round()
    quant_key = ((key_states / outlier_scale.to(torch.float32).unsqueeze(-2))+outlier_zp.to(torch.float32).unsqueeze(-2)).clamp(0, 255).round().to(torch.uint8)
    return quant_key,outlier_zp.half(),outlier_scale.half()
def test_my_score_cuda():
    # 输入维度设置
    batch_size = 1
    head_size = 32
    seq_len = 64000 # token length
    emb_dim = 128
    outlier_num = 0  # 假设有 8 个 outlier

    # 创建输入数据
    # key_states = torch.ones((batch_size, head_size, seq_len, emb_dim//8), device='cuda', dtype=torch.uint8)*255
    org_key = torch.randn((batch_size, head_size, seq_len, emb_dim), device='cuda',dtype=torch.float16)
    key_states,sign_tensor = extract_sign_and_compress(org_key)
    key_states = key_states.view((batch_size, head_size, seq_len, emb_dim//8))
    outlier_quant,outlier_zp,outlier_scale = mockoutlier(batch_size, head_size, seq_len, outlier_num)
    # outlier_quant = torch.randint(0, 128, (batch_size, head_size, seq_len, outlier_num), device='cuda', dtype=torch.uint8)
    # outlier_zp = torch.ones((batch_size, head_size, outlier_num), device='cuda', dtype=torch.float16)
    # outlier_scale = torch.ones((batch_size, head_size, outlier_num), device='cuda', dtype=torch.float16)
    outlier_idx = torch.randint(0, emb_dim-1, (batch_size, head_size, outlier_num), device='cuda', dtype=torch.uint8)
    query_states = torch.randn((batch_size, head_size, emb_dim), device='cuda', dtype=torch.float16)
    dequant = (outlier_quant.float()-outlier_zp.float().unsqueeze(-2))*outlier_scale.float().unsqueeze(-2)
    print(dequant)
    print(outlier_idx)
    broadcast_idx = outlier_idx.unsqueeze(-2).expand(batch_size, head_size, seq_len, outlier_num)
    print(dequant.shape)
    print(sign_tensor.shape)
    res = torch.scatter_add(sign_tensor, -1, broadcast_idx.to(torch.int64), dequant)
    # print("diff!!!!:",(res-sign_tensor).sum())
    scatter_query=query_states.gather(dim = -1,index = outlier_idx.to(torch.int64))
    print()
    # print((scatter_query-query_states).sum())
    # print(query_states)
    # 调用自定义的CUDA核函数，进行half精度量化测试
    for i in range(10):
        result = my_test(
            key_states,
            outlier_quant,
            outlier_idx,
            outlier_zp,
            outlier_scale,
            query_states,
            outlier_num
        )
        result2 = my_test2(
            key_states,
            outlier_quant,
            outlier_idx,
            outlier_zp,
            outlier_scale,
            query_states,
            outlier_num
        )
        true_result = simple_test(query_states=query_states.to(torch.float16), org_key=res.to(torch.float16))
        simple_test(query_states=query_states[...,:8].to(torch.float16), org_key=res[...,:8].to(torch.float16))
    print("-------------")
    # 打印结果
    print("diff:",(true_result.to(torch.float16).view(-1)-result.to(torch.float16).view(-1)).abs().mean())
    print("diff:",(result2.to(torch.float16).view(-1)-result.to(torch.float16).view(-1)).abs().mean())
    print(result.abs().mean())
    # 可在此处进行一些简单的验证，例如验证返回形状是否正确：
    
    diff_mask = (true_result-result.view(true_result.shape)).abs() > 0.001  # torch.ne: element-wise 'not equal'
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    # print(true_result[diff_indices[0]])
    # print("Indices of different elements:", diff_indices)

    # for item in diff_indices:
    #     cnt = item
    #     print(cnt)
    #     # print()
    #     a = dequant[0,0,item[-1],:]
    #     # b = 
    #     print(a)
    #     print(dequant[0,0,item[-1]-1,:])
    #     print("error:",scatter_query.float()@a)
    #     print("error:",scatter_query.float()@dequant[0,0,item[-1]-1,:])
    #     print("-----------")
    assert result.shape == (batch_size, head_size, seq_len), f"Unexpected result shape: {result.shape}"

if __name__ == '__main__':
    test_my_score_cuda()
