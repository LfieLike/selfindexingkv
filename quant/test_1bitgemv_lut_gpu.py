import torch
import cuda_my_score  # 假设编译后的模块名称为 cuda_my_quant
import cuda_my_score_half
import get_lut
import kvquantv3
import lut_gemv
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


@gpu_timer
def extract_sign_and_compress(tensor):
    """
    提取Tensor中每个元素的符号位，并将每8个符号位压缩成一个uint8数
    :param tensor: 输入Tensor，数据类型为浮点类型
    :return: 符号位压缩后的uint8 Tensor
    """
    # 提取符号位：正数为1，负数为0
    key_states = tensor
    sign_bits = (tensor > 0).to(torch.uint8)
    
    # 计算每8个符号位如何组合成一个uint8
    # 需要保证输入的长度是8的倍数，若不是8的倍数则进行填充
    num_elements = sign_bits.numel()
    padded_size = (num_elements + 7) // 8 * 8  # 向上取整为8的倍数
    padding = padded_size - num_elements
    
    # 对Tensor进行填充，保证符号位总数是8的倍数
    if padding > 0:
        sign_bits = torch.cat([sign_bits, torch.zeros(padding, dtype=torch.uint8, device=tensor.device)], dim=0)
    
    # 将符号位reshape成 (n, 8)，每一行代表8个符号位
    sign_bits = sign_bits.view(-1, 8)
    
    # 计算每行对应的uint8值，使用位移操作将8个符号位压缩为一个uint8
    compressed_signs = torch.zeros(sign_bits.size(0), dtype=torch.uint8, device=tensor.device)
    for i in range(8):
        compressed_signs |= (sign_bits[:, i] << i)
        min_val, _ = torch.min(key_states, dim=-2)

    return compressed_signs

@gpu_timer
def my_test(
        key_states,
        query_states
    ):
    return cuda_my_score_half.My_score_half_half(
        key_states,
        query_states,
        0
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
@gpu_timer
def mock_lut(query,tensor,shape=4):
    # return  get_lut.My_get_lut_half_float(query)
    return query.float().view(-1,shape)@tensor.T
@gpu_timer
def my_lut(query):
    return  get_lut.My_get_lut_half_float(query)
    # return query.float().view(-1,4)@tensor.T
@gpu_timer
def my_lutgemv(key1bit,query_states):
    lut = get_lut.My_get_lut_half_float(query_states)
    return lut_gemv.lut_gemv_half_half(key1bit,lut)
@gpu_timer
def gemv_test(org_key,query_states):
    return torch.matmul(org_key,query_states)
def test_my_score_cuda(K=16):
    # 输入维度设置
    batch_size = 10
    head_size = 8
    seq_len = K*1024 # token length
    emb_dim = 128
    outlier_num = 8  # 假设有 8 个 outlier

    # 创建输入数据
    # key_states = torch.ones((batch_size, head_size, seq_len, emb_dim//8), device='cuda', dtype=torch.uint8)*255
    org_key = torch.randn((batch_size, head_size, seq_len, emb_dim), device='cuda',dtype=torch.float16)
    key_states= extract_sign_and_compress(org_key)

    query_states = torch.randn((batch_size, head_size, 1,emb_dim), device='cuda', dtype=torch.float16)*10
    a = get_lut.My_get_lut_half_float(query_states)

    # print(a.view(-1,16))
    combinations = [
    [ -1, -1, -1, -1],  # 0
    [  1, -1, -1, -1],  # 1
    [ -1,  1, -1, -1],  # 2
    [  1,  1, -1, -1],  # 3
    [ -1, -1,  1, -1],  # 4
    [  1, -1,  1, -1],  # 5
    [ -1,  1,  1, -1],  # 6
    [  1,  1,  1, -1],  # 7
    [ -1, -1, -1,  1],  # 8
    [  1, -1, -1,  1],  # 9
    [ -1,  1, -1,  1],  # 10
    [  1,  1, -1,  1],  # 11
    [ -1, -1,  1,  1],  # 12
    [  1, -1,  1,  1],  # 13
    [ -1,  1,  1,  1],  # 14
    [  1,  1,  1,  1],  # 15
]

    # 将组合转换成PyTorch tensor
    tensor = torch.tensor(combinations).cuda().float()
    _,key1bit,_ = kvquantv3.My_quant_half_half(org_key,org_key)
    mock1bit = extract_sign_and_compress(org_key)
    org_key_1 = torch.randn((batch_size, head_size, seq_len, emb_dim//8), device='cuda',dtype=torch.float16)
    query_states = torch.randn((batch_size, head_size, 1,emb_dim), device='cuda', dtype=torch.float16)*10
    query_1 = torch.randn((batch_size, head_size, 1,emb_dim//8), device='cuda', dtype=torch.float16)*10
    # combinations = []
    
    for i in range(10):
        res = my_lutgemv(key1bit,query_states)
        # mock_res = torch.matmul(torch.sign(org_key).float(),query_states.transpose(-1,-2).float()).half()
        mock_res = gemv_test(org_key.float(),query_states.float().transpose(-1,-2))


        mock_res_1 = gemv_test(org_key_1,query_1.transpose(-1,-2))
        print("diff",(res.view(-1)-mock_res.view(-1)).abs().mean())
        print(key1bit.shape,a.shape,key1bit.view(-1)-mock1bit.view(-1))
    
    # print(torch.sign(org_key).view(-1)[:16],key1bit.view(-1)[:2])
    # print(a.view(-1,16)[0,:],query_states)
    # for i in range(256):
    #     # 获取当前数字的二进制表示，确保是8位
    #     bin_rep = format(i, '08b')
    #     # 将二进制表示转换为1和-1的组合，低位在右
    #     combination = [1 if bit == '1' else -1 for bit in bin_rep[::-1]]
    #     combinations.append(combination)

    # # 将组合转换成PyTorch tensor
    # tensor2 = torch.tensor(combinations).cuda().float()
    # for i in range(5):
    #     res = my_lutgemv(key1bit,query_states)
    #     print((mock_lut(query_states,tensor).view(a.shape)-my_lut(query_states).view(a.shape)).sum())
if __name__ == '__main__':
    for K in [16,32,64]:
        print("input len",K)
        test_my_score_cuda(K=K)

