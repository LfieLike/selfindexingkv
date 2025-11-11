import torch
# import my_extension  # 假设你的扩展名是my_extension，如果不是，替换为正确的名称
import cuda_my_quant
import cuda_my_key_dequant
import torch
from functools import wraps
def gpu_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 创建 CUDA 事件以记录时间
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        initial_allocated = torch.cuda.memory_allocated() / 1024**2
        initial_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"初始显存分配: {initial_allocated:.2f} MB, 预留: {initial_reserved:.2f} MB")
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
    # 打印最终显存占用
        final_allocated = torch.cuda.memory_allocated() / 1024**2
        final_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"最终显存分配: {final_allocated:.2f} MB, 预留: {final_reserved:.2f} MB")

        print(f"{func.__name__} executed in: {elapsed_time_ms:.3f} ms")
        return result

    return wrapper
@gpu_timer
def extract_sign_and_compress(tensor,outlier_scale,outlier_zp):
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

    quant_key = ((key_states / outlier_scale.to(torch.float32).unsqueeze(-2))+outlier_zp.to(torch.float32).unsqueeze(-2)).clamp(0, 255).round().to(torch.uint8)
    return compressed_signs,outlier_scale,outlier_zp,quant_key
@gpu_timer
def myquant(key_states,
                key_states_means,
        outlier_idx,
        outlier_zp,
        outlier_scale,
        outlier_num):
    return cuda_my_quant.My_quant_half_half(
        key_states,
        key_states_means,
        outlier_idx,
        outlier_zp,
        outlier_scale,
        outlier_num
    )
    #     torch::Tensor compressed_key,
    # torch::Tensor key_outlier_quant,
    # torch::Tensor dequant_dst,
    # torch::Tensor channel_mean,
    # torch::Tensor outlier_idx,
    # torch::Tensor quant_outlier_zp,
    # torch::Tensor quant_outlier_scale,
    # int outlier_num
@gpu_timer
def mydequant(
        compressed_key,
        key_outlier_quant,
        dequant_dst,
        channel_mean,
        outlier_idx,
        quant_outlier_zp,
        quant_outlier_scale,
        outlier_num
    ):
    
    
    cuda_my_key_dequant.My_key_dequant_half_half(
        compressed_key,
        key_outlier_quant,
        dequant_dst,
        channel_mean,
        outlier_idx,
        quant_outlier_zp,
        quant_outlier_scale,
        outlier_num
    )
@gpu_timer
def test_dequant(sign_key,channel_mean,outlier_idx,outlier_quant,my_outlier_zp_quant,my_outlier_scale_quant):
    dequant_outlier = (outlier_quant- my_outlier_zp_quant.unsqueeze(-2)) * my_outlier_scale_quant.unsqueeze(-2)
    true_res = restore_matrix(sign_key*channel_mean,-1,outlier_idx,dequant_outlier)
    return true_res
def test_quantization():
    # 测试参数
    batch_size = 4
    head_size = 32
    seq_len = 4096
    emb_dim = 128  # EMB_DIM固定为128

    # 创建输入张量，数据类型为float16，放置在CUDA设备上
    key_states = torch.randn(batch_size, head_size, seq_len, emb_dim, dtype=torch.float16, device='cuda')
    true_res = extract_sign_and_compress(key_states)
    key_quant = myquant(key_states)
    # 调用已绑定的CUDA内核函数进行量化处理
    key_quant = cuda_my_quant.My_quant_half_half(key_states)

    # 打印输出结果
    print("Input shape (key_states):", key_states.shape)
    print("Quantized output shape (key_quant):", key_quant.shape)
    # 验证输出张量的形状是否符合预期
    expected_shape = (batch_size, head_size, seq_len, emb_dim // 8)  # 因为8位压缩
    assert key_quant.shape == expected_shape, f"Expected shape {expected_shape}, but got {key_quant.shape}"


    
    # print("true",extract_sign_and_compress(key_states))  # 打印前10个输入的float16元素
    # print("ours:\n", key_quant)  # 打印前10个输出的量化结果
    # print("fuck",(true_res-key_quant.view(-1)).sum())
    # print("Test passed successfully!")
def zero_out_top_k_columns(matrix, dim,indices, k=10):
    # 找到绝对值最大的k列的索引
    indices = indices.expand(matrix.shape[0],-1,matrix.shape[2],-1)
    values = torch.gather(matrix, dim, indices.to(torch.int64))
    return  values
def restore_matrix(zeroed_matrix, dim, indices, values):
    restored_matrix = zeroed_matrix
    restored_matrix.scatter_(dim, indices.expand(restored_matrix.shape[0],-1,restored_matrix.shape[2],-1).to(torch.int64), values)
    return restored_matrix.half()
@gpu_timer
def get_idx(key_states,outlier_num):
    abs_matrix = torch.abs(key_states)
    abs_sum1 = abs_matrix.mean(dim=(2),keepdim=True)
    sorted_indices = torch.argsort(abs_sum1, dim=-1, descending=True)[...,:outlier_num].contiguous()
    outlier_idx = sorted_indices.to(torch.uint8)
    return outlier_idx
def test_quantize_with_outliers():
    # 设置测试数据
    batch_size = 2
    head_size = 32
    seq_len = 4096
    emb_dim = 128
    outlier_num = 12  # 假设每个embedding有12个outlier

    # 创建 key_states 张量 (随机初始化为 float16)
    key_states = torch.randn((batch_size, head_size, seq_len, emb_dim), device='cuda', dtype=torch.float16)
    

    # 创建 outlier 相关的张量
    # outlier_idx = torch.randint(0, emb_dim, (batch_size, head_size,1,outlier_num), device='cuda', dtype=torch.uint8)

    min_val, _ = torch.min(key_states, dim=-2)
    max_val, _ = torch.max(key_states, dim=-2)
    # 计算 scale 和 zp (量化范围是 [0, 255])
    outlier_scale = (max_val - min_val) / 255.0
    outlier_zp = -min_val / outlier_scale
    outlier_zp = torch.clamp(outlier_zp, 0, 255).round()
    channel_mean = key_states.abs().mean(dim=-2,keepdim=True)
    # print(outlier_scale[outlier_idx.to(torch.int32)])
    # outlier_zp = key_states.min(dim=-2)
    # outlier_zp = torch.zeros((batch_size, head_size, emb_dim), device='cuda', dtype=torch.float16)  # 零点初始化为 0
    # outlier_scale = torch.ones((batch_size, head_size, emb_dim), device='cuda', dtype=torch.float16)  # scale 初始化为 1
    # 调用量化函数
    for i in range(5):
        outlier_idx = get_idx(key_states,outlier_num)
    for i in range(5):
        print(outlier_idx.shape)
        key_quant, outlier_quant,my_outlier_zp_quant,my_outlier_scale_quant ,my_resduial = myquant(
            key_states,
            channel_mean,
            outlier_idx,
            outlier_zp,
            outlier_scale,
            outlier_num
        )
        true_res,outlier_scale_quant,outlier_zp_quant,quant_key = extract_sign_and_compress(key_states,outlier_scale,outlier_zp)
        outlier_org = zero_out_top_k_columns(quant_key,-1,outlier_idx,outlier_num)
        # outlier_org = ((outlier_org / my_outlier_scale_quant.to(torch.float32).unsqueeze(-2))+my_outlier_zp_quant.to(torch.float32).unsqueeze(-2)).clamp(0, 255).round().to(torch.uint8)
        # print((outlier_org-outlier_quant).abs().sum())
        # print()
    print("----")
    # quant_key=(key_states-outlier_zp.unsqueeze(-2))/(outlier_scale.unsqueeze(-2))
    get_idx
    true_outlier_scale = torch.gather(outlier_scale, -1, outlier_idx.squeeze(dim=-2).to(torch.int64))
    true_outlier_zp = torch.gather(outlier_zp, -1, outlier_idx.squeeze(dim=-2).to(torch.int64))
    print("1bit量化误差:", (true_res-key_quant.view(-1)).sum())
    print("outlier scale 误差",(true_outlier_scale-my_outlier_scale_quant).abs().sum())
    print("outlier zp 误差",(true_outlier_zp-my_outlier_zp_quant).abs().sum())
    print("提取的outlier的误差",((outlier_quant-outlier_org).abs()>1).sum())
    # print((outlier_quant-outlier_org).abs().sum())
    print()
    
    dst = torch.randn((batch_size, head_size, seq_len+10, emb_dim), device='cuda', dtype=torch.half)
    for i in range(5):
        mydequant(
            key_quant,
            outlier_quant,
            dst,
            channel_mean,
            outlier_idx,
            my_outlier_zp_quant,
            my_outlier_scale_quant,
            outlier_num
        )
    sign_key = torch.sign(key_states)
    sign_key[sign_key==0]=-1
    for i in range(5):
        true_res = test_dequant(sign_key,channel_mean,outlier_idx,outlier_quant,my_outlier_zp_quant,my_outlier_scale_quant)
    print(my_outlier_scale_quant.shape)
    # dequant_outlier = (outlier_quant- my_outlier_zp_quant.unsqueeze(-2)) * my_outlier_scale_quant.unsqueeze(-2)
    # true_res = restore_matrix(sign_key*channel_mean,-1,outlier_idx,dequant_outlier)
    

    print("dequant error",(dst[:,:,:seq_len,:].contiguous()-true_res).sum())
    # print(dst[:,:,:seq_len,:])
    # print(true_res)
    # print(dst[:,0,:seq_len,:].storage_offset())
    # print(dst[:,1,:seq_len,:].storage_offset())
    # print("dst",dst)
    # print(my_outlier_scale_quant.shape)
    # print(torch.sign(key_states)*channel_mean)
    # print(sorted_indices.shape)
        # print()
if __name__ == "__main__":
    test_quantize_with_outliers()
