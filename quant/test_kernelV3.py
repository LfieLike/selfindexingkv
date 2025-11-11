import torch
# import my_extension  # 假设你的扩展名是my_extension，如果不是，替换为正确的名称
import cuda_my_quant
import cuda_my_key_quant_V2
import cuda_my_key_dequant_V2
import cuda_my_key_value_quant_V2
import cuda_my_key_value_dequant_V2
import cuda_my_key_value_dequant_V3
import cuda_my_key_value_dequant_select
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
def myquant(key_states,value_states,zp_key,scale_key,zp_value,scale_value):
    return cuda_my_key_value_quant_V2.My_quant_half_half(
        key_states,value_states,zp_key,scale_key,zp_value,scale_value
    )
@gpu_timer
def mydequant(key_states,value_states,key_value_quant,key_1bit_quant,zp_key,scale_key,zp_value,scale_value,index):
    return cuda_my_key_value_dequant_V2.My_quant_half_half(key_states,value_states,key_value_quant,key_1bit_quant,zp_key,scale_key,zp_value,scale_value)
@gpu_timer
def mydequant_select(key_states,value_states,key_value_quant,key_1bit_quant,zp_key,scale_key,zp_value,scale_value,index):
    return cuda_my_key_value_dequant_select.My_quant_half_half(key_states,value_states,key_value_quant,key_1bit_quant,zp_key,scale_key,zp_value,scale_value,index)

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
def quantize_and_dequantize(input_matrix,bit=4):
    """
    对输入矩阵进行量化再反量化的函数
    :param input_matrix: 输入的torch.Tensor类型的矩阵，维度可以是二维等
    :return: 反量化后的矩阵，与输入矩阵形状相同
    """
    # 计算量化区间范围
    # print(input_matrix.shape)
    if bit==0:
        return input_matrix
    shape = input_matrix.shape
    # input_matrix = input_matrix.view(-1,block_size).float()
    ranges = 2**bit -1
    # input_matrix = input_matrix.view(-1,64*128)
    min_val = input_matrix.min(dim=-1,keepdim=True)[0].float()
    max_val = input_matrix.max(dim=-1,keepdim=True)[0].float()
    interval = (max_val - min_val) / ranges  # 划分成4个区间，所以是除以3
    # 进行量化
    quantized = torch.round((input_matrix - min_val) / interval)
    quantized = quantized.clamp(0, ranges) 

    # 进行反量化
    dequantized = (quantized * interval) + min_val
    # print(dequantized-input_matrix)
    return dequantized.reshape(shape).half()
@gpu_timer
def get_idx(key_states,outlier_num):
    abs_matrix = torch.abs(key_states)
    abs_sum1 = abs_matrix.mean(dim=(2),keepdim=True)
    sorted_indices = torch.argsort(abs_sum1, dim=-1, descending=True)[...,:outlier_num].contiguous()
    outlier_idx = sorted_indices.to(torch.uint8)
    return outlier_idx
def transposequant(key_states,slip=32):
    shape = key_states.shape
    sign = torch.sign(key_states)
    key_states = key_states.abs()
    key_states = key_states.view(shape[0],shape[1],shape[2]//slip,slip,shape[3])
    key_states = key_states.transpose(-1,-2).contiguous()
    
    dequant_key_states_4bit = quantize_and_dequantize(key_states,bit=2)
    key_states = dequant_key_states_4bit.transpose(-1,-2).contiguous()
    return key_states.view(shape)*sign
    # return key_states.view(shape)
def dequant(tensor):
    res=[]
    for i in range(16):
        res.append(tensor&0b11)
        tensor = tensor>>2
    return torch.cat(res,dim=-1)
def test_quantize_with_outliers():
    # 设置测试数据
    batch_size = 1
    head_size = 1
    seq_len = 32
    emb_dim = 128

    # 创建 key_states 张量 (随机初始化为 float16)
    key_states = torch.rand((batch_size, head_size, seq_len, emb_dim), device='cuda', dtype=torch.float16)-0.5
    key_states *=10
    # print(index.shape)
    # 创建 outlier 相关的张量
    # outlier_idx = torch.randint(0, emb_dim, (batch_size, head_size,1,outlier_num), device='cuda', dtype=torch.uint8)
    shape = key_states.shape
    key_states_needquant = key_states.abs().view(-1,32)
    _,index = torch.topk(key_states.mean(dim=-1,keepdim=True),k=seq_len,dim=-2)
    print(index.shape)
    sign = key_states.view(-1,32).sign()
    min_val, _ = torch.min(key_states_needquant, dim=-1,keepdim=True)
    max_val, _ = torch.max(key_states_needquant, dim=-1,keepdim=True)
    # 计算 scale 和 zp (量化范围是 [0, 255])
    scale = ((max_val - min_val) / 3).contiguous()
    zp = (min_val).contiguous()
    quantized = torch.round((key_states_needquant.float()-zp.float())/scale.float())
    quantized = (quantized.round()).clamp(0,3).to(torch.int8).float()
    
    # print(quantized[0,0,0,:])
    dequantized = (quantized*scale.float()+zp)*sign
    dequantized = dequantized.reshape(shape).half()
    # print("error:",(dequantized-dequantizedV2).abs().mean())
    # index = torch.range(0,shape[-2]-1).expand(shape[0],shape[1],-1).to(torch.int64).cuda()
    # print(index.shape)
    print(index)
    for i in range(1):
        key_value_quant,key_1bit = myquant(
            key_states,key_states,zp,scale,zp,scale
        )
        dequant_key = torch.zeros_like(key_states)
        dequant_value = torch.zeros_like(key_states)
        mydequant(dequant_key,dequant_value,key_value_quant,key_1bit,zp,scale,zp,scale,index)
        compressed_signs = extract_sign_and_compress(key_states)
        # select_dequant = torch.gather(dequantized,-2,index.expand(-1,-1,-1,128))
        res = torch.zeros_like(key_states).view(-1)
    print(torch.sign(dequant_key).sum(dim=-1))
    print(sign.reshape(shape).sum(dim=-1))
    # print(select_dequant)
    print((dequant_key))
    print(dequantized)
    # print(key_2bit)
    # print(format(key_2bit[0,0,0,0].to(torch.uint32).item(),'032b'))
    # print("----")
    # print(quantized[0,:])
    # print(key_2bit)
    # quant_key=(key_states-outlier_zp.unsqueeze(-2))/(outlier_scale.unsqueeze(-2))
    print("1bit量化误差:", (compressed_signs-key_1bit.view(-1)).sum())
    # print((res[:select_true_quant.numel()]))
if __name__ == "__main__":
    test_quantize_with_outliers()
