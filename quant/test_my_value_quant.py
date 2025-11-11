import torch
import cuda_my_value_quant,cuda_my_value_dequant
from test_kernel import gpu_timer
@gpu_timer
def uniform_quantize_value(tensor,maxvalue, level=1):
    org = tensor.clone().float()
    tensor = tensor.float()
    
    shapes = tensor.shape

    tensor = tensor/maxvalue
    # tensor = tensor.reshape(-1,128).float()
    # 确定量化的范围
    qmin = 0.
    qmax = level

    # 计算每一行的最小值和最大值
    min_vals = tensor.min(dim=-1, keepdim=True)[0]
    max_vals = tensor.max(dim=-1, keepdim=True)[0]
    # print("maxvalue:",maxvalue)
    # print(max_vals,min_vals)
    scales = (max_vals - min_vals) / (qmax - qmin)
    # scales[scales == 0] = 1  # 防止出现除以零的情况
    zero_points = qmin - min_vals / scales
    # print("float zp:",zero_points)
    zero_points = zero_points.round()
    # print(zero_points)
    # 进行量化
    quantized_tensor = ((tensor / scales) + zero_points).round().clamp(qmin, qmax)
    quantized_tensor = quantized_tensor.type(torch.uint8)

    # 进行反量化
    dequantized_tensor = (quantized_tensor - zero_points) * scales 
    dequantized_tensor = (dequantized_tensor)* maxvalue
    # print("dequant",dequantized_tensor)
    value_resduial = org - dequantized_tensor
        # 将符号位reshape成 (n, 8)，每一行代表8个符号位
    sign_bits = quantized_tensor.view(-1, 8)
    
    # 计算每行对应的uint8值，使用位移操作将8个符号位压缩为一个uint8
    compressed_signs = torch.zeros(sign_bits.size(0), dtype=torch.uint8, device=tensor.device)
    for i in range(8):
        compressed_signs |= (sign_bits[:, i] << i)
    return compressed_signs,quantized_tensor,zero_points.half(),scales.half(),value_resduial.half(),dequantized_tensor.half()
@gpu_timer
def my_value_quant(value_states,maxvalue):
    quant_param, value_quant,resduial = cuda_my_value_quant.My_quant_half_half(value_states,maxvalue)
    return quant_param, value_quant,resduial
@gpu_timer
def my_value_dequant(compressed_value,dequant_dst,channel_maxvalue,quant_param):
    cuda_my_value_dequant.My_value_dequant_half_half(compressed_value,dequant_dst,channel_maxvalue,quant_param)
# 定义测试用例
def test_my_quant_cuda():
    # 设置设备为CUDA
    device = torch.device("cuda")

    # 初始化输入数据
    batch_size = 1
    head_size = 1
    seq_len = 4096
    emb_dim = 128

    # 生成一个随机的float16张量作为输入
    value_states = torch.randn((batch_size, head_size, seq_len, emb_dim), device=device, dtype=torch.float16)+1
    maxvalue = value_states.abs().max(dim=-2,keepdim=True)[0]
    # print(maxvalue)
    # 调用CUDA扩展
    for i in range(5):
        quant_param, value_quant,resduial = my_value_quant(value_states,maxvalue.contiguous())
        dst = torch.zeros_like(value_states)
        compressed_signs,quantized_tensor,zero_points,scales,value_resduial,dequant_tensor = uniform_quantize_value(value_states,maxvalue.contiguous())
        my_zp = quant_param[...,0].view(zero_points.shape)
        my_scale = quant_param[...,1].view(scales.shape)
        my_value_dequant(value_quant,dst,maxvalue.contiguous(),quant_param)
        print("------------")
        print(dst)
        print(dequant_tensor)
        print((dst.float()-dequant_tensor.float()).abs().mean())
    # 打印结果
    # print((quant_param[...,0]))
    # print((value_quant-compressed_signs.reshape(value_quant.shape)).float().sum())
    # print(compressed_signs.shape)
    # print(quant_param)
    # print(zero_points.view(-1))
    print((quant_param[...,0].view(-1)-zero_points.view(-1)).sum())
    print((quant_param[...,1].view(-1)-scales.view(-1)).sum())
    print(value_resduial.float().abs().mean())
    print("myresduial",resduial.float().abs().mean())
    # print(zero_points)
    # # 验证结果
    # assert quant_param.shape == (batch_size, head_size, seq_len, 2), "Quantization parameters shape is incorrect"
    # assert value_quant.shape == (batch_size, head_size, seq_len, emb_dim // 8), "Quantized tensor shape is incorrect"
    print("Test passed!")

# 运行测试用例
if __name__ == "__main__":
    test_my_quant_cuda()
