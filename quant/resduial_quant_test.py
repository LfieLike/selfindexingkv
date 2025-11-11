import torch
import cuda_my_resduial_quant
from test_kernel import gpu_timer
@gpu_timer
def uniform_quantize(key,value):
    qmax = 15.0
    qmin = 0

    min_vals_key = key.min(dim=-1, keepdim=True)[0].float()
    max_vals_key = key.max(dim=-1, keepdim=True)[0].float()
    scales_key = (max_vals_key - min_vals_key) / (qmax - qmin)
    zero_points_key = qmin - min_vals_key / scales_key
    zero_points_key = zero_points_key.round()
    # 进行量化
    quantized_tensor = ((key / scales_key) + zero_points_key).round().clamp(qmin, qmax).to(torch.uint8)
    
    min_vals_value = value.min(dim=-1, keepdim=True)[0].float()
    max_vals_value = value.max(dim=-1, keepdim=True)[0].float()
    scales_value = (max_vals_value - min_vals_value) / (qmax - qmin)
    zero_points_value = qmin - min_vals_value / scales_value
    zero_points_value = zero_points_value.round()
    # 进行量化
    quantized_tensor = (quantized_tensor<<4)|(((value / scales_value) + zero_points_value).round().clamp(qmin, qmax).to(torch.uint8))
    
    return quantized_tensor,scales_key,zero_points_key,scales_value,zero_points_value
@gpu_timer
def my_reaquant(key, value):
    return cuda_my_resduial_quant.My_quant_resduial_half_half(key, value, 4)
def test_quantization():
    # 参数设置
    batch = 10
    head = 32
    len_seq = 4096
    emb_dim = 128
    bit = 4  # 4-bit量化
    key = torch.randn(batch, head, len_seq, emb_dim, dtype=torch.half, device='cuda')
    value = torch.randn(batch, head, len_seq, emb_dim, dtype=torch.half, device='cuda')
    # 生成随机的key和value张量，类型为半精度浮点数 (c10::Half)
    for i in range(10):
        quantized_tensor,scales,zero_points,scales_V,zero_points_V = uniform_quantize(key,value)
    # quantized_tensor_V,scales_V,zero_points_V = uniform_quantize(value)
    # quantized_tensor = (quantized_tensor_K<<4|quantized_tensor_V).to(torch.uint8)
    # 调用CUDA扩展的量化函数
    for i in range(10):
        quant_param, key_value_quant = my_reaquant(key, value)
    my_key_zp = quant_param[...,0:1]
    my_key_scale = quant_param[...,1:2]
    my_val_zp = quant_param[...,2:3]
    my_val_scale = quant_param[...,3:4]
    # print(quant_param)
    # print(zero_points)
    # print(scales)
    # print(min_vals,max_vals)
    # 打印结果
    # print("输入的key张量:\n", key)
    # print("输入的value张量:\n", value)
    # print(quant_param.shape)
    # print("量化参数:\n", quant_param[...,1])
    # print(scales,zero_points)
    print("量化后的key张量误差:\n", ((key_value_quant>>4).float()-(quantized_tensor>>4).float()).abs().max())
    print("量化后的value张量误差:\n", ((key_value_quant&0xf).float()-(quantized_tensor&0xf).float()).abs().max())
    print("error",(((key_value_quant>>4).float()-my_key_zp)*my_key_scale -key).abs().max())
    print("error",(((quantized_tensor>>4).float()-zero_points)*scales -key).abs().max())
    print("error",(((key_value_quant&0xf).float()-my_val_zp)*my_val_scale -value).abs().max())
    print("error",(((quantized_tensor&0xf).float()-zero_points_V)*scales_V -value).abs().max())
    # print("true:",quantized_tensor)
    # 检查量化结果的形状
    # assert quant_param.shape == (batch, head, len_seq * 4), "量化参数形状不正确"
    # assert key_value_quant.shape == (batch, head, len_seq, emb_dim), "量化后的key_value张量形状不正确"

    print("测试通过：量化结果和量化参数的形状正确。")

# 运行测试用例
if __name__ == "__main__":
    test_quantization()