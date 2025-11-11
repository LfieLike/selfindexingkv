import torch
# import my_extension  # 假设你的扩展名是my_extension，如果不是，替换为正确的名称
import cuda_my_scatter_add
import cuda_my_scatter_add_float
from functools import wraps
from test_kernel import gpu_timer,restore_matrix
@gpu_timer
def my_scatter_add(
        key_value_quantized_data,
        key_dequant_dst,
        value_dequant_dst,
        quant_param,
        quant_idx,
        num_elements,
        len_data
    ):
    cuda_my_scatter_add.My_scatter_add_half_half(
        key_value_quantized_data,
        key_dequant_dst,
        value_dequant_dst,
        quant_param,
        quant_idx,
        num_elements,
        len_data
    )
@gpu_timer
def my_scatter_add_float(
        key_value_quantized_data,
        key_dequant_dst,
        value_dequant_dst,
        quant_param,
        quant_idx,
        num_elements,
        len_data
    ):
    cuda_my_scatter_add_float.My_scatter_add_float(
        key_value_quantized_data,
        key_dequant_dst,
        value_dequant_dst,
        quant_param,
        quant_idx,
        num_elements,
        len_data
    )
@gpu_timer
def true_test(quant_idx,true_dequant_key,true_dequant_value,true_res_key,true_res_value,key_quant_zp,key_quant_scale,value_quant_zp,value_quant_scale):
    true_res_key = (true_res_key.float()-key_quant_zp)*key_quant_scale
    true_res_value =  (true_res_value.float()-value_quant_zp)*value_quant_scale
    indices_expanded = quant_idx.unsqueeze(-1).expand(-1, -1, -1, 128)
    cnt_true_dequant_key = true_dequant_key.float()
    cnt_true_dequant_key.scatter_add_(2,indices_expanded.to(torch.int64),true_res_key)
    cnt_true_dequant_value = true_dequant_value.float()
    cnt_true_dequant_value.scatter_add_(2,indices_expanded.to(torch.int64),true_res_value)
    return cnt_true_dequant_key,cnt_true_dequant_value
def get_quant_param(batch_size, head_size, seq_len, emb_dim):
    key_states = torch.randn(batch_size, head_size, seq_len, emb_dim, dtype=torch.float16, device='cuda')
    true_res = extract_sign_and_compress(key_states)
    key_quant = myquant(key_states)
    # 调用已绑定的CUDA内核函数进行量化处理
    key_quant = cuda_my_quant.My_quant_half_half(key_states)
    return key_states,outlier_zp,outlier_scale
def test_cuda_kernel():
    # 设置输入数据的尺寸
    batch_size = 10
    head_size = 32
    len_data = 12800  # 假设总长度为32
    emb_dim = 128
    num_elements = len_data//5
    pos = 0
    # 创建输入张量
    key_value_quantized_data = torch.randint(0, 256, (batch_size, head_size,num_elements, emb_dim), dtype=torch.uint8, device="cuda")
    # value_quantized_data = torch.randint(0, 256, (batch_size, head_size,num_elements,  emb_dim), dtype=torch.uint8, device="cuda")

    key_quant_zp = torch.randn(batch_size, head_size, num_elements,1, dtype=torch.float, device="cuda")
    key_quant_scale = torch.randint(3,8,(batch_size, head_size, num_elements,1), dtype=torch.float, device="cuda")
    value_quant_zp = torch.randn(batch_size, head_size, num_elements,1, dtype=torch.float, device="cuda")
    value_quant_scale = torch.randint(3,8,(batch_size, head_size, num_elements,1), dtype=torch.float, device="cuda")
    quant_param= torch.cat((key_quant_zp.view(-1,1),key_quant_scale.view(-1,1),value_quant_zp.view(-1,1),value_quant_scale.view(-1,1)),dim=-1).reshape(batch_size, head_size, num_elements*4)
    quant_idx = torch.argsort(quant_param[...,:num_elements],dim=-1)
    print(quant_idx.shape)
    # torch.randint(0, len_data, (batch_size, head_size, num_elements), dtype=torch.int32, device="cuda")
    key_dequant_dst = torch.randn((batch_size, head_size, len_data, emb_dim), dtype=torch.float, device="cuda")
    value_dequant_dst = torch.randn((batch_size, head_size, len_data, emb_dim), dtype=torch.float, device="cuda")
    true_res_value = key_value_quantized_data&0x0F
    true_res_key =  (key_value_quantized_data>>4)&0x0F
    true_dequant_key = key_dequant_dst.clone()
    true_dequant_value = value_dequant_dst.clone()
    # 调用CUDA内核
    my_scatter_add_float(
        key_value_quantized_data,
        key_dequant_dst,
        value_dequant_dst,
        quant_param,
        quant_idx.int(),
        num_elements,
        len_data
    )        
    indices_expanded = quant_idx.unsqueeze(-1).expand(-1, -1, -1, 128)

    # true_dequant.scatter_add_(2,indices_expanded.to(torch.int64),true_res_key.half())
    cnt_true_dequant_key,cnt_true_dequant_value = true_test(quant_idx,true_dequant_key,true_dequant_value,true_res_key,true_res_value,key_quant_zp,key_quant_scale,value_quant_zp,value_quant_scale)
    # 打印输出以验证结果
    print("Key Dequantized Output:")
    print("error_key",(key_dequant_dst-cnt_true_dequant_key).abs().mean())
    print("error_value",(value_dequant_dst-cnt_true_dequant_value).abs().mean())
    print(cnt_true_dequant_key.abs().mean())
    # print("Value Dequantized Output:")
    # print(value_dequant_dst)
    # print(true_dequant_value)
# 运行测试用例
for i in range(10):
    test_cuda_kernel()