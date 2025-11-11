import torch
# import my_extension  # 假设你的扩展名是my_extension，如果不是，替换为正确的名称
# import cuda_my_quant,cuda_my_score,cuda_my_scatter_add,cuda_my_score_half
# import cuda_my_key_dequant
# import cuda_my_resduial_quant
# import cuda_my_value_quant
# import cuda_my_value_dequant
# import cuda_my_key_quant_V2
# import cuda_my_key_dequant_V2
# import cuda_my_key_value_quant_V2
import cuda_my_key_value_dequant_V3,kvquantv3
import get_lut
import lut_gemv
import cuda_my_key_value_dequant_select,cuda_my_key_value_dequant_select_fuse
def my_cpu_gather(quant_kv, quant_param, index, output_kv, output_param):
    cpu_gather.my_cpu_gather(quant_kv, quant_param, index, output_kv, output_param)
def my_cpu_gather_all(quant_kv, quant_param,signs, index, output_kv, output_param,output_signs):
    cpu_gather_all.my_cpu_gather(quant_kv, quant_param,signs, index, output_kv, output_param,output_signs)
def my_cpu2gpu(key_value_quant_gpu, key_value_quant_cpu,quant_param_gpu,quant_param_cpu, select_index):
    return cpu2gpu.cpu2gpu(key_value_quant_gpu, key_value_quant_cpu,quant_param_gpu,quant_param_cpu, select_index)
def cpu_lut_gemvs(lut, key_states, res,true_len):
    return cpu_lut_gemv.cpu_lut_gemv(lut, key_states, res,true_len)
def my_key_value_quant_V3(key_states,value_states,zp_key,scale_key,zp_value,scale_value):
    return cuda_my_key_value_quant_V2.My_quant_half_half(
        key_states,value_states,zp_key,scale_key,zp_value,scale_value
    )
def my_key_value_dequant_V4(key_states,value_states,key_value_quant,key_1bit_quant,key_value_quant_param):
    # return cuda_my_key_value_dequant_V2.My_quant_half_half(key_states,value_states,key_value_quant,key_1bit_quant,zp_key,scale_key,zp_value,scale_value)
    return cuda_my_key_value_dequant_V3.My_dequant_half_half(key_states,value_states,key_value_quant,key_1bit_quant,key_value_quant_param)
def my_key_value_quant_V4(key_states,value_states):
    return kvquantv3.My_quant_half_half(key_states,value_states)
    # return key_value_quant,key_1bit_quant,key_value_quant_param
def my_key_value_dequant_V3(key_states,value_states,key_value_quant,key_1bit_quant,zp_key,scale_key,zp_value,scale_value):
    return cuda_my_key_value_dequant_V2.My_quant_half_half(key_states,value_states,key_value_quant,key_1bit_quant,zp_key,scale_key,zp_value,scale_value)
def mydequant_select(dequant_key,dequant_value,key_value_quant,key_1bit_quant,key_value_quant_param,index):
    return cuda_my_key_value_dequant_select.My_dequant_half_half(dequant_key,dequant_value,key_value_quant,key_1bit_quant,key_value_quant_param,index)
def mydequant_select_fuse(dequant_key,dequant_value,key_value_quant,key_1bit_quant,key_value_quant_param,kabs,vabs,index,full_len=0):
    return cuda_my_key_value_dequant_select_fuse.My_dequant_half_half(dequant_key,dequant_value,key_value_quant,key_1bit_quant,key_value_quant_param,kabs,vabs,index,full_len)
def mykeyquantV2(key_states,zp,scale,group_size):
    return cuda_my_key_quant_V2.My_key_quant_V2(
        key_states,zp,scale,group_size
    )
# def mykeydequantV2_sub(key_states_1bit,key_states_2bit,dequant_dst,zp,scale,index,group_size):
#     return cuda_my_key_dequant_V2.My_key_dequant_V2(key_states_1bit,key_states_2bit,dequant_dst,zp,scale,index,group_size)
def mykeydequantV2(key_states_1bit,key_states_2bit,dequant_dst,zp,scale,group_size):
    shape = key_states_1bit.shape
    index = torch.range(0,shape[-2]-1).expand(shape[0],shape[1],-1).to(torch.int64).cuda()
    return cuda_my_key_dequant_V2.My_key_dequant_V2(key_states_1bit,key_states_2bit,dequant_dst,zp,scale,index,group_size)
def ontbitquant(key_states,
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
def onebitgemv(
        key_states,
        outlier_quant,
        outlier_idx,
        outlier_zp,
        outlier_scale,
        query_states,
        outlier_num
    ):
    return cuda_my_score.My_score_half_half(
        key_states,
        outlier_quant,
        outlier_idx,
        outlier_zp,
        outlier_scale,
        query_states,
        outlier_num
    )
def onebitgemv(
        key_states,
        query_states
    ):
    return cuda_my_score_half.My_score_half_half(
        key_states,
        query_states
    )
def onebitgemv_withoutlier(
        key_states,
        outlier_quant,
        outlier_idx,
        outlier_zp,
        outlier_scale,
        query_states,
        outlier_num
    ):
    return cuda_my_score.My_score_half_half(
        key_states,
        outlier_quant,
        outlier_idx,
        outlier_zp,
        outlier_scale,
        query_states,
        outlier_num
    )
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
def my_lutgemv(key1bit,query_states):
    # print(query_states.shape)
    lut = get_lut.My_get_lut_half_float(query_states.contiguous())
    # print(lut)
    return lut_gemv.lut_gemv_half_half(key1bit.contiguous(),lut.contiguous())
def vq_lutgemv(key1bit,tables):
    # print(query_states.shape)
    # print(lut)
    return lut_gemv.lut_gemv_half_half(key1bit.contiguous(),tables.contiguous())
def my_resduialquant(key, value):
    return cuda_my_resduial_quant.My_quant_resduial_half_half(key, value, 4)
def my_value_quant(value_states,maxvalue):
    quant_param, value_quant,resduial = cuda_my_value_quant.My_quant_half_half(value_states,maxvalue)
    return quant_param, value_quant,resduial
def my_value_dequant(compressed_value,dequant_dst,channel_maxvalue,quant_param):
    cuda_my_value_dequant.My_value_dequant_half_half(compressed_value,dequant_dst,channel_maxvalue,quant_param)
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
    sign_bits = sign_bits.reshape(-1, 8)
    
    # 计算每行对应的uint8值，使用位移操作将8个符号位压缩为一个uint8
    compressed_signs = torch.zeros(sign_bits.size(0), dtype=torch.uint8, device=tensor.device)
    for i in range(8):
        compressed_signs |= (sign_bits[:, i] << i)
    
    return compressed_signs.view(*shape[:-1], -1), sign_tensor