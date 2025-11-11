import torch
import time
import cpu_lut_gemv,get_lut,lut_gemv
def my_lutgemv(key1bit,query_states):
    lut = get_lut.My_get_lut_half_float(query_states)
    return lut_gemv.lut_gemv_half_half(key1bit,lut)
for K in [16,32,64]:
    batch_size = 1
    num_heads = 8
    seq_len = K*1024
    dim = 16

    # 创建随机的LUT表和key_states
    combinations = []
    key_states = torch.randint(0, 256, (batch_size, num_heads, seq_len, dim), dtype=torch.uint8).pin_memory()
    key_states_cuda = key_states.cuda()
    query_states = torch.randn((batch_size, num_heads, 1,128), device='cuda', dtype=torch.float16)
    for i in range(256):
        # 获取当前数字的二进制表示，确保是8位
        bin_rep = format(i, '08b')
        # 将二进制表示转换为1和-1的组合，低位在右
        combination = [1 if bit == '1' else -1 for bit in bin_rep[::-1]]
        combinations.append(combination)
    tensor2 = torch.tensor(combinations).T.cuda().float().cpu()

    res_cuda = my_lutgemv(key_states_cuda,query_states)
    query_states = query_states.view(-1,8).float().cpu()
    # key_states = torch.cat((key_states,key_states),dim=-2)
    for i in range(10):
        # lut = torch.randn((batch_size * num_heads, 4096), dtype=torch.float32)

        res = torch.zeros((batch_size, num_heads, seq_len), dtype=torch.float32)
        start_time = time.time()
        lut = torch.matmul(query_states.view(-1,8).float(),tensor2).view(batch_size * num_heads, 4096)
        res = cpu_lut_gemv.cpu_lut_gemv(lut, key_states, res,seq_len)
        end_time = time.time()
        # 	# 输出执行时间
        # time.sleep(5)
        print(f"{K}K input, my cpu gemv operation took {end_time - start_time:.4f} seconds.")
        d,c = torch.rand(batch_size,num_heads,seq_len,128),torch.rand(batch_size,num_heads,128,1)
        start_time = time.time()
        torch.matmul(d,c)
        torch.cuda.synchronize()
        end_time = time.time()
        # 输出执行时间
        # time.sleep(5)
        print(f"{K}K input, naive cpu gemv operation took {end_time - start_time:.4f} seconds.")
    # 打印结果
    # print("Result tensor after lut_gemv:")
    # print(res)
    expected_res = torch.zeros((batch_size, num_heads, seq_len), dtype=torch.float32)

    # for batch_id in range(batch_size):
    #     for head_id in range(num_heads):
    #         lut_offset = (batch_id * num_heads + head_id) * 4096
    #         key_offset = (batch_id * num_heads + head_id) * seq_len * dim
    #         for len_id in range(seq_len):
    #             sum_val = 0.0
    #             for i in range(16):
    #                 index = key_states[batch_id, head_id, len_id, i].int().item() + i*256
    #                 sum_val += lut[lut_offset // 4096, index]
    #             expected_res[batch_id, head_id, len_id] = sum_val

    # 打印期望的结果
    print("error:")
    print((res_cuda.cpu()-res).abs().mean())
    print(res.abs().mean())
# for i in range(300):
# 	a = torch.rand(32,128000).float()
# 	b = a.cuda()
# 	start_time = time.time()
# 	_, now_indices = torch.topk(a,k=1024,dim = -1,sorted=False)
# 	# 记录结束时间

# 	end_time = time.time()
# 	torch.cuda.synchronize()
# 	# 输出执行时间
# 	print(f"Top-k operation took {end_time - start_time:.4f} seconds.")

# 	start_time = time.time()
# 	_, now_indices = torch.topk(b,k=1024,dim = -1,sorted=False)
# 	# 记录结束时间
# 	torch.cuda.synchronize()
# 	end_time = time.time()
# 	# 输出执行时间
# 	print(f"Top-k operation took {end_time - start_time:.4f} seconds.")
# 	# c = torch.rand(32,1,128)
# 	d,c = torch.rand(32,128000,128),torch.rand(32,128,1)
# 	start_time = time.time()
# 	torch.matmul(d,c)

# 	torch.cuda.synchronize()
# 	end_time = time.time()
# 	# 输出执行时间
# 	print(f"Top-k operation took {end_time - start_time:.4f} seconds.")