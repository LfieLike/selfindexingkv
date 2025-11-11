import torch
import cpu_gather  # Replace with the actual name of your compiled extension (e.g., my_extension)

# Initialize sample data
batch = 1
head = 32
length = 1024
kv_batch = 1
kv_head = 32
kv_len = 64000

# Create tensors for quant_kv and quant_param
quant_kv = torch.randint(low=-10, high=10, size=(kv_batch, kv_head, kv_len,16), dtype=torch.int32)
quant_param = torch.randn(kv_batch, kv_head, kv_len,16, dtype=torch.float16)

# Create index tensor (with appropriate shape for your case)
index = torch.randint(low=0, high=kv_len, size=(batch, head, length), dtype=torch.int64)
print(index)
# Create output tensors with correct shapes
output_kv = torch.zeros((batch, head, length, 16), dtype=torch.int32)
output_param = torch.zeros((batch, head, length, 16), dtype=torch.float16)

# Call the custom gather function
import time

# 假设 self 和 layer_idx 已经定义
for i in range(100):
    # 记录开始时间，单位为毫秒
    quant_kv = torch.randint(low=-10, high=10, size=(kv_batch, kv_head, kv_len,16), dtype=torch.int32)
    quant_param = torch.randn(kv_batch, kv_head, kv_len,16, dtype=torch.float16)

    # Create index tensor (with appropriate shape for your case)
    index = torch.randint(low=0, high=kv_len, size=(batch, head, length), dtype=torch.int64)
    # print(index)
    # Create output tensors with correct shapes
    output_kv = torch.zeros((batch, head, length, 16), dtype=torch.int32)
    output_param = torch.zeros((batch, head, length, 16), dtype=torch.float16)
    start_time = time.time() * 1000
    cpu_gather.my_cpu_gather(quant_kv, quant_param, index, output_kv, output_param)
    end_time = time.time() * 1000

    # 计算执行时间，单位为毫秒
    execution_time_ms = end_time - start_time
    print(f"Execution time of prefetch1: {execution_time_ms} milliseconds")
    output_kv = torch.zeros((batch, head, length, 16), dtype=torch.int32)
    output_param = torch.zeros((batch, head, length, 16), dtype=torch.float16)
    time.sleep(3)  # import time
    start_time = time.time() * 1000
    true_kv = torch.gather(quant_kv, 2, index.unsqueeze(-1).expand_as(output_kv),out=output_kv)
    true_param = torch.gather(quant_param, 2, index.unsqueeze(-1).expand_as(output_param),out=output_param)
    end_time = time.time() * 1000

    # 计算执行时间，单位为毫秒
    execution_time_ms = end_time - start_time
    print(f"Execution time of prefetch2: {execution_time_ms} milliseconds")
# Print output for inspection
print("Output KV:")
print((output_kv-true_kv).sum())
# print(true_kv)
print("Output Param:")
print((output_param-true_param).sum())
