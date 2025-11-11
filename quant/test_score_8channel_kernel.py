import torch
import cuda_my_8channel_score  # 假设编译后的模块名称为 cuda_my_quant
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
def my_test(
        key_states,
        query_states
    ):
    return cuda_my_8channel_score.My_score_8channle_half_half(
        key_states,
        query_states
    )
@gpu_timer
def simple_test(query_states,org_key):
    return torch.matmul(query_states.unsqueeze(-2),org_key.transpose(-1,-2)).to(torch.float16)
def test_my_score_cuda():
    # 输入维度设置
    batch_size = 1
    head_size = 32
    seq_len = 64000  # token length
    emb_dim = 8

    # 创建输入数据
    # key_states = torch.ones((batch_size, head_size, seq_len, emb_dim//8), device='cuda', dtype=torch.uint8)*255
    org_key = torch.randn((batch_size, head_size, seq_len, emb_dim), device='cuda',dtype=torch.float16)
    query_states = torch.randn((batch_size, head_size, emb_dim), device='cuda', dtype=torch.float16)
    for i in range(10):
        print(my_test(org_key,query_states)-simple_test(query_states,org_key).squeeze())
    print()
if __name__ == '__main__':
    test_my_score_cuda()
