import torch
import torch.multiprocessing as mp

# 共享内存管理函数
def offload_to_cpu(tensor_gpu):
    tensor_cpu = tensor_gpu.cpu()
    # 将数据存储到共享内存中
    return tensor_cpu

def fetch_from_cpu_and_move_to_gpu(tensor_cpu):
    # 将数据从CPU加载回GPU
    tensor_gpu = tensor_cpu.cuda()
    return tensor_gpu

def worker(shared_tensor):
    # 假设这里是从共享内存获取数据的地方
    tensor_cpu = shared_tensor
    # 将数据重新载入GPU进行处理
    tensor_gpu = fetch_from_cpu_and_move_to_gpu(tensor_cpu)
    return tensor_gpu

def main():
    # 在GPU上创建一个tensor
    mp.set_start_method('spawn')
    tensor_cpu = torch.randn(1000, 1000).share_memory_()
    tensor_gpu = torch.randn(1000, 1000).cuda()
    pool = mp.Pool(processes=4)
    # 将tensor offload到CPU，并放入共享内存
    results = pool.apply_async(worker, args=(tensor_cpu,))
    # aaa = results.get().clone()
    print(results.get()@tensor_gpu)
    # time.sleep(10)  # import time
    # print()
    # 手动关闭进程池
    # pool.close()
    # pool.join()

if __name__ == '__main__':
    main()
