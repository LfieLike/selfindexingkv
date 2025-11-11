import torch
import cpu2gpu  # Import the compiled extension (assuming your extension is named 'my_extension')
import time 
def test_cpu2gpu():
    # Define tensor dimensions
    batch_size = 1
    head_size = 32
    len_size = 40960
    index_len = 1024
    
    # Create sample input tensors
    key_value_quant_cpu = torch.randint(low=0, high=256, size=(batch_size, head_size, len_size, 128//8), dtype=torch.int32).cpu().pin_memory()
    quant_param_cpu = torch.randn((batch_size, head_size, len_size, 128//8),dtype=torch.half).cpu().pin_memory()
    select_index = torch.randint(low=0, high=len_size,size=(batch_size, head_size, index_len,1), dtype=torch.int64).cuda()
    print("cpu result tensor:")
    # print(key_value_quant_cpu)
    # Allocate GPU memory for key_value_quant_gpu
    key_value_quant_gpu = torch.zeros(size=(2*batch_size, head_size, index_len, 128//8), dtype=torch.int32).cuda().view(-1)
    quant_param_gpu = torch.zeros(size=(2*batch_size, head_size, index_len, 128//8), dtype=torch.half).cuda().view(-1)
    # Call the extension function
    for i in range(10):
        stream = torch.cuda.Stream()
        start_time = time.time()
        with torch.cuda.stream(stream):
            key_value_quant_gpu_result = cpu2gpu.cpu2gpu(key_value_quant_gpu, key_value_quant_cpu,quant_param_gpu,quant_param_cpu, select_index)
        # torch.cuda.synchronize()
        stream.synchronize()
        # key_value_quant_gpu_result = key_value_quant_gpu_result-1
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"GPU result tensor:")
        
        print(f"直接读取synchronize Execution time1: {execution_time:.4f} seconds")
        # Check the results (just print out the tensor for inspection)
        print("GPU result tensor:")
        start_time = time.time()
        cpu_index = select_index.cpu().expand(-1,-1,-1,128//8)
        gather_cpu = torch.gather(key_value_quant_cpu,-2,cpu_index)
        gather_param_cpu = torch.gather(quant_param_cpu,-2,cpu_index)
        gather_gpu = gather_cpu.cuda()
        gather_param_gpu = gather_param_cpu.cuda()
        torch.cuda.synchronize()
        end_time = time.time()
        cnt = key_value_quant_gpu[:gather_gpu.numel()].view(gather_gpu.shape)
        print((cnt-gather_gpu).sum())
        # print((quant_param_gpu-gather_param_gpu).sum())
        execution_time = end_time - start_time
        print(f"cpugahter 再传输synchronize Execution time2: {execution_time:.4f} seconds")
        print(f"GPU result tensor:")
    
if __name__ == "__main__":
    test_cpu2gpu()
