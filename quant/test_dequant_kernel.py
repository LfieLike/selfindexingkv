import torch
import cuda_my_key_dequant  # Assume this is the compiled extension with `TORCH_EXTENSION_NAME` from the code provided

def test_quantize_with_outliers_kernel():
    # Parameters
    batch_size = 1
    head_size = 1
    len_size = 2  # Number of elements in the embedding dimension
    outlier_num = 4  # Example number of outliers

    # Initialize inputs
    compressed_key = torch.randint(0, 2, (batch_size, head_size, len_size), dtype=torch.uint8, device='cuda')
    key_outlier_quant = torch.randint(0, 256, (batch_size, head_size, outlier_num), dtype=torch.uint8, device='cuda')
    dequant_dst = torch.zeros((batch_size, head_size, len_size), dtype=torch.float16, device='cuda')
    channel_mean = torch.rand(batch_size, head_size, len_size, dtype=torch.float16, device='cuda')
    outlier_idx = torch.randint(0, len_size, (batch_size, head_size, outlier_num), dtype=torch.uint8, device='cuda')
    quant_outlier_zp = torch.rand(batch_size, head_size, outlier_num, dtype=torch.float16, device='cuda')
    quant_outlier_scale = torch.rand(batch_size, head_size, outlier_num, dtype=torch.float16, device='cuda')

    # Run the CUDA function
    output = cuda_my_key_dequant.My_key_dequant_half_half(
        compressed_key,
        key_outlier_quant,
        dequant_dst,
        channel_mean,
        outlier_idx,
        quant_outlier_zp,
        quant_outlier_scale,
        outlier_num
    )

    # Check results
    print("Compressed Key:")
    print(compressed_key)
    print("Dequantized Output:")
    print(output)
    print("Channel Mean:")
    print(channel_mean)

    # Basic validation
    assert output.shape == dequant_dst.shape, "Output shape mismatch"
    print("Test passed!")

# Run the test
test_quantize_with_outliers_kernel()