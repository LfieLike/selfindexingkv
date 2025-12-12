# selfindexingkv

**Note:** This repository is not yet fully organized. Many hyperparameters (e.g.,  quantization bit‑width, KV‑cache configuration) are hard‑coded in source files or scripts and spread across multiple locations. We plan to centralize configuration and documentation in a future update.

### Custom Attention Kernel

Please note that the customized attention kernel we have implemented is at a toy level and is only guaranteed to be used for reproducing experimental results.



### Installation and Deployment

We have implemented the corresponding **GPU CUDA kernels**.

> ⚠️ Please make sure the Python version in your current environment is **3.10.x**.

Go into the project files and execute the following commands to install the required dependencies:

```
pip install -r requirements.txt
```

---

#### Install FlashAttention

This may take a lot of time. You can also download the `.whl` file for faster installation.

```
pip -v install flash-attn --no-build-isolation
```

---

### Compile Custom Kernels

#### Step 1: Compile the CUDA and CPU kernels in the `quant` directory:

```
cd quant
python setup.py install
python cpp_setup.py install
python cpp_gather_setup.py install
python cpp_gather_setupV2.py install
```

#### Step 2: Compile the custom FlashAttention kernel:

```
cd custom_flashatt
python setup.py install
```

---

### Reproduce Experiments

After compiling all kernels, you can run the following command to reproduce our **LongBench** experimental results. 

16+1bits:
```
bash scripts/long_test_trulycompress_mock.sh 0 \
NousResearch/Meta-Llama-3.1-8B-Instruct test
```
2+1bits:
```
bash scripts/long_test_truly.sh 0 \
NousResearch/Meta-Llama-3.1-8B-Instruct test
```
---





### Latency and KV Cache Memory Evaluation

The end-to-end latency and KV cache memory usage can be reproduced using:

```
python test_latency_llama3_ours.py
```

---


