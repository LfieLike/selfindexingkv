import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["NUMEXPR_MAX_THREADS"] = 64
import logging
logging.getLogger("config").setLevel(logging.WARNING)
import time
import numpy as np
from transformers import LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,LlamaConfig
from transformers.generation.utils import GenerationConfig
import torch.multiprocessing as mp
import pandas as pd
import tqdm
import torch
from loguru import logger
import json
from torch.profiler import profile, record_function, ProfilerActivity
# NOTE: We use Llama2-7b to benchmark the latency.
from transformers.cache_utils import OffloadedCache
def main():
    os.environ["SUBVEC"] = "2"
    os.environ["SUBBITS"] = "6"
    os.environ["MODE"] = "off"

    model_path = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    # model_path = "NousResearch/Llama-2-7b-chat-hf"
    # model_path = "./pqcache/llama-32k"
    # model_path = "./pqcache/Mistral-32k"
    # print(model_path)

    file_name = "passkey_examples.jsonl"
    df = pd.read_json(file_name, lines=True) # 读取文件

    # if config.compressor == "pq_search":
    #     initialize_objects(config, model=model_path)
    
    config = LlamaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                        use_fast=False, 
                                        trust_remote_code=True)
    # from transformers import LlamaForCausalLM
    # model = LlamaForCausalLM.from_pretrained(
    #     pretrained_model_name_or_path=model_path,
    #     cache_dir="./cached_models",
    #     config=config,
    #     torch_dtype=torch.half,
    #     low_cpu_mem_usage=True,
    #     use_flash_attention_2=True,
    #     device_map="auto",
    # )
    # config.k_bits = 2 # KiVi currently support 2/4 K/V bits
    # config.v_bits = 2
    # config.group_size = 32 
    # config.residual_length = 128 # corresponding to the number of recent fp16 tokens
    # config.use_flash = True # use flash-attention with KiVi for long context inference

    # model = LlamaForCausalLM_KIVI.from_pretrained(
    #     pretrained_model_name_or_path=model_path,
    #     config=config,
    #     cache_dir="./cached_models",
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    # ).cuda()
    
    from models.modeling_llama3V4 import LlamaForCausalLM
    # from models.cache_utils_llama3_truecompress_gatherbygpu import SkectchCache,pruneCache
    from models.cache_utils_llama3_selfindex import true_selfindex_Cache
    # from models.cache_utils_llama3_truecompress_allcpu import SkectchCache,pruneCache
    config._attn_implementation="flash_attention_2"
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        # cache_dir="./cached_models",
        config=config,
        torch_dtype=torch.half,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model = model.half().eval()

    print("We are loading model", model_path)
    peak_allocated_memory = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved_memory = torch.cuda.max_memory_reserved() / 1024**2

    print(f"New peak allocated memory: {peak_allocated_memory:.2f} MB")
    print(f"New peak reserved memory: {peak_reserved_memory:.2f} MB")
    batch_size = 1
    for line in open(file_name, "r"):
        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"]+example["input"]+example["input"]+example["input"]+example["input"]+example["input"]+example["input"]+example["input"] +example["input"]+example["input"]+example["input"]+example["input"]+example["input"]+example["input"]+example["input"]+ prompt_postfix
        input_ids = tokenizer([prompt]*batch_size, return_tensors="pt").input_ids.cuda()
        print("maxlen",len(input_ids[0]))
        gen_max_token = 128
        # 16*1024,32*1024,48*1024,
        test_set = [8*1024,16*1024,32*1024,48*1024,64*1024]
        for idx in range(5):
            # 12000,16000,32000,36000,42000,64000,
            for seqlen in tqdm.tqdm(test_set):
                # 预热
                torch.cuda.reset_max_memory_allocated()
                output = model.generate(
                                input_ids=input_ids[:, :seqlen],
                                attention_mask=None,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=1, 
                                num_beams=1,
                                do_sample=False,
                                temperature=1.0,
                                past_key_values = true_selfindex_Cache(device=model.device,maxlen = seqlen,batch_size=batch_size)
                            )[0]
            # for seqlen in [2000, 4000]:
                print("fuck",len(input_ids[0]))



                total_time = 0
                for i in range(5):
                    # past_key_values = SkectchCache()
                    torch.cuda.reset_peak_memory_stats()
                    cache = true_selfindex_Cache(device=model.device,maxlen = seqlen,batch_size=batch_size)
                    begin = time.perf_counter()
                    output = model.generate(
                                input_ids=input_ids[:, :seqlen],
                                attention_mask=None,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=1, 
                                num_beams=1,
                                do_sample=False,
                                temperature=1.0,
                                past_key_values = cache, eos_token_id=None, early_stopping=False
                            )[0]
                    end = time.perf_counter()
                    total_time+=(end - begin)
                    del cache
                print(f"{output.flatten()[-1]} \r")
                
                ttft = total_time/5
                
                time.sleep(1)
                
                total_time = 0
                for i in range(5):
                    cache = true_selfindex_Cache(device=model.device,maxlen = seqlen,batch_size=batch_size)
                    begin = time.perf_counter()
                    output = model.generate(
                                input_ids=input_ids[:, 40:seqlen+40],
                                attention_mask=None,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=2, 
                                num_beams=1,
                                do_sample=False,
                                temperature=1.0,
                                past_key_values = cache, eos_token_id=None, early_stopping=False
                            )[0]
                    end = time.perf_counter()
                    total_time+=(end - begin)
                    del cache

                
                tt2t = total_time/5
                print("tt2t len",tt2t,len(output))
                time.sleep(1)
                total_time = 0
                
                for i in range(5):
                    torch.cuda.reset_max_memory_allocated()
                    cache = true_selfindex_Cache(device=model.device,maxlen = seqlen,batch_size=batch_size)
                    begin = time.perf_counter()
                    output = model.generate(
                                input_ids=input_ids[:, 40:seqlen+40],
                                attention_mask=None,
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=gen_max_token, 
                                num_beams=1,
                                do_sample=False,
                                temperature=1.0,
                                past_key_values = cache, eos_token_id=None, early_stopping=False
                            )[0]
                    end = time.perf_counter()
                    total_time+=(end - begin)
                    peak_allocated_memory = torch.cuda.memory_allocated() / 1024**2
                    print(f"befor del cache: {peak_allocated_memory:.2f} MB")
                    del cache
                    peak_allocated_memory = torch.cuda.memory_allocated() / 1024**2
                    print(f"after del cache: {peak_allocated_memory:.2f} MB")
                print(f"{output.flatten()[-1]}")
                # print("len",len(output))
                decoding_elapsed = total_time/5 - tt2t
                print("decoding_elapsed len",decoding_elapsed,len(output))
                
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("LlamaAttention"):
                #         outputs = model.generate(
                #             input_ids=input_ids[:, :seqlen],
                #             attention_mask=None,
                #             pad_token_id=tokenizer.eos_token_id,
                #             max_new_tokens=1, 
                #             num_beams=1,
                #             do_sample=False,
                #             temperature=1.0,past_key_values = SkectchCache(device=model.device)
                #         )[0]
                # prof.export_chrome_trace(f"{seqlen}trace.json")
                # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                peak_allocated_memory = torch.cuda.max_memory_allocated() / 1024**2
                peak_reserved_memory = torch.cuda.max_memory_reserved() / 1024**2

                print(f"New peak allocated memory: {peak_allocated_memory:.2f} MB")
                print(f"New peak reserved memory: {peak_reserved_memory:.2f} MB")
                print(f"Given input len is:{seqlen}, gen seq_len:{gen_max_token},"
                        f"ttft is {ttft},"
                        f"tt2t is {tt2t},"
                        f"decoding elasped:{decoding_elapsed},"
                        f"{decoding_elapsed / (gen_max_token - 2)} per decoding token.")
                with open('allcpu_latency.txt', 'a') as file:
                    file.write(f"New peak allocated memory: {peak_allocated_memory:.2f} MB\n")
                    file.write(f"New peak reserved memory: {peak_reserved_memory:.2f} MB\n")
                    file.write(f"Given input len is:{seqlen}, gen seq_len:{gen_max_token},"
                            f"ttft is {ttft},"
                            f"tt2t is {tt2t},"
                            f"decoding elapsed:{decoding_elapsed},"
                            f"{decoding_elapsed / (gen_max_token - 2)} per decoding token.\n")
    del model
    logger.info(f"del objects done.")   
    exit()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()

