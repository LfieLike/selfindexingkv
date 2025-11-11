import torch
import time
import os
import numpy as np
class RetrievalBasedCompressor:
    def __init__(self, **kwargs) -> None:
        self.profile_metric = {
            "prefill_time" : 0,
            "prefill_attn_time" : 0,
            "prefill_cnt": 0,
            "prefill_per_layer_time": 0,

            "prepare_idx_elapsed" : 0,
            "prepare_idx_cnt" : 0,

            "decoding_time" : 0,
            "decoding_cnt" : 0,
            "decoding_attn_time" : 0,

            "offload_pq_ref_elapsed" : 0,
            "offload_kv_elapsed" : 0,
            "offload_cnt": 0,
            "offload_pq_ref_bytes" : 0,
            "offload_kv_bytes" : 0,

            "fetch_ref_elapsed" : 0,
            "fetch_kv_elapsed" : 0,
            "cpu_gather_elapsed" : 0,
            "calc_dummy_weight_elapsed" : 0,
            "fetch_ref_data_bytes" : 0,
            "fetch_kv_data_bytes" : 0,
        }

        self.device = kwargs["cur_device"]
    
    def profile_ckpt(self):
        torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def reset(self):
        for k,_ in self.profile_metric.items():
            self.profile_metric[k] = 0

    def showtime(self):
        result_str = "\n".join([f"{key} : {value}" for key, value in self.profile_metric.items()])
        print("-----profile result show:\n", result_str)
        with open(f"./profile_result/mistral_profile_{os.getpid()}","a") as f:
            f.write(result_str)
            
class MyCompressor(RetrievalBasedCompressor):
    all_pq_compressors = []
    def __init__(self,  **kwargs):

        self.prefetch_event = torch.cuda.Event()
    
        if self.layer_idx <= 1:
            print(f"GQA is {self.GQA}")
        super().__init__(**kwargs)
        
        # Used for prefetch
        PqBasedSearchCompressor.all_pq_compressors.append(self)
        
    def prefill_attn(
        self,
        query,
        past_key_value: torch.Tensor,
        use_gpu = True
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        self.centroids = None
        self.code_book = None
        self.gpu_centroids = None
        self.centroids = None
        self.code_book = None
        self.gpu_code_book = None
        self.km_done = False
        self.ip2l2_phi = None
        self.past_token_cnt = 0
        self.seq_cnt += 1
        if self.layer_idx == 0:
            global_compressor.refresh_pool()

        key_states, value_states = past_key_value
        bsz, kv_heads, kv_seq_len, dim = key_states.shape

        assert bsz == 1, "Do not support bsz > 1 in adaptive compression mode yet."
        self.recent_size = int((kv_seq_len-self.sink_size) * self.compress_ratio * self.recent_ratio)
        self.prefill_length = kv_seq_len
        self.topk_size = int((kv_seq_len-self.sink_size) * self.compress_ratio * (1 - self.recent_ratio))

        # There is no need to compress sink token
        xb = key_states[:,:, self.sink_size:, :]
        n_xb = kv_seq_len - self.sink_size

        subvec_d = dim // self.n_subvec_per_head
        centroid_cnt = 2 ** self.n_subbits
        xb = xb.reshape(bsz, kv_heads, n_xb, self.n_subvec_per_head, subvec_d).transpose(2,3)
        
        cache_manager.init(key_states, value_states, self.layer_idx,self.topk_size) 
        # Do compression, in async manner
        self.build_index_cpu_multi_core_sklr(xb, centroid_cnt)
     

        attn_output = flash_attn_func(
            query.transpose(1,2),
            key_states.transpose(1,2),
            value_states.transpose(1,2),
            causal = True
        ).transpose(1,2)

        self.kv_cache_cnt = np.zeros([bsz*kv_heads], dtype=np.int64)
        self.past_token_cnt = key_states.shape[-2]
        return attn_output, self.kv_cache_cnt