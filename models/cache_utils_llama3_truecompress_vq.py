from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from quant.myquant import ontbitquant,onebitgemv,extract_sign_and_compress,my_scatter_add,mydequant,my_resduialquant,my_value_quant,my_value_dequant,mykeyquantV2,mykeydequantV2
from quant.myquant import my_key_value_quant_V3,my_key_value_dequant_V3,my_lutgemv,my_key_value_quant_V4,my_key_value_dequant_V4,my_cpu_gather,mydequant_select,mydequant_select_fuse
import matplotlib.pyplot as plt
# from custom_flashatt import custom_kernels_extension
from torch.nn.attention import SDPBackend, sdpa_kernel
# from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
batch_size = 1
def true_key_value_quant(key_states,value_states,key_mask,value_mask):
    value_states = value_states/value_mask
    key_states = key_states/key_mask
    # key_zp,key_scale = get_zp_scale(key_states.view(-1,32).abs(),bit=2,shape=key_states.shape)
    # value_zp,value_scale = get_zp_scale((value_states).view(-1,32),bit=2,shape=key_states.shape)
    # key_value_quant,key_1bit = my_key_value_quant_V3(key_states=key_states,
    #                       value_states=value_states,
    #                       zp_key=key_zp,
    #                       scale_key=key_scale,
    #                       zp_value=value_zp,
    #                       scale_value=value_scale)
    key_value_quant,key_1bit,quant_param = my_key_value_quant_V4(key_states=key_states,
                          value_states=value_states)
    # print(((key_value_quant-key_value_quant1)!=0).sum()/key_value_quant1.numel())
    return key_value_quant,key_1bit,quant_param
def get_tensor_mem(tensor):
    tensor_memory_size = tensor.element_size() * tensor.numel()

    # 转换为 MB
    tensor_memory_size_mb = tensor_memory_size / 1024**2
    print("fuck",tensor_memory_size_mb)
    return tensor_memory_size_mb
def mock_prune(key_states,topk):
    W_metric = key_states.abs()
    # 在分数张量的最后一维进行排序并获取前 k 个值及其索引
    top_scores, top_indices = torch.topk(W_metric, topk, dim=-1)
    # print(top_scores)
    # 创建一个零张量用于存储结果
    result = torch.zeros_like(key_states)
    # 使用 scatter 将值填入结果张量
    result = result.scatter_(3, top_indices, key_states.gather(3, top_indices))
    return result
def quantize_and_dequantize(input_matrix,bit=4,block_size=32):
    """
    对输入矩阵进行量化再反量化的函数
    :param input_matrix: 输入的torch.Tensor类型的矩阵，维度可以是二维等
    :return: 反量化后的矩阵，与输入矩阵形状相同
    """
    # 计算量化区间范围
    # print(input_matrix.shape)
    dtypes = input_matrix.dtype
    if bit==0:
        return input_matrix
    input_matrix = input_matrix.float()
    shape = input_matrix.shape
    if block_size == -1:
        block_size = input_matrix.shape[-1]
    input_matrix = input_matrix.reshape(-1,block_size)
    ranges = 2**bit -1
    # input_matrix = input_matrix.view(-1,64*128)
    if block_size <= 32:
        min_val = input_matrix.min(dim=-1,keepdim=True)[0]
        max_val = input_matrix.max(dim=-1,keepdim=True)[0]
    else:
        max_val = input_matrix.max(dim=-1,keepdim=True)[0]
        min_val = input_matrix.min(dim=-1,keepdim=True)[0]
    interval = (max_val - min_val) / ranges  # 划分成4个区间，所以是除以3
    # 进行量化
    quantized = torch.round((input_matrix - min_val) / interval)
    quantized = quantized.clamp(0, ranges) 
    # 进行反量化
    dequantized = (quantized * interval) + min_val
    # print(dequantized-input_matrix)
    return dequantized.reshape(shape).to(dtypes)
def quant_key(key_states,bit = 4,block_size=64):
    shape = key_states.shape
    # cnt = key_states.view(shape[0],shape[1],2,-1).transpose(-1,-2)
    # print(key_states.mean(dim=-1),key_states.mean(dim=-1).mean(dim=-1))
    # print("---------------")
    return quantize_and_dequantize(key_states,bit=bit,block_size=block_size)
def quant_value(value_states,bit = 4,block_size=32):
    return quantize_and_dequantize(value_states,bit=bit,block_size=block_size)

def vq_hash(key_states):
    # 先转向
    types = key_states.dtype
    key_states = key_states.float()
    shape = key_states.shape
    device = key_states.device
    key_states_transpose = key_states.transpose(-1,-2).reshape(-1,32,4,shape[-2]).transpose(-1,-2)
    sign_bits = torch.signbit(key_states_transpose)
    powers_of_2 = 2 ** torch.arange(3, -1, -1, device=device)
    group_indices = torch.sum(sign_bits.int() * powers_of_2, dim=-1,keepdim=True).expand(-1,-1,-1,4)
    codebook = torch.zeros(shape[0]*shape[1],32,16,4,device = device,dtype = key_states_transpose.dtype)
    codebook.scatter_add_(-2, group_indices, key_states_transpose)
    labels = torch.sum(sign_bits.int() * powers_of_2, dim=-1)
    counts_per_label = torch.zeros(shape[0]*shape[1], 32, 16,
                                device=device, dtype=torch.long)
    ones_to_add = torch.ones_like(labels)
    counts_per_label.scatter_add_(2, labels, ones_to_add)
    codebook = codebook/counts_per_label.unsqueeze(-1)
    reconstructed_data = torch.gather(codebook, dim=-2, index=group_indices)
    reconstructed_data_trans = reconstructed_data.transpose(-1,-2).reshape(-1,128,shape[-2]).transpose(-1,-2).reshape(shape)
    return reconstructed_data_trans.to(types)

def reconstruct(key_states,codebook):
    # 先转向
    types = key_states.dtype
    key_states = key_states.float()
    shape = key_states.shape
    device = key_states.device
    key_states_transpose = key_states.transpose(-1,-2).reshape(-1,32,4,shape[-2]).transpose(-1,-2)
    sign_bits = torch.signbit(key_states_transpose)
    powers_of_2 = 2 ** torch.arange(3, -1, -1, device=device)
    group_indices = torch.sum(sign_bits.int() * powers_of_2, dim=-1,keepdim=True).expand(-1,-1,-1,4)
    codebook = torch.zeros(shape[0]*shape[1],32,16,4,device = device,dtype = key_states_transpose.dtype)
    codebook.scatter_add_(-2, group_indices, key_states_transpose)
    labels = torch.sum(sign_bits.int() * powers_of_2, dim=-1)
    counts_per_label = torch.zeros(shape[0]*shape[1], 32, 16,
                                device=device, dtype=torch.long)
    ones_to_add = torch.ones_like(labels)
    counts_per_label.scatter_add_(2, labels, ones_to_add)
    codebook = codebook/counts_per_label.unsqueeze(-1)
    reconstructed_data = torch.gather(codebook, dim=-2, index=group_indices)
    reconstructed_data_trans = reconstructed_data.transpose(-1,-2).reshape(-1,128,shape[-2]).transpose(-1,-2).reshape(shape)
    return reconstructed_data_trans.to(types)
def vq_hash_fu(key_states):
    # 先转向
    types = key_states.dtype
    key_states = key_states.float()
    shape = key_states.shape
    device = key_states.device
    key_states_transpose = key_states.transpose(-1,-2).reshape(-1,32,4,shape[-2]).transpose(-1,-2)
    sign_bits = ~torch.signbit(key_states_transpose)
    powers_of_2 = 2 ** torch.arange(3, -1, -1, device=device)
    group_indices = torch.sum(sign_bits.int() * powers_of_2, dim=-1,keepdim=True).expand(-1,-1,-1,4)
    codebook = torch.zeros(shape[0]*shape[1],32,16,4,device = device,dtype = key_states_transpose.dtype)
    codebook.scatter_add_(-2, group_indices, key_states_transpose)
    labels = torch.sum(sign_bits.int() * powers_of_2, dim=-1)
    counts_per_label = torch.zeros(shape[0]*shape[1], 32, 16,
                                device=device, dtype=torch.long)
    ones_to_add = torch.ones_like(labels)
    counts_per_label.scatter_add_(2, labels, ones_to_add)
    codebook = codebook/counts_per_label.unsqueeze(-1)
    reconstructed_data = torch.gather(codebook, dim=-2, index=group_indices)
    reconstructed_data_trans = reconstructed_data.transpose(-1,-2).reshape(-1,128,shape[-2]).transpose(-1,-2).reshape(shape)
    return reconstructed_data_trans.to(types)
    # print(counts_per_label)
    # 每4行一组
def draw(tensor,path="/root/tensor_heatmap.png"):
    tensor_np = tensor.cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(tensor_np, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    plt.title("Tensor Heatmap")
    
    # 4. 保存图像
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
class pruneCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,layers=None,device = None,layer_num = 32,maxlen=138100,batch_size=1,prefetch_rate = 0.15) -> None:
        super().__init__()
        self.rate = prefetch_rate
        self.batch_size=batch_size
        self.layer_num = layer_num
        self.maxlen = maxlen+128
        self.prefetch_len = []
        self.value_norm: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.compress_key_cache: List[torch.Tensor] = []
        self.compress_value_cache: List[torch.Tensor] = []
        self.means:List[torch.Tensor] = []
        self.sign:List[torch.Tensor] = []
        self.abs_key:List[torch.Tensor] = []
        self.abs_value:List[torch.Tensor] = []
        self.abs_query:List[torch.Tensor] = []
        self.pruneCache_seen_tokens=0
        self.neaset_len = 4
        self.sink_len = 60
        # print("true sketchkv")
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)
    def select_and_dequant(self,layer_idx,index,out_key_buffer = None):
        torch.gather(self.key_cache_1bit[layer_idx], 2, index.expand(-1, -1, -1, 16),out = out_key_buffer)
        # key_zp = torch.gather(self.key_zp[layer_idx],2,index.expand(-1, -1, -1, 4))
        # key_scale = torch.gather(self.key_scale[layer_idx],2,index.expand(-1, -1, -1, 4))
        # value_zp = torch.gather(self.value_zp[layer_idx],2,index.expand(-1, -1, -1, 4))
        # value_scale = torch.gather(self.value_scale[layer_idx],2,index.expand(-1, -1, -1, 4))
        return out_key_buffer

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        next_layer_query_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        # torch.cuda.synchronize()
        if layer_idx == 0:
            self.pruneCache_seen_tokens += key_states.shape[-2]
        if len(self.key_cache) <= layer_idx:
            # sink_token = key_states[:,:,0:1,:]
            # sink_token = sink_token/torch.norm(sink_token,dim=-1,keepdim=True)
            # sim_with_sink = torch.matmul(key_states/torch.norm(key_states,dim=-1,keepdim=True),sink_token.transpose(-1,-2)).squeeze()
            # M = key_states.abs().max(dim=-1)[0]
            M_median = torch.median(key_states, dim=-2,keepdim=True)[0]
            self.means.append(M_median)
            key_states = key_states-self.means[layer_idx]
            q_1 = query_states[:,:,-1:,:]
            shape = q_1.shape
            q_1 = q_1.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            score_now = torch.matmul(key_states,q_1.transpose(-1,-2)).squeeze(-1)
            _, now_indices = torch.sort(score_now,dim = -1)
            full_token = now_indices[...,-64:].reshape(shape[0],key_states.shape[1],-1,1).expand(-1,-1,-1,128)
            compress_token = now_indices[...,:-64].reshape(shape[0],key_states.shape[1],-1,1).expand(-1,-1,-1,128)
            full_k = torch.gather(key_states,dim=-2,index=full_token)
            full_v = torch.gather(value_states,dim=-2,index=full_token)
            self.key_cache.append(full_k)
            self.value_cache.append(full_v)
            need_compress_keycache = torch.gather(key_states,dim=-2,index=compress_token)
            need_compress_valuecache = torch.gather(value_states,dim=-2,index=compress_token)
            
            
            self.abs_key.append(need_compress_keycache.abs().max(dim=-2,keepdim=True)[0])
            vq_recon = vq_hash(need_compress_keycache)
            self.sign.append(vq_recon)

            # need_compress_keycache = torch.round((key_states.abs()*rounds/self.abs_key[layer_idx]).clamp(0, rounds))*key_states.sign()/rounds*self.abs_key[layer_idx]
            need_compress_keycache = quant_value(need_compress_keycache.abs()/self.abs_key[layer_idx],block_size=32,bit=2)*need_compress_keycache.sign()*self.abs_key[layer_idx]
            need_compress_valuecache = quant_value(need_compress_valuecache,block_size=32,bit=2)
            
            self.compress_key_cache.append(need_compress_keycache)
            self.compress_value_cache.append(need_compress_valuecache)
            return key_states,value_states
        else:
            key_states = key_states-self.means[layer_idx]
            rounds = 2**3-1
            # need_compress_keycache = torch.round((key_states.abs()*rounds/self.abs_key[layer_idx]).clamp(0, rounds))*key_states.sign()/rounds*self.abs_key[layer_idx]
            # need_compress_keycache = quant_key(key_states/self.abs_key[layer_idx],block_size=128,bit=4)*self.abs_key[layer_idx]
            need_compress_keycache = quant_value(key_states.abs()/self.abs_key[layer_idx],block_size=32,bit=2)*key_states.sign()*self.abs_key[layer_idx]
            need_compress_valuecache = quant_value(value_states,block_size=32,bit=2)
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], need_compress_keycache], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], need_compress_valuecache], dim=-2)
            q_1 = query_states[:,:,-1:,:]
            shape = q_1.shape
            q_1 = q_1.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            score_now = torch.matmul(self.sign[layer_idx],q_1.transpose(-1,-2)).squeeze(-1)
            # score_now = score_now.view(shape[0],key_states.shape[1],-1,16).max(dim=-1)[0]
            _, now_indices = torch.topk(score_now,k=min(96,score_now.shape[-1]),dim = -1,sorted=False)
            
            # now_indices = now_indices.unsqueeze(-1).expand(-1,-1,-1,16)*16
            # offsets = torch.arange(0, 16, dtype=now_indices.dtype, device=now_indices.device)
            # offsets = offsets.view(1, 1, 1, 16)
            # now_indices = now_indices+offsets
            now_indices = now_indices.view(shape[0],key_states.shape[1],-1,1).expand(-1,-1,-1,128)
            cnt_k = torch.gather(self.compress_key_cache[layer_idx],dim=-2,index=now_indices)
            cnt_v = torch.gather(self.compress_value_cache[layer_idx],dim=-2,index=now_indices)
            q_len = self.sign[layer_idx].shape[-2]
        return torch.cat([cnt_k,self.key_cache[layer_idx]],dim=-2), torch.cat([cnt_v,self.value_cache[layer_idx]],dim=-2)
    def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
        thres_cumsum = sum_before * alpha 
        sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
        thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
        W_mask = (W_metric <= thres)
        cur_sparsity = (W_mask==True).sum() / W_mask.numel()
        return W_mask, cur_sparsity
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache



class true_compress_cache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,layers=None,device = None,layer_num = 32,maxlen=138100,batch_size=1,prefetch_rate = 0.15) -> None:
        super().__init__()
        self.rate = prefetch_rate
        self.batch_size=batch_size
        self.layer_num = layer_num
        self.maxlen = maxlen+128
        self.prefetch_len = []
        self.value_norm: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.compress_key_cache: List[torch.Tensor] = []
        self.compress_value_cache: List[torch.Tensor] = []
        self.means:List[torch.Tensor] = []
        self.sign:List[torch.Tensor] = []
        self.abs_key:List[torch.Tensor] = []
        self.abs_value:List[torch.Tensor] = []
        self.pruneCache_seen_tokens=0
        self.neaset_len = 28
        self.sink_len = 4
        # print("true sketchkv")
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)
    def select_and_dequant(self,layer_idx,index,out_key_buffer = None):
        torch.gather(self.key_cache_1bit[layer_idx], 2, index.expand(-1, -1, -1, 16),out = out_key_buffer)
        # key_zp = torch.gather(self.key_zp[layer_idx],2,index.expand(-1, -1, -1, 4))
        # key_scale = torch.gather(self.key_scale[layer_idx],2,index.expand(-1, -1, -1, 4))
        # value_zp = torch.gather(self.value_zp[layer_idx],2,index.expand(-1, -1, -1, 4))
        # value_scale = torch.gather(self.value_scale[layer_idx],2,index.expand(-1, -1, -1, 4))
        return out_key_buffer

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        next_layer_query_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        # torch.cuda.synchronize()
        if layer_idx == 0:
            self.pruneCache_seen_tokens += key_states.shape[-2]
        # print("fuck")
        # Update the cache
        # print("fuck")
        if len(self.key_cache) <= layer_idx:
            
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            
            return key_states,value_states
        else:
           
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            q_res = custom_kernels_extension.flash_attention_decode(query_states.contiguous(), self.key_cache[layer_idx].contiguous(), self.value_cache[layer_idx].contiguous())
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                q_res1 = F.scaled_dot_product_attention(query_states.contiguous(), self.key_cache[layer_idx].contiguous(), self.value_cache[layer_idx].contiguous(), enable_gqa=True)
            # print(q_res - q_res1)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
        thres_cumsum = sum_before * alpha 
        sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
        thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
        W_mask = (W_metric <= thres)
        cur_sparsity = (W_mask==True).sum() / W_mask.numel()
        return W_mask, cur_sparsity
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
