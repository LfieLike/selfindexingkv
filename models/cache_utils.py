from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
import torch
from .kmeans import fit_kmeans
from flash_attn import flash_attn_func
from quant.myquant import ontbitquant,onebitgemv,extract_sign_and_compress,my_scatter_add,mydequant,my_resduialquant,my_value_quant,my_value_dequant
class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length



class KVquant_unit:
    def __init__(self):
        self.quantized_tensor_KV = None
        self.zero_points_K = None
        self.scales_K = None
        self.zero_points_V = None
        self.scales_V = None
        self.device = 'cpu'
        self.bit_num = 4
        self.head_dim = 128
        self.quant_param = None
    def cat_new_cache(self,quantized_kv,quant_param):
        if self.quantized_tensor_KV is None:
            self.head_dim = quantized_kv.shape[-1]
            self.quantized_tensor_KV = quantized_kv
            self.device = quantized_kv.device
        else:
            # quantized_tensor = self.pact_to_int8(quantized_tensor_K,quantized_tensor_V)
            self.quantized_tensor_KV = torch.cat((self.quantized_tensor_KV,quantized_kv),dim=-2)
        if self.quant_param is None:
            self.quant_param = quant_param
        else:
            self.quant_param = torch.cat((self.quant_param,quant_param),dim=-2)
    def select(self,indices):
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        quantized_tensor_slip = torch.gather(self.quantized_tensor_KV, 2, indices_expanded)
        quant_param = torch.gather(self.quant_param, 2, indices.unsqueeze(-1).expand(-1,-1,-1,4)).contiguous()
        # quant_param= torch.cat((zero_points_slip_K.view(-1,1),scales_slip_K.view(-1,1),zero_points_slip_V.view(-1,1),scales_slip_V.view(-1,1)),dim=-1).reshape(batch_size, head_size, num_elements*4)
        return quantized_tensor_slip,quant_param
    def prefetch(self,non_blocking=True):
        if self.quantized_tensor_KV is None:
            return
        self.quantized_tensor_KV = self.quantized_tensor_KV.to(device=self.device, non_blocking=non_blocking)
    def evcit(self,non_blocking=True):
        if self.quantized_tensor_KV is None:
            return
        self.quantized_tensor_KV = self.quantized_tensor_KV.to(device='cpu', non_blocking=non_blocking)
    def get_cpu_memory(self):
        memory_size = self.quantized_tensor_KV.element_size() * self.quantized_tensor_KV.numel() +self.quant_param.element_size() * self.quant_param.numel()

        # print(f"Memory size of the tensor: {memory_size / 1024**2:.2f} MB")
class quant_unit_V2:
    def __init__(self):
        self.quantized_tensor = None
        self.zero_points = None
        self.scales = None
        self.quant_param = None
        self.device = 'cpu'
        self.head_dim = 128
    def cat_new_cache_V2(self,quantized_tensor,quant_param):
        # print(unpact_tensor-quantized_tensor)
        if self.quantized_tensor is None:
            # self.head_dim = quantized_tensor.shape[-1]
            # quantized_tensor = self.pact_to_int8(quantized_tensor)
            self.quantized_tensor = quantized_tensor
            self.device = quantized_tensor.device
        else:
            # quantized_tensor = self.pact_to_int8(quantized_tensor)
            self.quantized_tensor = torch.cat((self.quantized_tensor,quantized_tensor),dim=-2)
        if self.quant_param is None:
            self.quant_param = quant_param
        else:
            self.quant_param = torch.cat((self.quant_param,quant_param),dim=-2)
class outline_quant_unit:
    def __init__(self,scales,zero_points,num_bits=8):
        self.qmin = 0.
        self.qmax = 2.**num_bits - 1.
        self.scales = scales
        self.zero_points = zero_points
class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,layernum=32,device='cpu') -> None:
        self.layernum=layernum
        self.key_outline = [torch.empty(0) for _ in range(layernum)]
        self.key_cache_full = [torch.empty(0) for _ in range(layernum)]
        self.key_cache_index = [torch.empty(0) for _ in range(layernum)]
        self.key_cache_means = [torch.empty(0) for _ in range(layernum)]
        self.KV_cache_quant_unit = [KVquant_unit() for _ in range(layernum)]
        self.key_cache_channel_zp = {_:torch.empty(0) for _ in range(layernum)}
        self.key_cache_channel_scale = {_:torch.empty(0) for _ in range(layernum)}
        self.value_cache_full = {_:torch.empty(0) for _ in range(layernum)}
        self.value_cache_max = {_:torch.empty(0) for _ in range(layernum)}

        # self.value_cache_quant_unit = [quant_unit() for _ in range(layernum)]
        self.value_cache_quant_unit_V2 = [quant_unit_V2() for _ in range(layernum)]
        self.value_sink = {_:torch.empty(0) for _ in range(layernum)}
        self.key_sink = {_:torch.empty(0) for _ in range(layernum)}
        self.key_sign_new = {_:torch.empty(0) for _ in range(layernum)}
        self.attn_cache = {_:torch.empty(0) for _ in range(layernum)}
        self.outline_quant = {_:None for _ in range(layernum)}
        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.window_size = 128
        self.sink_size = 4
        self.imp_rate = 0.3
        self.prefetch_stream={_:torch.cuda.Stream() for _ in range(layernum)}
        self.key_cache_temp  = {_:None for _ in range(2)}
        self.value_cache_temp  = {_:None for _ in range(2)}
        self.num = 30
        self.default_stream = torch.cuda.default_stream()
        self.topK = 12
    def resize_buff(self,device,num_key_value_heads=32,dtype=torch.float16,shapes=None):
        if self.key_cache_temp[0] is not None:
            return
        shape=(1,num_key_value_heads,4050,128)
        # for i in range(2):
        self.key_cache_temp[0] = torch.empty(shape,device=device, dtype=dtype)
        self.value_cache_temp[0] = torch.empty(shape,device=device, dtype=dtype)
        
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache_full[layer_idx], self.value_cache_full[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache_full[layer_idx], self.value_cache_full[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache_full)   
    def prefetch_layer(self,layer_idx:int,device,non_blocking=True):
        with torch.cuda.stream(self.prefetch_stream[layer_idx]):
            self.KV_cache_quant_unit[layer_idx].prefetch(non_blocking=non_blocking)
            # self.value_cache_quant_unit[layer_idx].prefetch()
    def evict_previous_layer(self, layer_idx: int,non_blocking=True):
        with torch.cuda.stream(self.prefetch_stream[layer_idx]):
            self.KV_cache_quant_unit[layer_idx].evcit(non_blocking=non_blocking)
        # self.value_cache_quant_unit[layer_idx].evcit()
    def get_kv(self,
        layer_idx: int
    )-> Tuple[torch.Tensor, torch.Tensor]:
        device = self.attn_cache[layer_idx].device
        windows_size = self.window_size

        token_num = self.key_sign_new[layer_idx].shape[-2]
        
        mydequant(
            self.key_sign_new[layer_idx].contiguous(),
            self.key_outline[layer_idx].contiguous(),
            self.key_cache_temp[0],
            self.key_cache_means[layer_idx].contiguous(),
            self.key_cache_index[layer_idx][...,:self.topK].contiguous().to(torch.uint8),
            self.outline_quant[layer_idx].zero_points.half(),
            self.outline_quant[layer_idx].scales.half(),
            self.topK
        )
        
        key_value_quantized_data,quant_param = self.KV_cache_quant_unit[layer_idx].select(self.attn_cache[layer_idx])
        my_value_dequant(self.value_cache_quant_unit_V2[layer_idx].quantized_tensor,
                         self.value_cache_temp[0],
                         self.value_cache_max[layer_idx],
                         self.value_cache_quant_unit_V2[layer_idx].quant_param)
        # print((test_quant_value-quant_value).abs().mean())
        my_scatter_add(
            key_value_quantized_data.contiguous(),
            self.key_cache_temp[0],
            self.value_cache_temp[0],
            quant_param.contiguous(),
            self.attn_cache[layer_idx].int(),
            key_value_quantized_data.shape[-2],
            self.value_cache_quant_unit_V2[layer_idx].quantized_tensor.shape[-2]
        )
        
        self.key_cache_temp[0][:,:,:self.sink_size,:].copy_(self.key_sink[layer_idx],non_blocking=True)
        self.key_cache_temp[0][:,:,token_num-windows_size:token_num,:].copy_(self.key_cache_full[layer_idx][:,:,-windows_size:,:],non_blocking=True)
        self.value_cache_temp[0][:,:,:self.sink_size,:].copy_(self.value_sink[layer_idx],non_blocking=True)
        self.value_cache_temp[0][:,:,token_num-windows_size:token_num,:].copy_(self.value_cache_full[layer_idx][:,:,-windows_size:,:],non_blocking=True)
    def update_new(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]
            self.num +=1
        # Update the cache
        value_states = value_states.contiguous()
        if key_states.shape[-2] != 1:
            self.resize_buff(device = key_states.device)
            means = key_states.abs().mean(dim=-2,keepdim=True)
            self.key_cache_means[layer_idx] = means
            abs_matrix = torch.abs(key_states)
            abs_sum1 = abs_matrix.mean(dim=(0,2),keepdim=True)
            sorted_indices = torch.argsort(abs_sum1, dim=-1, descending=True)
            maxvalue = value_states.abs().max(dim=-2,keepdim=True)[0].contiguous()
            self.value_cache_max[layer_idx] = maxvalue
            self.key_cache_index[layer_idx] = sorted_indices[...,:self.topK].contiguous()
            
            qmin = 0.
            qmax = 2.**8 - 1.
            min_val = key_states.min(dim=-2)[0]
            max_val = key_states.max(dim=-2)[0]
            # 计算 scale 和 zp (量化范围是 [0, 255])
            outlier_scale = (max_val - min_val) / (qmax - qmin)
            outlier_scale[outlier_scale==0]=1
            outlier_zp = (-min_val / outlier_scale)
            outlier_zp = outlier_zp.round().half()
            self.key_cache_channel_zp[layer_idx] = outlier_zp.contiguous()
            self.key_cache_channel_scale[layer_idx] = outlier_scale.contiguous()
            test_key_states = key_states.clone().contiguous()
            key_quant1, outlier_quant,my_outlier_zp_quant,my_outlier_scale_quant,my_resduial  = ontbitquant(
                test_key_states.contiguous(),
                self.key_cache_means[layer_idx].contiguous(),
                self.key_cache_index[layer_idx].to(torch.uint8),
                self.key_cache_channel_zp[layer_idx],
                self.key_cache_channel_scale[layer_idx],
                self.topK
            )
            self.key_sign_new[layer_idx]=key_quant1
            score =  onebitgemv(key_quant1,query_states[:,:,-1,:]).unsqueeze(dim=-2).to(query_states.device)
            hlaf_len = int(key_states.shape[-2]*self.imp_rate)
            _,sort_index = torch.topk(score[...,:-self.window_size],k=hlaf_len,dim = -1,sorted=False)
            sort_index = sort_index.squeeze(dim=-2)
            self.attn_cache[layer_idx] = sort_index
            self.key_cache_full[layer_idx]=(key_states[:,:,-self.window_size:,:])
            self.value_cache_full[layer_idx]=(value_states[:,:,-self.window_size:,:])
            self.key_sink[layer_idx] = key_states[:,:,:self.sink_size,:]
            self.value_sink[layer_idx] = value_states[:,:,:self.sink_size,:]
            clone_key_states = key_states
            clone_value_states = value_states
            value_quant_param, value_quant,my_value_resduial = my_value_quant(value_states,maxvalue)
            quant_param, key_value_quant = my_resduialquant(my_resduial,my_value_resduial)
            
            self.KV_cache_quant_unit[layer_idx].cat_new_cache(key_value_quant,quant_param)
            # self.KV_cache_quant_unit[layer_idx].get_cpu_memory()
            self.value_cache_quant_unit_V2[layer_idx].cat_new_cache_V2(value_quant,value_quant_param)
            self.outline_quant[layer_idx] = outline_quant_unit(my_outlier_scale_quant.unsqueeze(dim=-2),my_outlier_zp_quant.unsqueeze(dim=-2))
            self.key_outline[layer_idx] = outlier_quant
            # print("outlier_quant",self.key_outline[layer_idx].shape)
            # print("my_outlier_zp_quant",my_outlier_zp_quant.shape)
            # self.evict_previous_layer((layer_idx)%self.layernum,non_blocking=True)
            # self.prefetch_layer((layer_idx + 1) % self.layernum,value_states.device,,non_blocking=True)
            attn_output = flash_attn_func(
                query_states.transpose(1,2),
                key_states.transpose(1,2),
                value_states.transpose(1,2),
                causal = True
            ).transpose(1,2)
            return attn_output
        else:
            # self.evict_previous_layer((layer_idx-1)%self.layernum,non_blocking=True)
            
            self.get_kv(layer_idx=(layer_idx)%self.layernum)
            
            self.key_cache_full[layer_idx] = torch.cat([self.key_cache_full[layer_idx][:,:,1:,:], key_states], dim=-2)
            self.value_cache_full[layer_idx] = torch.cat([self.value_cache_full[layer_idx][:,:,1:,:], value_states], dim=-2)
            value_quant_param, value_quant,my_value_resduial = my_value_quant(value_states,self.value_cache_max[layer_idx])
            self.value_cache_quant_unit_V2[layer_idx].cat_new_cache_V2(value_quant,value_quant_param)
            key_quant1, outlier_quant,my_outlier_zp_quant,my_outlier_scale_quant,my_resduial  = ontbitquant(
                key_states.contiguous(),
                self.key_cache_means[layer_idx].contiguous(),
                self.key_cache_index[layer_idx].to(torch.uint8),
                self.key_cache_channel_zp[layer_idx],
                self.key_cache_channel_scale[layer_idx],
                self.topK
            )
            quant_param, key_value_quant = my_resduialquant(my_resduial.contiguous(),my_value_resduial.contiguous())
            self.key_sign_new[layer_idx] = torch.cat((self.key_sign_new[layer_idx],key_quant1),dim=-2)
            self.KV_cache_quant_unit[layer_idx].cat_new_cache(key_value_quant,quant_param)
            self.key_outline[layer_idx] = torch.cat((self.key_outline[layer_idx],outlier_quant),dim=-2)
        # self.prefetch_layer((layer_idx + 1) % self.layernum,value_states.device) 
        my_score =  onebitgemv(self.key_sign_new[layer_idx],query_states[:,:,-1,:]).unsqueeze(dim=-2).to(query_states.device)
        hlaf_len = int(self.seen_tokens*self.imp_rate)
        _,sort_index = torch.topk(my_score[...,:-self.window_size],k=hlaf_len,dim = -1,sorted=False)
        sort_index = sort_index.squeeze(dim=-2)
        self.attn_cache[layer_idx] = sort_index 
        self.key_cache_temp[0][:,:,self.seen_tokens-1:self.seen_tokens,:].copy_(key_states)
        self.value_cache_temp[0][:,:,self.seen_tokens-1:self.seen_tokens,:].copy_(value_states)
        attn_output = flash_attn_func(
                query_states.transpose(1,2),
                self.key_cache_temp[0][:,:,:self.seen_tokens,:].transpose(1,2),
                self.value_cache_temp[0][:,:,:self.seen_tokens,:].transpose(1,2),
                causal = True
            ).transpose(1,2)
        return attn_output
        # return self.get_kv(layer_idx=layer_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache_full) <= layer_idx:
            return 0
        return self.seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        print("fuck")
        for layer_idx in range(len(self.key_cache_full)):
            device = self.key_cache_full[layer_idx].device
            self.key_cache_full[layer_idx] = self.key_cache_full[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache_full[layer_idx].device
            self.value_cache_full[layer_idx] = self.value_cache_full[layer_idx].index_select(0, beam_idx.to(device))


