from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from quant.myquant import ontbitquant,onebitgemv,extract_sign_and_compress,my_scatter_add,mydequant,my_resduialquant,my_value_quant,my_value_dequant,mykeyquantV2,mykeydequantV2
from quant.myquant import my_key_value_quant_V3,my_key_value_dequant_V3,my_lutgemv,my_key_value_quant_V4,my_key_value_dequant_V4,my_cpu_gather
import matplotlib.pyplot as plt

# from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
batch_size = 1
def get_tensor_mem(tensor):
    tensor_memory_size = tensor.element_size() * tensor.numel()

    # 转换为 MB
    tensor_memory_size_mb = tensor_memory_size / 1024**2
    print("fuck",tensor_memory_size_mb)
    return tensor_memory_size_mb

class KVquant_unit:
    def __init__(self,device='cpu'):
        len = 16050
        # self.quantized_tensor_KV = torch.empty((batch_size, 32, len, 128),device='cuda', dtype=torch.uint8 ,pin_memory=True)
        self.quantized_tensor_KV = torch.empty((batch_size, 32, len, 128),device='cuda', dtype=torch.uint8 )
        self.device = 'cpu'
        self.bit_num = 4
        self.head_dim = 128
        # self.quant_param = torch.empty((batch_size, 32, len, 4),device='cuda', dtype=torch.float,pin_memory=True)
        self.quant_param = torch.empty((batch_size, 32, len, 4),device='cuda', dtype=torch.float)
        self.first = True
        self.quantized_tensor_KV_gpu = None
        self.quant_param_gpu = None
        self.token_len = 0
        self.quantized_tensor_KV_cpu = None
        self.quant_param_cpu = None
        self.prepare_future =None
        self.event = torch.cuda.Event()
    def cat_new_cache(self,quantized_kv,quant_param):
        # print(quantized_kv)
        if self.first:
            self.token_len = quantized_kv.shape[-2]
            self.quantized_tensor_KV[:,:,:self.token_len,:].copy_(quantized_kv,non_blocking=True)
            self.quant_param[:,:,:self.token_len,:].copy_(quant_param,non_blocking=True)
            self.first = False
        else:
            self.token_len+=1
            self.quantized_tensor_KV[:,:,self.token_len-1:self.token_len,:].copy_(quantized_kv,non_blocking=True)
            self.quant_param[:,:,self.token_len-1:self.token_len,:].copy_(quant_param,non_blocking=True)
    def prepare_for_next_gen_in_cpu(self,indices,stream):
        with torch.cuda.stream(stream):
            # self.event.synchronize()
            indices = indices.cpu()
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.quantized_tensor_KV.shape[-1])
            # print(self.quantized_tensor_KV)
            torch.cuda.current_stream().synchronize()
            self.quantized_tensor_KV_cpu = torch.gather(self.quantized_tensor_KV, 2, indices_expanded).contiguous().pin_memory()
            self.quant_param_cpu = torch.gather(self.quant_param, 2, indices.unsqueeze(-1).expand(-1,-1,-1,4)).contiguous().pin_memory()
    def select(self,indices):
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        # print(self.quantized_tensor_KV)
        quantized_tensor_slip = torch.gather(self.quantized_tensor_KV, 2, indices_expanded).contiguous()
        quant_param = torch.gather(self.quant_param, 2, indices.unsqueeze(-1).expand(-1,-1,-1,4)).contiguous()
        return quantized_tensor_slip,quant_param
    def prefetch(self,device):
        self.device = device
        if self.prepare_future is None:
            return
        self.prepare_future.result()
        self.quantized_tensor_KV_gpu = self.quantized_tensor_KV_cpu.to(device=device, non_blocking=True)
        # .to(device=self.device, non_blocking=non_blocking)
        # self.quantized_tensor_KV_gpu = self.quantized_tensor_KV[:, :,:self.token_len, :].to(device, non_blocking=non_blocking)
        # self.quant_param_gpu = self.quant_param_cpu.to(device=device, non_blocking=True)
    def evcit(self):
        if self.quantized_tensor_KV is None:
            return
        self.quantized_tensor_KV_gpu =None
        self.quant_param_gpu = None
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
class SkectchCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,
                 layernum=32
                 ,buff_shape=(batch_size,32,16050,128),dtype=torch.float16,
                 device='cpu') -> None:
        super().__init__()
        self.skip_layer = [0]
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
        self.skectch_seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.window_size = 128
        self.sink_size = 4
        self.imp_rate = 0.3
        self.prefetch_stream={_:torch.cuda.Stream() for _ in range(layernum)}
        self.key_cache_temp=torch.empty(buff_shape,device=device, dtype=dtype)
        self.value_cache_temp=torch.empty(buff_shape,device=device, dtype=dtype)
        self.key_cache_first_layer=torch.empty(buff_shape,device=device, dtype=dtype)
        self.value_cache_first_layer=torch.empty(buff_shape,device=device, dtype=dtype)
        self.device = device
        torch._dynamo.mark_static_address(self.key_cache_temp)
        torch._dynamo.mark_static_address(self.value_cache_temp)
        # print(self.value_cache_temp)
        self.default_stream = torch.cuda.default_stream()
        self.topK = 12
        self.executor = ThreadPoolExecutor(max_workers=layernum+1)
    def resize_buff(self,device,num_key_value_heads=32,dtype=torch.float16,shapes=None):
        if self.key_cache_temp is not None:
            return
        shape=(1,num_key_value_heads,3250,128)
        # for i in range(2):
    def get_total_memory(self,layer_num):
        totla_size = 0
        for i in range(layer_num):
            totla_size += get_tensor_mem(self.key_outline[i])
            totla_size += get_tensor_mem(self.key_cache_full[i])
            totla_size += get_tensor_mem(self.key_cache_index[i])
            totla_size += get_tensor_mem(self.key_cache_means[i])
            totla_size += get_tensor_mem(self.key_cache_channel_zp[i])
            totla_size += get_tensor_mem(self.key_cache_channel_scale[i])
            totla_size += get_tensor_mem(self.value_cache_full[i])
            totla_size += get_tensor_mem(self.value_cache_max[i])
            totla_size += get_tensor_mem(self.value_sink[i])
            totla_size += get_tensor_mem(self.key_sign_new[i])
            totla_size += get_tensor_mem(self.attn_cache[i])
            totla_size += get_tensor_mem(self.outline_quant[i].scales)
            totla_size += get_tensor_mem(self.outline_quant[i].zero_points)
            totla_size += get_tensor_mem(self.KV_cache_quant_unit[i].quant_param)
            totla_size += get_tensor_mem(self.KV_cache_quant_unit[i].quantized_tensor_KV)
            totla_size += get_tensor_mem(self.value_cache_quant_unit_V2[i].quant_param)
            totla_size += get_tensor_mem(self.value_cache_quant_unit_V2[i].quantized_tensor)
        # totla_size+=get_tensor_mem(self.key_cache_temp)*2
        print(f"Tensor 显存占用: {totla_size:.2f} MB")
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
    def prefetch_layer(self,layer_idx:int,non_blocking=True):
        with torch.cuda.stream(self.prefetch_stream[layer_idx]):
            self.KV_cache_quant_unit[layer_idx].prefetch(device = self.device)
            # self.value_cache_quant_unit[layer_idx].prefetch()
    def evict_previous_layer(self, layer_idx: int,non_blocking=True):
        # with torch.cuda.stream(self.prefetch_stream[layer_idx]):
        self.KV_cache_quant_unit[layer_idx].evcit()
        # self.value_cache_quant_unit[layer_idx].evcit()
    def get_kv(self,
        layer_idx: int
    )-> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self.skip_layer:
            return
        device = self.attn_cache[layer_idx].device
        windows_size = self.window_size

        token_num = self.key_sign_new[layer_idx].shape[-2]
        
        mydequant(
            self.key_sign_new[layer_idx].contiguous(),
            self.key_outline[layer_idx].contiguous(),
            self.key_cache_temp,
            self.key_cache_means[layer_idx].contiguous(),
            self.key_cache_index[layer_idx],
            self.outline_quant[layer_idx].zero_points,
            self.outline_quant[layer_idx].scales,
            self.topK
        )
        # torch.cuda.default_stream().synchronize()
        key_value_quantized_data,quant_param = self.KV_cache_quant_unit[layer_idx].select(self.attn_cache[layer_idx])
        my_value_dequant(self.value_cache_quant_unit_V2[layer_idx].quantized_tensor,
                         self.value_cache_temp,
                         self.value_cache_max[layer_idx],
                         self.value_cache_quant_unit_V2[layer_idx].quant_param)
        # print((test_quant_value-quant_value).abs().mean())
        my_scatter_add(
            key_value_quantized_data.contiguous(),
            self.key_cache_temp,
            self.value_cache_temp,
            quant_param.contiguous(),
            self.attn_cache[layer_idx].int(),
            key_value_quantized_data.shape[-2],
            self.value_cache_quant_unit_V2[layer_idx].quantized_tensor.shape[-2]
        )
        
        self.key_cache_temp[:,:,:self.sink_size,:].copy_(self.key_sink[layer_idx],non_blocking=True)
        self.key_cache_temp[:,:,token_num-windows_size:token_num,:].copy_(self.key_cache_full[layer_idx][:,:,-windows_size:,:],non_blocking=True)
        self.value_cache_temp[:,:,:self.sink_size,:].copy_(self.value_sink[layer_idx],non_blocking=True)
        self.value_cache_temp[:,:,token_num-windows_size:token_num,:].copy_(self.value_cache_full[layer_idx][:,:,-windows_size:,:],non_blocking=True)
    def update_new(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            start = self.skectch_seen_tokens
            end = key_states.shape[-2]+self.skectch_seen_tokens
            # self.key_cache_first_layer[:,:,start:end,:].copy_(key_states)
            # self.value_cache_first_layer[:,:,start:end,:].copy_(value_states)
            self.skectch_seen_tokens =end

        # get_tensor_mem(key_states)
        # Update the cache
        self.evict_previous_layer((layer_idx-1)%self.layernum,non_blocking=True)
        self.prefetch_layer((layer_idx + 1) % self.layernum,value_states.device)
        if key_states.shape[-2] != 1:
            self.evict_previous_layer((layer_idx-1)%self.layernum,non_blocking=True)
            means = key_states.abs().mean(dim=-2,keepdim=True)
            self.key_cache_means[layer_idx] = means
            abs_matrix = torch.abs(key_states)
            abs_sum1 = abs_matrix.mean(dim=(2),keepdim=True)
            sorted_indices = torch.argsort(abs_sum1, dim=-1, descending=True)
            maxvalue = value_states.abs().max(dim=-2,keepdim=True)[0].contiguous()
            self.value_cache_max[layer_idx] = maxvalue
            self.key_cache_index[layer_idx] = sorted_indices[...,:self.topK].contiguous().to(torch.uint8)
            qmin = 0.
            qmax = 2.**8 - 1.
            min_val = key_states.min(dim=-2)[0]
            max_val = key_states.max(dim=-2)[0]
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
                self.key_cache_index[layer_idx].contiguous(),
                self.key_cache_channel_zp[layer_idx].contiguous(),
                self.key_cache_channel_scale[layer_idx].contiguous(),
                self.topK
            )
            self.key_sign_new[layer_idx]=key_quant1
            score =  onebitgemv(key_quant1,query_states[:,:,-1,:]).unsqueeze(dim=-2).to(query_states.device)
            hlaf_len = int(key_states.shape[-2]*self.imp_rate)
            _,sort_index = torch.topk(score[...,:-self.window_size],k=hlaf_len,dim = -1,sorted=False)
            sort_index = sort_index.squeeze(dim=-2)
            self.attn_cache[layer_idx] = sort_index
            self.key_cache_full[layer_idx]=(key_states[:,:,-self.window_size:,:]).clone()
            self.value_cache_full[layer_idx]=(value_states[:,:,-self.window_size:,:]).clone()
            self.key_sink[layer_idx] = key_states[:,:,:self.sink_size,:].clone()
            self.value_sink[layer_idx] = value_states[:,:,:self.sink_size,:].clone()
            # print("fuck",key_states.shape)
            
            value_quant_param, value_quant,my_value_resduial = my_value_quant(value_states,maxvalue)
            quant_param, key_value_quant = my_resduialquant(my_resduial,my_value_resduial)
            self.KV_cache_quant_unit[layer_idx].cat_new_cache(key_value_quant,quant_param)
            self.value_cache_quant_unit_V2[layer_idx].cat_new_cache_V2(value_quant,value_quant_param)
            self.outline_quant[layer_idx] = outline_quant_unit(my_outlier_scale_quant.unsqueeze(dim=-2),my_outlier_zp_quant.unsqueeze(dim=-2))
            self.key_outline[layer_idx] = outlier_quant
            
            # self.KV_cache_quant_unit[layer_idx].prepare_future = self.executor.submit(self.KV_cache_quant_unit[layer_idx].prepare_for_next_gen_in_cpu,
            #                                                                      self.attn_cache[layer_idx],self.prefetch_stream[layer_idx])
            # if layer_idx==31:
            # self.get_total_memory(1)
            return key_states,value_states
        else:
            if layer_idx == 0:
                my_score =  onebitgemv(self.key_sign_new[layer_idx],query_states[:,:,-1,:]).unsqueeze(dim=-2).to(query_states.device)
                hlaf_len = int(self.skectch_seen_tokens*self.imp_rate)
                _,sort_index = torch.topk(my_score[...,:-self.window_size],k=hlaf_len,dim = -1,sorted=False)
                sort_index = sort_index.squeeze(dim=-2)
                self.attn_cache[layer_idx] = sort_index 
            elif layer_idx != self.layernum-1:
                my_score =  onebitgemv(self.key_sign_new[layer_idx+1],query_states[:,:,-1,:]).unsqueeze(dim=-2).to(query_states.device)
                hlaf_len = int(self.skectch_seen_tokens*self.imp_rate)
                _,sort_index = torch.topk(my_score[...,:-self.window_size],k=hlaf_len,dim = -1,sorted=False)
                sort_index = sort_index.squeeze(dim=-2)
                self.attn_cache[layer_idx] = sort_index 
            self.get_kv(layer_idx=(layer_idx)%self.layernum)

            value_quant_param, value_quant,my_value_resduial = my_value_quant(value_states,self.value_cache_max[layer_idx])
            self.value_cache_quant_unit_V2[layer_idx].cat_new_cache_V2(value_quant,value_quant_param)
            key_quant1, outlier_quant,my_outlier_zp_quant,my_outlier_scale_quant,my_resduial  = ontbitquant(
                key_states.contiguous(),
                self.key_cache_means[layer_idx].contiguous(),
                self.key_cache_index[layer_idx],
                self.key_cache_channel_zp[layer_idx],
                self.key_cache_channel_scale[layer_idx],
                self.topK
            )
            quant_param, key_value_quant = my_resduialquant(my_resduial.contiguous(),my_value_resduial.contiguous())
            self.key_sign_new[layer_idx] = torch.cat((self.key_sign_new[layer_idx],key_quant1),dim=-2)
            self.KV_cache_quant_unit[layer_idx].cat_new_cache(key_value_quant,quant_param)
            self.key_outline[layer_idx] = torch.cat((self.key_outline[layer_idx],outlier_quant),dim=-2) 
        # print("fuck")

        self.key_cache_temp[:,:,self.skectch_seen_tokens-1:self.skectch_seen_tokens,:].copy_(key_states)
        self.value_cache_temp[:,:,self.skectch_seen_tokens-1:self.skectch_seen_tokens,:].copy_(value_states)
        self.key_cache_full[layer_idx].copy_(self.key_cache_temp[:,:,self.skectch_seen_tokens-self.window_size:self.skectch_seen_tokens,:])
        self.value_cache_full[layer_idx].copy_(self.value_cache_temp[:,:,self.skectch_seen_tokens-self.window_size:self.skectch_seen_tokens,:])
        # self.KV_cache_quant_unit[layer_idx].prepare_future = self.executor.submit(self.KV_cache_quant_unit[layer_idx].prepare_for_next_gen_in_cpu,
        #                                                                          self.attn_cache[layer_idx],self.prefetch_stream[layer_idx])

        return self.key_cache_temp[:,:,:self.skectch_seen_tokens,:],self.value_cache_temp[:,:,:self.skectch_seen_tokens,:]
        # return self.get_kv(layer_idx=layer_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache_full) <= layer_idx:
            return 0
        return self.skectch_seen_tokens

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
    if bit==0:
        return input_matrix
    input_matrix = input_matrix.float()
    shape = input_matrix.shape
    # input_matrix = input_matrix.view(-1,block_size).float()
    ranges = 2**bit -1
    # input_matrix = input_matrix.view(-1,64*128)
    min_val = input_matrix.min(dim=-1,keepdim=True)[0]
    max_val = input_matrix.max(dim=-1,keepdim=True)[0]
    interval = (max_val - min_val) / ranges  # 划分成4个区间，所以是除以3
    # 进行量化
    quantized = torch.round((input_matrix - min_val) / interval)
    quantized = quantized.clamp(0, ranges) 
    # 进行反量化
    dequantized = (quantized * interval) + min_val
    # print(dequantized-input_matrix)
    return dequantized.reshape(shape).half()
def get_zp_scale(input_matrix,bit,shape):
    # input_matrix = input_matrix.view(-1,block_size).float()
    ranges = 2**bit -1
    # input_matrix = input_matrix.view(-1,64*128)
    min_val = input_matrix.min(dim=-1,keepdim=True)[0]
    max_val = input_matrix.max(dim=-1,keepdim=True)[0]
    scale = (max_val - min_val) / ranges  # 划分成4个区间，所以是除以3
    return min_val.view(shape[0],shape[1],shape[2],-1),scale.view(shape[0],shape[1],shape[2],-1)
def mock_spare(key_states,channel_abs,top_indices,topk):
    result = torch.sign(key_states)
    # # # 使用 scatter 将值填入结果张量
    top_indices = top_indices.expand(-1,-1,key_states.shape[-2],-1)
    result = result.scatter(3, top_indices[...,topk:], 0)
    return result
def mock_share_compress(key_states,mask,bit=3):
    the_mask = torch.sign(key_states)*mask
    return quantize_and_dequantize(key_states-the_mask, bit=bit)+the_mask
def mock_value_compress(value_states,mask,bit=2):
    value_states = value_states/mask
    shape = value_states.shape
    value_states = value_states.view(-1,32)
    dequant_value_states = quantize_and_dequantize(value_states, bit=bit)
    dequant_value_states = dequant_value_states.reshape(shape)
    return dequant_value_states*mask
def mock_share_compress_V2(key_states,slip=32):
    shape = key_states.shape
    sign = torch.sign(key_states)
    absmean = torch.abs(key_states).mean(-1,keepdim=True)
    key_states = key_states.abs()
    key_states = key_states.view(shape[0],shape[1],shape[2]//slip,slip,shape[3])
    key_states = key_states.transpose(-1,-2).contiguous()
    
    dequant_key_states_4bit = quantize_and_dequantize(key_states,bit=2)
    key_states = dequant_key_states_4bit.transpose(-1,-2).contiguous()
    return key_states.view(shape)*sign
def mock_share_compress_V3(key_states,slip=32):
    shape = key_states.shape
    # sign = torch.sign(key_states)
    # key_states = key_states.abs()
    key_states = key_states.view(-1,32)
    dequant_key_states = quantize_and_dequantize(key_states, bit=2)
    dequant_key_states = dequant_key_states.reshape(shape)
    return dequant_key_states.view(shape)
    # return dequant_key_states.view(shape)
def true_key1bitquant(key_states,slip=32):
    shape = key_states.shape
    key_states_needquant = key_states.view(shape[0],shape[1],shape[2]//slip,slip,shape[3])
    sign = key_states_needquant.sign()
    trnas_key_states = key_states_needquant.abs()
    min_val, _ = torch.min(trnas_key_states, dim=-2,keepdim=True)
    max_val, _ = torch.max(trnas_key_states, dim=-2,keepdim=True)
    # 计算 scale 和 zp (量化范围是 [0, 255])
    scale = ((max_val - min_val) / 3).contiguous()
    zp = (min_val).contiguous()
    key_1bit,key_2bit = mykeyquantV2(
        key_states,
        zp,scale,group_size=32
    )
    return key_1bit,key_2bit,zp,scale
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
class Prune_KV_unit:
    def __init__(self,maxlen=36000,batch_size=4):
        len = maxlen
        self.quantized_tensor_KV = torch.zeros((batch_size, 8, len, 16),device='cpu', dtype=torch.int32 ,pin_memory=True)
        self.device = 'cpu'
        self.bit_num = 4
        self.head_dim = 128
        self.quant_param = torch.zeros((batch_size, 8, len, 16),device='cpu', dtype=torch.half,pin_memory=True)
        self.first = True
        self.quantized_tensor_KV_gpu = None
        self.quant_param_gpu = None
        self.token_len = 0
        self.quantized_tensor_KV_cpu = torch.zeros((batch_size*32*8192*16),device='cpu', dtype=torch.int32 ,pin_memory=True)
        self.quant_param_cpu = torch.zeros((batch_size*32*8192*16),device='cpu', dtype=torch.half,pin_memory=True)
        self.prepare_future =None
        self.event = torch.cuda.Event()
        self.shape = None
        self.numel = None
    def cat_new_cache(self,quantized_kv,quant_param=None):
        # print(quantized_kv)
        if self.first:
            self.token_len = quantized_kv.shape[-2]
            self.quantized_tensor_KV[:,:,:self.token_len,:].copy_(quantized_kv.to('cpu'))
            self.quant_param[:,:,:self.token_len,:].copy_(quant_param.to('cpu'))
            self.first = False
        else:
            self.quantized_tensor_KV[:,:,self.token_len:self.token_len+quantized_kv.shape[-2],:].copy_(quantized_kv.to('cpu'))
            self.token_len+=quantized_kv.shape[-2]
            self.quant_param[:,:,self.token_len:self.token_len+quantized_kv.shape[-2],:].copy_(quant_param.to('cpu'))
        # self.event.record()
        # if self.quant_param is None:
        #     self.quant_param = quant_param
        # else:
        #     self.quant_param = torch.cat((self.quant_param,quant_param),dim=-2)
    def prepare_for_next_gen_in_cpu(self,indices,stream):
        # with torch.cuda.stream(stream):
            # self.event.synchronize()
        indices = indices.to('cpu',non_blocking=False)
        my_cpu_gather(self.quantized_tensor_KV, self.quant_param, indices, self.quantized_tensor_KV_cpu, self.quant_param_cpu)
        # print(cnt_kv2.sub(cnt_kv).abs().max())
        # print(cnt_param2.sub(cnt_param).abs().max())
        self.numel = indices.numel()*16
        return self.quantized_tensor_KV, self.quant_param
            # self.quant_param_cpu = torch.gather(self.quant_param, 2, indices.expand(-1,-1,-1,4)).contiguous().pin_memory()
    def select(self,indices):
        return self.quantized_tensor_KV_gpu,self.quant_param_gpu
    def prefetch(self,quantized_tensor_KV_buffer,quant_param_buffer,device,stream):
        self.device = device
        with torch.cuda.stream(stream):
            if self.prepare_future is None:
                return
            self.prepare_future.result()

            quantized_tensor_KV_buffer[:self.numel].copy_(self.quantized_tensor_KV_cpu[:self.numel],non_blocking=True)
            quant_param_buffer[:self.numel].copy_(self.quant_param_cpu[:self.numel],non_blocking=True)
            self.quantized_tensor_KV_gpu = quantized_tensor_KV_buffer[:self.numel]
            self.quant_param_gpu =quant_param_buffer[:self.numel]
    def evcit(self):
        if self.quantized_tensor_KV is None:
            return
        self.quantized_tensor_KV_gpu =None
        self.quant_param_gpu = None
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
def pro_query(qnow,qpre):
    mask = torch.sign(qnow)!=torch.sign(qpre)
    a = qnow.clone()
    a[mask]=0
    return a
class pruneCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,layers=None,device = None,layer_num = 32,maxlen=138100,batch_size=1) -> None:
        super().__init__()
        self.batch_size=batch_size
        self.layer_num = layer_num
        self.maxlen = maxlen+128
        self.prefetch_len = 12200
        self.KV_unit_cpu: List[Prune_KV_unit] = []
        self.value_norm: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.query_cache: List[torch.Tensor] = []
        self.org_value_cache: List[torch.Tensor] = []
        self.org_key_cache: List[torch.Tensor] = []
        self.sink_value_cache: List[torch.Tensor] = []
        self.sink_key_cache: List[torch.Tensor] = []
        self.key_cache_1bit: List[torch.Tensor] = []
        self.key_value_cache_2bit: List[torch.Tensor] = []
        self.key_zp: List[torch.Tensor] = []
        self.key_scale: List[torch.Tensor] = []
        self.value_zp: List[torch.Tensor] = []
        self.value_scale: List[torch.Tensor] = []
        self.pruneCache_seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.topk = 64
        self.neaset_len = 124
        self.sink_len = 4
        self.channel_abs: List[torch.Tensor] = []
        self.key_channel_abs: List[torch.Tensor] = []
        self.key_channel_order: List[torch.Tensor] = []
        self.key_channel_abs_mean: List[torch.Tensor] = []
        self.the_key: List[torch.Tensor] = []
        self.quant_param:List[torch.Tensor] = []
        self.key_buffer=None
        self.value_buffer=None
        self.out_key_buffer=None
        self.out_value_buffer=None
        self.layers = layers
        self.slip=32
        self.att_score: List[torch.Tensor] = []
        self.executor = ThreadPoolExecutor(max_workers=self.layer_num+1)
        self.prefetch_stream={_:torch.cuda.Stream() for _ in range(self.layer_num)}
        self.kernel_size = 3
        self.quantized_tensor_KV_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 )]
        self.quant_param_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half )]
        self.key1bit2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 )]
        print("true sketchkv")
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
            if layer_idx == 0:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.prefetch_len = int(key_states.shape[-2]*0.15)
                # self.prefetch_len = 384
                
                # self.prefetch_len = 4600
                # if <60000:
                #     self.prefetch_len = 3200
                # else:
                    
            else:
                self.key_cache.append(None)
            q_1 = query_states[:,:,-1:,:]
            q_1 = q_1/torch.norm(q_1,dim=-1,p=2,keepdim=True)
            shape = q_1.shape
            q_1 = q_1.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            self.query_cache.append(q_1)
            
            # median, indices = torch.median(key_states,dim=-2,keepdim=True)
            median = torch.mean(key_states,dim=-2,keepdim=True)
            self.key_channel_order.append(median)
            key_states = key_states-self.key_channel_order[layer_idx]
            
            self.KV_unit_cpu.append(Prune_KV_unit(maxlen=self.maxlen,batch_size = self.batch_size))
            self.org_key_cache.append(key_states[:,:,-self.neaset_len:,:].clone())
            self.sink_key_cache.append(key_states[:,:,:self.sink_len,:].clone())
            self.sink_value_cache.append(value_states[:,:,:self.sink_len,:].clone())
            self.org_value_cache.append(value_states[:,:,-self.neaset_len:,:].clone())
            now_abs = torch.max(key_states.abs(),dim=-2,keepdim=True)[0]
            # now_abs = torch.std(key_states.abs(),dim=-2,keepdim=True)[0]
            self.channel_abs.append(torch.max(value_states.abs(),dim=-2,keepdim=True)[0])

            self.key_channel_abs.append(now_abs)
            self.key_buffer = torch.zeros((self.batch_size*key_states.shape[1]*key_states.shape[2]*128),device = key_states.device,dtype = key_states.dtype)
            self.value_buffer = torch.zeros((self.batch_size*key_states.shape[1]*key_states.shape[2]*128),device = key_states.device,dtype = key_states.dtype)
            need_compress_keycache = key_states[:,:,self.sink_len:-self.neaset_len,:].contiguous()
            need_compress_valuecache = value_states[:,:,self.sink_len:-self.neaset_len,:].contiguous()
            # self.

            # *torch.sum(need_compress_keycache.abs().view(-1,128)*key_weight,dim=-1,keepdim=True)
            self.the_key.append(torch.mean(need_compress_keycache.abs(),dim=-2,keepdim=True))
            # need_compress_keycache = torch.matmul(need_compress_keycache,self.key_channel_order[layer_idx]/now_abs)
            key_value_quant,key_1bit,quant_param = true_key_value_quant(key_states=need_compress_keycache,
                                                                                                  value_states=need_compress_valuecache,
                                                                                                  key_mask=now_abs,
                                                                                                  value_mask=self.channel_abs[layer_idx])
            self.KV_unit_cpu[layer_idx].cat_new_cache(key_value_quant,quant_param)
            # print(quant_param[0,0,12,:])
            
            score_now =  my_lutgemv(key_1bit,self.query_cache[layer_idx]*self.the_key[layer_idx]).to(query_states.device)
            # score_now = torch.matmul(self.the_key[layer_idx],(self.query_cache[layer_idx]).transpose(-1,-2)).squeeze(-1)
            # print(self.value_norm[layer_idx].shape)
            # print(score_now.shape)
            # print(self.key_buffer.shape)
            # print(self.key_buffer)
            # print("fuck",layer_idx,self.key_buffer.shape)
            score_now = F.avg_pool1d(score_now, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1).unsqueeze(dim=-2)
            
            hlaf_len = min(self.prefetch_len,score_now.shape[-1])
            top_scores, top_indices = torch.topk(score_now,k=hlaf_len,dim = -1)
            self.att_score.append(top_indices)
            self.KV_unit_cpu[layer_idx].prepare_future = self.executor.submit(self.KV_unit_cpu[layer_idx].prepare_for_next_gen_in_cpu,
                                                                                 top_indices.squeeze(-2).unsqueeze(-1),self.prefetch_stream[layer_idx])
            # if layer_idx==self.layer_num-1:
            #     self.KV_unit_cpu[0].prefetch(self.quantized_tensor_KV_gpu2[0],self.quant_param_gpu2[0] ,key_states.device,self.prefetch_stream[0])
            # my_key_value_dequant_V3(dequant_key,dequant_value,key_value_quant,key_1bit,key_zp,key_scale,value_zp,value_scale)
            self.key_cache_1bit.append(key_1bit)
            
            return key_states,value_states
        else:
            
            self.KV_unit_cpu[(layer_idx+1)%self.layer_num].prefetch(self.quantized_tensor_KV_gpu2[(layer_idx+1)%2],self.quant_param_gpu2[(layer_idx+1)%2],key_states.device,self.prefetch_stream[(layer_idx+1)%self.layer_num])
            
            if layer_idx == 0:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                return self.key_cache[layer_idx],self.value_cache[layer_idx] 
            key_states = key_states-self.key_channel_order[layer_idx]
            pre_query = self.query_cache[layer_idx]
            q_1 = query_states[:,:,-1:,:]/torch.norm(query_states[:,:,-1:,:],dim=-1,p=2,keepdim=True)
            shape = q_1.shape
            q_1 = q_1.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            self.query_cache[layer_idx] =q_1
            self.org_key_cache[layer_idx] = torch.cat([self.org_key_cache[layer_idx], key_states], dim=-2)
            self.org_value_cache[layer_idx] = torch.cat([self.org_value_cache[layer_idx], value_states], dim=-2)
            pre_indices = self.att_score[layer_idx] 
            # duibi =  onebitgemv(self.key_cache_1bit[layer_idx],self.query_cache[layer_idx]*self.key_channel_abs[layer_idx])[...,:self.pruneCache_seen_tokens+1-self.neaset_len].to(query_states.device)
            
            score_now =  my_lutgemv(self.key_cache_1bit[layer_idx],self.query_cache[layer_idx]*self.the_key[layer_idx]).to(query_states.device)
            # score_now = torch.matmul(self.the_key[layer_idx],(self.query_cache[layer_idx]).transpose(-1,-2)).squeeze(-1)
            score_now = F.avg_pool1d(score_now, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1).unsqueeze(dim=-2)
            hlaf_len = min(self.prefetch_len,score_now.shape[-1])
            
            now_len = min(self.prefetch_len,score_now.shape[-1])
            
            _, now_indices = torch.topk(score_now,k=now_len,dim = -1)
            self.KV_unit_cpu[layer_idx].prepare_future = self.executor.submit(self.KV_unit_cpu[layer_idx].prepare_for_next_gen_in_cpu,
                                                                                 now_indices.squeeze(-2).unsqueeze(-1),self.prefetch_stream[layer_idx])
            self.att_score[layer_idx] = now_indices
            cnt =  pre_indices.squeeze(-2).unsqueeze(-1).contiguous()
            # print(cnt.shape,cnt.dtype)
            k_1bit = self.select_and_dequant(layer_idx,cnt,self.key1bit2[layer_idx%2][:pre_indices.numel()*16].view(-1,key_states.shape[1],hlaf_len,16))
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream[layer_idx])
            kv_2bit = self.KV_unit_cpu[layer_idx].quantized_tensor_KV_gpu.view(self.batch_size,8,-1,16)
            quant_param = self.KV_unit_cpu[layer_idx].quant_param_gpu.view(self.batch_size,8,-1,16)
            # print(kv_2bit.shape)
            my_key_value_dequant_V4(self.key_buffer,self.value_buffer,kv_2bit,k_1bit,quant_param)
            out_k_pahse1 = self.key_buffer[:self.batch_size*key_states.shape[1]*hlaf_len*128].view(-1,key_states.shape[1],hlaf_len,128)*self.key_channel_abs[layer_idx]
            out_v_pahse1 = self.value_buffer[:self.batch_size*key_states.shape[1]*hlaf_len*128].view(-1,key_states.shape[1],hlaf_len,128)*self.channel_abs[layer_idx]
            # print(out_k_pahse1.shape,self.org_key_cache[layer_idx].shape)
            out_k = torch.cat((out_k_pahse1,self.sink_key_cache[layer_idx],self.org_key_cache[layer_idx]),dim=-2)
            out_v = torch.cat((out_v_pahse1,self.sink_value_cache[layer_idx],self.org_value_cache[layer_idx]),dim=-2)
            
            # self.KV_unit_cpu[layer_idx].evcit()

        return out_k.contiguous(), out_v.contiguous()
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

class mock_pruneCacheV2(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,layers=None,device = None,layer_num = 28,maxlen=138100,batch_size=1) -> None:
        super().__init__()
        self.batch_size=batch_size
        self.layer_num = layer_num
        self.maxlen = maxlen+128
        self.prefetch_len = 384
        self.KV_unit_cpu: List[Prune_KV_unit] = []
        self.value_norm: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.query_cache: List[torch.Tensor] = []
        self.org_value_cache: List[torch.Tensor] = []
        self.org_key_cache: List[torch.Tensor] = []
        self.sink_value_cache: List[torch.Tensor] = []
        self.sink_key_cache: List[torch.Tensor] = []
        self.key_cache_1bit: List[torch.Tensor] = []
        self.compress_key: List[torch.Tensor] = []
        self.compress_value: List[torch.Tensor] = []
        self.pruneCache_seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.topk = 64
        self.neaset_len = 124
        self.sink_len = 4
        self.channel_abs: List[torch.Tensor] = []
        self.key_channel_abs: List[torch.Tensor] = []
        self.key_channel_order: List[torch.Tensor] = []
        self.key_channel_abs_mean: List[torch.Tensor] = []
        self.the_key: List[torch.Tensor] = []
        self.quant_param:List[torch.Tensor] = []
        self.hid:List[torch.Tensor] = []
        self.key_buffer=None
        self.value_buffer=None
        self.out_key_buffer=None
        self.out_value_buffer=None
        self.layers = layers
        self.slip=32
        self.att_score: List[torch.Tensor] = []
        self.executor = ThreadPoolExecutor(max_workers=self.layer_num+1)
        self.prefetch_stream={_:torch.cuda.Stream() for _ in range(self.layer_num)}
        self.kernel_size = 5
        self.quantized_tensor_KV_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 )]
        self.quant_param_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half )]
        self.key1bit2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 )]
        
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
        hid : torch.Tensor,
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
        # print("mock")
        if layer_idx == 0:
            self.pruneCache_seen_tokens += key_states.shape[-2]
            # print("fuck")
        # Update the cache
        # print("fuck")
        # print(len(self.key_cache))
        if len(self.key_cache) <= layer_idx:
            if layer_idx ==0:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.prefetch_len = int(key_states.shape[-2]*0.15)
            # print(self.prefetch_len)
            else:
                self.key_cache.append(None)
            q_1 = query_states
            shape = q_1.shape
            q_1 = q_1.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            self.query_cache.append(q_1)
            self.hid.append(hid)
            _,index = torch.topk(q_1.abs().mean(dim=-2,keepdim=True)+key_states.abs().mean(dim=-2,keepdim=True),k=12,dim=-1)
            a = torch.zeros_like(q_1.abs().mean(dim=-2,keepdim=True))
            a.scatter_(-1,index,1)
            self.key_channel_order.append(a)
            org_key_cache = key_states[:,:,-self.neaset_len:,:].clone()
            sink_key_cache = (key_states[:,:,:self.sink_len,:].clone())
            sink_value_cache = value_states[:,:,:self.sink_len,:].clone()
            org_value_cache = value_states[:,:,-self.neaset_len:,:].clone()
            now_abs = torch.max(key_states.abs(),dim=-2,keepdim=True)[0]
            self.org_key_cache.append(org_key_cache)
            self.org_value_cache.append(org_value_cache)
            self.sink_value_cache.append(sink_value_cache)
            self.sink_key_cache.append(sink_key_cache)
            # now_abs = torch.std(key_states.abs(),dim=-2,keepdim=True)[0]
            self.channel_abs.append(torch.max(value_states.abs(),dim=-2,keepdim=True)[0])

            self.key_channel_abs.append(now_abs)
            need_compress_keycache = key_states[:,:,self.sink_len:-self.neaset_len,:].contiguous()
            need_compress_valuecache = value_states[:,:,self.sink_len:-self.neaset_len,:].contiguous()
            # self.
            # print(key_states.shape,need_compress_keycache.shape)
            # print(self.key_cache_1bit[layer_idx].shape)            # print("fucl")
            self.compress_key.append(need_compress_keycache)
            self.compress_value.append(need_compress_valuecache)
            # *torch.sum(need_compress_keycache.abs().view(-1,128)*key_weight,dim=-1,keepdim=True)
            self.the_key.append(torch.mean(need_compress_keycache.abs(),dim=-2,keepdim=True))

            return key_states,value_states
        else:
            self.hid[layer_idx] = hid
            # if layer_idx!=0:
            #     sim = F.cosine_similarity(hid.view(-1),self.hid[layer_idx-1].view(-1),dim=0)
            #     sim2 = F.cosine_similarity(next_layer_query_states.view(-1),query_states.view(-1),dim=0)
            #     print(layer_idx,sim,sim2)
            # print(layer_idx)
            shape = next_layer_query_states.shape
            next_layer_query_states = next_layer_query_states.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            self.query_cache[(layer_idx+1)%32] = next_layer_query_states
            if layer_idx == 0:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

                return self.key_cache[layer_idx],self.value_cache[layer_idx] 
            
            self.org_key_cache[layer_idx] = torch.cat([self.org_key_cache[layer_idx], key_states], dim=-2)
            self.org_value_cache[layer_idx] = torch.cat([self.org_value_cache[layer_idx], value_states], dim=-2)
            # duibi =  onebitgemv(self.key_cache_1bit[layer_idx],self.query_cache[layer_idx]*self.key_channel_abs[layer_idx])[...,:self.pruneCache_seen_tokens+1-self.neaset_len].to(query_states.device)
            # print(self.query_cache[layer_idx])
            # score_now =  my_lutgemv(self.key_cache_1bit[layer_idx],self.query_cache[layer_idx]*self.the_key[layer_idx]).to(query_states.device)
            score_now = torch.matmul(self.compress_key[layer_idx],(self.key_channel_order[layer_idx]*self.query_cache[layer_idx]).transpose(-1,-2)).squeeze(-1)
            # score_now = F.avg_pool1d(score_now, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1).unsqueeze(dim=-2)
            # print("111111")
            hlaf_len = min(self.prefetch_len,score_now.shape[-1])
            
            now_len = min(self.prefetch_len,score_now.shape[-1])
            
            _, now_indices = torch.topk(score_now,k=now_len,dim = -1)
            # print(now_indices.shape)
            now_indices = now_indices.squeeze(-2).unsqueeze(-1).expand(-1,-1,-1,128)
            # print(now_indices.shape)
            out_k_pahse1 = torch.gather(self.compress_key[layer_idx],-2,now_indices)
            # print(out_k_pahse1.shape)
            out_v_pahse1 = torch.gather(self.compress_value[layer_idx],-2,now_indices)
            out_k = torch.cat((self.sink_key_cache[layer_idx],out_k_pahse1,self.org_key_cache[layer_idx]),dim=-2)
            out_v = torch.cat((self.sink_value_cache[layer_idx],out_v_pahse1,self.org_value_cache[layer_idx]),dim=-2)
        # print(out_v.shape)
            # self.KV_unit_cpu[layer_idx].evcit()
        # print(out_k.shape)
        return out_k.contiguous(), out_v.contiguous()
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
        return self.key_cache[0].shape[-2]

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

class mock_pruneCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,layers=None,device = None,layer_num = 28,maxlen=138100,batch_size=1) -> None:
        super().__init__()
        self.batch_size=batch_size
        self.layer_num = layer_num
        self.maxlen = maxlen+128
        self.prefetch_len = 12200
        self.KV_unit_cpu: List[Prune_KV_unit] = []
        self.value_norm: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.query_cache: List[torch.Tensor] = []
        self.org_value_cache: List[torch.Tensor] = []
        self.org_key_cache: List[torch.Tensor] = []
        self.sink_value_cache: List[torch.Tensor] = []
        self.sink_key_cache: List[torch.Tensor] = []
        self.key_cache_1bit: List[torch.Tensor] = []
        self.compress_key: List[torch.Tensor] = []
        self.compress_value: List[torch.Tensor] = []
        self.pruneCache_seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.topk = 64
        self.neaset_len = 124
        self.sink_len = 4
        self.channel_abs: List[torch.Tensor] = []
        self.key_channel_abs: List[torch.Tensor] = []
        self.key_channel_order: List[torch.Tensor] = []
        self.key_channel_abs_mean: List[torch.Tensor] = []
        self.the_key: List[torch.Tensor] = []
        self.quant_param:List[torch.Tensor] = []
        self.key_buffer=None
        self.value_buffer=None
        self.out_key_buffer=None
        self.out_value_buffer=None
        self.layers = layers
        self.slip=32
        self.att_score: List[torch.Tensor] = []
        self.executor = ThreadPoolExecutor(max_workers=self.layer_num+1)
        self.prefetch_stream={_:torch.cuda.Stream() for _ in range(self.layer_num)}
        self.kernel_size = 3
        self.quantized_tensor_KV_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 )]
        self.quant_param_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half )]
        self.key1bit2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 )]
        
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
        # print("mock")
        if layer_idx == 0:
            self.pruneCache_seen_tokens += key_states.shape[-2]
        # print("fuck")
        # Update the cache
        # print("fuck")
        # print(len(self.key_cache))
        if len(self.key_cache) <= layer_idx:
            if layer_idx ==0:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                # self.prefetch_len = int(key_states.shape[-2]*0.15)
                self.prefetch_len = int(key_states.shape[-2]*0.15)
            else:
                self.key_cache.append(None)
            q_1 = query_states[:,:,-1:,:]
            q_1 = q_1/torch.norm(q_1,dim=-1,p=2,keepdim=True)
            shape = q_1.shape
            q_1 = q_1.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            self.query_cache.append(q_1)
            
            # median, indices = torch.median(key_states,dim=-2,keepdim=True)
            median = torch.mean(key_states,dim=-2,keepdim=True)
            self.key_channel_order.append(median)
            key_states = key_states-self.key_channel_order[layer_idx]
            
            self.org_key_cache.append(key_states[:,:,-self.neaset_len:,:].clone())
            self.sink_key_cache.append(key_states[:,:,:self.sink_len,:].clone())
            self.sink_value_cache.append(value_states[:,:,:self.sink_len,:].clone())
            self.org_value_cache.append(value_states[:,:,-self.neaset_len:,:].clone())
            now_abs = torch.max(key_states.abs(),dim=-2,keepdim=True)[0]
            # now_abs = torch.std(key_states.abs(),dim=-2,keepdim=True)[0]
            self.channel_abs.append(torch.max(value_states.abs(),dim=-2,keepdim=True)[0])

            self.key_channel_abs.append(now_abs)
            need_compress_keycache = key_states[:,:,self.sink_len:-self.neaset_len,:].contiguous()
            need_compress_valuecache = value_states[:,:,self.sink_len:-self.neaset_len,:].contiguous()
            # self.
            # print("fuck")
            
            # print(key_states.shape,need_compress_keycache.shape)
            # print(self.key_cache_1bit[layer_idx].shape)
            need_compress_keycache_compress = mock_share_compress_V3(need_compress_keycache.abs()/now_abs)*now_abs*need_compress_keycache.sign()
            # print((need_compress_keycache1-need_compress_keycache).abs().mean())
            need_compress_valuecache_compress = mock_share_compress_V3(need_compress_valuecache/self.channel_abs[layer_idx].float())*self.channel_abs[layer_idx]
            # print("fucl")
            
            # *torch.sum(need_compress_keycache.abs().view(-1,128)*key_weight,dim=-1,keepdim=True)
            # key_value_quant,key_1bit,quant_param = true_key_value_quant(key_states=need_compress_keycache,
                                                                                        # value_states=need_compress_valuecache,
                                                                                        # key_mask=now_abs,
                                                                                        # value_mask=self.channel_abs[layer_idx])
            # my_key_value_dequant_V4(need_compress_keycache_compress,need_compress_valuecache_compress,key_value_quant,key_1bit,quant_param)
            self.compress_key.append(need_compress_keycache_compress)
            self.compress_value.append(need_compress_valuecache_compress)
            # self.compress_key.append(need_compress_keycache)
            # self.compress_value.append(need_compress_valuecache)
            self.key_cache_1bit.append(torch.sign(need_compress_keycache))
            self.the_key.append(torch.mean(need_compress_keycache.abs(),dim=-2,keepdim=True))
            # need_compress_keycache = torch.matmul(need_compress_keycache,self.key_channel_order[layer_idx]/now_abs)
            # key_value_quant,key_1bit,quant_param = true_key_value_quant(key_states=need_compress_keycache,
            #                                                                                       value_states=need_compress_valuecache,
            #                                                                                       key_mask=now_abs,
            #                                                                                       value_mask=self.channel_abs[layer_idx])
            # self.KV_unit_cpu[layer_idx].cat_new_cache(key_value_quant,quant_param)
            # print("fuck")
            return key_states,value_states
        else:
            
            # print(layer_idx)
            
            if layer_idx == 0:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                return self.key_cache[layer_idx],self.value_cache[layer_idx] 
            key_states = key_states-self.key_channel_order[layer_idx]
            pre_query = self.query_cache[layer_idx]
            q_1 = query_states[:,:,-1:,:]/torch.norm(query_states[:,:,-1:,:],dim=-1,p=2,keepdim=True)
            shape = q_1.shape
            q_1 = q_1.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            self.query_cache[layer_idx] =q_1
            self.org_key_cache[layer_idx] = torch.cat([self.org_key_cache[layer_idx], key_states], dim=-2)
            self.org_value_cache[layer_idx] = torch.cat([self.org_value_cache[layer_idx], value_states], dim=-2)
            # duibi =  onebitgemv(self.key_cache_1bit[layer_idx],self.query_cache[layer_idx]*self.key_channel_abs[layer_idx])[...,:self.pruneCache_seen_tokens+1-self.neaset_len].to(query_states.device)
            
            # score_now =  my_lutgemv(self.key_cache_1bit[layer_idx],self.query_cache[layer_idx]*self.the_key[layer_idx]).to(query_states.device)
            score_now = torch.matmul(self.key_cache_1bit[layer_idx],(self.the_key[layer_idx]*pre_query).transpose(-1,-2)).squeeze(-1)
            # score_now =  my_lutgemv(self.key_cache_1bit[layer_idx],pre_query*self.the_key[layer_idx]).to(query_states.device)
            score_now = F.avg_pool1d(score_now, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1).unsqueeze(dim=-2)
            hlaf_len = min(self.prefetch_len,score_now.shape[-1])
            
            now_len = min(self.prefetch_len,score_now.shape[-1])
            
            _, now_indices = torch.topk(score_now,k=now_len,dim = -1)
            # print(now_indices.shape)
            now_indices = now_indices.squeeze(-2).unsqueeze(-1).expand(-1,-1,-1,128)
            out_k_pahse1 = torch.gather(self.compress_key[layer_idx],-2,now_indices)
            # print(out_k_pahse1.shape)
            out_v_pahse1 = torch.gather(self.compress_value[layer_idx],-2,now_indices)
            out_k = torch.cat((out_k_pahse1,self.sink_key_cache[layer_idx],self.org_key_cache[layer_idx]),dim=-2)
            out_v = torch.cat((out_v_pahse1,self.sink_value_cache[layer_idx],self.org_value_cache[layer_idx]),dim=-2)
            # print("fuck")
            # self.KV_unit_cpu[layer_idx].evcit()
        # print(out_k.shape)
        return out_k.contiguous(), out_v.contiguous()
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
        return self.key_cache[0].shape[-2]

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
class snapkvCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,layers=None,device = None,layer_num = 28,maxlen=138100,batch_size=1) -> None:
        super().__init__()
        self.batch_size=batch_size
        self.layer_num = layer_num
        self.maxlen = maxlen+128
        self.prefetch_len = 384
        self.KV_unit_cpu: List[Prune_KV_unit] = []
        self.value_norm: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.query_cache: List[torch.Tensor] = []
        self.org_value_cache: List[torch.Tensor] = []
        self.org_key_cache: List[torch.Tensor] = []
        self.sink_value_cache: List[torch.Tensor] = []
        self.sink_key_cache: List[torch.Tensor] = []
        self.key_cache_1bit: List[torch.Tensor] = []
        self.compress_key: List[torch.Tensor] = []
        self.compress_value: List[torch.Tensor] = []
        self.pruneCache_seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.topk = 64
        self.neaset_len = 124
        self.sink_len = 4
        self.channel_abs: List[torch.Tensor] = []
        self.key_channel_abs: List[torch.Tensor] = []
        self.key_channel_order: List[torch.Tensor] = []
        self.key_channel_abs_mean: List[torch.Tensor] = []
        self.the_key: List[torch.Tensor] = []
        self.quant_param:List[torch.Tensor] = []
        self.key_buffer=None
        self.value_buffer=None
        self.out_key_buffer=None
        self.out_value_buffer=None
        self.layers = layers
        self.slip=32
        self.att_score: List[torch.Tensor] = []
        self.executor = ThreadPoolExecutor(max_workers=self.layer_num+1)
        self.prefetch_stream={_:torch.cuda.Stream() for _ in range(self.layer_num)}
        self.kernel_size = 5
        self.quantized_tensor_KV_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 )]
        self.quant_param_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half )]
        self.key1bit2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 )]
        
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
        # print("mock")
        if layer_idx == 0:
            self.pruneCache_seen_tokens += key_states.shape[-2]
        # print("fuck")
        # Update the cache
        # print("fuck")
        # print(len(self.key_cache))
        if len(self.key_cache) <= layer_idx:
            if layer_idx ==0:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.prefetch_len = int(key_states.shape[-2]*0.15)
            else:
                pre_query = query_states[:,:,-20:,:]
                shape = pre_query.shape
                pre_query = pre_query.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2).mean(dim=-2,keepdim=True)
                score_now = torch.matmul(key_states,pre_query.transpose(-1,-2)).squeeze(-1)
                # print(score_now.shape)
                score_now = F.avg_pool1d(score_now, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1).unsqueeze(dim=-2)
                now_len = min(self.prefetch_len,score_now.shape[-1])
                _, now_indices = torch.topk(score_now,k=now_len,dim = -1)
                # print(now_indices.shape)
                now_indices = now_indices.squeeze(-2).unsqueeze(-1).expand(-1,-1,-1,128)
                out_k_pahse1 = torch.gather(key_states,-2,now_indices)
                # print(out_k_pahse1.shape)
                out_v_pahse1 = torch.gather(value_states,-2,now_indices)
                self.key_cache.append(out_k_pahse1)
                self.value_cache.append(out_v_pahse1)
            return key_states,value_states
        else:
            # print("fuck")
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx],self.value_cache[layer_idx] 
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
        return self.key_cache[0].shape[-2]

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
    
class mock_compressim(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,layers=None,device = None,layer_num = 28,maxlen=138100,batch_size=1) -> None:
        super().__init__()
        self.batch_size=batch_size
        self.layer_num = layer_num
        self.maxlen = maxlen+128
        self.prefetch_len = 384
        self.KV_unit_cpu: List[Prune_KV_unit] = []
        self.value_norm: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.query_cache: List[torch.Tensor] = []
        self.org_value_cache: List[torch.Tensor] = []
        self.org_key_cache: List[torch.Tensor] = []
        self.sink_value_cache: List[torch.Tensor] = []
        self.sink_key_cache: List[torch.Tensor] = []
        self.key_cache_1bit: List[torch.Tensor] = []
        self.compress_key: List[torch.Tensor] = []
        self.compress_value: List[torch.Tensor] = []
        self.pruneCache_seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.topk = 64
        self.neaset_len = 124
        self.sink_len = 4
        self.channel_abs: List[torch.Tensor] = []
        self.key_channel_abs: List[torch.Tensor] = []
        self.key_channel_order: List[torch.Tensor] = []
        self.key_channel_abs_mean: List[torch.Tensor] = []
        self.the_key: List[torch.Tensor] = []
        self.quant_param:List[torch.Tensor] = []
        self.hid:List[torch.Tensor] = []
        self.key_buffer=None
        self.value_buffer=None
        self.out_key_buffer=None
        self.out_value_buffer=None
        self.layers = layers
        self.slip=32
        self.att_score: List[torch.Tensor] = []
        self.executor = ThreadPoolExecutor(max_workers=self.layer_num+1)
        self.prefetch_stream={_:torch.cuda.Stream() for _ in range(self.layer_num)}
        self.kernel_size = 5
        self.quantized_tensor_KV_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 )]
        self.quant_param_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half )]
        self.key1bit2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 )]
        
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
        hid : torch.Tensor,
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
        # print("mock")
        if layer_idx == 0:
            self.pruneCache_seen_tokens += key_states.shape[-2]
        # print("fuck")
        # Update the cache
        # print("fuck")
        # print(len(self.key_cache))
        if len(self.key_cache) <= layer_idx:
            if layer_idx <=1:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache.append(None)
            q_1 = query_states[:,:,-1:,:]
            shape = q_1.shape
            q_1 = q_1.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            self.query_cache.append(q_1)
            self.hid.append(hid)
            _,index = torch.topk(q_1.abs().mean(dim=-2,keepdim=True)+key_states.abs().mean(dim=-2,keepdim=True),k=12,dim=-1)
            a = torch.zeros_like(q_1.abs().mean(dim=-2,keepdim=True))
            a.scatter_(-1,index,1)
            self.key_channel_order.append(a)
            org_key_cache = key_states[:,:,-self.neaset_len:,:].clone()
            sink_key_cache = (key_states[:,:,:self.sink_len,:].clone())
            sink_value_cache = value_states[:,:,:self.sink_len,:].clone()
            org_value_cache = value_states[:,:,-self.neaset_len:,:].clone()
            now_abs = torch.max(key_states.abs(),dim=-2,keepdim=True)[0]
            self.org_key_cache.append(org_key_cache)
            self.org_value_cache.append(org_value_cache)
            self.sink_value_cache.append(sink_value_cache)
            self.sink_key_cache.append(sink_key_cache)
            # now_abs = torch.std(key_states.abs(),dim=-2,keepdim=True)[0]
            self.channel_abs.append(torch.max(value_states.abs(),dim=-2,keepdim=True)[0])

            self.key_channel_abs.append(now_abs)
            need_compress_keycache = key_states[:,:,self.sink_len:-self.neaset_len,:].contiguous()
            need_compress_valuecache = value_states[:,:,self.sink_len:-self.neaset_len,:].contiguous()
            # self.
            # print(key_states.shape,need_compress_keycache.shape)
            # print(self.key_cache_1bit[layer_idx].shape)            # print("fucl")
            self.compress_key.append(need_compress_keycache)
            self.compress_value.append(need_compress_valuecache)
            # *torch.sum(need_compress_keycache.abs().view(-1,128)*key_weight,dim=-1,keepdim=True)
            self.the_key.append(torch.mean(need_compress_keycache.abs(),dim=-2,keepdim=True))

            return key_states,value_states
        else:
            shape = query_states.shape
            q_1 = query_states.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            if layer_idx!=0:

                sim = F.cosine_similarity(hid.view(-1),self.hid[layer_idx-1].view(-1),dim=0)
                sim2 = F.cosine_similarity(q_1.view(-1),self.query_cache[layer_idx].view(-1),dim=0)
                print("hid",layer_idx,sim,sim2)
            self.hid[layer_idx] = hid
            # print(layer_idx)
            shape = next_layer_query_states.shape
            next_layer_query_states = next_layer_query_states.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            self.query_cache[layer_idx] = q_1
            if layer_idx <= 1:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

                return self.key_cache[layer_idx],self.value_cache[layer_idx] 
            
            self.org_key_cache[layer_idx] = torch.cat([self.org_key_cache[layer_idx], key_states], dim=-2)
            self.org_value_cache[layer_idx] = torch.cat([self.org_value_cache[layer_idx], value_states], dim=-2)
            # duibi =  onebitgemv(self.key_cache_1bit[layer_idx],self.query_cache[layer_idx]*self.key_channel_abs[layer_idx])[...,:self.pruneCache_seen_tokens+1-self.neaset_len].to(query_states.device)
            # print(self.query_cache[layer_idx])
            # score_now =  my_lutgemv(self.key_cache_1bit[layer_idx],self.query_cache[layer_idx]*self.the_key[layer_idx]).to(query_states.device)
            score_now = torch.matmul(self.compress_key[layer_idx],(self.key_channel_order[layer_idx]*self.query_cache[layer_idx]).transpose(-1,-2)).squeeze(-1)
            # score_now = F.avg_pool1d(score_now, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1).unsqueeze(dim=-2)
            # print("111111")
            hlaf_len = min(self.prefetch_len,score_now.shape[-1])
            
            now_len = min(self.prefetch_len,score_now.shape[-1])
            
            _, now_indices = torch.topk(score_now,k=now_len,dim = -1)
            # print(now_indices.shape)
            now_indices = now_indices.squeeze(-2).unsqueeze(-1).expand(-1,-1,-1,128)
            # print(now_indices.shape)
            out_k_pahse1 = torch.gather(self.compress_key[layer_idx],-2,now_indices)
            # print(out_k_pahse1.shape)
            out_v_pahse1 = torch.gather(self.compress_value[layer_idx],-2,now_indices)
            out_k = torch.cat((self.sink_key_cache[layer_idx],out_k_pahse1,self.org_key_cache[layer_idx]),dim=-2)
            out_v = torch.cat((self.sink_value_cache[layer_idx],out_v_pahse1,self.org_value_cache[layer_idx]),dim=-2)
        # print(out_v.shape)
            # self.KV_unit_cpu[layer_idx].evcit()
        # print(out_k.shape)
        return out_k.contiguous(), out_v.contiguous()
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
        return self.key_cache[0].shape[-2]

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
    
class infinigenCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,layers=None,device = None,layer_num = 28,maxlen=138100,batch_size=1) -> None:
        super().__init__()
        self.batch_size=batch_size
        self.layer_num = layer_num
        self.maxlen = maxlen+128
        self.prefetch_len = 384
        self.KV_unit_cpu: List[Prune_KV_unit] = []
        self.value_norm: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.query_cache: List[torch.Tensor] = []
        self.org_value_cache: List[torch.Tensor] = []
        self.org_key_cache: List[torch.Tensor] = []
        self.sink_value_cache: List[torch.Tensor] = []
        self.sink_key_cache: List[torch.Tensor] = []
        self.key_cache_1bit: List[torch.Tensor] = []
        self.compress_key: List[torch.Tensor] = []
        self.compress_value: List[torch.Tensor] = []
        self.pruneCache_seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.topk = 64
        self.neaset_len = 124
        self.sink_len = 4
        self.channel_abs: List[torch.Tensor] = []
        self.key_channel_abs: List[torch.Tensor] = []
        self.key_channel_order: List[torch.Tensor] = []
        self.key_channel_abs_mean: List[torch.Tensor] = []
        self.the_key: List[torch.Tensor] = []
        self.quant_param:List[torch.Tensor] = []
        self.key_buffer=None
        self.value_buffer=None
        self.out_key_buffer=None
        self.out_value_buffer=None
        self.layers = layers
        self.slip=32
        self.att_score: List[torch.Tensor] = []
        
        self.hidden_states: List[torch.Tensor] = []
        self.executor = ThreadPoolExecutor(max_workers=self.layer_num+1)
        self.prefetch_stream={_:torch.cuda.Stream() for _ in range(self.layer_num)}
        self.kernel_size = 5
        self.quantized_tensor_KV_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.int32 )]
        self.quant_param_gpu2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.half )]
        self.key1bit2 = [torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 ),torch.zeros((batch_size*32*self.prefetch_len*16),device='cuda:0', dtype=torch.uint8 )]
        
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
        pre_hidden_states: torch.Tensor,
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
        # print("mock")
        
        if layer_idx == 0:
            self.pruneCache_seen_tokens += key_states.shape[-2]
        # print("fuck")
        # Update the cache
        # print("fuck")
        # print(len(self.key_cache))
        if len(self.key_cache) <= layer_idx:
            
            self.hidden_states.append(None)
            if layer_idx ==0:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.prefetch_len = int(key_states.shape[-2]*0.15)
            else:
                self.key_cache.append(None)
            q_1 = query_states
            shape = q_1.shape
            q_1 = q_1.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            self.query_cache.append(q_1)
            
            
            _,index = torch.topk(q_1.abs().mean(dim=-2,keepdim=True)+key_states.abs().mean(dim=-2,keepdim=True),k=12,dim=-1)
            a = torch.zeros_like(q_1.abs().mean(dim=-2,keepdim=True))
            a.scatter_(-1,index,1)
            self.key_channel_order.append(a)
            org_key_cache = key_states[:,:,-self.neaset_len:,:].clone()
            sink_key_cache = (key_states[:,:,:self.sink_len,:].clone())
            sink_value_cache = value_states[:,:,:self.sink_len,:].clone()
            org_value_cache = value_states[:,:,-self.neaset_len:,:].clone()
            now_abs = torch.max(key_states.abs(),dim=-2,keepdim=True)[0]
            self.org_key_cache.append(org_key_cache)
            self.org_value_cache.append(org_value_cache)
            self.sink_value_cache.append(sink_value_cache)
            self.sink_key_cache.append(sink_key_cache)
            # now_abs = torch.std(key_states.abs(),dim=-2,keepdim=True)[0]
            need_compress_keycache = key_states[:,:,self.sink_len:-self.neaset_len,:].contiguous()
            need_compress_valuecache = value_states[:,:,self.sink_len:-self.neaset_len,:].contiguous()
            # self.
            # print(key_states.shape,need_compress_keycache.shape)
            # print(self.key_cache_1bit[layer_idx].shape)            # print("fucl")
            self.compress_key.append(need_compress_keycache)
            self.compress_value.append(need_compress_valuecache)
            # *torch.sum(need_compress_keycache.abs().view(-1,128)*key_weight,dim=-1,keepdim=True)
            self.the_key.append(torch.mean(need_compress_keycache.abs(),dim=-2,keepdim=True))

            return key_states,value_states
        else:
            
            
            self.hidden_states[(layer_idx+1)%32] = pre_hidden_states
            # print(layer_idx)
            if layer_idx == 0:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

                return self.key_cache[layer_idx],self.value_cache[layer_idx] 
            shape = next_layer_query_states.shape
            next_layer_query_states = next_layer_query_states.view(shape[0],key_states.shape[1],-1,shape[2],shape[3]).mean(dim=2)
            self.org_key_cache[layer_idx] = torch.cat([self.org_key_cache[layer_idx], key_states], dim=-2)
            self.org_value_cache[layer_idx] = torch.cat([self.org_value_cache[layer_idx], value_states], dim=-2)
            # duibi =  onebitgemv(self.key_cache_1bit[layer_idx],self.query_cache[layer_idx]*self.key_channel_abs[layer_idx])[...,:self.pruneCache_seen_tokens+1-self.neaset_len].to(query_states.device)
            # print(self.query_cache[layer_idx].shape)
            # score_now =  my_lutgemv(self.key_cache_1bit[layer_idx],self.query_cache[layer_idx]*self.the_key[layer_idx]).to(query_states.device)
            score_now = torch.matmul(self.compress_key[layer_idx],(self.key_channel_order[layer_idx]*next_layer_query_states).transpose(-1,-2)).squeeze(-1)
            # score_now = F.avg_pool1d(score_now, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1).unsqueeze(dim=-2)
            # print("111111")
            hlaf_len = min(self.prefetch_len,score_now.shape[-1])
            
            now_len = min(self.prefetch_len,score_now.shape[-1])
            
            _, now_indices = torch.topk(score_now,k=now_len,dim = -1)
            # print(now_indices.shape)
            now_indices = now_indices.squeeze(-2).unsqueeze(-1).expand(-1,-1,-1,128)
            out_k_pahse1 = torch.gather(self.compress_key[layer_idx],-2,now_indices)
            # print(out_k_pahse1.shape)
            out_v_pahse1 = torch.gather(self.compress_value[layer_idx],-2,now_indices)
            out_k = torch.cat((self.sink_key_cache[layer_idx],out_k_pahse1,self.org_key_cache[layer_idx]),dim=-2)
            out_v = torch.cat((self.sink_value_cache[layer_idx],out_v_pahse1,self.org_value_cache[layer_idx]),dim=-2)
            
            # self.KV_unit_cpu[layer_idx].evcit()
        # print(out_k.shape)
        return out_k.contiguous(), out_v.contiguous()
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
        return self.key_cache[0].shape[-2]

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