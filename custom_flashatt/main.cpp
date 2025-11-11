#include <torch/extension.h>
std::tuple<torch::Tensor, torch::Tensor> flash_attention_decode(torch::Tensor query, torch::Tensor keys, torch::Tensor values);
std::tuple<torch::Tensor, torch::Tensor> quant_flash_attention_decode(torch::Tensor query,    torch::Tensor key_value_quant,torch::Tensor key_1bit_quant,torch::Tensor quant_param);
std::tuple<torch::Tensor, torch::Tensor> select_quant_flash_attention_decode(torch::Tensor query,    torch::Tensor key_value_quant,torch::Tensor key_1bit_quant,torch::Tensor quant_param,torch::Tensor topk_index);
std::tuple<torch::Tensor, torch::Tensor> mix_select_quant_flash_attention_decode(torch::Tensor query,    torch::Tensor full_key,torch::Tensor full_value,     torch::Tensor key_value_quant,torch::Tensor key_1bit_quant,torch::Tensor quant_param,torch::Tensor topk_index);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("flash_attention_decode", torch::wrap_pybind_function(flash_attention_decode), "flash_attention_decode");
m.def("quant_flash_attention_decode", torch::wrap_pybind_function(quant_flash_attention_decode), "quant_flash_attention_decode");
m.def("select_quant_flash_attention_decode", torch::wrap_pybind_function(select_quant_flash_attention_decode), "select_quant_flash_attention_decode");
m.def("mix_select_quant_flash_attention_decode", torch::wrap_pybind_function(mix_select_quant_flash_attention_decode), "mix_select_quant_flash_attention_decode");

}