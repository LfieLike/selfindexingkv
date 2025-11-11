
#meta-llama/Llama-2-7b-hf
#huggyllama/llama-7b
gpuid=$1
model=$2
e=0
name=$3
CUDA_VISIBLE_DEVICES=$gpuid python pred_long_bench_llama3.py --model_name_or_path $model \
    --k_bits 16 \
    --v_bits 16 \
    --e ${e} \
    --name $name
    # bash scripts/long_test.sh 1 16 16 32 32  /root/model/Llama-2-7b-hf