#!/bin/bash
#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME=/groups/chiyuan/.cache/huggingface
export HF_TOKEN=YOUR_HF_TOKEN
source /home/chiyuan/.bashrc
conda activate /home/chiyuan/llama3/llama3_env
ln -sf /usr/lib/libstdc++.so.6 /home/chiyuan/llama3/llama3_env/lib/libstdc++.so.6

#gpus=2;
#distributed="-m torch.distributed.launch --nproc_per_node ${gpus}";
#python3 $distributed train.py
#accelerate launch --multi_gpu --num_processes 2 train_1.py
#torchrun --nproc_per_node 2 ./nlp_example.py --mixed_precision bf16
#accelerate launch --config_file ./acc_config.yaml --mixed_precision bf16 ./nlp_example.py
accelerate launch --config_file ./acc_config.yaml ./train_1.py
