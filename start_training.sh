#!/bin/bash
# sft trainer
curl https://raw.githubusercontent.com/lvwerra/trl/main/examples/scripts/sft_trainer.py -o sft_trainer.py
#Younes's script
curl https://gist.githubusercontent.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da/raw/6f027020ed295447a2c476c610b99f2441b45e39/finetune_llama_v2.py -o finetune_llama_v2.py
python finetune_llama_v2.py --model_name meta-llama/Llama-2-7b-hf --dataset_name szymonindy/ociepa-raw-self-generated-instructions-pl --per_device_train_batch_size 8 --use_4bit --bf16 --bnb_4bit_compute_dtype bfloat16 --merge_and_push
