#!/bin/bash
#Younes's script
adaptername="szymonrucinski/krakowiak-7b"
model_name="meta-llama/Llama-2-7b-hf"
dataset_name="szymonindy/ociepa-raw-self-generated-instructions-pl"
quant_type = "q4_k_m"

curl https://gist.githubusercontent.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da/raw/6f027020ed295447a2c476c610b99f2441b45e39/finetune_llama_v2.py -o finetune_llama_v2.py
python finetune_llama_v2.py --model_name $model_name --dataset_name $dataset_name --per_device_train_batch_size 8 --use_4bit --bf16 --bnb_4bit_compute_dtype bfloat16 --merge_and_push
python merge_peft_adapters.py --base_model_name_or_path meta-llama/Llama-2-7b-hf --peft_model_path szymonrucinski/krakowiak-7b --output_dir ./models/merged/ --device auto
# python merge_peft_adapters.py --base_model_name_or_path meta-llama/Llama-2-7b-hf --peft_model_path szymonrucinski/krakowiak-7b --output_dir szymonrucinski/krakowiak-7b-merged --push_to_hub 

# Install and compile llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && git pull && make clean && LLAMA_CUBLAS=1 make
pip install -r llama.cpp/requirements.txt
#DOWNLOAD MODEL and install lfs
## Convert model to gguf 16bit
python llama.cpp/convert.py ./models/merged --outtype f16 --outfile  ./models/gguf/merged-model.gguf.fp16.bin
## quantize model
./build/Release/quantize.exe ./merged-model/ q4_k_m  ./gguf/merged-model.q4_k_m.gguf.bin
python .\llama.cpp\convert.py ./merged-model/ --outtype f16 --outfile  ./gguf/merged-model.fp16.bin

C:\Users\SRU\Desktop\finetune-llm\llama.cpp\build\bin\Release\main.exe .\gguf\merged-model.q4_k_m.fp16.bin q4_k_m -n 128
.\llama.cpp\build\bin\Release\quantize.exe .\models\gguf\krakowiak-7b.gguf.fp16.bin .\models\gguf\krakowiak-7b.gguf.q4_k_m.bin $quant_type