#!/usr/bin/env python
# coding: utf-8

# #### Before we begin: A note on OOM errors
# 
# If you get an error like this: `OutOfMemoryError: CUDA out of memory`, tweak your parameters to make the model less computationally intensive. I will help guide you through that in this guide, and if you have any additional questions you can reach out on the [Discord channel](https://discord.gg/vREMUGyP) or on [X](https://x.com/harperscarroll).
# 
# To re-try after you tweak your parameters, open a Terminal ('Launcher' or '+' in the nav bar above -> Other -> Terminal) and run the command `nvidia-smi`. Then find the process ID `PID` under `Processes` and run the command `kill [PID]`. You will need to re-start your notebook from the beginning. (There may be a better way to do this... if so please do let me know!)

# ## Let's begin!
# I used a GPU and dev environment from [brev.dev](https://brev.dev). Provision a pre-configured GPU in one click [here](https://console.brev.dev/environment/new?instance=A10G:g5.xlarge&name=llama2-7b-finetune) (a single A10G or L4 should be enough for this dataset; anything with >= 24GB GPU Memory. You may need more GPUs and/or Memory if your sequence max_length is larger than 512). Once you've checked out your machine and landed in your instance page, select the specs you'd like (I used **Python 3.10** and CUDA 11.7) and click the "Build" button to build your autogpu container. Give this a few minutes.
# 
# A few minutes after your model has started Running, click the 'Notebook' button on the top right of your screen once it illuminates (you may need to refresh the screen). You will be taken to a Jupyter Lab environment, where you can upload this Notebook.
# 
# 
# Note: You can connect your cloud credits (AWS or GCP) by clicking "Org: " on the top right, and in the panel that slides over, click "Connect AWS" or "Connect GCP" under "Connect your cloud" and follow the instructions linked to attach your credentials.
# 
# 

# In[1]:


# import locale
# locale.getpreferredencoding = lambda: "UTF-8"


# In[2]:
# In[3]:


# ! export CUDA_VISIBLE_DEVICES=0,1


# ### 0. Accelerator
# 
# Set up the Accelerator. I'm not sure if we really need this for a QLoRA given its [description](https://huggingface.co/docs/accelerate/v0.19.0/en/usage_guides/fsdp) (I have to read more about it) but it seems it can't hurt, and it's helpful to have the code for future reference. You can always comment out the accelerator if you want to try without.

# In[4]:


from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


# ### 1. Load Dataset

# Let's load a meaning representation dataset, and fine-tune Mistral on that. This is a great fine-tuning dataset as it teaches the model a unique form of desired output on which the base model performs poorly out-of-the box, so it's helpful to easily and inexpensively gauge whether the fine-tuned model has learned well. (Sources: [here](https://ragntune.com/blog/gpt3.5-vs-llama2-finetuning) and [here](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts)) (In contrast, if you fine-tune on a fact-based dataset, the model may already do quite well on that, and gauging learning is less obvious / may be more computationally expensive.)

# In[5]:


from datasets import load_dataset

train_dataset = load_dataset('szymonrucinski/krakowiak-pl', split='train')
eval_dataset = load_dataset('szymonrucinski/krakowiak-pl', split='validation')


# In[6]:


print(train_dataset)
print(eval_dataset)


# ### 2. Load Base Model

# Let's now load Mistral - `mistralai/Mistral-7B-v0.1` - using 4-bit quantization!

# In[7]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)


# ### 3. Tokenization
# 
# Set up the tokenizer. Add padding on the left as it [makes training use less memory](https://ai.stackexchange.com/questions/41485/while-fine-tuning-a-decoder-only-llm-like-llama-on-chat-dataset-what-kind-of-pa).
# 

# In[8]:


tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=2048,
    padding_side="left",
    add_eos_token=True)

tokenizer.pad_token = tokenizer.eos_token


# Setup the tokenize function to make labels and input_ids the same. This is basically what [self-supervised fine-tuning is](https://neptune.ai/blog/self-supervised-learning):

# In[9]:


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


# Reformat the prompt and tokenize each sample:

# In[10]:


tokenized_train_dataset = tokenize(train_dataset["text"])
tokenized_val_dataset = tokenize(eval_dataset["text"])


# Check that `input_ids` is padded on the left with the `eos_token` (2) and there is an `eos_token` 2 added to the end, and the prompt starts with a `bos_token` (1).
# 

# Check that a sample has the max length, i.e. 512.

# #### How does the base model do?

# Let's grab a test input (`meaning_representation`) and desired output (`target`) pair to see how the base model does on it.

# In[11]:


eval_prompt = """### Użytkownik: Który amerykański polityk jest synem kubańskiego plantatora ananasów?### Asystent:
"""


# In[12]:


model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=64, pad_token_id=2)[0], skip_special_tokens=True))


# We can see it doesn't do very well out of the box.

# ### 4. Set Up LoRA

# Now, to start our fine-tuning, we have to apply some preprocessing to the model to prepare it for training. For that use the `prepare_model_for_kbit_training` method from PEFT.

# In[13]:


from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# In[14]:


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# Let's print the model to examine its layers, as we will apply QLoRA to all the linear layers of the model. Those layers are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and `lm_head`.

# In[15]:


print(model)


# Here we define the LoRA config.
# 
# `r` is the rank of the low-rank matrix used in the adapters, which thus controls the number of parameters trained. A higher rank will allow for more expressivity, but there is a compute tradeoff.
# 
# `alpha` is the scaling factor for the learned weights. The weight matrix is scaled by `alpha/r`, and thus a higher value for `alpha` assigns more weight to the LoRA activations.
# 
# The values used in the QLoRA paper were `r=64` and `lora_alpha=16`, and these are said to generalize well, but we will use `r=8` and `lora_alpha=16` so that we have more emphasis on the new fine-tuned data while also reducing computational complexity.

# In[16]:


from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)


# See how the model looks different now, with the LoRA adapters added:

# In[17]:


print(model)


# 
# Let's use Weights & Biases to track our training metrics. You'll need to apply an API key when prompted. Feel free to skip this if you'd like, and just comment out the `wandb` parameters in the `Trainer` definition below.

# In[18]:

import wandb, os
wandb.login()

wandb_project = "finetune-polish-llms"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project


# ### 5. Run Training!

# I used 500 steps, but I found the model should have trained for longer as it had not converged by then, so I upped the steps to 1000 below.
# 
# A note on training. You can set the `max_steps` to be high initially, and examine at what step your model's performance starts to degrade. There is where you'll find a sweet spot for how many steps to perform. For example, say you start with 1000 steps, and find that at around 500 steps the model starts overfitting - the validation loss goes up (bad) while the training loss goes down significantly, meaning the model is learning the training set really well, but is unable to generalize to new datapoints. Therefore, 500 steps would be your sweet spot, so you would use the `checkpoint-500` model repo in your output dir (`mistral-finetune-viggo`) as your final model in step 6 below.
# 
# You can interrupt the process via Kernel -> Interrupt Kernel in the top nav bar once you realize you didn't need to train anymore.

# In[19]:


if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True


# In[20]:


from trl import SFTTrainer


# In[21]:


train_dataset = load_dataset("szymonrucinski/krakowiak-pl", split="train")
val_dataset = load_dataset("szymonrucinski/krakowiak-pl", split="validation")


# In[ ]:


import transformers
from datetime import datetime

project = "finetune-polish-llms"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=2048,
    padding_side="left",
    add_eos_token=True)

tokenizer.pad_token = tokenizer.eos_token

tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    neftune_noise_alpha=2,
    max_seq_length=2048,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=10000,
        learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
        warmup_steps=100,
        logging_steps=100,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=250,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=200,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


# ### 6. Drum Roll... Try the Trained Model!
# 
# By default, the PEFT library will only save the QLoRA adapters, so we need to first load the base Mistral model from the Huggingface Hub:
# 

# In[ ]:


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# Now load the QLoRA adapter from the appropriate checkpoint directory, i.e. the best performing model checkpoint:

# In[ ]:


from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "mistral-viggo-finetune/checkpoint-1000")


# and run your inference!

# Let's try the same `eval_prompt` and thus `model_input` as above, and see if the new finetuned model performs better.

# In[ ]:


ft_model.eval()
with torch.no_grad():
    print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, pad_token_id=2)[0], skip_special_tokens=True))


# ### Sweet... it worked! The fine-tuned model now understands the meaning representation!
# 
# I hope you enjoyed this tutorial on fine-tuning Mistral. If you have any questions, feel free to reach out to me on [X](https://x.com/harperscarroll) or on the [Discord channel](https://discord.gg/vREMUGyP).
# 
# 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙 🤙
