from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
import transformers
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
import numpy as np

from transformers import EvalPrediction

# def compute_metrics(eval_pred: EvalPrediction):
#     logits, labels = eval_pred
#     logits = torch.from_numpy(logits)  # Convert logits to tensor
#     labels = torch.from_numpy(labels)  # Convert labels to tensor

#     labels = labels.reshape(-1)
#     logits = logits.reshape(-1, logits.shape[-1])

#     # Exclude ignored index (if any) from calculations
#     active_labels = labels != -100
#     active_logits = logits[active_labels, :]
#     active_labels = labels[active_labels]

#     # Calculate Cross Entropy Loss
#     loss_fct = torch.nn.CrossEntropyLoss()
#     loss = loss_fct(active_logits, active_labels)

#     # Calculate Perplexity
#     perplexity = torch.exp(loss).item()
#     return {"perplexity": perplexity}

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=2048,
    add_eos_token=True)

tokenizer.pad_token = tokenizer.eos_token

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
    r=8,
    lora_alpha=16,
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

print(model)
# Let's use Weights & Biases to track our training metrics. You'll need to apply an API key when prompted. Feel free to skip this if you'd like, and just comment out the `wandb` parameters in the `Trainer` definition below.

# In[18]:

import wandb, os
wandb.login()

wandb_project = "finetune-polish-llms"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project


if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

from trl import SFTTrainer
from  datasets import Dataset
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
train_dataset = load_dataset('szymonrucinski/krakowiak-instructions', split='train')
val_dataset = load_dataset('szymonrucinski/krakowiak-instructions', split='validation')
val_dataset = load_dataset('szymonrucinski/krakowiak-instructions', split='validation')
val_dataset=val_dataset[:50]
val_dataset = Dataset.from_dict(val_dataset)

train_dataset = train_dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["prompt"], tokenize=False, add_generation_prompt=False)})
val_dataset = val_dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["prompt"], tokenize=False, add_generation_prompt=False)})
print(val_dataset['formatted_chat'][0])


project = "finetune-polish-llms"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=2048,
    # truncation=True,
    # padding=True,
    add_eos_token=True)

tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="formatted_chat",
    neftune_noise_alpha=2,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=100000,
        learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
        warmup_steps=200,
        logging_steps=100,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="epoch",       # Save the model checkpoint every logging step
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=100,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # compute_metrics=compute_metrics
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
