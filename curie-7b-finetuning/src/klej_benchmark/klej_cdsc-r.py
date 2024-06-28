from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
import bitsandbytes as bnb

from torch.utils.data import DataLoader
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset

import wandb
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)
challenge_name = "klej_cdsc-r"
base_file_path = f"./klej_data/{challenge_name}"
wandb.init(
    project="klej-benchmark-lodzianin",
    name=challenge_name,
)

tsv_train_file_path = base_file_path + "/train.tsv"
train_file_path = base_file_path + "/train.csv"
tsv_test_file_path = base_file_path + "/test_features.tsv"
test_file_path = base_file_path + "/test_features.csv"
tsv_val_file_path = base_file_path + "/dev.tsv"
val_file_path = base_file_path + "/dev.csv"


# Read the TSV file
df_train = pd.read_csv(tsv_train_file_path, sep="\t")
df_train.to_csv(train_file_path, index=False)
df_test = pd.read_csv(tsv_test_file_path, sep="\t")
df_test.to_csv(test_file_path, index=False)
df_val = pd.read_csv(tsv_val_file_path, sep="\t")
df_val.to_csv(val_file_path, index=False)

model_name = "Azurro/APT-1B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
Lodzianin_Model = AutoModelForCausalLM.from_pretrained(model_name)


class RelatednessDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=2):
        self.df = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence_a = self.df.iloc[idx]["sentence_A"]
        sentence_b = self.df.iloc[idx]["sentence_B"]
        score = float(self.df.iloc[idx]["relatedness_score"]) / 5

        # Encode sentences
        encoded_pair = self.tokenizer.encode_plus(
            sentence_a,
            sentence_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded_pair["input_ids"].flatten(),
            "attention_mask": encoded_pair["attention_mask"].flatten(),
            "labels": torch.tensor(score, dtype=torch.float),
        }


class TestRelatednessDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=32):
        self.df = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence_a = self.df.iloc[idx]["sentence_A"]
        sentence_b = self.df.iloc[idx]["sentence_B"]

        # Encode sentences
        encoded_pair = self.tokenizer.encode_plus(
            sentence_a,
            sentence_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded_pair["input_ids"].flatten(),
            "attention_mask": encoded_pair["attention_mask"].flatten(),
        }


# Prepare the dataset and dataloader
train = RelatednessDataset(train_file_path, tokenizer)
train_dataloader = DataLoader(train, batch_size=16, shuffle=True)
test = TestRelatednessDataset(test_file_path, tokenizer)
test_dataloader = DataLoader(test, batch_size=16, shuffle=True)
val = RelatednessDataset(val_file_path, tokenizer)
val_dataloader = DataLoader(val, batch_size=16, shuffle=True)

model = AutoModelForCausalLM.from_pretrained(model_name)
model = prepare_model_for_kbit_training(model)
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
    # task_type="SEQ_CLS",
)
model = get_peft_model(model, config)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
model = accelerator.prepare_model(model)
model = model.to(device)


class LodzianinForSentenceRelatedness(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lodzianin = model
        self.regressor = nn.Linear(self.lodzianin.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.lodzianin(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )  # Request hidden states

        last_hidden_state = outputs.hidden_states[-1]
        cls_representation = last_hidden_state[:, 0]
        score = self.regressor(cls_representation)
        score = self.sigmoid(score)  # Normalize score to be between 0 and 1
        return score.squeeze(-1)


model = LodzianinForSentenceRelatedness(model)
model = model.to(device)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=2e-5)


NUM_EPOCHS = 1
# Training loop
model.train()
for epoch in tqdm(range(NUM_EPOCHS)):
    train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.MSELoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Train_Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    actuals = []
    predictions = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask).squeeze()
            # Calculate loss
            loss = nn.MSELoss()(outputs, labels)
            val_loss += loss.item()

            # Move outputs and labels to CPU for metric calculation
            actuals.extend(labels.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f"Validation Loss: {val_loss / len(val_dataloader)}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    wandb.log(
        {
            "epoch": epoch,
            "loss": avg_train_loss,
            "eval_loss": val_loss,
            "eval_r2": r2,
            "eval_mse": mse,
        }
    )


##### TEST #####
predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask).squeeze()
        predictions.extend(outputs.cpu().numpy())
        predictions = [x * 5 for x in predictions]


results = pd.DataFrame.from_dict({"relatedness_score": predictions})
print(results)
# Save the results to a CSV file
result_file_path = base_file_path + f"pred_{challenge_name}.tsv"
results.to_csv(result_file_path, index=False, sep="\t")
artifact = wandb.Artifact("test_results", type="result")
artifact.add_file(result_file_path)

# Log the artifact to wandb
wandb.log_artifact(artifact)
