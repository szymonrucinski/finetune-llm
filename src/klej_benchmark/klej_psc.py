from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import prepare_model_for_kbit_training
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import torch

from sklearn.model_selection import train_test_split
import wandb
from peft import LoraConfig, get_peft_model


challenge_name = "klej_psc"
base_file_path = f"./klej_data/{challenge_name}"
wandb.init(
    project="klej-benchmark-lodzianin",
    name=challenge_name,
)


model_id = "Azurro/APT3-1B-Base"

id2label = {0: "0", 1: "1"}
label2id = {"0": 0, "1": 1}

# tokenizer = AutoTokenizer.from_pretrained(model_id, padding="max_length")
tokenizer = AutoTokenizer.from_pretrained(
    model_id, model_max_length=1024, truncation=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=2, id2label=id2label, label2id=label2id
)
model.config.pad_token_id = model.config.eos_token_id

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


config = LoraConfig(
    r=2,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        # "k_proj",
        "v_proj",
        # "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj",
        # "lm_head",
    ],
    bias="none",
    lora_dropout=0.1,  # Conventional
    task_type="SEQ_CLS",
)

model = get_peft_model(model, config)
print(model.print_trainable_parameters())

tsv_train_file_path = base_file_path + "/train.tsv"
train_file_path = base_file_path + "/train.csv"
tsv_test_file_path = base_file_path + "/test_features.tsv"
test_file_path = base_file_path + "/test_features.csv"

# Read the TSV file
df_train = pd.read_csv(tsv_train_file_path, sep="\t")
df_train.to_csv(train_file_path, index=False)
df_test = pd.read_csv(tsv_test_file_path, sep="\t")
df_test.to_csv(test_file_path, index=False)

train_df, val_df = train_test_split(df_train, test_size=0.2)  # 20% for validation
my_train_file_path = base_file_path + "/my_train.csv"
my_val_file_path = base_file_path + "/my_val.csv"
train_df.to_csv(my_train_file_path, index=False)
val_df.to_csv(my_val_file_path, index=False)


# Load the datasets
klej_psc_train = load_dataset("csv", data_files=my_train_file_path, split="train")
klej_psc_val = load_dataset("csv", data_files=my_val_file_path, split="train")
klej_psc_test = load_dataset("csv", data_files=test_file_path, split="train")


# Combine them into a single DatasetDict
klej_psc_dataset = DatasetDict(
    {"train": klej_psc_train, "test": klej_psc_test, "validation": klej_psc_val}
)
print(klej_psc_dataset["train"])
print(klej_psc_dataset["validation"])
print(klej_psc_dataset["test"])

# Load Mistral 7B Tokenizer

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


def preprocess_dataset(dataset, col_to_delete, col_to_rename, new_col_name):
    dataset = dataset.map(
        lambda x: {
            "text": "[TEXT]" + x["extract_text"] + "[SUMMARY]" + x["summary_text"]
        }
    )

    def mistral_preprocessing_function(examples):
        return tokenizer(examples["text"], max_length=1024)

    # Apply preprocessing function and remove specified columns
    dataset = dataset.map(
        mistral_preprocessing_function, batched=False, remove_columns=col_to_delete
    )

    return dataset


# Usage of the function
klej_psc_dataset["train"] = preprocess_dataset(
    klej_psc_dataset["train"], ["extract_text", "summary_text"], "extract_text", "text"
)

klej_psc_dataset["validation"] = preprocess_dataset(
    klej_psc_dataset["validation"],
    ["extract_text", "summary_text"],
    "extract_text",
    "text",
)

klej_psc_dataset["test"] = preprocess_dataset(
    klej_psc_dataset["test"],
    ["extract_text", "summary_text"],
    "extract_text",
    "text",
)

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    probabilities = F.softmax(torch.from_numpy(logits), dim=1).numpy()[
        :, 1
    ]  # Probability of the positive class
    print(probabilities)
    print(labels, predictions)
    return {
        "accuracy": accuracy_score(predictions, labels),
        "f1": f1_score(predictions, labels, average="macro"),
        "auroc": roc_auc_score(labels, probabilities),
    }


# )
pos_weights = len(klej_psc_dataset["train"].to_pandas()) / (
    2 * klej_psc_dataset["train"].to_pandas().label.value_counts()[1]
)
neg_weights = len(klej_psc_dataset["train"].to_pandas()) / (
    2 * klej_psc_dataset["train"].to_pandas().label.value_counts()[0]
)


class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(
                [neg_weights, pos_weights], device=model.device, dtype=logits.dtype
            )
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="models/klej_psc",
    learning_rate=1e-4,
    lr_scheduler_type="constant",
    warmup_ratio=0.1,
    max_grad_norm=0.3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.001,
    eval_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_auroc",
    greater_is_better=True,
    # report_to="wandb",
    # optim="paged_adamw_8bit",
    bf16=True,
    # gradient_checkpointing=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)

trainer = WeightedCELossTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=klej_psc_dataset["train"],
    eval_dataset=klej_psc_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


trainer.train()


### TEST ###
predictions = trainer.predict(klej_psc_dataset["test"])
predicted_labels = np.argmax(predictions.predictions, axis=1)

result_df = df_test.copy()
result_df["label"] = predicted_labels
# Convert the predicted labels from ids to actual labels
result_df["label"] = result_df["label"].map(id2label)
print(result_df)
# Save the results to a CSV file
result_file_path = base_file_path + "/test_pred_psc.tsv"
print(result_file_path)
result_df["label"].to_csv(result_file_path, index=False, sep="\t")

artifact = wandb.Artifact("test_pred_psc", type="result")
artifact.add_file(result_file_path)
wandb.log_artifact(artifact)
