from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import prepare_model_for_kbit_training
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
import pandas as pd
import numpy as np

import wandb
from peft import LoraConfig, get_peft_model


fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)
challenge_name = "klej_nkjp-ner"
base_file_path = f"./klej_data/{challenge_name}"
wandb.init(
    project="klej-benchmark-lodzianin",
    name=challenge_name,
)
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"


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

# Load the datasets
klej_psc_train = load_dataset("csv", data_files=train_file_path, split="train")
klej_psc_val = load_dataset("csv", data_files=val_file_path, split="train")
klej_psc_test = load_dataset("csv", data_files=test_file_path, split="train")


# Combine them into a single DatasetDict
klej_psc_dataset = DatasetDict(
    {"train": klej_psc_train, "test": klej_psc_test, "validation": klej_psc_val}
)
print(klej_psc_dataset["train"])
print(klej_psc_dataset["validation"])
print(klej_psc_dataset["test"])

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


def preprocess_dataset(dataset, col_to_delete, col_to_rename, new_col_name):
    dataset = dataset.map(lambda x: {"text": x["sentence"]})

    def mistral_preprocessing_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=32,
        )

    # Apply preprocessing function and remove specified columns
    dataset = dataset.map(mistral_preprocessing_function, remove_columns=col_to_delete)
    try:
        dataset = dataset.rename_column(col_to_rename, new_col_name)
    except:
        print("no_col")
        return dataset
    return dataset


# Usage of the function
klej_psc_dataset["train"] = preprocess_dataset(
    klej_psc_dataset["train"],
    [
        "sentence",
    ],
    "target",
    "labels",
)

klej_psc_dataset["validation"] = preprocess_dataset(
    klej_psc_dataset["validation"],
    ["sentence"],
    "target",
    "labels",
)

klej_psc_dataset["test"] = preprocess_dataset(
    klej_psc_dataset["test"],
    ["sentence"],
    "target",
    "labels",
)

labels = list(set(df_train["target"].to_list()))
ids = [i for i, e in enumerate(labels)]
id2label = dict(zip(ids, labels))
label2id = dict(zip(labels, ids))


def map_labels_to_ids(example):
    example["labels"] = label2id[example["labels"]]
    return example


klej_psc_dataset["train"] = klej_psc_dataset["train"].map(map_labels_to_ids)
klej_psc_dataset["validation"] = klej_psc_dataset["validation"].map(map_labels_to_ids)


model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=len(labels), id2label=id2label, label2id=label2id
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable()
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
    task_type="SEQ_CLS",
)

model = get_peft_model(model, config)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
model = accelerator.prepare_model(model)

### metrics

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    print(labels, predictions)
    return {
        "accuracy": accuracy_score(predictions, labels),
        "f1": f1_score(predictions, labels, average="macro"),
    }


print(klej_psc_dataset["train"][1])
print(klej_psc_dataset["validation"][1])


training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2.5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_f1",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=klej_psc_dataset["train"],
    eval_dataset=klej_psc_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

print(klej_psc_dataset["test"])
predictions = trainer.predict(klej_psc_dataset["test"])
predicted_labels = np.argmax(predictions.predictions, axis=1)
# Create a DataFrame with the original test data and the predicted labels
result_df = df_test.copy()
result_df["predicted_label"] = predicted_labels
# Convert the predicted labels from ids to actual labels
result_df["predicted_label"] = result_df["predicted_label"].map(id2label)

# Save the results to a CSV file
result_file_path = base_file_path + f"test_pred_{challenge_name}.tsv"
result_df.to_csv(result_file_path, index=False, sep="\t")


artifact = wandb.Artifact("test_results", type="result")
artifact.add_file(result_file_path)

# Log the artifact to wandb
wandb.log_artifact(artifact)
