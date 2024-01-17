from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

from peft import prepare_model_for_kbit_training
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score
from datasets import Dataset as HF_Dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

from sklearn.model_selection import train_test_split
import wandb
from peft import LoraConfig, get_peft_model


challenge_name = "klej_cdsc-e"
base_file_path = f"./klej_data/{challenge_name}"
wandb.init(
    project="klej-benchmark-lodzianin",
    name=challenge_name,
)


model_id = "Azurro/APT3-1B-Base"
max_len = 1024

tokenizer = AutoTokenizer.from_pretrained(
    model_id, model_max_length=max_len, truncation=True
)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=3, problem_type="multi_class_classification"
)
model.config.pad_token_id = model.config.eos_token_id

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


config = LoraConfig(
    r=2,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    bias="none",
    lora_dropout=0.1,  # Conventional
    task_type="SEQ_CLS",
)

model = get_peft_model(model, config)
print(model.print_trainable_parameters())

# Read the TSV file
df_train = pd.read_csv(base_file_path + "/train.tsv", sep="\t")
df_test = pd.read_csv(base_file_path + "/test_features.tsv", sep="\t")
print(len(df_test))
df_val = pd.read_csv(base_file_path + "/dev.tsv", sep="\t")

# df_train, df_val = train_test_split(
#     df_train,
#     test_size=0.1,
#     random_state=42,
#     stratify=df_train["target"],
# )  # 20% for validation


labels = list(set(df_train["entailment_judgment"].to_list()))
ids = [i for i, e in enumerate(labels)]
id2label = dict(zip(ids, labels))
label2id = dict(zip(labels, ids))


# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, id2label, label2id, isTest=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
        self.isTest = isTest
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.isTest is False:
            text = (
                "[Zdanie A]: "
                + self.dataframe.iloc[idx]["sentence_A"]
                + "[Zdanie B]: "
                + self.dataframe.iloc[idx]["sentence_B"]
            )
            inputs = self.tokenizer(text, max_length=self.max_len, truncation=True)
            label = label2id[self.dataframe.iloc[idx]["entailment_judgment"]]
            return {"input_ids": inputs["input_ids"], "label": label}
        else:
            text = (
                "[Zdanie A]: "
                + self.dataframe.iloc[idx]["sentence_A"]
                + "[Zdanie B]: "
                + self.dataframe.iloc[idx]["sentence_B"]
            )
            inputs = self.tokenizer(text, max_length=self.max_len, truncation=True)
            return {"input_ids": inputs["input_ids"]}


train_dataset = TextDataset(df_train, tokenizer, id2label, label2id)
val_dataset = TextDataset(df_val, tokenizer, id2label, label2id)
test_dataset = TextDataset(df_test, tokenizer, id2label, label2id, isTest=True)


# Create datasets
def convert_to_hf(pytorch_dataset, isTest=False):
    if isTest is True:
        hf_data = {"input_ids": []}
        for item in pytorch_dataset:
            hf_data["input_ids"].append(item["input_ids"])
        return HF_Dataset.from_dict(hf_data)
    else:
        hf_data = {"input_ids": [], "label": []}
        for item in pytorch_dataset:
            hf_data["input_ids"].append(item["input_ids"])
            hf_data["label"].append(item["label"])
        return HF_Dataset.from_dict(hf_data)


train_dataset = convert_to_hf(train_dataset)
val_dataset = convert_to_hf(val_dataset)
test_dataset = convert_to_hf(test_dataset, isTest=True)

print(len(train_dataset), len(val_dataset), len(test_dataset))


def get_weights(train_dataset):
    # Convert to pandas DataFrame and count label occurrences
    label_counts = train_dataset.to_pandas()["label"].value_counts()
    label_count_dict = label_counts.to_dict()
    sorted_dict = {key: label_count_dict[key] for key in sorted(label_count_dict)}
    weights = [(len(train_dataset)) / (4 * i) for i in sorted_dict.values()]
    print("get weights", weights)
    return weights


class_weights = get_weights(train_dataset)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities
    probabilities = F.softmax(torch.from_numpy(logits), dim=1).numpy()

    # Predictions are the class with the highest probability
    predictions = np.argmax(logits, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")

    # Calculate ROC AUC for each class and average
    # Note: This requires one-hot encoding of the labels
    num_classes = probabilities.shape[1]
    one_hot_labels = np.eye(num_classes)[labels]

    # Calculate ROC AUC per class and average
    auroc = roc_auc_score(
        one_hot_labels, probabilities, multi_class="ovr", average="macro"
    )

    precision = precision_score(labels, predictions, average="macro")

    return {"accuracy": accuracy, "f1": f1, "auroc": auroc, "precision": precision}


class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=model.device, dtype=logits.dtype)
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="models/klej_polemo2.0-out",
    learning_rate=1e-4,
    lr_scheduler_type="constant",
    warmup_ratio=0.1,
    max_grad_norm=0.3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_auroc",
    remove_unused_columns=False,
    bf16=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)

trainer = WeightedCELossTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


trainer.train()


### TEST ###
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

result_df = df_test.copy()
result_df["entailment_judgment"] = predicted_labels
# Convert the predicted labels from ids to actual labels
result_df["entailment_judgment"] = result_df["entailment_judgment"].map(id2label)
print(result_df)
# Save the results to a CSV file
result_file_path = base_file_path + "/test_pred_polemo2.0-out.tsv"
print(result_file_path)
result_df["entailment_judgment"].to_csv(result_file_path, index=False, sep="\t")

artifact = wandb.Artifact("test_pred_polemo2.0-out", type="result")
artifact.add_file(result_file_path)
wandb.log_artifact(artifact)
