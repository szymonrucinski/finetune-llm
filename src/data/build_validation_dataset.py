from datasets import load_dataset
from tqdm import tqdm
from datasets import DatasetDict, Dataset
import pandas as pd
import time
import requests
import json


def make_call_to_deepl(text: str):
    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Authorization": "DeepL-Auth-Key ca39ba8f-fc60-1428-066c-7d98987271b5:fx",
        "Content-Type": "application/json",
    }

    data = {"text": [text], "source_lang": "en", "target_lang": "pl"}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        translated_data = response.json()
        return translated_data["translations"][0]["text"]
    else:
        print(f"Request failed with status code {response.status_code}")


if __name__ == "__main__":
    qa_generation_raw = load_dataset("truthful_qa", "generation")
    qa_mc_raw = load_dataset("truthful_qa", "generation")

    qa_generation_raw_df = qa_generation_raw["validation"].to_pandas()

    results = {
        "question": [],
        "best_answer": [],
        "correct_answers": [],
        "incorrect_answers": [],
    }
    for column in tqdm(
        ["question", "best_answer", "correct_answers", "incorrect_answers"]
    ):
        term_list = qa_generation_raw_df[column].to_list()
        if column in ["question", "best_answer"]:
            for term in tqdm(term_list):
                time.sleep(1)
                make_call_to_deepl(term)
                results[column].append(make_call_to_deepl(term))
        if column in ["correct_answers", "incorrect_answers"]:
            for term in tqdm(term_list):
                subterms = []
                for subterm in tqdm(term):
                    subterms.append(make_call_to_deepl(subterm))
                results[column].append(subterms)
        print(results)
    qa = pd.DataFrame.from_dict(results)
    qa.to_csv("../../data/pl_eval/thruthul_qa_pl.csv")
    qa["type"] = qa_generation_raw_df["type"]
    qa["category"] = qa_generation_raw_df["category"]
    qa["source"] = qa_generation_raw_df["source"]
    qa.reset_index(inplace=True)

    ds = DatasetDict()
    ds["validation"] = Dataset.from_pandas(qa)
    ds.push_to_hub("szymonrucinski/truthful_qa_pl")
