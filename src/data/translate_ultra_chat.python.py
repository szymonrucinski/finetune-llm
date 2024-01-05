import requests
import tqdm
from datasets import load_dataset
import time


url = "https://translate.apostroph.ch/api/unstable/v3/translate"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://translate.apostroph.ch",
    "Referer": "https://translate.apostroph.ch/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "TE": "trailers",
}


def translate_text(text: str, source_language: str = "en", target_language: str = "pl"):
    data = {
        "segments": [{"text": text}],
        "source_language": source_language,
        "target_language": target_language,
        "politeness": "informal",
        "highlight_terms": False,
    }
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response = response.json()["segments"][0]["text"]
        return response
    except:
        try:
            time.sleep(30)
            response = requests.post(url, json=data, headers=headers, timeout=10)
            print(response.json())
            response = response.json()["segments"][0]["text"]
        except:
            response = "PROBLEM Z TŁUMACZENIEM"
            print(response)

        return response


def check_2000_char_limit(conversations, max_length=2850):
    cur_chunk_len = 0
    chunks = []
    cur_left_index = 0

    for i, conversation in enumerate(conversations):
        message_length = len(conversation["content"])
        if cur_chunk_len + message_length > max_length and cur_chunk_len > 0:
            # Finish the current chunk before this message, as it would exceed the limit
            chunks.append((cur_left_index, i - 1))
            cur_left_index = i
            cur_chunk_len = (
                message_length  # Start the next chunk with the current message
            )
        else:
            # Add the current message's length to the current chunk
            cur_chunk_len += message_length

    # After processing all messages, add the remaining messages as the last chunk
    if cur_chunk_len > 0:
        chunks.append((cur_left_index, i))

    return chunks


def parse_chunks(chunk, conversation, max_chunk_length=2850):
    text_chunks = []
    for idx in chunk:
        concat_conversation = ""
        for j in range(idx[0], idx[1] + 1):
            content = conversation[j]["content"]
            if len(concat_conversation) + len(content) <= max_chunk_length:
                concat_conversation += f"{content}<META_TAG>"
            else:
                remaining_chars = max_chunk_length - len(concat_conversation)
                concat_conversation += f"{content[:remaining_chars]}<META_TAG>"
                break

        text_chunks.append(concat_conversation)
    return text_chunks


ultra_zephyr = load_dataset("HuggingFaceH4/ultrachat_200k")

from datasets import load_dataset

zephyr_df = ultra_zephyr["train_sft"].to_pandas()
zephyr_df["messages_pl"] = zephyr_df["messages"]
ALREAD_TRANSLATED = 0

# zephyr_df = zephyr_df[ALREAD_TRANSLATED:]
zephyr_df = zephyr_df
zephyr_df.reset_index(inplace=True, drop=True)

for zephyr_df_idx, row in tqdm.tqdm(zephyr_df.iterrows(), total=len(zephyr_df)):
    msg = row["messages"]
    chunk = check_2000_char_limit(msg)
    # print(chunk)
    concated_chunks = parse_chunks(chunk, msg)
    # print(concated_chunks)
    # print(len(concated_chunks[0]), len(concated_chunks[1]))
    translations = [translate_text(j) for j in concated_chunks]
    concatenated_string = " ".join(translations)
    translated_string = concatenated_string.split("<META_TAG>")
    if translated_string[-1] == "":
        translated_string.pop(-1)
    # print(len(msg), len(translated_string))
    if len(translated_string) != len(msg):
        print("ERROR")
        zephyr_df.at[zephyr_df_idx, "messages_pl"] = []
        continue

    for i, message in enumerate(msg):
        msg[i]["content"] = translated_string[i]

    zephyr_df.at[zephyr_df_idx, "messages_pl"] = msg
    if zephyr_df_idx % 35001 == 0:
        zephyr_df.to_csv(
            f"../../data/pl/zephyr_pl_backup_{int(zephyr_df_idx)+ALREAD_TRANSLATED}.csv"
        )
        break
# zephyr_df.to_csv("../../data/pl/zephyr_pl.csv")
