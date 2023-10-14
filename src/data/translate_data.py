import requests
import pandas as pd
import tqdm
import argparse


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


def translate_text(
    file_path: str,
    output_path: str,
    source_language: str,
    target_language: str,
    columns_to_translate: list,
):
    ## create dictonary of lists for columns
    df = pd.read_csv(file_path)

    translation_dict = {}
    for column_to_translate in columns_to_translate:
        translation_dict[f"{column_to_translate}_{target_language}"] = []

    for index, row in tqdm.tqdm(df.iterrows()):
        for column_to_translate in columns_to_translate:
            data = {
                "segments": [{"text": row[column_to_translate]}],
                "source_language": source_language,
                "target_language": target_language,
                "politeness": "informal",
                "highlight_terms": False,
            }
            response = requests.post(url, json=data, headers=headers)
            translation_dict[f"{column_to_translate}_{target_language}"].append(
                response.json()["segments"][0]["text"]
            )
            print(response.json()["segments"][0]["text"])

    translation_df = pd.DataFrame.from_dict(translation_dict)
    translation_df.to_csv("translation_" + output_path, index=False)
    df = pd.concat([df, translation_df], axis=1)
    df.to_csv(output_path, index=False)


def main():
    """Translate text in a file."""
    parser = argparse.ArgumentParser(description="Translate text in a file.")
    parser.add_argument("file_path", help="Path to the input file.")
    parser.add_argument("output_path", help="Path to the output file.")
    parser.add_argument(
        "--source-language",
        required=True,
        help="Source language code (e.g., 'en' for English).",
    )
    parser.add_argument(
        "--target-language",
        required=True,
        help="Target language code (e.g., 'fr' for French).",
    )
    parser.add_argument(
        "--columns-to-translate",
        nargs="+",
        help="Columns to translate (e.g., '1 2 3').",
    )

    args = parser.parse_args()
    translate_text(
        args.file_path,
        args.output_path,
        args.source_language,
        args.target_language,
        args.columns_to_translate,
    )


if __name__ == "__main__":
    main()
