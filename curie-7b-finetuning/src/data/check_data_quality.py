from comet import download_model, load_from_checkpoint
import torch
import pandas as pd
import tqdm
import argparse
import logging

logging.getLogger("Python").setLevel(logging.WARNING)


torch.set_float32_matmul_precision("high")
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)


def rate_translation(input_file, output_file):
    df = pd.read_csv(input_file)
    ### get all of the columns that dodo not have string pl in them
    columns = [col for col in df.columns if "pl" not in col]
    ## remove category from columns
    try:
        columns.remove("category")
    except:
        pass

    trans_score_pl = []
    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        data = []
        for col in columns:
            data.append(
                {
                    "src": row[col],
                    "mt": row[f"{col}_pl"],
                }
            )
        prediction = model.predict(
            data,
            batch_size=8,
            gpus=1,
            num_workers=0,
            progress_bar=False,
        )
        score = round(prediction[1], 4)
        trans_score_pl.append(score)
    df["trans_score_pl"] = trans_score_pl
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rate translations in a CSV file.")
    parser.add_argument(
        "input_file", type=str, help="Input CSV file containing translations."
    )
    parser.add_argument(
        "output_file", type=str, help="Output file to save the results."
    )

    args = parser.parse_args()
    rate_translation(args.input_file, args.output_file)
