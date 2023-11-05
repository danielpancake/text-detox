import os
import pandas as pd
import torch
import warnings

warnings.filterwarnings("ignore")

from detoxGPT2 import detoxGPT2
from tqdm import tqdm

FILEPATH = os.path.dirname(__file__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # Load test dataset
    df = pd.read_csv(
        os.path.join(FILEPATH, "../../data/interim/test.tsv"),
        sep="\t",
        header=None,
        names=["tox", "detox"],
    )

    # Load model
    detoxGPT = detoxGPT2()
    df_detoxGPT_scores = pd.DataFrame(columns=["wo", "cs", "bleu", "detox_score"])

    # Inference on test dataset
    for i in tqdm(range(len(df))):
        ref = df.iloc[i]["tox"]
        _, scores = detoxGPT.get_best_suggestion(ref, device=DEVICE)
        df_detoxGPT_scores.loc[i] = scores

    df_detoxGPT_scores.to_csv(
        os.path.join(FILEPATH, "../../data/interim/detoxGPT_scores.csv"),
        index=False,
    )
