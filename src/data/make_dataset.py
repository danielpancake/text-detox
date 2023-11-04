"""
The purpose of this script is to automate the same process that the notebook `02-data-preps.ipynb` does.
It reads data from a .tsv file, processes it by filtering and modifying certain columns,
and finally save the processed data back into a new .tsv file.
"""

import os
import pandas as pd

MAX_TEXT_LENGTH = 700
FILEPATH = os.path.dirname(__file__)


if __name__ == "__main__":
    # Reading the data from the file
    df = pd.read_csv(
        os.path.join(FILEPATH, "../../data/raw/paranmt_for_detox_500k.tsv"),
        sep="\t",
        index_col=0,
    )

    # Drop rows with reference or translation text length more than 700 symbols
    df = df[df["reference"].str.len() <= MAX_TEXT_LENGTH]
    df = df[df["translation"].str.len() <= MAX_TEXT_LENGTH]

    # Add toxicity difference column
    df["tox_diff"] = df["ref_tox"] - df["trn_tox"]

    # Drop rows with toxicity difference absolute value less than 0.5
    df = df[df["tox_diff"].abs() >= 0.5]

    # Prepare for packing back to file
    # First, let's swap reference and translation texts if the translation is more toxic
    index_mask = df["tox_diff"] < 0

    df.loc[index_mask, "reference"], df.loc[index_mask, "translation"] = (
        df.loc[index_mask, "translation"],
        df.loc[index_mask, "reference"],
    )

    # ..and change the toxicity difference sign
    df.loc[index_mask, "tox_diff"] = -df.loc[index_mask, "tox_diff"]

    # Save only reference and translation texts
    df[["reference", "translation"]].to_csv(
        os.path.join(FILEPATH, "../../data/interim/processed.tsv"),
        sep="\t",
        index=False,
        header=False,
    )
    print("Done!")
