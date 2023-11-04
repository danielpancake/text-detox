import matplotlib.pyplot as plt
import os
import pandas as pd

FILEPATH = os.path.dirname(__file__)


def generate_dataset_text_lengths_distribution():
    """
    Generates a plot of text lengths distribution from the dataset.
    """
    df = pd.read_csv(
        os.path.join(FILEPATH, "../../data/raw/paranmt_for_detox_500k.tsv"),
        sep="\t",
        index_col=0,
    )

    plt.hist(df["reference"].str.len(), bins=100, label="reference")
    plt.hist(df["translation"].str.len(), bins=100, label="translation")
    plt.title("Distribution of text lengths")
    plt.xlabel("Text length")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.legend()
    plt.savefig(
        os.path.join(
            FILEPATH, "../../reports/figures/detox_500k_text_lengths_distribution.png"
        ),
        format="png",
    )
    plt.close()


if __name__ == "__main__":
    generate_dataset_text_lengths_distribution()
    print("Done!")
