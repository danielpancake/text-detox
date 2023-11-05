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


def analyze_scores():
    df_delete_scores = pd.read_csv(
        os.path.join(FILEPATH, "../../data/interim/delete_baseline_scores.csv"),
    )
    df_replace_scores = pd.read_csv(
        os.path.join(FILEPATH, "../../data/interim/replace_baseline_scores.csv"),
    )
    df_detoxGPT_scores = pd.read_csv(
        os.path.join(FILEPATH, "../../data/interim/detoxGPT_scores.csv"),
    )

    # Count mean scores for each model
    delete_mean = df_delete_scores.mean(axis=0)
    replace_mean = df_replace_scores.mean(axis=0)
    detoxGPT_mean = df_detoxGPT_scores.mean(axis=0)

    df_total = pd.concat([delete_mean, replace_mean, detoxGPT_mean], axis=1)
    df_total.columns = ["delete", "replace", "detoxGPT"]

    # Plot the scores
    df_total.plot.bar()
    plt.title("Scores for different models")
    plt.xlabel("Score")
    plt.ylabel("Value")

    # Draw numbers above bars
    for idx, model in enumerate(df_total.columns):
        for index, value in enumerate(df_total[model]):
            plt.text(
                index + idx / 4 - 0.3,
                value + 0.01,
                str(round(value, 2)),
                color="black",
                fontweight="bold",
            )

    # Rotate x-axis labels
    plt.xticks(rotation=0)

    plt.savefig(
        os.path.join(FILEPATH, "../../reports/figures/scores_for_different_models.png"),
        format="png",
    )


if __name__ == "__main__":
    generate_dataset_text_lengths_distribution()
    analyze_scores()
    print("Done!")
