import pandas as pd
import pickle
import re

from scipy.sparse import csr_matrix, hstack


def clean_text(text: str) -> str:
    """
    Clean text from special characters and some of contractions.
    """
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub("\W", " ", text)
    text = re.sub("\s+", " ", text)
    text = text.strip(" ")
    return text


class STAToxic:
    """
    STAToxic class -- Style Transfer Accuracy for Toxicity

    Models are trained based on the: https://www.kaggle.com/code/rhodiumbeng/classifying-multi-label-comments-0-9741-lb/notebook
    For more information see: `notebooks/04-metrics-STA.ipynb`
    """

    def __init__(self):
        import os

        filepath = os.path.dirname(__file__)
        model_files = [
            os.path.join(filepath, "../../models/vectorizer.pkl"),
            os.path.join(filepath, "../../models/binary_logreg.pkl"),
            os.path.join(filepath, "../../models/chains_logreg.pkl"),
        ]

        self.vec, self.binary_logreg, self.chains_logreg = [
            pickle.load(open(file, "rb")) for file in model_files
        ]

        self.cols_target = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]

    @staticmethod
    def add_feature(X, feature_to_add):
        """
        Returns sparse feature matrix with added feature.
        feature_to_add can also be a list of features.
        """
        return hstack([X, csr_matrix(feature_to_add).T], "csr")

    def toxicity_report(self, input_data):
        """
        Returns a dataframe with toxicity probabilities for each label.
        """
        input_data = [clean_text(x) for x in input_data]
        input_data = self.vec.transform(input_data)

        # Create empty dataframe
        df = pd.DataFrame(columns=self.cols_target)

        # Predict using binary classifier
        for label in self.cols_target:
            prob = self.binary_logreg[label].predict_proba(input_data)[:, 1]
            df[label] = prob

        # Chain predictions
        for label in self.cols_target:
            y = self.chains_logreg[label].predict(input_data)
            prob = self.chains_logreg[label].predict_proba(input_data)[:, 1]

            # Average the probability
            df[label] = (df[label] + prob) / 2

            input_data = self.add_feature(input_data, y)

        return df
