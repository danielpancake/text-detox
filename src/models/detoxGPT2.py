"""
detoxGPT2 model
Pretrained GPT2 model fine-tuned on the detox dataset
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    TrainingArguments,
    Trainer,
    pipeline,
)

if __name__ == "__main__":
    from evaluation import STAToxic, Similarity
else:
    from .evaluation import STAToxic, Similarity

from typing import Tuple

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

METRIC_SIM = Similarity()
METRIC_TOX = STAToxic()


class detoxGPT2:
    special_tokens_dict = {
        "tox_begin": "[TOX]",
        "tox_end": "[/TOX]",
        "detox_begin": "[DETOX]",
        "detox_end": "[/DETOX]",
        "separator": "»»",
        "split": "[SPLIT]",
    }

    def __init__(self, model_dir: str = "danielpancake/detoxGPT2-pmldl") -> None:
        # The format is: [TOX]text[/TOX]»»[DETOX]text[/DETOX]
        # [TOX]   - source text
        # [DETOX] - target text
        # »»      - separator

        # So, we will add 5 custom tokens to the vocabulary:
        # [TOX], [/TOX], [DETOX], [/DETOX], »»
        self.model = None
        self.tokenizer = None
        self.generator = None

        self.model_dir = model_dir

    def train(
        self,
        train_data_path: str,
        block_size: int,
        batch_size: int = 8,
        epochs: int = 1,
        warmup_steps: int = 500,
        save_steps: int = 10_000,
        logging_steps: int = 500,
    ) -> None:
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir="cache")

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="cache")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        # Add special tokens
        num_added_toks = self.tokenizer.add_tokens(
            list(self.special_tokens_dict.values())
        )
        assert num_added_toks == len(self.special_tokens_dict), "Error adding tokens"
        self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)

        datacollator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        train_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            block_size=block_size,
            file_path=train_data_path,
        )

        training_args = TrainingArguments(
            output_dir=self.model_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=datacollator,
            train_dataset=train_dataset,
        )

        trainer.train()

        # Save model and tokenizer
        trainer.save_model(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

    def generate(
        self,
        input_text: str,
        max_length: int = 150,
        sequences: int = 3,
        device: str = "cuda",
    ) -> list:
        if self.model is None:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir, cache_dir="cache"
                )
            except:
                raise Exception("Model not found")

        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir, cache_dir="cache"
                )
            except:
                raise Exception("Tokenizer not found")

        if self.generator is None:
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
            )

        prompt_text = (
            self.special_tokens_dict["tox_begin"]
            + input_text
            + self.special_tokens_dict["tox_end"]
            + self.special_tokens_dict["separator"]
        )

        generated_texts = self.generator(
            prompt_text,
            max_length=max_length,
            num_return_sequences=sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return [text["generated_text"] for text in generated_texts]

    def get_detoxed_suggestions(
        self,
        input_text: str,
        max_length: int = 150,
        sequences: int = 3,
        device: str = "cuda",
    ) -> list:
        suggestions = []

        generated_texts = self.generate(input_text, max_length, sequences, device)
        for text in generated_texts:
            suggestions += self._parse_generated_text(text)

        return list(set(suggestions))

    def _parse_generated_text(self, text: str) -> list:
        # Remove the first part of the text
        suggestions = text.split(
            self.special_tokens_dict["separator"]
            + self.special_tokens_dict["detox_begin"]
        )[1]

        # Replace all the special tokens with [SPLIT]
        for token in self.special_tokens_dict.values():
            suggestions = suggestions.replace(token, self.special_tokens_dict["split"])

        # Split by [SPLIT]
        suggestions = suggestions.split(self.special_tokens_dict["split"])

        # Trim if any of \", \n, or whitespace is present
        suggestions = [s.strip('"\n ') for s in suggestions]

        # Remove empty strings
        suggestions = list(filter(None, suggestions))

        return suggestions

    def get_best_suggestion(
        self,
        input_text: str,
        max_length: int = 150,
        sequences: int = 3,
        device: str = "cuda",
    ) -> Tuple[str, pd.DataFrame]:
        """
        Returns the best suggestion based on the following criteria:
        - Toxicity
        - Word Overlap
        - Cosine Similarity
        - BLEU Score

        and the associated scores
        """
        suggestions = self.get_detoxed_suggestions(
            input_text, max_length, sequences, device
        )

        if len(suggestions) == 0:
            return input_text
        else:
            df = pd.DataFrame(
                suggestions,
                columns=["suggestion"],
            )

            # Add empty column for each metric
            metrics = ["wo", "cs", "bleu"]
            df[metrics] = pd.DataFrame(
                [[0] * len(metrics)], index=df.index, dtype=float
            )

            # Generate toxicity report for each suggestion
            toxicity_report = METRIC_TOX.toxicity_report(df["suggestion"])

            for index, row in df.iterrows():
                df.loc[index, "wo"] = METRIC_SIM.get_wo_score(
                    input_text, row["suggestion"]
                )
                df.loc[index, "cs"] = METRIC_SIM.get_cosine_score(
                    input_text, row["suggestion"]
                )
                df.loc[index, "bleu"] = METRIC_SIM.get_bleu_score(
                    input_text, row["suggestion"]
                )

            # Concat with toxicity report
            df = pd.concat([df, toxicity_report], axis=1)

            # Calculate the score
            metric_weights = {"wo": 0.1, "cs": 0.5, "bleu": 0.4}

            # Toxicity report should be as low as possible
            # Similarity metrics should be as high as possible
            df["detox_score"] = 1 - df[["toxic"]].mean(axis=1)
            df["similarity"] = df[metrics].dot(pd.Series(metric_weights))

            # Final score
            df["score"] = df[["detox_score", "similarity"]].mean(axis=1)

            # Find the suggestion with the highest score
            df.sort_values(by=["score"], ascending=False, inplace=True)

            # Split into two parts suggestion and everything but it
            return (
                df.iloc[0]["suggestion"],
                df.iloc[0][[*metrics, "detox_score"]],
            )
