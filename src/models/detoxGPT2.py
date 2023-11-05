from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    TrainingArguments,
    Trainer,
    pipeline,
)


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

    def generate(self, input_text: str, device: str = "cuda") -> str:
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

        generated_text = self.generator(
            prompt_text,
            max_length=100,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        return generated_text

    def get_detoxed_suggestions(self, input_text: str, device: str = "cuda") -> list:
        generated_text = self.generate(input_text, device=device)

        suggestions = generated_text.split(
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
