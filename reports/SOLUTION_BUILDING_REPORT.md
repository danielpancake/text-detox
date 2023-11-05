# Solution Building Report

This report is dedicated to the process of building a solution for the task of detecting profanity in text (or text detoxification).

## Baseline 1: Delete profanity words

This baseline is based on the idead of simply deleting words that are considered profanity. Resulting sentences might be grammatically incorrect, but they closely resemble the original sentences, which can be considered a good starting point for further improvement.

Example:

```text
Original: What the hell do they expect me to say?
Delete:   What the **** do they expect me to say?
```

### Implementation

To find profanity words, we use a dictionary of profanity words from [here](https://www.kaggle.com/datasets/nicapotato/bad-bad-words). This dataset contains 1616 unique profanity words.

To make use of this dictionary, we implement kgram similarity search. For each word in the sentence, we find the most similar word in the dictionary. If the similarity is above a certain threshold, we consider the word profanity and delete it from the sentence.

## Baseline 2: Replace profanity words with synonyms

This baseline is based on the idea of replacing profanity words with their synonyms. This approach is similar to the previous one, but instead of deleting profanity words, we replace them with similar words.

This baseline is expected to produce better results than the previous one, since the resulting sentences are more likely to be grammatically correct. However, the problem arises when the profanity consists of more than one word. In this case, we make replacements per each word, which far from ideal.

Example:

```text
Replace:  What's it like to make love someone to dying
```

### Implementation

Again, we use the same dictionary of profanity words from the previous baseline. To find synonyms, we use the [WordNet](https://wordnet.princeton.edu/) database. For each word in the sentence, we find some number of synonyms. Then, we filter synonyms that are considered profanity and replace the word with the first synonym in the list. If none found, we delete the word.

## Hypothesis: Using pretrained model: GPT-2

GPT-2 is a transformer-based language model trained on a large text corpus. We hypothesize it can generate profanity-free text by framing it as a style transfer problem with two styles: toxic (profane) and non-toxic (profanity-free).

We can train GPT-2 for this by preparing a parallel corpus of toxic and non-toxic sentences, using special tokens to indicate style, so GPT-2 learns to transfer text from the toxic to the non-toxic style when predicting the next word.

```text
Parallel corpus:
  Toxic:     What the hell do they expect me to say?
  Non-toxic: What did they expect me to say?

Style-transfer corpus:
  [TOX]What the hell do they expect me to say?[/TOX]»»[DETOX]What did they expect me to say?[/DETOX]
```

To prompt for style transfer, we use the following template as input:

```text
[TOX] ... [/TOX]»»[DETOX]
```

## Results

As a result, we have a GPT-2 based model that can generate profanity-free text from toxic text. It has the following advantages:

- It can be trained on a small portion of the data.

It has the following disadvantages:

- It requires a lot of computational resources to train.
- Often generates text that is not grammatically correct or does not make sense.
- It is not guaranteed to generate text that is free of profanity.
