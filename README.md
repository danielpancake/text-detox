# Practical Machine Learning and Deep Learning - Assignment 1 - Text De-toxification

- Daniel Vakhrushev
- d.satarov@innopolis.university
- BS21-DS-02

## How to run the model

### Training

```python
from src.models.detoxGPT2 import detoxGPT2

detoxGPT2().train("data/interim/processed.tsv", 128)
# Note: that it may be better to use max length from the dataset
# as the block size, but it will take more time to train.
```

For more information on the training process, please refer to [this notebook](./notebooks/03_1__GPT2_training.ipynb).

### Inference

```python
"""
This is a minimal example of how to use the detoxGPT2 model.
"""
import torch
from src.models.detoxGPT2 import detoxGPT2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [0] is a suggestion, [1] is the metric scores
print(
    detoxGPT2().get_best_suggestion(
        "What a fucking stupid thing to say!", device=DEVICE
    )[0]
)
```

For more information on the inference process, please refer to [this notebook](./notebooks/03_2__GPT2_inference.ipynb).

For more information on the ranked suggestions from the inference, please refer to [this notebook](./notebooks/05_GPT2_inference_with_metrics.ipynb).
