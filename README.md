# Practical Machine Learning and Deep Learning - Assignment 1 - Text De-toxification

Daniel Vakhrushev <>@innopolis.university BS21-DS-02

## How to run the model

### Training

```python
from src.models.detoxGPT2 import detoxGPT2

detoxGPT2().train("data/interim/processed.tsv", 128)
```

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
