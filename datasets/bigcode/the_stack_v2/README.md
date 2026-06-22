# The Stack v2 — Access Instructions

## Gated Dataset
1. Visit https://huggingface.co/datasets/bigcode/the-stack-v2
2. Accept agreement
3. `huggingface-cli login`

## Streaming usage
```python
from datasets import load_dataset
ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
for item in ds:
    print(item.keys())
    break
```
