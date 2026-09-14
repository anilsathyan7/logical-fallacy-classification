# Logical Fallacy Classifier

Small Hugging Face/Accelerate training pipeline for multi-class logical fallacy
classification.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Train

```bash
python3 main.py
```

Current default config:

```text
dataset: kuwrom/fallacy classification
model: microsoft/deberta-v3-base
mixed precision: bf16
batch size: 16
epochs: 10
```

## W&B Sweeps

```bash
wandb sweep sweep.yaml
wandb agent --count 6 <entity>/<project>/<sweep_id>
```

Single-model sweep files are also available:

```text
sweep_deberta.yaml
sweep_modernbert.yaml
sweep_minilm.yaml
```

## Outputs

Training writes local artifacts to:

```text
checkpoints/
plots/
wandb/
```

Those outputs are ignored by Git. Raw local downloads are ignored; the cleaned
combined CSV can be committed.

## Current Results

Best completed Kuwrom runs so far:

| Model | Run | Best Dev Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| `microsoft/deberta-v3-base` | `l1k8t4wj` | 0.9391 | 0.9411 | 0.9409 |
| `answerdotai/ModernBERT-base` | `7bpgpfq2` | 0.9377 | 0.9395 | 0.9391 |
| `sentence-transformers/all-MiniLM-L6-v2` | `8wfncto3` | 0.9336 | 0.9358 | 0.9358 |

Next phase: use realistic data such as CoCoLoFa/Touché for fine-tuning and
evaluation, because the Kuwrom benchmark may overestimate real-world behavior.
