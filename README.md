# Logical Fallacy Classifier

**Logical fallacies** are errors in reasoning that can make an argument seem
convincing without adequately supporting its conclusion. Examples include
attacking the person instead of their argument or treating a popular belief
as proof.

This project treats fallacy detection as a **text classification task**,
fine-tuning **BERT-based encoder models** (DeBERTa, ModernBERT, and MiniLM) to
assign arguments to **14 fallacy categories**. It uses **Hugging Face
Transformers and PyTorch** for modeling, **Accelerate** for training, and
**Weights & Biases** for experiment tracking and hyperparameter optimization.

The experiments **compare models**, **analyze their errors**, and test a selected
model on more realistic arguments to assess **how well it generalizes**.

## Setup

This project uses Python 3.13 and `uv` for dependency management. Run the
commands below to create the virtual environment, install the dependencies
specified in `uv.lock`, and activate the environment.

```bash
uv sync
source .venv/bin/activate
```

The code is built around:

- **PyTorch** for the training loop and predictions.
- **Hugging Face Transformers, Datasets, and Evaluate** for loading pretrained
  models, tokenizing text, preparing datasets, and computing metrics.
- **Accelerate** for moving models and batches to the GPU and running training
  in BF16 mixed precision.
- **Weights & Biases** for logging experiments, comparing runs, and tuning
  hyperparameters with Bayesian sweeps.

All reported training runs used a single NVIDIA GeForce RTX 5070 Ti Laptop GPU
(12 GB VRAM), CUDA 13.2, BF16 mixed precision, and a batch size of 16.

## Dataset

The models are trained and compared on the `classification` version of
[`kuwrom/fallacy`](https://huggingface.co/datasets/kuwrom/fallacy). It contains
138,574 short arguments, each assigned to one of 14 fallacy categories.
According to its dataset card, about 97% of the examples are GPT-4-generated;
the rest are human-written. This makes evaluation on other sources particularly
useful for checking how well the models generalize.

| Split | Examples | Purpose |
| --- | ---: | --- |
| Training | 110,859 | Fine-tune the models |
| Validation | 13,857 | Tune hyperparameters and select checkpoints |
| Test | 13,858 | Evaluate the selected configurations |

The plots below show the Kuwrom label distribution and token lengths.

![Label distribution](plots/data/label_distribution.png)

![Token-length distribution](plots/data/token_lengths.png)

We also combined CoCoLoFa and Touché for external evaluation. See
[Tests](#tests) for the dataset details and results.

## Models

We compare three pretrained encoders on the same 14-class task. Each model gets
a new classification head that scores the fallacy labels, and training updates
both the encoder and the head. Each model also has its own hyperparameter search.

- **[DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base)** has
  **12 transformer layers** and uses disentangled attention, which represents token
  content and position separately when computing attention. Its pretraining
  includes learning to detect replaced tokens. It is the default in `config.py`
  and achieved the **highest validation Macro-F1** and **highest test accuracy**
  in these experiments.
- **[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)** is a
  newer BERT-style encoder with **22 layers** and **long-context support up to
  8,192 tokens**.
  It alternates local attention with attention across the full input to handle
  longer sequences efficiently. Here it provides another architecture to compare
  on the same short arguments; these experiments do not establish an advantage
  on long documents.
- **[All-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**
  is a **lightweight encoder with 6 layers** and a **hidden size of 384**. The checkpoint was
  trained for sentence similarity; here we fine-tune its encoder with a new
  fallacy classification head. It lets us compare a much smaller model with the
  two larger encoders. Its saved checkpoint is also used for the embedding
  analysis and external evaluation below.

## Hyperparameter Optimization

**Hyperparameter optimization (HPO)** means trying different training settings
and comparing their validation results. Here, we tune the **learning rate**,
**weight decay**, and **warmup ratio** for each model.

**Weights & Biases (W&B)** records the settings and metrics for each run.
**W&B Sweeps** runs the search using a YAML configuration. We use **Bayesian
search**, which uses earlier results to choose promising settings for later runs.

- **Search space:** learning rates of `1e-5`, `2e-5`, or `3e-5`; weight decay of
  `0.0` or `0.01`; and warmup ratios of `0.0` or `0.06` in `configs/sweep.yaml`.
- **Selection metric:** maximize **validation Macro-F1**, which averages the
  F1 scores of all 14 classes with equal weight.
- **Test data stays out of tuning:** sweep runs set `run_test: false`. Test
  results are computed after the configuration has been selected.

Create a sweep, then use its returned ID to start an agent that runs up to six
experiments:

```bash
wandb sweep configs/sweep.yaml
wandb agent --count 6 <entity>/<project>/<sweep-id>
```

After choosing the best configuration by validation Macro-F1, put its settings
in `config.py` and run `main.py` to train and evaluate it on the test split.

For a separate search per model, use the corresponding YAML file in the
`wandb sweep` command:

```text
configs/sweep_deberta.yaml
configs/sweep_modernbert.yaml
configs/sweep_minilm.yaml
```

## Training

Training settings live in `config.py`. The current configuration is:

| Setting | Value |
| --- | --- |
| Model | `microsoft/deberta-v3-base` |
| Dataset | `kuwrom/fallacy` (`classification`) |
| Learning rate | `3e-5` |
| Weight decay | `0.01` |
| Warmup ratio | `0.06` |
| Batch size | `16` |
| Maximum epochs | `10` |
| Early-stopping patience | `3` epochs |
| Mixed precision | `bf16` |

Start a training run with:

```bash
python3 main.py
```

The script tokenizes the dataset, creates the train, validation, and test data
loaders, and adds a 14-class classification head to the selected encoder. The
whole model is fine-tuned with **AdamW** and a **linear learning-rate schedule**.
Gradients are clipped at `1.0` before each optimizer step to limit unstable
updates.

After every epoch, the run records training loss and accuracy alongside
validation loss, accuracy, and Macro-F1. The model state with the **highest
validation Macro-F1** is kept in memory. Training stops early after three epochs
without improvement, and the best state is restored before any final evaluation.

With `RUN_TEST = True`, the restored model is evaluated once on the Kuwrom test
split. The run then saves:

- the model and tokenizer under `checkpoints/best_model/<run-id>/`;
- the classification report and confusion matrix under
  `plots/evaluation/<run-id>/`;
- training settings and epoch-level metrics in W&B.

The curves below show how this selection worked for the best All-MiniLM run,
`8wfncto3`. The plots use "dev" as another name for the validation split.

### Accuracy

![All-MiniLM training and validation accuracy](plots/training/8wfncto3/accuracy.png)

Training accuracy continues to improve while validation accuracy levels off near
93%, showing a growing generalization gap in later epochs.

### Loss

![All-MiniLM training and validation loss](plots/training/8wfncto3/loss.png)

Validation loss reaches its minimum around epoch 2 and then rises as training
loss falls. This indicates that the model becomes increasingly overconfident
without comparable validation improvement.

### Macro-F1

![All-MiniLM validation Macro-F1](plots/training/8wfncto3/macro_f1.png)

Validation Macro-F1 peaks at `0.9336` in epoch 7. The training loop selects and
restores this checkpoint instead of retaining the final epoch.

## Inference

`predict.py` runs a saved classifier over every row in a CSV file. Set the model
and dataset near the bottom of the script:

```python
model_path = Path("checkpoints/best_model/8wfncto3")
dataset_name = "hard_real_test"
```

The input file is read from `datasets/<dataset_name>.csv` and must contain a
`text` column. Any other columns, including expected labels or split names, are
kept in the output. The repository includes two ready-to-run inputs:

| Dataset | Use |
| --- | --- |
| `easy_synthetic_test` | A small 13-example sanity check |
| `hard_real_test` | The full 3,746-example CoCoLoFa/Touché evaluation set |

Run inference with:

```bash
python3 predict.py
```

The model uses the GPU when one is available and otherwise falls back to the
CPU. Texts are processed in batches using `BATCH_SIZE` from `config.py`; no rows
are sampled or skipped. For each input, the script prints the predicted class
and confidence, then adds two columns to the saved CSV:

- `predicted_label`: the fallacy class with the highest score;
- `confidence`: the softmax probability assigned to that class.

Results are written to
`results/<run-id>/<dataset_name>_predictions.csv`.

## Outputs

Training, evaluation, and inference artifacts are kept in separate directories:

| Path | Contents |
| --- | --- |
| `checkpoints/best_model/<run-id>/` | Best model weights, configuration, and tokenizer |
| `plots/training/<run-id>/` | Training and validation curves |
| `plots/evaluation/<run-id>/` | Classification report, confusion matrix, and optional UMAP |
| `results/<run-id>/` | CSV files produced by `predict.py` |
| `wandb/` | Local W&B run data and logs |

The W&B dashboard also stores each run's configuration, epoch-level metrics,
final test scores, and evaluation images.

## Results

Each model was tuned separately and selected by its best validation Macro-F1.
The selected checkpoint was then evaluated once on the Kuwrom test split.
Accuracy shows the overall proportion of correct predictions, while Macro-F1
gives equal weight to every fallacy class.

The table reports the strongest completed run for each model:

| Model | Run | Best Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| `microsoft/deberta-v3-base` | `l1k8t4wj` | 0.9391 | 0.9411 | 0.9409 |
| `answerdotai/ModernBERT-base` | `7bpgpfq2` | 0.9377 | 0.9395 | 0.9391 |
| `sentence-transformers/all-MiniLM-L6-v2` | `8wfncto3` | 0.9336 | 0.9358 | 0.9358 |

All three models score above 93.5% test accuracy on Kuwrom. DeBERTa ranks first
at **94.11% accuracy** and **0.9409 Macro-F1**, followed closely by ModernBERT.
The gap between DeBERTa and MiniLM is only 0.53 percentage points in accuracy,
which makes the six-layer MiniLM a strong compact model for this dataset. The
detailed analysis below uses its saved checkpoint, `8wfncto3`.

### Hyperparameters

These are the settings selected by the model-specific sweeps:

| Model | Learning Rate | Weight Decay | Warmup Ratio | Batch Size |
| --- | ---: | ---: | ---: | ---: |
| `microsoft/deberta-v3-base` | `3e-5` | `0.01` | `0.00` | `16` |
| `answerdotai/ModernBERT-base` | `3e-5` | `0.01` | `0.06` | `16` |
| `sentence-transformers/all-MiniLM-L6-v2` | `2e-5` | `0.01` | `0.00` | `16` |

All three runs used a maximum of 10 epochs, early-stopping patience of 3,
gradient clipping at `1.0`, seed `42`, and BF16 mixed precision.

## Analysis

The detailed analysis uses the saved All-MiniLM run, `8wfncto3`. It reaches
**93.58% accuracy and Macro-F1** on the Kuwrom test set, but the aggregate score
hides one important class boundary.

### Kuwrom Errors

![All-MiniLM normalized test confusion matrix](plots/evaluation/8wfncto3/test_confusion_matrix.png)

**Twelve of the fourteen classes** have an F1 score above 0.92, and ten are above
0.96. The clear exceptions are **`ad_populum` and `the_bandwagon`**:

| Performance | Classes |
| --- | --- |
| Highest | `appeal_to_ignorance` 0.998, `loaded_question` 0.996, `slippery_slope` 0.990, `false_dilemma` 0.989 |
| Strong | `circular_reasoning` 0.980, `false_causality` 0.977, `equivocation` 0.975, `ad_hominem` 0.964, `appeal_to_authority` 0.962, `red_herring` 0.960 |
| Moderate | `hasty_generalization` 0.930, `cherry_picking` 0.928 |
| Weak | `the_bandwagon` 0.727, `ad_populum` 0.721 |

The confusion matrix shows that **roughly a quarter** of the examples from each of
the two weakest classes are assigned to the other. The distinction is narrow:

- **Ad populum** uses widespread belief as evidence that a claim is true.
- **Bandwagon** argues that someone should adopt a belief or behavior because
  many other people already have.

Bandwagon is often treated as a form of ad populum, and some arguments fit both
descriptions. The symmetric error therefore points to an **overlapping label
definition** as well as a model limitation.

### Embedding Structure

![All-MiniLM test embedding UMAP](plots/evaluation/8wfncto3/test_embedding_umap.png)

Each point in the UMAP is a Kuwrom test example, colored by its true label. The
left panel uses the original All-MiniLM encoder; the right uses the fine-tuned
encoder.

**Before fine-tuning**, most labels occupy the same broad region. **After
fine-tuning**, the examples form much tighter groups organized around the
training labels. This matches the strong in-domain classification scores and
shows that fine-tuning
substantially changed the representation space. UMAP is a two-dimensional
projection, so the spacing between clusters should be read as a visual summary,
not as a precise measure of semantic distance.

### Tests

The external test asks whether those in-domain results carry over to arguments
written in a different setting. It combines two sources:

- **[CoCoLoFa](https://github.com/Crowd-AI-Lab/cocolofa):** news-article comments
  written by crowd workers with LLM assistance, labelled for fallacy presence
  and type.
- **[Touché Fallacy Detection 2026](https://touche.webis.de/clef26/touche26-web/fallacy-detection.html):**
  a shared-task dataset based on Reddit comments, with labels for fallacy
  detection and classification.

We combined these sources into `datasets/cocolofa_touche_combined.csv`, then
selected and mapped six fallacy categories to Kuwrom labels to create
`datasets/hard_real_test.csv`. This file contains 3,746 examples with `split`,
`text`, and `label` columns. It includes the original train, validation, and test
splits, so it is an **external stress test** rather than a new held-out benchmark.

The saved All-MiniLM checkpoint (`8wfncto3`) was evaluated on the easy examples
and the mapped CoCoLoFa/Touché dataset using `predict.py`.

| Dataset | Examples | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| `easy_synthetic_test.csv` | 13 | 12 | **92.31%** |
| `hard_real_test.csv` | 3,746 | 1,775 | **47.38%** |

#### Observations

- **Easy-set check:** The easy set contains nine synthetic examples and four
  external samples. Its only error is a `hasty_generalization` predicted as
  `appeal_to_authority` with **99.83% confidence**. Thirteen selected examples
  are useful as a sanity check, but are too few for a general performance claim.
- **Generalization gap:** Performance drops sharply on the hard set. The original
  test split scores **44.39%** (178/401), close to the **47.38%** obtained across
  all splits. Recall also varies widely by class:

| Label | Recall |
| --- | ---: |
| `slippery_slope` | 89.6% |
| `appeal_to_authority` | 69.9% |
| `false_dilemma` | 42.6% |
| `red_herring` | 31.8% |
| `ad_populum` | 30.2% |
| `hasty_generalization` | 16.0% |

- **Class bias:** The model predicts `slippery_slope` 1,565 times even though the
  hard set contains 711 such examples. Its largest error groups are `red_herring`
  (270), `hasty_generalization` (234), and `false_dilemma` (227), all predicted
  as `slippery_slope`.
- **Confidence:** The model is often very sure when it is wrong. **1,199 of the
  1,971 errors** have at least 90% confidence.
- **Likely causes:** The external examples are longer and less templated, and
  some contain cues for more than one fallacy. No input exceeds the model's
  512-token limit, so input truncation does not explain the errors.
- **Mapping limits:** `appeal_to_majority` is mapped to `ad_populum`, while
  `appeal_to_worse_problems` is mapped to `red_herring`; both are approximate.
  On the four categories whose names match directly, accuracy improves to
  **55.62%** across 2,492 examples. The hard set excludes `none`,
  `appeal_to_nature`, and `appeal_to_tradition` because Kuwrom has no direct
  counterparts.
- **Conclusion:** The model learns the Kuwrom label structure well but **does not
  transfer reliably to the external data**. The next useful experiment is to
  fine-tune on the external training split and reserve its test split for
  evaluation.
