"""Extract encoder embeddings and generate UMAP comparison plots."""

from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

from config import (
    BATCH_SIZE,
    DATASET_CONFIG,
    DATASET_NAME,
    RESULTS_DIR,
    LABEL_COLUMN,
    SEED,
    TEXT_COLUMN,
)
from data import load_data


def mean_pool_embeddings(last_hidden_state, attention_mask):
    """
    Mean-pool token embeddings while ignoring padding tokens.

    Args:
        last_hidden_state: Final hidden states from the encoder.
        attention_mask: Token mask where real tokens are 1 and padding is 0.

    Returns:
        One pooled embedding per input example.
    """

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed_embeddings = (last_hidden_state * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed_embeddings / token_counts


def extract_encoder_embeddings(
    texts,
    model_path,
    tokenizer_path=None,
    batch_size=32,
    device=None,
):
    """
    Extract normalized final-layer encoder embeddings for texts.

    Args:
        texts: Input strings to embed.
        model_path: Hugging Face checkpoint name or local checkpoint path.
        tokenizer_path: Optional tokenizer checkpoint or path.
        batch_size: Number of texts to embed per forward pass.
        device: Optional device override such as "cpu" or "cuda".

    Returns:
        NumPy array with one L2-normalized embedding per text.
    """

    texts = list(texts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model_path = str(model_path)
    tokenizer_path = str(tokenizer_path or model_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        batch_embeddings = mean_pool_embeddings(
            outputs.last_hidden_state,
            inputs["attention_mask"],
        )
        # Normalize so cosine-based UMAP compares directions, not vector scale.
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        embeddings.append(batch_embeddings.cpu())

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return torch.cat(embeddings).numpy()


def save_umap_comparison_plot(
    texts,
    labels,
    label_names,
    original_model_path,
    fine_tuned_model_path,
    output_dir,
    batch_size=32,
    random_state=42,
    n_neighbors=30,
    min_dist=0.05,
    metric="cosine",
    device=None,
):
    """
    Save a side-by-side UMAP of original and fine-tuned embeddings.

    Args:
        texts: Test examples to embed with both checkpoints.
        labels: Integer true labels for coloring points.
        label_names: Label names ordered by integer label id.
        original_model_path: Base model checkpoint before fine-tuning.
        fine_tuned_model_path: Saved fine-tuned checkpoint.
        output_dir: Directory where the UMAP image is saved.
        batch_size: Number of texts to embed per forward pass.
        random_state: Seed for reproducible UMAP layout.
        n_neighbors: UMAP neighborhood size.
        min_dist: UMAP minimum distance between nearby points.
        metric: Distance metric used by UMAP.
        device: Optional device override such as "cpu" or "cuda".

    Returns:
        Path to the saved UMAP image.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_embedding_umap.png"

    texts = list(texts)
    labels = np.asarray(labels)

    if len(texts) != len(labels):
        raise ValueError("UMAP text and label counts must match.")
    if len(texts) < 2:
        raise ValueError("UMAP comparison requires at least two texts.")

    print("Extracting original checkpoint embeddings...", flush=True)
    original_embeddings = extract_encoder_embeddings(
        texts=texts,
        model_path=original_model_path,
        batch_size=batch_size,
        device=device,
    )

    print("Extracting fine-tuned checkpoint embeddings...", flush=True)
    fine_tuned_embeddings = extract_encoder_embeddings(
        texts=texts,
        model_path=fine_tuned_model_path,
        tokenizer_path=fine_tuned_model_path,
        batch_size=batch_size,
        device=device,
    )

    combined_embeddings = np.vstack(
        [original_embeddings, fine_tuned_embeddings]
    )
    n_neighbors = min(n_neighbors, len(combined_embeddings) - 1)

    # Fit one projection so the two panels share the same UMAP coordinate space.
    print("Fitting shared UMAP projection...", flush=True)
    coordinates = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    ).fit_transform(combined_embeddings)

    split_index = len(original_embeddings)
    original_coordinates = coordinates[:split_index]
    fine_tuned_coordinates = coordinates[split_index:]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 8),
        sharex=True,
        sharey=True,
    )
    colors = plt.get_cmap("tab20", len(label_names))

    for ax, panel_coordinates, title in [
        (axes[0], original_coordinates, "Original Encoder"),
        (axes[1], fine_tuned_coordinates, "Fine-Tuned Encoder"),
    ]:
        for label_id, label_name in enumerate(label_names):
            label_mask = labels == label_id

            if not label_mask.any():
                continue

            ax.scatter(
                panel_coordinates[label_mask, 0],
                panel_coordinates[label_mask, 1],
                s=6,
                alpha=0.55,
                linewidths=0,
                color=colors(label_id),
                label=label_name,
            )

        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Test Embedding UMAP by True Label")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"UMAP plot saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    original_model_path = "sentence-transformers/all-MiniLM-L6-v2"
    fine_tuned_model_path = Path("checkpoints/best_model/8wfncto3")
    output_dir = Path(RESULTS_DIR) / "8wfncto3" / "evaluation"
    split = "test"

    dataset = load_data(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        label_column=LABEL_COLUMN,
    )

    label_names = dataset["train"].features["labels"].names

    save_umap_comparison_plot(
        texts=dataset[split][TEXT_COLUMN],
        labels=dataset[split]["labels"],
        label_names=label_names,
        original_model_path=original_model_path,
        fine_tuned_model_path=fine_tuned_model_path,
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        random_state=SEED,
    )
