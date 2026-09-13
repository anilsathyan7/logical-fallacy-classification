from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_label_distribution_plot(
    dataset,
    labels_column,
    output_dir,
):
    """
    Save a plot with the number of examples for each label.
    """

    label_feature = (
        dataset["train"]
        .features[labels_column]
    )

    label_names = getattr(
        label_feature,
        "names",
        None,
    )

    label_ids = range(
        label_feature.num_classes
    )

    if label_names is None:
        label_names = [
            str(label_id)
            for label_id in label_ids
        ]

    counts_by_split = {
        split: Counter(
            dataset[split][labels_column]
        )
        for split in dataset
    }

    left = [
        0
        for _ in label_ids
    ]

    output_path = (
        Path(output_dir)
        / "label_distribution.png"
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    for split, counts in counts_by_split.items():
        values = [
            counts.get(label_id, 0)
            for label_id in label_ids
        ]

        ax.barh(
            label_names,
            values,
            left=left,
            label=split,
        )

        left = [
            current + value
            for current, value in zip(left, values)
        ]

    ax.set_xlabel("Number of examples")
    ax.set_ylabel("Label")
    ax.set_title("Label Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_token_lengths_plot(
    tokenized_dataset,
    output_dir,
):
    """
    Save a plot with tokenized input lengths for each split.
    """

    output_path = (
        Path(output_dir)
        / "token_lengths.png"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for split in tokenized_dataset:
        lengths = [
            len(input_ids)
            for input_ids in tokenized_dataset[split]["input_ids"]
        ]

        ax.hist(
            lengths,
            bins=30,
            alpha=0.45,
            label=split,
        )

    ax.set_xlabel("Token length")
    ax.set_ylabel("Number of examples")
    ax.set_title("Token Length Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_data_plots(
    dataset,
    tokenized_dataset,
    labels_column,
    output_dir,
):
    """
    Save dataset label-count and token-length plots.
    """

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    save_label_distribution_plot(
        dataset,
        labels_column,
        output_dir,
    )

    save_token_lengths_plot(
        tokenized_dataset,
        output_dir,
    )

    print(
        f"Data plots saved to: {output_dir}"
    )
