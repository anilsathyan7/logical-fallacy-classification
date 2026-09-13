from pathlib import Path

import evaluate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import TRAINING_PLOTS_DIR


def evaluate_model(
    model,
    dataloader,
):

    model.eval()

    total_loss = 0.0
    total_examples = 0

    accuracy_metric = evaluate.load(
        "accuracy"
    )

    f1_metric = evaluate.load(
        "f1"
    )

    with torch.no_grad():

        for batch in dataloader:

            outputs = model(**batch)

            loss = outputs.loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite evaluation loss."
                )

            predictions = outputs.logits.argmax(
                dim=-1
            )

            batch_size = batch["labels"].size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_examples += batch_size

            accuracy_metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

            f1_metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

    loss = (
        total_loss / total_examples
    )

    accuracy = (
        accuracy_metric.compute()["accuracy"]
    )

    macro_f1 = (
        f1_metric.compute(
            average="macro"
        )["f1"]
    )

    return loss, accuracy, macro_f1


def plot_training_history(
    history,
    output_dir=TRAINING_PLOTS_DIR,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # ========================================================
    # LOSS
    # ========================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Train Loss",
    )

    ax.plot(
        epochs,
        history["dev_loss"],
        marker="o",
        label="Dev Loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Dev Loss")

    ax.legend()
    ax.grid(True)
    ax.set_xticks(list(epochs))

    fig.tight_layout()
    fig.savefig(
        output_dir / "loss.png",
        dpi=200,
    )
    plt.close(fig)

    # ========================================================
    # ACCURACY
    # ========================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs,
        history["train_accuracy"],
        marker="o",
        label="Train Accuracy",
    )

    ax.plot(
        epochs,
        history["dev_accuracy"],
        marker="o",
        label="Dev Accuracy",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Dev Accuracy")

    ax.legend()
    ax.grid(True)
    ax.set_xticks(list(epochs))

    fig.tight_layout()
    fig.savefig(
        output_dir / "accuracy.png",
        dpi=200,
    )
    plt.close(fig)

    # ========================================================
    # MACRO F1
    # ========================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs,
        history["dev_macro_f1"],
        marker="o",
        label="Dev Macro-F1",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Dev Macro-F1")

    ax.legend()
    ax.grid(True)
    ax.set_xticks(list(epochs))

    fig.tight_layout()
    fig.savefig(
        output_dir / "macro_f1.png",
        dpi=200,
    )
    plt.close(fig)

    print(
        f"Training plots saved to: {output_dir}"
    )
