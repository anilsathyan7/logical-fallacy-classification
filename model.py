from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    get_scheduler,
)


def create_model(
    checkpoint,
    num_labels,
    id2label=None,
    label2id=None,
):
    """
    Create the sequence classification model.

    Args:
        checkpoint: Hugging Face model checkpoint.
        num_labels: Number of output labels.
        id2label: Optional id-to-label mapping.
        label2id: Optional label-to-id mapping.

    Returns:
        Sequence classification model.
    """

    model_kwargs = {
        "num_labels": num_labels,
    }

    # Save readable label names in the model config.
    if id2label is not None:
        model_kwargs["id2label"] = id2label
        model_kwargs["label2id"] = label2id

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            checkpoint,
            **model_kwargs,
        )
    )

    model = model.float()

    return model


def create_optimizer(
    model,
    learning_rate,
    weight_decay,
):
    """
    Create optimizer.

    Args:
        model: Model to optimize.
        learning_rate: Optimizer learning rate.
        weight_decay: AdamW weight decay.

    Returns:
        AdamW optimizer.
    """

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return optimizer


def create_scheduler(
    optimizer,
    num_training_steps,
    warmup_ratio,
):
    """
    Create learning-rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        num_training_steps: Total optimizer steps.
        warmup_ratio: Fraction of steps used for warmup.

    Returns:
        Linear learning-rate scheduler.
    """

    num_warmup_steps = int(
        num_training_steps * warmup_ratio
    )

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return scheduler
