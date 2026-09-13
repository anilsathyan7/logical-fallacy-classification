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
    """

    model_kwargs = {
        "num_labels": num_labels,
    }

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
