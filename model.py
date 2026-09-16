from torch.optim import AdamW
from transformers import (
    AutoConfig,
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

    saved_config = AutoConfig.from_pretrained(checkpoint)
    previous_num_labels = saved_config.num_labels
    has_classification_head = any(
        name.endswith("ForSequenceClassification")
        for name in (saved_config.architectures or [])
    )
    saved_config.num_labels = num_labels

    # Save readable label names in the model config.
    if id2label is not None:
        saved_config.id2label = id2label
        saved_config.label2id = label2id

    model_kwargs = {"config": saved_config}
    if has_classification_head and previous_num_labels != num_labels:
        # Keep the trained encoder while initializing a head for the new labels.
        model_kwargs["ignore_mismatched_sizes"] = True

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
