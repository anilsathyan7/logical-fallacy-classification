from datasets import ClassLabel, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)

from config import DATA_PLOTS_DIR
from utils import save_data_plots


def load_data(
    dataset_name,
    dataset_config,
    label_column,
):
    """
    Load and prepare the logical fallacy dataset.

    Args:
        dataset_name: Hugging Face dataset name.
        dataset_config: Dataset config name.
        label_column: Source label column.

    Returns:
        Dataset with a ClassLabel "labels" column.
    """

    dataset = load_dataset(
        dataset_name,
        dataset_config,
    )

    label_feature = (
        dataset["train"]
        .features[label_column]
    )

    # Convert string labels to ClassLabel when needed.
    if not isinstance(label_feature, ClassLabel):
        dataset = dataset.class_encode_column(
            label_column
        )

    # Transformers expects the label column to be named labels
    if label_column != "labels":
        dataset = dataset.rename_column(
            label_column,
            "labels"
        )

    return dataset


def tokenize_data(
    dataset,
    checkpoint,
    text_column,
):
    """
    Tokenize the text data.

    Args:
        dataset: Dataset with text and labels.
        checkpoint: Tokenizer checkpoint.
        text_column: Source text column.

    Returns:
        Tokenized dataset and tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    def tokenize_function(example):
        return tokenizer(
            example[text_column],
            truncation=True,
        )

    # Keep only tensors accepted by model(**batch).
    columns_to_remove = [
        column
        for column in dataset["train"].column_names
        if column != "labels"
    ]

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
    )

    return tokenized_dataset, tokenizer


def create_dataloaders(
    tokenized_dataset,
    tokenizer,
    batch_size,
):
    """
    Create train, validation and test DataLoaders.

    Args:
        tokenized_dataset: Tokenized train/validation/test dataset.
        tokenizer: Tokenizer for dynamic padding.
        batch_size: Batch size per DataLoader.

    Returns:
        Train, validation and test DataLoaders.
    """

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    validation_dataloader = DataLoader(
        tokenized_dataset["validation"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    test_dataloader = DataLoader(
        tokenized_dataset["test"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
    )


def prepare_data(
    dataset_name,
    dataset_config,
    text_column,
    label_column,
    checkpoint,
    batch_size,
):
    """
    Complete data preparation pipeline.

    Args:
        dataset_name: Hugging Face dataset name.
        dataset_config: Dataset config name.
        text_column: Source text column.
        label_column: Source label column.
        checkpoint: Tokenizer checkpoint.
        batch_size: Batch size per DataLoader.

    Returns:
        Dataset, tokenized dataset, tokenizer, dataloaders and label count.
    """

    dataset = load_data(
        dataset_name,
        dataset_config,
        label_column,
    )

    tokenized_dataset, tokenizer = tokenize_data(
        dataset,
        checkpoint,
        text_column,
    )

    # Save label distribution and token length plots.
    save_data_plots(
        dataset,
        tokenized_dataset,
        "labels",
        DATA_PLOTS_DIR,
    )

    dataloaders = create_dataloaders(
        tokenized_dataset,
        tokenizer,
        batch_size,
    )

    num_labels = (
        tokenized_dataset["train"]
        .features["labels"]
        .num_classes
    )

    return (
        dataset,
        tokenized_dataset,
        tokenizer,
        dataloaders,
        num_labels,
    )


if __name__ == "__main__":
    from config import (
        BATCH_SIZE,
        CHECKPOINT,
        DATASET_CONFIG,
        DATASET_NAME,
        LABEL_COLUMN,
        TEXT_COLUMN,
    )

    # Prepare data and save summary plots.
    (
        dataset,
        tokenized_dataset,
        _,
        _,
        num_labels,
    ) = prepare_data(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        checkpoint=CHECKPOINT,
        batch_size=BATCH_SIZE,
    )

    # Basic dataset checks.
    print(f"num_labels: {num_labels}")

    print(
        f"label_names: "
        f"{dataset['train'].features['labels'].names}"
    )

    print(f"columns: {dataset['train'].column_names}")

    # Tokenization checks.
    print(
        f"tokenized_columns: "
        f"{tokenized_dataset['train'].column_names}"
    )

    print(
        f"max_token_len: "
        f"{max(len(input_ids) for input_ids in tokenized_dataset['train']['input_ids'])}"
    )

    # Split sizes.
    for split in dataset:
        print(
            f"{split}: {len(dataset[split])}"
        )
