from torch.utils.data import DataLoader

from config import DATA_PLOTS_DIR
from datasets import ClassLabel, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)
from utils import save_data_plots


LABELS_COLUMN = "labels"


def load_data(
    dataset_name,
    dataset_config,
    label_column,
):
    """
    Load and prepare the logical fallacy dataset.
    """

    dataset = load_dataset(
        dataset_name,
        dataset_config,
    )

    # Keep the rest of the code using train/dev/test names.
    if (
        "dev" not in dataset
        and "validation" in dataset
    ):
        dataset["dev"] = dataset.pop(
            "validation"
        )

    label_feature = (
        dataset["train"]
        .features[label_column]
    )

    # Convert string labels -> ClassLabel when needed.
    if not isinstance(label_feature, ClassLabel):
        dataset = dataset.class_encode_column(
            label_column
        )

    # Transformers expects the label column to be named labels
    if label_column != LABELS_COLUMN:
        dataset = dataset.rename_column(
            label_column,
            LABELS_COLUMN
        )

    return dataset


def tokenize_data(
    dataset,
    checkpoint,
    text_column,
):
    """
    Tokenize the text data.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    def tokenize_function(example):
        return tokenizer(
            example[text_column],
            truncation=True,
        )

    columns_to_remove = [
        column
        for column in dataset["train"].column_names
        if column != LABELS_COLUMN
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
    Create train, dev and test DataLoaders.
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

    dev_dataloader = DataLoader(
        tokenized_dataset["dev"],
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
        dev_dataloader,
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

    save_data_plots(
        dataset,
        tokenized_dataset,
        LABELS_COLUMN,
        DATA_PLOTS_DIR,
    )

    dataloaders = create_dataloaders(
        tokenized_dataset,
        tokenizer,
        batch_size,
    )

    num_labels = (
        tokenized_dataset["train"]
        .features[LABELS_COLUMN]
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
        f"{dataset['train'].features[LABELS_COLUMN].names}"
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
