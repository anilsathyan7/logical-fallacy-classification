"""Run fallacy-classifier training and W&B experiment tracking."""

from pathlib import Path

import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINT,
    DATASET_CONFIG,
    DATASET_NAME,
    EARLY_STOPPING_PATIENCE,
    LABEL_COLUMN,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MIXED_PRECISION,
    NUM_EPOCHS,
    RUN_TEST,
    SEED,
    TEXT_COLUMN,
    WANDB_MODE,
    WANDB_PROJECT,
    WEIGHT_DECAY,
    WARMUP_RATIO,
)
from data import prepare_data
from metrics import evaluate_model
from model import create_model, create_optimizer, create_scheduler
from train import train


def main():
    """Run the fallacy classifier training pipeline."""

    wandb_kwargs = {
        "project": WANDB_PROJECT,
        "config": {
            "checkpoint": CHECKPOINT,
            "dataset_name": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "text_column": TEXT_COLUMN,
            "label_column": LABEL_COLUMN,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "seed": SEED,
            "max_grad_norm": MAX_GRAD_NORM,
            "mixed_precision": MIXED_PRECISION,
            "run_test": RUN_TEST,
        },
    }

    if WANDB_MODE is not None:
        wandb_kwargs["mode"] = WANDB_MODE

    run = wandb.init(**wandb_kwargs)

    cfg = run.config

    checkpoint = cfg["checkpoint"]
    dataset_name = cfg["dataset_name"]
    dataset_config = cfg["dataset_config"]
    text_column = cfg["text_column"]
    label_column = cfg["label_column"]
    batch_size = int(cfg["batch_size"])
    num_epochs = int(cfg["num_epochs"])
    learning_rate = float(cfg["learning_rate"])
    weight_decay = float(cfg["weight_decay"])
    warmup_ratio = float(cfg["warmup_ratio"])
    early_stopping_patience = cfg["early_stopping_patience"]
    seed = int(cfg["seed"])
    max_grad_norm = float(cfg["max_grad_norm"])
    mixed_precision = cfg["mixed_precision"]
    run_test = bool(cfg["run_test"])

    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)

    set_seed(seed)

    try:

        # ====================================================
        # 1. DATA
        # ====================================================

        (
            dataset,
            tokenized_dataset,
            tokenizer,
            dataloaders,
            num_labels,
        ) = prepare_data(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            text_column=text_column,
            label_column=label_column,
            checkpoint=checkpoint,
            batch_size=batch_size,
        )

        train_dataloader, validation_dataloader, test_dataloader = dataloaders

        print(f"Number of labels: {num_labels}")
        print(f"Train size: {len(dataset['train'])}")
        print(f"Validation size: {len(dataset['validation'])}")
        print(f"Test size: {len(dataset['test'])}")

        run.summary["num_labels"] = num_labels
        run.summary["train_size"] = len(dataset["train"])
        run.summary["validation_size"] = len(dataset["validation"])
        run.summary["test_size"] = len(dataset["test"])

        label_names = dataset["train"].features["labels"].names
        id2label = {label_id: label for label_id, label in enumerate(label_names)}
        label2id = {label: label_id for label_id, label in id2label.items()}

        run.summary["label_names"] = label_names

        # ====================================================
        # 2. MODEL
        # ====================================================

        model = create_model(
            checkpoint=checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # ====================================================
        # 3. OPTIMIZER
        # ====================================================

        optimizer = create_optimizer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # ====================================================
        # 4. SCHEDULER
        # ====================================================

        num_training_steps = num_epochs * len(train_dataloader)

        lr_scheduler = create_scheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            warmup_ratio=warmup_ratio,
        )

        # ====================================================
        # 5. ACCELERATE
        # ====================================================

        accelerator = Accelerator(mixed_precision=mixed_precision)

        (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
            model,
            optimizer,
            lr_scheduler,
        ) = accelerator.prepare(
            train_dataloader,
            validation_dataloader,
            test_dataloader,
            model,
            optimizer,
            lr_scheduler,
        )

        # ====================================================
        # 6. TRAIN
        # ====================================================

        model, history = train(
            model=model,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            num_epochs=num_epochs,
            evaluate_fn=evaluate_model,
            wandb_run=run,
            early_stopping_patience=early_stopping_patience,
            max_grad_norm=max_grad_norm,
        )

        run.summary["best_validation_macro_f1"] = max(history["validation_macro_f1"])

        # ====================================================
        # 7. FINAL TEST
        # ====================================================

        if run_test:
            test_loss, test_accuracy, test_macro_f1 = evaluate_model(model, test_dataloader)

            run.log(
                {
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "test_macro_f1": test_macro_f1,
                }
            )

            print("\n" + "=" * 40)
            print("FINAL TEST RESULTS")
            print("=" * 40)

            print(f"Test Loss:      {test_loss:.4f}")
            print(f"Test Accuracy:  {test_accuracy:.4f}")
            print(f"Test Macro-F1:  {test_macro_f1:.4f}")

        # ====================================================
        # 8. SAVE
        # ====================================================

        model_path = Path(BEST_MODEL_PATH) / run.id

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        run.summary["model_path"] = str(model_path)
        print(f"\nModel saved to: {model_path}")

    finally:

        run.finish()


if __name__ == "__main__":
    main()
