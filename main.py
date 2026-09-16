"""Run fallacy-classifier training and W&B experiment tracking."""

import gc
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

from embedding import save_umap_comparison_plot
from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINT,
    DATASET_CONFIG,
    DATASET_NAME,
    EARLY_STOPPING_PATIENCE,
    RESULTS_DIR,
    LABEL_COLUMN,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MIXED_PRECISION,
    NUM_EPOCHS,
    RUN_TEST,
    RUN_UMAP_ANALYSIS,
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
from utils import (
    save_classification_report,
    save_confusion_matrix,
)


def main(
    checkpoint=CHECKPOINT,
    dataset_name=DATASET_NAME,
    learning_rate=LEARNING_RATE,
    run_test=RUN_TEST,
):
    """Run the fallacy classifier training pipeline."""

    wandb_kwargs = {
        "project": WANDB_PROJECT,
        "config": {
            "checkpoint": checkpoint,
            "dataset_name": dataset_name,
            "dataset_config": DATASET_CONFIG,
            "text_column": TEXT_COLUMN,
            "label_column": LABEL_COLUMN,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "seed": SEED,
            "max_grad_norm": MAX_GRAD_NORM,
            "mixed_precision": MIXED_PRECISION,
            "run_test": run_test,
            "run_umap_analysis": RUN_UMAP_ANALYSIS,
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
    run_umap_analysis = bool(cfg["run_umap_analysis"])

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
            data_plots_dir=Path(RESULTS_DIR) / run.id / "data",
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

        model_path = Path(BEST_MODEL_PATH) / run.id
        evaluation_path = Path(RESULTS_DIR) / run.id / "evaluation"

        if run_test:
            (
                test_loss,
                test_accuracy,
                test_macro_f1,
                test_references,
                test_predictions,
            ) = evaluate_model(model, test_dataloader, return_predictions=True)

            report, report_path = save_classification_report(
                test_references,
                test_predictions,
                label_names,
                evaluation_path,
            )
            confusion_matrix_path = save_confusion_matrix(
                test_references,
                test_predictions,
                label_names,
                evaluation_path,
            )

            per_class_metrics = [
                [
                    label,
                    report[label]["precision"],
                    report[label]["recall"],
                    report[label]["f1-score"],
                    int(report[label]["support"]),
                ]
                for label in label_names
            ]

            run.log(
                {
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "test_macro_f1": test_macro_f1,
                    "test_per_class_metrics": wandb.Table(
                        columns=["label", "precision", "recall", "f1", "support"],
                        data=per_class_metrics,
                    ),
                    "test_confusion_matrix": wandb.Image(
                        str(confusion_matrix_path)
                    ),
                }
            )

            print("\n" + "=" * 40)
            print("FINAL TEST RESULTS")
            print("=" * 40)

            print(f"Test Loss:      {test_loss:.4f}")
            print(f"Test Accuracy:  {test_accuracy:.4f}")
            print(f"Test Macro-F1:  {test_macro_f1:.4f}")

            print("\nPer-Class Metrics")
            print(
                f"{'Label':<24} {'Precision':>9} {'Recall':>9} "
                f"{'F1':>9} {'Support':>9}"
            )
            for label, precision, recall, f1, support in per_class_metrics:
                print(
                    f"{label:<24} {precision:>9.4f} {recall:>9.4f} "
                    f"{f1:>9.4f} {support:>9}"
                )

            run.summary["classification_report_path"] = str(report_path)
            run.summary["confusion_matrix_path"] = str(confusion_matrix_path)

        # ====================================================
        # 8. SAVE
        # ====================================================

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        run.summary["model_path"] = str(model_path)
        print(f"\nModel saved to: {model_path}")

        # ====================================================
        # 9. UMAP ANALYSIS
        # ====================================================

        if run_umap_analysis:
            del model
            del optimizer
            del lr_scheduler
            del unwrapped_model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            umap_path = save_umap_comparison_plot(
                texts=dataset["test"][text_column],
                labels=dataset["test"]["labels"],
                label_names=label_names,
                original_model_path=checkpoint,
                fine_tuned_model_path=model_path,
                output_dir=evaluation_path,
                batch_size=batch_size,
                random_state=seed,
            )

            run.log(
                {
                    "test_embedding_umap": wandb.Image(str(umap_path)),
                }
            )
            run.summary["umap_path"] = str(umap_path)

    finally:

        run.finish()


if __name__ == "__main__":
    checkpoint = "checkpoints/best_model/8wfncto3"  # Original: "microsoft/deberta-v3-base"
    dataset_name = "datasets/cocolofa_touche_full.csv"  # Original: "kuwrom/fallacy"
    learning_rate = 2e-5
    run_test = True  # Use False while tuning on the validation split.
    main(
        checkpoint=checkpoint,
        dataset_name=dataset_name,
        learning_rate=learning_rate,
        run_test=run_test,
    )
