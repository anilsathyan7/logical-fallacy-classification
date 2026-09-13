import copy

import torch


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    accelerator,
    max_grad_norm,
):

    model.train()

    total_loss = 0.0
    total_examples = 0
    correct = 0

    for batch in dataloader:

        # -------------------------
        # Forward
        # -------------------------

        outputs = model(**batch)

        loss = outputs.loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss. "
                "Try lower learning_rate, smaller batch_size, "
                "or MIXED_PRECISION = 'no'."
            )

        predictions = outputs.logits.argmax(
            dim=-1
        )

        # -------------------------
        # Statistics
        # -------------------------

        batch_size = batch["labels"].size(0)

        total_loss += (
            loss.item() * batch_size
        )

        correct += (
            predictions == batch["labels"]
        ).sum().item()

        total_examples += batch_size

        # -------------------------
        # Backward
        # -------------------------

        accelerator.backward(loss)

        accelerator.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(
            set_to_none=True
        )

    epoch_loss = (
        total_loss / total_examples
    )

    epoch_accuracy = (
        correct / total_examples
    )

    return epoch_loss, epoch_accuracy


def train(
    model,
    train_dataloader,
    dev_dataloader,
    optimizer,
    lr_scheduler,
    accelerator,
    num_epochs,
    evaluate_fn,
    wandb_run,
    early_stopping_patience,
    max_grad_norm,
):

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "dev_loss": [],
        "dev_accuracy": [],
        "dev_macro_f1": [],
    }

    best_dev_macro_f1 = 0.0
    epochs_without_improvement = 0

    best_model_state = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(num_epochs):

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}"
        )

        print("-" * 40)

        # ====================================================
        # TRAIN
        # ====================================================

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            max_grad_norm=max_grad_norm,
        )

        # ====================================================
        # DEV
        # ====================================================

        (
            dev_loss,
            dev_accuracy,
            dev_macro_f1,
        ) = evaluate_fn(
            model,
            dev_dataloader,
        )

        # ====================================================
        # HISTORY
        # ====================================================

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["dev_loss"].append(
            dev_loss
        )

        history["dev_accuracy"].append(
            dev_accuracy
        )

        history["dev_macro_f1"].append(
            dev_macro_f1
        )

        if wandb_run is not None:

            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "dev_loss": dev_loss,
                    "dev_accuracy": dev_accuracy,
                    "dev_macro_f1": dev_macro_f1,
                }
            )

        # ====================================================
        # PRINT
        # ====================================================

        print(f"Train Loss:     {train_loss:.4f}")

        print(f"Train Accuracy: {train_accuracy:.4f}")

        print(f"Dev Loss:       {dev_loss:.4f}")

        print(f"Dev Accuracy:   {dev_accuracy:.4f}")

        print(f"Dev Macro-F1:   {dev_macro_f1:.4f}")

        # ====================================================
        # BEST MODEL
        # ====================================================

        if dev_macro_f1 > best_dev_macro_f1:

            best_dev_macro_f1 = dev_macro_f1
            epochs_without_improvement = 0

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            print("-> New best model")

        else:

            epochs_without_improvement += 1

            if (
                early_stopping_patience is not None
                and epochs_without_improvement
                >= early_stopping_patience
            ):

                print(
                    f"Early stopping at epoch "
                    f"{epoch + 1}"
                )

                break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_model_state
    )

    print(
        f"\nBest Dev Macro-F1: "
        f"{best_dev_macro_f1:.4f}"
    )

    return model, history
