import torch


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    accelerator,
    max_grad_norm,
):
    """
    Train the model for one epoch.

    Args:
        model: Model to train.
        dataloader: Training DataLoader.
        optimizer: Optimizer.
        lr_scheduler: Learning-rate scheduler.
        accelerator: Accelerate runtime.
        max_grad_norm: Gradient clipping norm.

    Returns:
        Average loss and accuracy for the epoch.
    """

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

        # Stop early if training becomes numerically unstable.
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss. "
                "Try lower learning_rate, smaller batch_size, "
                "or MIXED_PRECISION = 'no'."
            )

        predictions = outputs.logits.argmax(dim=-1)

        # -------------------------
        # Statistics
        # -------------------------

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == batch["labels"]).sum().item()
        total_examples += batch_size

        # -------------------------
        # Backward
        # -------------------------

        accelerator.backward(loss)

        # Limit gradient spikes before the optimizer update.
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    epoch_loss = total_loss / total_examples
    epoch_accuracy = correct / total_examples

    return epoch_loss, epoch_accuracy


def train(
    model,
    train_dataloader,
    validation_dataloader,
    optimizer,
    lr_scheduler,
    accelerator,
    num_epochs,
    evaluate_fn,
    wandb_run,
    early_stopping_patience,
    max_grad_norm,
):
    """
    Train the model and keep the best validation checkpoint.

    Args:
        model: Model to train.
        train_dataloader: Training DataLoader.
        validation_dataloader: Validation DataLoader.
        optimizer: Optimizer.
        lr_scheduler: Learning-rate scheduler.
        accelerator: Accelerate runtime.
        num_epochs: Maximum number of epochs.
        evaluate_fn: Evaluation function.
        wandb_run: Optional W&B run.
        early_stopping_patience: Epochs without improvement before stopping.
        max_grad_norm: Gradient clipping norm.

    Returns:
        Best model and training history.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "validation_macro_f1": [],
    }

    best_validation_macro_f1 = float("-inf")
    epochs_without_improvement = 0

    # Keep best weights on CPU to avoid extra GPU memory.
    best_model_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

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
        # VALIDATION
        # ====================================================

        (
            validation_loss,
            validation_accuracy,
            validation_macro_f1,
        ) = evaluate_fn(
            model,
            validation_dataloader,
        )

        # ====================================================
        # HISTORY
        # ====================================================

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["validation_macro_f1"].append(validation_macro_f1)

        # Log epoch metrics when W&B is enabled.
        if wandb_run is not None:

            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "validation_macro_f1": validation_macro_f1,
                }
            )

        # ====================================================
        # PRINT
        # ====================================================

        print(f"Train Loss:     {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")

        print(f"Validation Loss:       {validation_loss:.4f}")
        print(f"Validation Accuracy:   {validation_accuracy:.4f}")
        print(f"Validation Macro-F1:   {validation_macro_f1:.4f}")

        # ====================================================
        # BEST MODEL
        # ====================================================

        if validation_macro_f1 > best_validation_macro_f1:

            best_validation_macro_f1 = validation_macro_f1
            epochs_without_improvement = 0

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            # Select best model by validation Macro-F1.
            print(f"-> New best model ({best_validation_macro_f1:.4f})")

        else:

            epochs_without_improvement += 1

            # Early stopping, if no improvement for several epochs.
            if (
                early_stopping_patience is not None
                and epochs_without_improvement
                >= early_stopping_patience
            ):

                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best validation Macro-F1: "
                    f"{best_validation_macro_f1:.4f})"
                )

                break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(best_model_state)

    # Report metrics from the best validation epoch.
    best_epoch_index = max(
        range(len(history["validation_macro_f1"])),
        key=history["validation_macro_f1"].__getitem__,
    )

    print(f"\n\nBest Epoch: {best_epoch_index + 1}")
    print(f"Train Loss:     {history['train_loss'][best_epoch_index]:.4f}")
    print(f"Train Accuracy: {history['train_accuracy'][best_epoch_index]:.4f}")
    print(f"Validation Loss:       {history['validation_loss'][best_epoch_index]:.4f}")
    print(f"Validation Accuracy:   {history['validation_accuracy'][best_epoch_index]:.4f}")
    print(f"Validation Macro-F1:   {history['validation_macro_f1'][best_epoch_index]:.4f}")

    return model, history
