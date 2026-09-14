import evaluate
import torch


def evaluate_model(model, dataloader, return_predictions=False):
    """
    Evaluate the model on one dataloader.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation DataLoader.
        return_predictions: Whether to return labels and predictions.

    Returns:
        Loss, accuracy, and macro-F1, optionally followed by labels and
        predictions.
    """

    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_references = []
    all_predictions = []

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    with torch.inference_mode():

        for batch in dataloader:

            outputs = model(**batch)
            loss = outputs.loss

            # Stop early if evaluation becomes numerically unstable.
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite evaluation loss."
                )

            predictions = outputs.logits.argmax(dim=-1)

            if return_predictions:
                all_references.extend(batch["labels"].detach().cpu().tolist())
                all_predictions.extend(predictions.detach().cpu().tolist())

            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            accuracy_metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

            f1_metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

    loss = total_loss / total_examples
    accuracy = accuracy_metric.compute()["accuracy"]
    macro_f1 = f1_metric.compute(average="macro")["f1"]

    if return_predictions:
        return loss, accuracy, macro_f1, all_references, all_predictions

    return loss, accuracy, macro_f1
