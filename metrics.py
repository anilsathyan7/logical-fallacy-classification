import evaluate
import torch


def evaluate_model(model, dataloader):
    """
    Evaluate the model on one dataloader.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation DataLoader.

    Returns:
        Loss, accuracy, and macro-F1.
    """

    model.eval()

    total_loss = 0.0
    total_examples = 0

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    with torch.no_grad():

        for batch in dataloader:

            outputs = model(**batch)
            loss = outputs.loss

            # Stop early if evaluation becomes numerically unstable.
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite evaluation loss."
                )

            predictions = outputs.logits.argmax(dim=-1)

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

    return loss, accuracy, macro_f1
