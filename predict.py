"""Run inference with a locally trained fallacy classifier."""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import BATCH_SIZE


def predict(sentences, model_path, batch_size=BATCH_SIZE):
    """Predict the fallacy label and confidence for each sentence."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = inputs.to(device)
            probabilities = model(**inputs).logits.softmax(dim=-1)
            confidences, label_ids = probabilities.max(dim=-1)
            predictions.extend(
                {
                    "text": text,
                    "label": model.config.id2label[label_id.item()],
                    "confidence": confidence.item(),
                }
                for text, label_id, confidence in zip(batch, label_ids, confidences)
            )

    return predictions


if __name__ == "__main__":
    model_path = Path("checkpoints/best_model/8wfncto3")
    dataset_name = "hard_real_test"  # "easy_synthetic_test" or "hard_real_test"
    input_path = Path("datasets") / f"{dataset_name}.csv"
    examples = pd.read_csv(input_path)
    sentences = examples["text"].tolist()
    predictions = predict(sentences, model_path)

    for prediction in predictions:
        print(f"{prediction['label']} ({prediction['confidence']:.2%})")
        print(f"  {prediction['text']}\n")

    examples["predicted_label"] = [prediction["label"] for prediction in predictions]
    examples["confidence"] = [prediction["confidence"] for prediction in predictions]
    output_path = Path("results") / model_path.name / "predictions" / input_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
