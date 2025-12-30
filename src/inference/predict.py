import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 5
MAX_LEN = 256

LABEL_MAP = {
    0: "business",
    1: "entertainment",
    2: "politics",
    3: "sports",
    4: "technology"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

model.to(device)
model.eval()

# Prediction function
def predict(text: str) -> str:
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    return LABEL_MAP[pred]


if __name__ == "__main__":
    sample_text = """
    The government announced new economic reforms aimed at boosting
    foreign investment and stabilizing the financial markets.
    """

    category = predict(sample_text)
    print(f"Predicted Category: {category}")
