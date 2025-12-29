import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from transformers import AutoTokenizer
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm

from src.models.classifier import NewsClassifier


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
VAL_PATH = BASE_DIR / "data" / "processed" / "val.csv"
MODEL_DIR = BASE_DIR / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 5
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5


#Device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")


#Dataset Class
class NewsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.df=pd.read_csv(csv_path)
        self.tokenizer=tokenizer
        self.max_len=max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        text=self.df.loc[idx, "text"]
        label=self.df.loc[idx, "label"]

        encoding=self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return{
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

#Training Function
def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = NewsDataset(TRAIN_PATH, tokenizer, MAX_LEN)
    val_dataset = NewsDataset(VAL_PATH, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = NewsClassifier(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Val loss:   {avg_val_loss:.4f}")
        print(f"Val acc:    {val_accuracy:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print("✅ Saved best model")


if __name__=="__main__":
    train()

