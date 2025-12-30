import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm

from src.models.classifier import NewsClassifier


BASE_DIR = Path(__file__).resolve().parents[2]

TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_model.pt"

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 5
MAX_LEN = 256
BATCH_SIZE = 8


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
            max_length=self.max_len,
            padding="max_length",
            return_tensors='pt'
        )

        return{
            "input_ids":encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "labels":torch.tensor(label, dtype=torch.long)
        }
    

def evaluate():
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

    test_dataset=NewsDataset(TEST_PATH,tokenizer,MAX_LEN)
    test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE)

    model=NewsClassifier(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    criterion=nn.CrossEntropyLoss()

    total_loss=0
    correct=0
    total=0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)

            logits=model(input_ids,attention_mask)
            loss=criterion(logits,labels)

            total_loss+=loss.item()

            pred=torch.argmax(logits, dim=1)
            correct+=(pred==labels).sum().item()
            total+=labels.size(0)

        avg_loss=total_loss/len(test_loader)
        accuracy=correct/total

        print("\nTest Results")
        print("-" * 30)
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")


if __name__=="__main__":
    evaluate()
