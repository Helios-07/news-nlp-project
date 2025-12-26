import torch
import torch.nn as nn
from transformers import AutoModel


class NewsClassifier(nn.Module):
    def __init__(self, model_name:str, num_labels:int, dropout: float=0.1):
        super().__init__()

        # Pretrained Transformer Encoder
        self.encoder=AutoModel.from_pretrained(model_name)
        hidden_size=self.encoder.config.hidden_size

         #Classification Head
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(hidden_size, num_labels)

    

    def forward(self, input_ids, attention_mask):
        #Forward pass through the encoder
        outputs=self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        pooled_output=outputs.last_hidden_state[:,0,:]

        # Classification
        pooled_output=self.dropout(pooled_output)
        logits=self.classifier(pooled_output)

        return logits
    