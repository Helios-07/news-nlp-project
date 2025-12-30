import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path


MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_LEN = 1024
MAX_SUMMARY_LEN = 150
MIN_SUMMARY_LEN = 40


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()


def summarize(text:str)->str:
    if not isinstance(text,str) or len(text.strip())==0:
        return ""
    

    inputs=tokenizer(
        text,
        truncation=True,
        max_length=MAX_INPUT_LEN,
        return_tensors='pt'
    )

    input_ids=inputs.input_ids.to(device)
    attention_mask=inputs.attention_mask.to(device)


    #Generate Summary
    summary_ids=model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=MAX_SUMMARY_LEN,
        min_length=MIN_SUMMARY_LEN,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )


    #Decode Summary
    summary=tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return summary



if __name__=="__main__":
    article = """
    The government announced a series of economic reforms aimed at boosting
    foreign investment and strengthening the country's financial sector.
    Experts believe these measures could lead to long-term growth, although
    some concerns remain about short-term market volatility.
    """

    print("SUMMARY:\n")
    print(summarize(article))