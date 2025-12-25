import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "BBC news dataset.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "bbc_news_clean.csv"


valid_categories=[
    "business", "entertainment", "politics", "sports", "technology"
]


def clean_text(text:str)->str:
    """
    Minimal text cleaning.
    Transformers do NOT require heavy preprocessing.
    """

    if not isinstance(text,str):
        return ""
    
    text=text.strip()
    text=text.replace("\n", " ")
    text=" ".join(text.split())
    return text

def preprocess():
    print("Loading raw data...")
    df=pd.read_csv(RAW_DATA_PATH)

    # Drop useless index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Handle missing values
    df["description"] = df["description"].fillna("")
    df["tags"] = df["tags"].fillna("")

    # Extract category from first tag
    df["category"] = df["tags"].apply(
        lambda x: x.split(",")[0].strip().lower() if x else None
    )

    # Drop rows with invalid or missing category
    df = df[df["category"].isin(valid_categories)]

    # Clean text
    df["text"] = df["description"].apply(clean_text)

    # Remove very short texts
    df = df[df["text"].str.split().str.len() >= 20]

    # Encode labels
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["category"])

    # Keep only required columns
    df = df[["text", "category", "label"]]

    # Create processed directory if not exists
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Preprocessing complete")
    print(f"Saved cleaned data to: {PROCESSED_DATA_PATH}")
    print("\nLabel mapping:")
    for idx, cls in enumerate(encoder.classes_):
        print(f"{cls} -> {idx}")





if __name__=="__main__":
    preprocess()

