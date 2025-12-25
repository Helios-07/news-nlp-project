import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "bbc_news_clean.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = OUTPUT_DIR / "train.csv"
VAL_PATH = OUTPUT_DIR / "val.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"


TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

def split_data():
    print("Loading cleaned data...")
    df=pd.read_csv(CLEAN_DATA_PATH)


    # First split: Train + Temp
    train_df, temp_df=train_test_split(df, test_size=(TEST_SIZE+VAL_SIZE),
                                       stratify=df['label'],
                                       random_state=RANDOM_STATE)
    

    # Second split: Validation + Test
    val_df, test_df=train_test_split(
        temp_df,
        test_size=TEST_SIZE/(TEST_SIZE+VAL_SIZE),
        stratify=temp_df['label'],
        random_state=RANDOM_STATE
    )


    # Save splits
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Data split completed:")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")



if __name__=="__main__":
    split_data()