# 📰 News Classification & Summarization using Transformers

An end-to-end NLP application that **classifies news articles into categories** and **generates abstractive summaries** using state-of-the-art Transformer models.  
The system is deployed as an interactive **Streamlit web app**.

---

## 🚀 Features

- 🔹 News category classification (Business, Politics, Sports, Tech, Entertainment)
- 🔹 Abstractive news summarization
- 🔹 Transformer-based models (DistilBERT & BART)
- 🔹 GPU support (PyTorch)
- 🔹 Interactive Streamlit interface
- 🔹 Clean, modular project structure

---

## 🧠 Models Used

### 1️⃣ News Classification
- **Model**: `distilbert-base-uncased`
- **Architecture**: Transformer encoder + classification head
- **Framework**: PyTorch + Hugging Face Transformers
- **Accuracy on test set**: **~95.6%**

### 2️⃣ News Summarization
- **Model**: `facebook/bart-large-cnn`
- **Type**: Encoder–Decoder Transformer
- **Task**: Abstractive summarization
- **Inference-only** (no fine-tuning)

---

## 🗂️ Project Structure
news_nlp_project/
│
├── app/
│ └── app.py # Streamlit application
│
├── src/
│ ├── data/ # Data loading & preprocessing
│ ├── models/ # Transformer models
│ ├── training/ # Training & evaluation scripts
│ ├── inference/ # Prediction pipeline
│ └── utils/
│
├── data/
│ ├── raw/ # Raw dataset
│ └── processed/ # Train / Val / Test splits
│
├── outputs/
│ └── models/ # Saved trained models
│
├── notebooks/ # Exploration notebooks
├── requirements.txt
└── README.md



---

## 📊 Dataset

- **Dataset**: BBC News Dataset
- **Text column**: `description`
- **Labels**: Business, Politics, Sports, Technology, Entertainment
- Dataset was cleaned, filtered, and split into train/validation/test sets.

---

## ⚙️ Setup Instructions

### 1️⃣ Create environment
```bash
conda create -n news-transformer python=3.10
conda activate news-transformer



## ⚙️ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ How to Run

### 🔹 Train the Classifier

```bash
python -m src.training.train_classifier
```

### 🔹 Evaluate the Model

```bash
python -m src.training.evaluate
```

### 🔹 Run the Streamlit App

```bash
streamlit run app/app.py
```

---

## 🖥️ Streamlit App

The web application allows users to:

* 📄 Paste a full news article
* 🏷️ Predict the article’s category
* 📝 Generate a concise, abstractive summary

---

## 📈 Results

* ✅ **Test Accuracy:** ~95.6%
* 📉 Low test loss, indicating strong generalization
* ✨ High‑quality summaries for long‑form news articles

---

## 🔮 Future Improvements

* Add confidence scores to predictions
* Introduce ROUGE evaluation for summaries
* Cache models for faster inference
* Deploy to Streamlit Cloud / Hugging Face Spaces
* Extend to multi‑label news classification

---

## 🛠️ Tech Stack

* Python 3.10
* PyTorch
* Hugging Face Transformers
* Scikit‑learn
* Pandas
* Streamlit

---

## 👤 Author

**Aman Natial**
GitHub: [Helios‑07](https://github.com/Helios-07)

