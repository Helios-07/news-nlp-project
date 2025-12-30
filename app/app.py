import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


import streamlit as st
import torch
from src.inference.predict import predict
from src.models.summarizer import summarize


st.set_page_config(
    page_title="News NLP App",
    page_icon="📰",
    layout='centered'
)

st.title("📰 News Classification & Summarization")
st.write(
    "Paste a news article below to **classify** it and generate a **summary** using transformers."
)



#Text Input
article_text=st.text_area(
    "Enter News Article:",
    height=300,
    placeholder='Place a full news article here...'
)


#Analyze Button
if st.button("Analyze"):
    if article_text.strip()=="":
        st.warning("Please enter some text")

    else:
        with st.spinner("Running models..."):
            category=predict(article_text)
            summary=summarize(article_text)

        st.success('Analysis Complete!')



        #Display Results
        st.subheader("📌 Predicted Category")
        st.write(f"**{category.capitalize()}**")

        st.subheader("📝 Generated Summary")
        st.write(summary)