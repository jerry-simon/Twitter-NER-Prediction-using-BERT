import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Twitter Tag Prediction using BERT",
    layout="centered"
)

# ---------------------------------------------------
# Load model & tokenizer (cached for performance)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("JerrySimon/ner-bert-twitter")
    model = AutoModelForTokenClassification.from_pretrained("JerrySimon/ner-bert-twitter")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
id2label = model.config.id2label

# ---------------------------------------------------
# Preprocess function
# ---------------------------------------------------
def clean_and_tokenize(text):
    # Remove punctuation except @ and #, and split into words
    text = re.sub(r"[^\w\s@#]", "", text)
    return text.split()

# ---------------------------------------------------
# Prediction function
# ---------------------------------------------------
def predict_sentence(words):
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    word_ids = inputs.word_ids()

    results = []
    prev_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id:
            continue
        label = id2label[predictions[idx]]
        results.append((words[word_id], label))
        prev_word_id = word_id

    return results



# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("Twitter Tag Prediction using BERT")

tweet_text = st.text_area(
    "Enter Twitter text",
    placeholder="Type your tweet here..."
)

if st.button("Post"):
    if tweet_text.strip():
        words = clean_and_tokenize(tweet_text)
        
        if not words:
            st.warning("No valid tokens found after preprocessing.")
            st.stop()

        with st.spinner("Running BERT inference..."):
            results = predict_sentence(words)


        st.subheader("Predicted Named Entity Tags")
        for word, tag in results:
            st.markdown(f"**{word}** → `{tag}`")
    else:
        st.warning("Please enter some text.")
# ---------------------------------------------------