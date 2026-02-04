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
    tokenizer = AutoTokenizer.from_pretrained("ner_bert_model")
    model = AutoModelForTokenClassification.from_pretrained("ner_bert_model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
id2label = model.config.id2label

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
    previous_word_id = None

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == previous_word_id:
            continue

        label_id = predictions[token_idx]
        label = id2label[label_id]
        results.append((words[word_id], label))
        previous_word_id = word_id

    return results

def preprocess_tweet(tweet_text: str):
    tweet_text = tweet_text.strip()

    # Separate punctuation
    tweet_text = re.sub(r"([.,!?;:()\"'])", r" \1 ", tweet_text)

    # Normalize mentions and hashtags
    tweet_text = re.sub(r"([@#]\w+)", r" \1 ", tweet_text)

    # Remove extra spaces
    tweet_text = re.sub(r"\s+", " ", tweet_text)

     # Split
    words = tweet_text.split()

    # Remove standalone punctuation tokens
    words = [w for w in words if re.search(r"[A-Za-z0-9@#]", w)]

    return words


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("Twitter Tag Prediction using BERT")

tweet_text = st.text_area(
    "Enter Twitter text",
    placeholder="Type or paste a tweet here...",
    height=150
)

if st.button("Post"):
    if not tweet_text.strip():
        st.warning("Please enter some text.")
    else:
        # ---------------------------------------------------
        # Convert text → comma-separated → list of words
        # ---------------------------------------------------
        words = preprocess_tweet(tweet_text)

        # If user didn't enter commas, fall back to space split
        # if len(words) == 1:
        #     words = tweet_text.split()

        # ---------------------------------------------------
        # Run prediction
        # ---------------------------------------------------
        results = predict_sentence(words)

        # ---------------------------------------------------
        # Display results
        # ---------------------------------------------------
        st.subheader("Predicted Named Entity Tags")

        for word, tag in results:
            st.write(f"**{word}** → `{tag}`")