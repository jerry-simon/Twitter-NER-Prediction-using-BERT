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
    # Remove punctuation except @ and #
    text = re.sub(r"[^\w\s@#]", "", text)

    tokens = text.split()

    # Normalize mentions and hashtags
    normalized_tokens = []
    for t in tokens:
        if t.startswith("@") or t.startswith("#"):
            normalized_tokens.append(t[1:])  # strip @ or #
        else:
            normalized_tokens.append(t)

    return normalized_tokens

# ---------------------------------------------------
# Prediction function
# ---------------------------------------------------
def predict_sentence(words, radio_status):
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

    token_results = []
    prev_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id:
            continue
        label = id2label[predictions[idx]]
        token_results.append((words[word_id], label))
        prev_word_id = word_id

    # ---------------------------------------
    # MODE 1: Single entity (token-level)
    # ---------------------------------------
    if radio_status == "Single entity":
        return token_results

    # ---------------------------------------
    # MODE 2: Complete entity (merged spans)
    # ---------------------------------------
    merged_entities = []
    current_entity = []
    current_label = None

    for word, label in token_results:
        if label.startswith("B-"):
            if current_entity:
                merged_entities.append(
                    (" ".join(current_entity), current_label)
                )
            current_entity = [word]
            current_label = label[2:]

        elif label.startswith("I-") and current_label == label[2:]:
            current_entity.append(word)

        else:
            if current_entity:
                merged_entities.append(
                    (" ".join(current_entity), current_label)
                )
                current_entity = []
                current_label = None

    # Flush last entity
    if current_entity:
        merged_entities.append(
            (" ".join(current_entity), current_label)
        )

    return merged_entities




# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("Twitter Tag Prediction using BERT")

tweet_text = st.text_area(
    "Enter Twitter text",
    placeholder="Type your tweet here..."
)

radio_status = st.radio(
    "Prediction mode",
    ("Single entity", "Complete entity"),
    horizontal=True
)

if st.button("Post"):
    if tweet_text.strip():
        words = clean_and_tokenize(tweet_text)
        results = predict_sentence(words, radio_status)

        st.subheader("Predicted Named Entity Tags")

        if radio_status == "Single entity":
            for word, tag in results:
                st.markdown(f"**{word}** → `{tag}`")
        else:
            for entity, tag in results:
                st.markdown(f"**{entity}** → `{tag}`")
    else:
        st.warning("Please enter some text.")
# ---------------------------------------------------