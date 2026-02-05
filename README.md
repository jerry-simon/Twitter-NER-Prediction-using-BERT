# 🐦 Twitter Tag Prediction using BERT

This project implements a Named Entity Recognition (NER) system for Twitter data using a fine-tuned BERT model.
It is designed to handle noisy, real-world social media text (mentions, hashtags, informal language) and is deployed as an interactive Streamlit web application.

🚀 Project Overview

Task: Named Entity Recognition (NER)

Domain: Twitter / Social Media text

Model: bert-base-uncased fine-tuned for token classification

Frameworks: PyTorch, Hugging Face Transformers

Deployment: Streamlit Cloud

Model Hosting: Hugging Face Hub

The application allows users to paste raw tweet text and receive entity-level predictions, such as:

person

geo-loc

facility

product

other

🧠 Key Features

✅ Handles hashtags (#) and mentions (@) gracefully

✅ Supports token-level (B/I tags) and merged entity-level outputs

✅ Uses BIO tagging scheme

✅ Model loaded efficiently using Streamlit caching

✅ Production-ready inference pipeline

✅ Model versioned and hosted on Hugging Face Hub

🏗️ Architecture
User Input (Tweet Text)
        ↓
Text Cleaning & Tokenisation
        ↓
BERT Token Classification Model
        ↓
Post-processing (BIO Tag Handling)
        ↓
NER Output (Single / Complete Entity Mode)

🧪 Example

Input

@Joey Have you listened to the new song by #Justin Bieber?


Output (Complete Entity Mode)

Justin Bieber → PERSON

📂 Repository Structure
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Ignored files & folders
├── huggingface_upload.ipynb   # Model upload notebook
└── data/
    ├── wnut16.txt.conll
    └── wnut16test.txt.conll


⚠️ Large model artefacts are not stored in GitHub and are fetched dynamically from Hugging Face Hub.

🔗 Model Details

Model Name: JerrySimon/ner-bert-twitter

Hosted On: Hugging Face Hub

Format: safetensors

Inference Only (no training on Streamlit)

🖥️ Running Locally
1️⃣ Clone the repository
git clone https://github.com/jerry-simon/Twitter-NER-Prediction-using-BERT.git
cd Twitter-NER-Prediction-using-BERT

2️⃣ Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit app
streamlit run app.py

☁️ Deployment

The application is deployed using Streamlit Cloud, and the model is fetched directly from Hugging Face Hub, avoiding GitHub size limitations.

📌 Why This Project Matters

Demonstrates end-to-end ML system design

Shows real-world NLP handling (social media noise)

Covers training → model versioning → deployment

Aligns with industry-grade ML & MLOps practices

📈 Future Improvements

Add confidence scores per entity

Support batch tweet processing

Add entity visualisation (colour-coded spans)

Extend to multilingual Twitter NER

👤 Author

Jerry Simon
Data Scientist | Machine Learning Engineer | NLP Enthusiast

🔗 LinkedIn: https://www.linkedin.com/in/jerry-simon-v/
🔗 Hugging Face: https://huggingface.co/JerrySimon
