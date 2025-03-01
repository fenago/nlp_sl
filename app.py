import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import re

# Download necessary NLTK resources (cached)
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_data()

# Load stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Custom tokenizer to avoid NLTK punkt issues
def custom_tokenize(text):
    return re.split(r'\W+', text)  # Splitting on non-word characters

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = custom_tokenize(text)  # Use regex-based tokenizer instead of word_tokenize
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load saved model and TF-IDF vectorizer
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Streamlit UI
st.title("Multiclass Text Classification")

user_input = st.text_area("Enter text to classify")

if st.button("Predict"):
    processed = preprocess_text(user_input)
    vectorized = tfidf.transform([processed])
    prediction = model.predict(vectorized)
    prediction_proba = model.predict_proba(vectorized)
    confidence = np.max(prediction_proba) * 100

    st.write(f"**Predicted label**: {prediction[0]}")
    st.write(f"**Confidence**: {confidence:.2f}%")
