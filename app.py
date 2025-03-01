import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import os

# Manually set the NLTK data path
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Ensure required NLTK resources are downloaded
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

# Load NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
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
