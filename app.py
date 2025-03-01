import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
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
