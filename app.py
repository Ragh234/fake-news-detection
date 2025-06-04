!pip install streamlit
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detector")
st.write("Enter a news article or headline to check whether it's **Real or Fake**.")

user_input = st.text_area("Paste your news content here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)
        result = "🚨 Fake News!" if prediction[0] == 1 else "✅ Real News"
        st.success(f"Prediction: **{result}**")
