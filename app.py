import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
from streamlit_option_menu import option_menu

# Load dataset
dataset = pd.read_csv("dataset/labeled_data.csv")
dataset["labels"] = dataset["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate or Offensive Language"})

# Text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def clean_text(text):
    text = str(text).lower()
    text = re.sub("https?://\\S+|www\\.S+", " ", text)
    text = re.sub("\\[.*?\\]", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\\n", "", text)
    text = re.sub("\\w*\\d\\w*", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

dataset["tweet"] = dataset["tweet"].apply(clean_text)

# Feature extraction
X = dataset["tweet"].values
y = dataset["labels"].values
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model & vectorizer
joblib.dump(model, "hate_speech_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Hate Speech Detection", page_icon="🛑")

# Sidebar navigation
# st.sidebar.title("")
# page = st.sidebar.radio("Go to", ["Home", "Hate Speech Detection"])
with st.sidebar:
    page = option_menu(
        menu_title="Main Menu",
        options=["Home", "Hate Speech Detection"],
        icons=["house", "robot"],
        menu_icon="cast",
        default_index=0,
    )

if page == "Home":
    st.title("Welcome to Detecting Hate Speech and Offensive Language ")
    st.write("This tool helps detect hate speech and offensive language in text.")
    st.write("Navigate to 'Hate Speech Detection' to test the model.")
    st.write("Developed By Subramanyam Rekhandar")
    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: subramanyam@namunahai.com')
    st.write('LinkedIn: [Subramanyam Rekhandar](https://www.linkedin.com/in/subramanyamrekhandar/)')

elif page == "Hate Speech Detection":
    st.title("Detecting Hate Speech and Offensive Language")
    user_input = st.text_area("Enter text to analyze:")
    if st.button("Predict"):
        if user_input:
            # Load model & vectorizer
            model = joblib.load("hate_speech_model.pkl")
            vectorizer = joblib.load("vectorizer.pkl")
            
            # Preprocess input
            cleaned_text = clean_text(user_input)
            vectorized_text = vectorizer.transform([cleaned_text])
            
            # Predict
            prediction = model.predict(vectorized_text)[0]
            st.write(f"Prediction: **{prediction}**")
        else:
            st.warning("Please enter text to analyze.")
