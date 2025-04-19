import os
import random
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------- STEP 1: Load IMDb Dataset --------
@st.cache_data  # Cache to avoid reloading
def load_imdb_dataset(data_path='aclImdb'):
    def load_reviews_from_folder(folder_path, label):
        reviews = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), encoding='utf-8') as file:
                    text = file.read().strip()
                    reviews.append((text, label))
        return reviews

    pos_train = load_reviews_from_folder(os.path.join(data_path, 'train/pos'), 1)
    neg_train = load_reviews_from_folder(os.path.join(data_path, 'train/neg'), 0)
    pos_test = load_reviews_from_folder(os.path.join(data_path, 'test/pos'), 1)
    neg_test = load_reviews_from_folder(os.path.join(data_path, 'test/neg'), 0)
    
    all_data = pos_train + neg_train + pos_test + neg_test
    random.shuffle(all_data)
    
    return pd.DataFrame(all_data, columns=['review', 'sentiment'])

# -------- STEP 2: Clean Text --------
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# -------- STEP 3: Load Model --------
@st.cache_resource  # Cache the model to avoid reloading
def load_sentiment_model():
    return pipeline("sentiment-analysis")

# -------- STEP 4: Predict Sentiment --------
def predict_sentiment(text, model):
    result = model(text[:512])[0]  # Trim to 512 tokens (BERT limit)
    return {
        "label": 1 if result['label'] == 'POSITIVE' else 0,
        "score": result['score']
    }

# -------- STREAMLIT UI --------
st.title("🎬 IMDb Sentiment Analysis")
st.write("Analyze movie reviews using Hugging Face's transformer model!")

# Sidebar controls
st.sidebar.header("Options")
sample_size = st.sidebar.slider("Number of IMDb reviews to evaluate", 100, 5000, 1000)
show_raw_data = st.sidebar.checkbox("Show raw data")

# Load data and model
df = load_imdb_dataset()
model = load_sentiment_model()

# Tab layout
tab1, tab2 = st.tabs(["📊 Evaluate Model", "🔍 Test Custom Review"])

with tab1:
    # Evaluate on IMDb dataset
    st.subheader("Model Evaluation on IMDb Reviews")
    df_sample = df.head(sample_size).copy()
    
    # Predict sentiments
    df_sample['predicted'] = df_sample['review'].apply(lambda x: predict_sentiment(x, model)['label'])
    
    # Metrics
    accuracy = accuracy_score(df_sample['sentiment'], df_sample['predicted'])
    st.metric("Accuracy", f"{accuracy:.2%}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(df_sample['sentiment'], df_sample['predicted'])
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive']).plot(ax=ax, cmap='Blues')
    st.pyplot(fig)

    # Distribution Plot
    st.subheader("Predicted Sentiment Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_sample, x='predicted', ax=ax2)
    ax2.set_title("0 = Negative, 1 = Positive")
    st.pyplot(fig2)

    if show_raw_data:
        st.dataframe(df_sample[['review', 'sentiment', 'predicted']])

with tab2:
    # Test custom text
    st.subheader("Test Your Own Review")
    user_input = st.text_area("Enter a movie review:", "This movie was fantastic!")
    
    if st.button("Analyze Sentiment"):
        prediction = predict_sentiment(user_input, model)
        st.write("---")
        col1, col2 = st.columns(2)
        col1.metric("Predicted", "Positive" if prediction['label'] == 1 else "Negative")
        col2.metric("Confidence", f"{prediction['score']:.2%}")
        
        # Show cleaned text
        st.write("**Cleaned Text:**")
        st.code(clean_text(user_input))