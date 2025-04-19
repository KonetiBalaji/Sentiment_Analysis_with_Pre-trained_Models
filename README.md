# IMDb Sentiment Analysis using Pre-trained Transformers

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project demonstrates real-time sentiment analysis on IMDb movie reviews using Hugging Face’s pre-trained transformer models. Users can either evaluate the model on a large dataset or enter custom reviews to test the sentiment prediction interactively via a Streamlit web interface.

The application uses the `distilbert-base-uncased-finetuned-sst-2-english` model for binary classification (positive or negative sentiment) without requiring any model training or fine-tuning.

---

## Features

- Analyze 100 to 5000 IMDb reviews with evaluation metrics and visualization.
- View accuracy, confusion matrix, and sentiment distribution plots.
- Test your own custom reviews interactively.
- Clean and modern UI built with Streamlit.

---

## Dataset

**IMDb Large Movie Review Dataset**  
Source: [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)  
- 50,000 reviews: 25,000 for training and 25,000 for testing
- Structured in folders: `aclImdb/train/pos`, `train/neg`, `test/pos`, `test/neg`
- Each review is a `.txt` file

---

## How to Run

### 1. Clone the Repository

```bash

### 2. Download the IMDb Dataset
Download and extract the dataset into the root directory: https://ai.stanford.edu/~amaas/data/sentiment/

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Launch the Streamlit App
streamlit run sentiment_analysis_pipeline.py

### 5. File Structure
imdb-sentiment-analysis/
│
├── aclImdb/                        # Extracted IMDb dataset
├── sentiment_analysis_pipeline.py # Complete Streamlit app
├── requirements.txt               # Project dependencies
├── LICENSE                        # MIT License
└── README.md                      # Project documentation

## Future Scope
Extend to multi-class emotion classification (e.g., joy, anger, sadness)
Add model fine-tuning on domain-specific data
Deploy the app to Streamlit Cloud or Hugging Face Spaces
Add support for multi-language sentiment classification
Add charts for confidence score distributions