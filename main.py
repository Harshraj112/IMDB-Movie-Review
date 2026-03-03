# Step 1: Import Libraries and Load the Model
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Step 2: Load Resources
# Load the IMDB dataset word index
@st.cache_resource
def load_resources():
    """Load word index and model once, cached across reruns."""
    word_idx = imdb.get_word_index()
    reverse_word_idx = {value: key for key, value in word_idx.items()}

    model_path = os.path.join(os.path.dirname(__file__), 'simple_rnn_imdb.h5')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    rnn_model = load_model(model_path)
    return word_idx, reverse_word_idx, rnn_model

word_index, reverse_word_index, model = load_resources()

# Step 3: Helper Functions
# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Step 4: Streamlit App UI
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if not user_input or not user_input.strip():
        st.warning('Please enter a non-empty movie review.')
    else:
        try:
            preprocessed_input = preprocess_text(user_input)

            # Make prediction
            prediction = model.predict(preprocessed_input)
            score = float(prediction[0][0])
            sentiment = 'Positive' if score > 0.5 else 'Negative'

            # Display the result
            st.write(f'Sentiment: {sentiment}')
            st.write(f'Prediction Score: {score:.4f}')
        except Exception as e:
            st.error(f'Error during prediction: {e}')
else:
    st.write('Please enter a movie review.')

