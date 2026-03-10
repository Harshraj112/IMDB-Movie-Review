# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN

# Patch SimpleRNN to handle legacy kwargs from TF 2.15 saved models
_original_simple_rnn_init = SimpleRNN.__init__
def _patched_simple_rnn_init(self, *args, **kwargs):
    kwargs.pop('time_major', None)
    _original_simple_rnn_init(self, *args, **kwargs)
SimpleRNN.__init__ = _patched_simple_rnn_init

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Use st.cache_resource to load the model only once
import streamlit as st
@st.cache_resource
def load_my_model():
    st.write("Loading model...")
    model = load_model('simple_rnn_imdb.h5')
    st.write("Model loaded.")
    return model

model = load_my_model()

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

# Ensure session state for result
if "result" not in st.session_state:
    st.session_state.result = None

if st.button('Classify'):
    st.write("Input:", user_input)
    preprocessed_input = preprocess_text(user_input)
    st.write("Preprocessed input shape:", preprocessed_input.shape)

    # Make prediction
    prediction = model.predict(preprocessed_input)
    st.write("Raw prediction:", prediction)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Store and display the result
    st.session_state.result = (sentiment, float(prediction[0][0]))
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

