# IMDB Movie Review Sentiment Analysis using Simple RNN

A deep learning project that builds a **Simple RNN** model to classify IMDB movie reviews as **Positive** or **Negative**. It includes notebooks for understanding word embeddings, training the RNN model, running predictions, and a Streamlit web app for interactive use.

---

## Project Structure

| File | Description |
|------|-------------|
| `embedding.ipynb` | Demonstrates word embedding concepts (one-hot encoding, Keras Embedding layer) |
| `simplernn.ipynb` | Trains a Simple RNN model on the IMDB dataset for sentiment classification |
| `prediction.ipynb` | Loads the trained model and runs sentiment predictions on sample reviews |
| `main.py` | Streamlit web application for interactive sentiment analysis |
| `simple_rnn_imdb.h5` | Pre-trained Simple RNN model file |
| `test_model.py` | Test suite to verify model predictions on sample reviews |
| `requirements.txt` | Python dependencies |

---

## Notebooks Overview

### 1. `embedding.ipynb` — Word Embedding Basics

Introduces text representation techniques used as the foundation for the RNN model:

- **One-Hot Encoding** — Converts sample sentences into one-hot integer representations using `tensorflow.keras.preprocessing.text.one_hot` with a vocabulary size of 10,000.
- **Padding** — Pads sequences to a uniform length of 8 using `pad_sequences`.
- **Embedding Layer** — Builds a simple `Sequential` model with a Keras `Embedding` layer (10-dimensional embeddings) and visualizes the dense vector representations.

### 2. `simplernn.ipynb` — Model Training (End-to-End)

Full pipeline to train the sentiment analysis model:

1. **Data Loading** — Loads the IMDB dataset (`keras.datasets.imdb`) with a vocabulary size of 10,000.
2. **Data Inspection** — Decodes integer-encoded reviews back to text using a reverse word index.
3. **Preprocessing** — Pads all sequences to a max length of 500.
4. **Model Architecture:**
   - `Embedding` layer — 10,000 vocab, 32-dimensional embeddings
   - `SimpleRNN` layer — 32 units, ReLU activation
   - `Dense` output layer — 1 unit, Sigmoid activation (binary classification)
5. **Training** — Compiled with Adam optimizer and binary crossentropy loss. Trained for 10 epochs with batch size 32, 20% validation split, and `EarlyStopping` (patience=5, restore best weights).
6. **Model Saving** — Saves the trained model as `simple_rnn_imdb.h5`.

### 3. `prediction.ipynb` — Inference & Prediction

Loads the saved model and performs sentiment predictions:

- Loads `simple_rnn_imdb.h5` and the IMDB word index.
- **`decode_review()`** — Converts integer-encoded reviews back to human-readable text.
- **`preprocess_text()`** — Tokenizes and pads raw text input to match the model's expected format (max length 500).
- **`predict_sentiment()`** — Returns sentiment label (Positive/Negative) and prediction score.
- Includes example prediction on a sample review.

---

## Streamlit Web App (`main.py`)

An interactive web interface for real-time sentiment analysis:

- Enter any movie review in the text area.
- Click **Classify** to get the sentiment (Positive/Negative) and the prediction score.

### Run the App

```bash
streamlit run main.py
```

---

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd simple_rnn_imdb

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. **Explore embeddings:**
   Open and run `embedding.ipynb` to understand word embedding concepts.

2. **Train the model:**
   Open and run `simplernn.ipynb` to train the Simple RNN model (or use the provided `simple_rnn_imdb.h5`).

3. **Run predictions:**
   Open and run `prediction.ipynb` to test predictions on sample reviews.

4. **Launch the web app:**
   ```bash
   streamlit run main.py
   ```

---

## Dependencies

- TensorFlow 2.15.0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- TensorBoard
- SciKeras

---

## Model Summary

| Layer | Type | Output Shape | Details |
|-------|------|-------------|---------|
| 1 | Embedding | (None, 500, 32) | 10,000 vocab, 32-dim vectors |
| 2 | SimpleRNN | (None, 32) | 32 units, ReLU activation |
| 3 | Dense | (None, 1) | Sigmoid activation |

---

## License

This project is for educational purposes.
