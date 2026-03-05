# IMDB Movie Review Sentiment Analysis using Simple RNN

## Table of Contents

1. [Project Overview](#project-overview)
2. [What is Sentiment Analysis?](#what-is-sentiment-analysis)
3. [Theoretical Foundations](#theoretical-foundations)
   - [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
   - [Simple RNN Architecture](#simple-rnn-architecture)
   - [Word Embeddings](#word-embeddings)
   - [Binary Classification with Sigmoid](#binary-classification-with-sigmoid)
4. [Dataset: IMDB Movie Reviews](#dataset-imdb-movie-reviews)
5. [Step-by-Step Implementation](#step-by-step-implementation)
   - [Step 1: Loading the Dataset](#step-1-loading-the-dataset)
   - [Step 2: Understanding Word Encoding](#step-2-understanding-word-encoding)
   - [Step 3: Padding Sequences](#step-3-padding-sequences)
   - [Step 4: Building the Model](#step-4-building-the-model)
   - [Step 5: Compiling the Model](#step-5-compiling-the-model)
   - [Step 6: Training with Early Stopping](#step-6-training-with-early-stopping)
   - [Step 7: Saving and Loading the Model](#step-7-saving-and-loading-the-model)
   - [Step 8: Prediction Pipeline](#step-8-prediction-pipeline)
   - [Step 9: Streamlit Web Application](#step-9-streamlit-web-application)
6. [Understanding the Embedding Notebook](#understanding-the-embedding-notebook)
7. [Complete Project Workflow](#complete-project-workflow)
8. [Key Concepts Recap](#key-concepts-recap)
9. [Tech Stack](#tech-stack)

---

## Project Overview

This project performs **Sentiment Analysis** on IMDB movie reviews using a **Simple Recurrent Neural Network (Simple RNN)**. The goal is to classify a given movie review as either **Positive** or **Negative** based on the text content of the review. The trained model is deployed as an interactive web application using **Streamlit**.

---

## What is Sentiment Analysis?

Sentiment Analysis is a sub-field of **Natural Language Processing (NLP)** that focuses on identifying and extracting the emotional tone or opinion expressed in a piece of text. In our case:

- **Input**: A movie review written in English (e.g., *"This movie was fantastic! The acting was great."*)
- **Output**: A label — **Positive** or **Negative**

This is a **binary classification** problem where:
- Label `1` → Positive review
- Label `0` → Negative review

### Why is it useful?

- Businesses use sentiment analysis to gauge customer opinions from reviews, social media, and feedback.
- It helps automate the process of understanding large volumes of textual data.

---

## Theoretical Foundations

### Recurrent Neural Networks (RNN)

Traditional neural networks (like feedforward networks) treat each input independently. But text is **sequential** — the meaning of a word depends on the words that came before it. For example:

> *"The movie was **not** good"* vs *"The movie was good"*

The word "not" completely flips the sentiment. A standard neural network would miss this sequential dependency.

**RNNs solve this** by introducing a **hidden state** that acts as a memory. At each time step, the RNN:

1. Takes the current input (a word)
2. Combines it with the previous hidden state (memory of past words)
3. Produces a new hidden state and an output

Mathematically, for time step $t$:

$$h_t = \text{activation}(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

Where:
- $x_t$ = input at time step $t$ (word embedding)
- $h_{t-1}$ = hidden state from previous time step
- $W_{xh}$ = weight matrix for input
- $W_{hh}$ = weight matrix for hidden state (recurrent weights)
- $b_h$ = bias term

The final hidden state $h_T$ (after processing the entire review) captures the overall meaning of the sequence and is passed to a Dense layer for classification.

### Simple RNN Architecture

A **Simple RNN** (also called Vanilla RNN) is the most basic form of RNN. In this project, our model architecture is:

```
Input → Embedding Layer → SimpleRNN Layer → Dense Layer → Output
```

| Layer           | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| **Embedding**   | Converts integer-encoded words into dense vector representations        |
| **SimpleRNN**   | Processes the sequence of word vectors, maintaining a hidden state      |
| **Dense**       | Takes the final hidden state and outputs a probability (0 to 1)         |

#### Limitations of Simple RNN

Simple RNNs suffer from the **vanishing gradient problem** — during backpropagation through time (BPTT), gradients can shrink exponentially as they propagate back through many time steps, making it hard for the network to learn long-range dependencies. More advanced architectures like **LSTM** and **GRU** address this, but Simple RNN works well for learning foundational concepts.

### Word Embeddings

Computers don't understand text — they understand numbers. **Word Embeddings** convert words into dense, low-dimensional vectors that capture semantic meaning.

#### From Words to Numbers — The Pipeline:

1. **Vocabulary Index**: Each unique word in the IMDB dataset is assigned a unique integer (e.g., "movie" → 17, "great" → 84).

2. **One-Hot Encoding** (explored in `embedding.ipynb`): A naive approach where each word is represented as a vector of size `vocab_size` with a 1 at the word's index and 0s elsewhere. This is **sparse** and **inefficient** for large vocabularies.

3. **Word Embeddings** (used in the model): Instead of sparse one-hot vectors, an **Embedding Layer** maps each integer to a **dense vector** of fixed dimension (128 in our case). These vectors are **learned during training** — words with similar meanings end up with similar vectors.

   Example:
   ```
   "great"  → [0.25, -0.13, 0.78, ..., 0.42]   (128 dimensions)
   "fantastic" → [0.23, -0.11, 0.80, ..., 0.40]  (similar vector!)
   ```

The Embedding layer in Keras acts as a **lookup table** — given integer index `i`, it returns the `i`-th row of a trainable weight matrix of shape `(vocab_size, embedding_dim)`.

### Binary Classification with Sigmoid

The final Dense layer uses the **sigmoid activation function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

This squashes the output to a value between 0 and 1, which we interpret as:
- Output **> 0.5** → Positive sentiment
- Output **≤ 0.5** → Negative sentiment

---

## Dataset: IMDB Movie Reviews

The **IMDB dataset** is a benchmark dataset for binary sentiment classification, provided directly by Keras/TensorFlow.

| Property               | Detail                                |
|------------------------|---------------------------------------|
| **Total Reviews**      | 50,000                                |
| **Training Set**       | 25,000 reviews                        |
| **Test Set**           | 25,000 reviews                        |
| **Classes**            | 2 (Positive, Negative)                |
| **Class Distribution** | Balanced (50% positive, 50% negative) |
| **Format**             | Pre-tokenized as integer sequences    |

Each review is already encoded as a **list of integers**, where each integer represents a word (mapped via a word index dictionary). For example:

```python
[1, 14, 22, 16, 43, 530, 973, ...]  # Label: 1 (Positive)
```

The parameter `num_words=10000` limits the vocabulary to the **top 10,000 most frequent words**, replacing rarer words with an out-of-vocabulary token.

---

## Step-by-Step Implementation

### Step 1: Loading the Dataset

```python
from tensorflow.keras.datasets import imdb

max_features = 10000  # Vocabulary size (top 10,000 words)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
```

- `X_train` / `X_test`: Lists of integer-encoded reviews
- `y_train` / `y_test`: Binary labels (0 = Negative, 1 = Positive)
- Each review is a **variable-length** list of integers

### Step 2: Understanding Word Encoding

The IMDB dataset provides a **word index** — a dictionary mapping words to their integer IDs:

```python
word_index = imdb.get_word_index()
# Example: {'the': 1, 'and': 2, 'a': 3, ...}
```

To convert integer sequences back to human-readable text:

```python
reverse_word_index = {value: key for key, value in word_index.items()}
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in sample_review])
```

> **Why `i - 3`?** The first three indices are reserved:
> - `0` → Padding
> - `1` → Start of sequence
> - `2` → Unknown/out-of-vocabulary word
>
> So the actual word indices start from 3.

### Step 3: Padding Sequences

Reviews have different lengths, but neural networks require **fixed-size inputs**. We use **padding** to make all reviews the same length:

```python
from tensorflow.keras.preprocessing import sequence

max_len = 500  # Maximum review length
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)
```

- Reviews **shorter** than 500 words are **padded with zeros** at the beginning (pre-padding).
- Reviews **longer** than 500 words are **truncated** (earlier words are cut off).
- After padding, every review is a vector of exactly 500 integers.

### Step 4: Building the Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))  # Embedding Layer
model.add(SimpleRNN(128, activation='relu'))                     # SimpleRNN Layer
model.add(Dense(1, activation='sigmoid'))                        # Output Layer
```

**Layer-by-Layer Breakdown:**

| Layer | Configuration | Output Shape | Parameters | Description |
|-------|--------------|--------------|------------|-------------|
| **Embedding** | `input_dim=10000, output_dim=128, input_length=500` | `(batch, 500, 128)` | 1,280,000 | Converts each of the 500 word indices into a 128-dim dense vector |
| **SimpleRNN** | `units=128, activation='relu'` | `(batch, 128)` | 32,896 | Processes the sequence of 500 embeddings; outputs the last hidden state |
| **Dense** | `units=1, activation='sigmoid'` | `(batch, 1)` | 129 | Maps the 128-dim hidden state to a single probability value |

**Total trainable parameters ≈ 1,313,025**

> **Why ReLU in RNN?** The `relu` (Rectified Linear Unit) activation function ($f(x) = \max(0, x)$) helps mitigate the vanishing gradient problem to some extent compared to `tanh` (the default for SimpleRNN), though it can sometimes cause the "exploding gradient" issue.

### Step 5: Compiling the Model

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

| Component | Choice | Reason |
|-----------|--------|--------|
| **Optimizer** | Adam | Adaptive learning rate optimizer; efficient and widely used |
| **Loss Function** | Binary Crossentropy | Standard loss for binary classification problems |
| **Metric** | Accuracy | Percentage of correctly classified reviews |

**Binary Cross-Entropy Loss:**

$$L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

Where $y_i$ is the true label and $\hat{y}_i$ is the predicted probability.

### Step 6: Training with Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)
```

**Training Configuration:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `epochs` | 10 | Maximum 10 passes over the training data |
| `batch_size` | 32 | Process 32 reviews at a time |
| `validation_split` | 0.2 | Use 20% of training data for validation |
| `EarlyStopping` | patience=5 | Stop training if validation loss doesn't improve for 5 consecutive epochs |
| `restore_best_weights` | True | After stopping, revert model to the epoch with the lowest validation loss |

**Why Early Stopping?**

Without early stopping, the model may **overfit** — it memorizes the training data but performs poorly on unseen data. Early stopping monitors the validation loss and stops training when it starts to increase, preventing overfitting.

### Step 7: Saving and Loading the Model

**Saving** (after training):
```python
model.save('simple_rnn_imdb.h5')
```

**Loading** (for prediction):
```python
from tensorflow.keras.models import load_model
model = load_model('simple_rnn_imdb.h5')
```

The `.h5` file stores:
- Model architecture (layers, configurations)
- Trained weights and biases
- Optimizer state
- Compilation configuration

### Step 8: Prediction Pipeline

To classify a new review, we need to preprocess it the same way as the training data:

```python
def preprocess_text(text):
    words = text.lower().split()                              # 1. Lowercase + Tokenize
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2. Encode words to integers
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)  # 3. Pad to 500
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]
```

**Preprocessing Steps for New Reviews:**

1. **Lowercase & Split**: Convert text to lowercase and split into individual words.
2. **Integer Encoding**: Look up each word in the IMDB word index. Unknown words default to index `2` (OOV). Add `3` to align with the dataset's reserved indices.
3. **Padding**: Pad the sequence to exactly 500 integers.
4. **Predict**: Feed the padded array into the model to get a probability score.

### Step 9: Streamlit Web Application

The `main.py` file wraps everything into an interactive web app using **Streamlit**:

```python
import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
```

**To run the app:**
```bash
streamlit run main.py
```

This launches a local web server where users can type a review and get real-time sentiment predictions.

---

## Understanding the Embedding Notebook

The `embedding.ipynb` notebook is a **standalone tutorial** demonstrating how word embeddings work from scratch:

1. **Sample Sentences**: A small set of sentences is defined manually.
2. **One-Hot Encoding**: Each sentence is converted using `one_hot()`, which hashes words to integer indices within a vocabulary of size 10,000.
3. **Padding**: Sentences are padded to a uniform length of 8 using `pad_sequences()` with pre-padding.
4. **Embedding Layer**: An `Embedding(10000, 10, input_length=8)` layer is created. This maps each word integer to a 10-dimensional dense vector.
5. **Prediction**: The model's `predict()` method shows the learned embedding vectors for each padded sentence.

This notebook helps build intuition about how the Embedding layer transforms sparse integer representations into dense, meaningful vectors — the same concept used in the main IMDB model (but with `embedding_dim=128` instead of 10).

---

## Complete Project Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROJECT WORKFLOW                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. LOAD DATA (IMDB Dataset - 50K reviews)                         │
│     └──> 25K Train + 25K Test, already tokenized as integers       │
│                                                                     │
│  2. PREPROCESS                                                      │
│     ├──> Limit vocabulary to top 10,000 words                      │
│     └──> Pad/truncate all sequences to length 500                  │
│                                                                     │
│  3. BUILD MODEL                                                     │
│     ├──> Embedding(10000, 128) — word → dense vector               │
│     ├──> SimpleRNN(128, relu) — sequential processing              │
│     └──> Dense(1, sigmoid)  — binary output                        │
│                                                                     │
│  4. TRAIN MODEL                                                     │
│     ├──> Adam optimizer + Binary Crossentropy loss                 │
│     ├──> 80/20 train/validation split                              │
│     └──> Early stopping (patience=5)                               │
│                                                                     │
│  5. SAVE MODEL → simple_rnn_imdb.h5                                │
│                                                                     │
│  6. DEPLOY via Streamlit (main.py)                                 │
│     ├──> User enters review text                                   │
│     ├──> Text preprocessed (lowercase → encode → pad)              │
│     ├──> Model predicts sentiment probability                      │
│     └──> Display: Positive/Negative + confidence score             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts Recap

| Concept | What It Means in This Project |
|---------|-------------------------------|
| **Sentiment Analysis** | Classifying reviews as Positive or Negative |
| **RNN** | Neural network with memory for sequential data (text) |
| **Simple RNN** | Basic RNN variant used here; processes words one by one |
| **Word Embedding** | Dense vector representation of words learned during training |
| **Padding** | Making all input sequences the same length (500) |
| **Binary Crossentropy** | Loss function for binary (2-class) classification |
| **Sigmoid Activation** | Squashes output to [0, 1] for probability interpretation |
| **Early Stopping** | Prevents overfitting by stopping training when validation loss plateaus |
| **Streamlit** | Python framework for building interactive ML web apps |

---

## Tech Stack

| Tool / Library | Version | Purpose |
|----------------|---------|---------|
| Python | 3.x | Programming language |
| TensorFlow / Keras | 2.15.0 | Deep learning framework |
| NumPy | — | Numerical computations |
| Streamlit | — | Web application deployment |
| Pandas | — | Data manipulation |
| Scikit-learn | — | ML utilities |
| Matplotlib | — | Plotting and visualization |
| TensorBoard | — | Training visualization |

---

## File Structure

```
simple_rnn_imdb/
├── embedding.ipynb      # Tutorial: Word embeddings from scratch
├── simplernn.ipynb      # Main notebook: Model building & training
├── prediction.ipynb     # Notebook: Loading model & making predictions
├── main.py              # Streamlit web app for real-time predictions
├── simple_rnn_imdb.h5   # Saved trained model file
├── requirements.txt     # Python dependencies
└── README.md            # This documentation file
```

---

> **Summary**: This project takes raw movie reviews, converts them into fixed-length integer sequences, passes them through an Embedding layer to get dense word vectors, processes the sequence with a Simple RNN to capture context, and finally classifies the review as Positive or Negative using a sigmoid output layer. The model is trained with early stopping to avoid overfitting and deployed as a Streamlit web app for easy interaction.
