"""
Test suite for IMDB Movie Review Sentiment Analysis model.

Runs the model on sample positive and negative reviews and verifies
the outputs are correct. Requires internet access for the IMDB word index.

Usage:
    python test_model.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


def load_test_resources():
    """Load word index and model for testing."""
    word_index = imdb.get_word_index()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simple_rnn_imdb.h5')
    model = load_model(model_path)
    return word_index, model


def preprocess_text(text, word_index):
    """Preprocess a review string into model input (matches main.py logic)."""
    max_features = 10000
    words = text.lower().split()
    encoded_review = []
    for word in words:
        index = word_index.get(word, None)
        if index is not None and (index + 3) < max_features:
            encoded_review.append(index + 3)
        else:
            encoded_review.append(2)  # OOV token
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(text, word_index, model):
    """Return (sentiment, score) for a review string."""
    preprocessed = preprocess_text(text, word_index)
    prediction = model.predict(preprocessed, verbose=0)
    score = float(prediction[0][0])
    sentiment = 'Positive' if score > 0.5 else 'Negative'
    return sentiment, score


def test_weights_not_nan(model):
    """Test that no model weights are NaN."""
    for i, w in enumerate(model.get_weights()):
        nan_count = np.isnan(w).sum()
        assert nan_count == 0, f"Layer {i} has {nan_count} NaN values"
    print("  PASS: All model weights are valid (no NaN)")


def test_predictions_valid(model):
    """Test that predictions are valid floats in [0, 1]."""
    inputs = np.random.randint(0, 10000, (10, 500)).astype('int32')
    preds = model.predict(inputs, verbose=0)
    for i, p in enumerate(preds):
        score = float(p[0])
        assert not np.isnan(score), f"Prediction {i} is NaN"
        assert 0.0 <= score <= 1.0, f"Prediction {i} = {score} out of [0,1]"
    print("  PASS: All predictions are valid floats in [0, 1]")


def test_predictions_vary(model):
    """Test that different inputs produce different outputs."""
    inputs = np.random.randint(0, 10000, (20, 500)).astype('int32')
    preds = model.predict(inputs, verbose=0)
    scores = set(round(float(p[0]), 6) for p in preds)
    assert len(scores) > 1, "All predictions are identical"
    print(f"  PASS: {len(scores)} unique scores from 20 inputs")


def test_positive_reviews(word_index, model):
    """Test that clearly positive reviews are classified as Positive."""
    positive_reviews = [
        "This movie was fantastic the acting was great and the plot was thrilling",
        "I loved this film it was the best movie i have ever seen wonderful acting",
        "A truly amazing movie with great performances and a beautiful story",
        "One of the best films of all time excellent direction and superb acting",
        "This is a masterpiece the story is incredible and the performances are outstanding",
    ]
    results = []
    for review in positive_reviews:
        sentiment, score = predict_sentiment(review, word_index, model)
        results.append((review[:60], sentiment, score))
        print(f"    '{review[:60]}...' -> {sentiment} ({score:.4f})")

    positive_count = sum(1 for _, s, _ in results if s == 'Positive')
    assert positive_count >= 4, (
        f"Only {positive_count}/5 positive reviews classified correctly"
    )
    print(f"  PASS: {positive_count}/5 positive reviews classified as Positive")


def test_negative_reviews(word_index, model):
    """Test that clearly negative reviews are classified as Negative."""
    negative_reviews = [
        "This movie was terrible the acting was awful and the plot made no sense",
        "I hated this film it was the worst movie i have ever seen horrible acting",
        "A truly bad movie with poor performances and a boring story waste of time",
        "One of the worst films ever terrible direction and awful writing",
        "This is garbage the story is stupid and the performances are embarrassing",
    ]
    results = []
    for review in negative_reviews:
        sentiment, score = predict_sentiment(review, word_index, model)
        results.append((review[:60], sentiment, score))
        print(f"    '{review[:60]}...' -> {sentiment} ({score:.4f})")

    negative_count = sum(1 for _, s, _ in results if s == 'Negative')
    assert negative_count >= 4, (
        f"Only {negative_count}/5 negative reviews classified correctly"
    )
    print(f"  PASS: {negative_count}/5 negative reviews classified as Negative")


def test_empty_and_edge_cases(word_index, model):
    """Test edge cases: single words, unknown words, etc."""
    # Single positive word
    _, score_good = predict_sentiment("good", word_index, model)
    assert not np.isnan(score_good), "NaN for single word 'good'"

    # Single negative word
    _, score_bad = predict_sentiment("bad", word_index, model)
    assert not np.isnan(score_bad), "NaN for single word 'bad'"

    # All unknown words (OOV)
    _, score_oov = predict_sentiment("xyzzy foobarbaz quuxquux", word_index, model)
    assert not np.isnan(score_oov), "NaN for all-OOV input"

    print(f"  PASS: Edge cases produce valid predictions")
    print(f"    'good' -> {score_good:.4f}, 'bad' -> {score_bad:.4f}, OOV -> {score_oov:.4f}")


def main():
    print("=" * 70)
    print("IMDB Movie Review Model — Test Suite")
    print("=" * 70)

    print("\nLoading resources...")
    try:
        word_index, model = load_test_resources()
    except Exception as e:
        print(f"FATAL: Could not load resources: {e}")
        sys.exit(1)
    print(f"  Model loaded: {model.count_params()} parameters")
    print(f"  Word index loaded: {len(word_index)} words")

    tests = [
        ("Model weights are valid", lambda: test_weights_not_nan(model)),
        ("Predictions are valid floats", lambda: test_predictions_valid(model)),
        ("Predictions vary across inputs", lambda: test_predictions_vary(model)),
        ("Positive reviews classified correctly", lambda: test_positive_reviews(word_index, model)),
        ("Negative reviews classified correctly", lambda: test_negative_reviews(word_index, model)),
        ("Edge cases handled", lambda: test_empty_and_edge_cases(word_index, model)),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
