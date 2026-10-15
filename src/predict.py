"""
Prediction Module for Emotion Detection and Sentiment Analysis.

Provides inference functions for:
- Single text emotion classification with confidence scores and class probability distributions.
- Single text sentiment classification with confidence scores and probabilities.
- Batch processing for DataFrames (CSV batch analysis).
"""

import os
from typing import Dict, Any, Tuple, Optional
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import preprocess_text

# Default paths to model artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.pkl")
EMOTION_VEC_PATH = os.path.join(MODELS_DIR, "emotion_vectorizer.pkl")
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
SENTIMENT_VEC_PATH = os.path.join(MODELS_DIR, "sentiment_vectorizer.pkl")


def load_artifacts():
    """
    Load all trained models and TF-IDF vectorizers from disk.
    
    Returns:
        tuple: (emotion_model, emotion_vectorizer, sentiment_model, sentiment_vectorizer)
    
    Raises:
        FileNotFoundError: If any model or vectorizer artifact is missing.
    """
    required_files = [
        ("Emotion Model", EMOTION_MODEL_PATH),
        ("Emotion Vectorizer", EMOTION_VEC_PATH),
        ("Sentiment Model", SENTIMENT_MODEL_PATH),
        ("Sentiment Vectorizer", SENTIMENT_VEC_PATH),
    ]

    for name, path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {name} artifact at '{path}'. Please run 'python train_model.py' first."
            )

    emotion_model = joblib.load(EMOTION_MODEL_PATH)
    emotion_vectorizer = joblib.load(EMOTION_VEC_PATH)
    sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
    sentiment_vectorizer = joblib.load(SENTIMENT_VEC_PATH)

    return emotion_model, emotion_vectorizer, sentiment_model, sentiment_vectorizer


def predict_emotion(
    text: str,
    model=None,
    vectorizer=None
) -> Dict[str, Any]:
    """
    Predict the emotion of a given text string.

    Args:
        text (str): Input raw text.
        model: Trained scikit-learn emotion classifier (loaded if None).
        vectorizer: Trained TF-IDF vectorizer for emotion (loaded if None).

    Returns:
        dict: {
            "text": str,
            "cleaned_text": str,
            "emotion": str,
            "confidence": float (percentage, 0.0 - 100.0),
            "probabilities": dict {class_name: percentage_float}
        }
    """
    if model is None or vectorizer is None:
        model = joblib.load(EMOTION_MODEL_PATH)
        vectorizer = joblib.load(EMOTION_VEC_PATH)

    cleaned = preprocess_text(text)
    if not cleaned:
        # Fallback for empty or non-text input
        return {
            "text": text,
            "cleaned_text": "",
            "emotion": "Neutral",
            "confidence": 0.0,
            "probabilities": {cls: 0.0 for cls in getattr(model, "classes_", ["Neutral"])},
        }

    # Transform text using TF-IDF
    features = vectorizer.transform([cleaned])

    # Predict emotion class
    pred_label = model.predict(features)[0]

    # Calculate probabilities if model supports predict_proba
    probabilities = {}
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        proba_matrix = model.predict_proba(features)[0]
        classes = model.classes_
        for cls_name, prob in zip(classes, proba_matrix):
            probabilities[str(cls_name)] = round(float(prob) * 100.0, 2)
        
        # Sort probabilities in descending order
        probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        confidence = probabilities.get(str(pred_label), 0.0)
    else:
        # Fallback if probability prediction is unavailable
        confidence = 100.0
        probabilities = {str(pred_label): 100.0}

    return {
        "text": text,
        "cleaned_text": cleaned,
        "emotion": str(pred_label),
        "confidence": confidence,
        "probabilities": probabilities,
    }


def predict_sentiment(
    text: str,
    model=None,
    vectorizer=None
) -> Dict[str, Any]:
    """
    Predict the sentiment (Positive, Negative, Neutral) of a given text string.

    Args:
        text (str): Input raw text.
        model: Trained scikit-learn sentiment classifier (loaded if None).
        vectorizer: Trained TF-IDF vectorizer for sentiment (loaded if None).

    Returns:
        dict: {
            "text": str,
            "cleaned_text": str,
            "sentiment": str,
            "confidence": float (percentage, 0.0 - 100.0),
            "probabilities": dict {class_name: percentage_float}
        }
    """
    if model is None or vectorizer is None:
        model = joblib.load(SENTIMENT_MODEL_PATH)
        vectorizer = joblib.load(SENTIMENT_VEC_PATH)

    cleaned = preprocess_text(text)
    if not cleaned:
        return {
            "text": text,
            "cleaned_text": "",
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probabilities": {cls: 0.0 for cls in getattr(model, "classes_", ["Neutral"])},
        }

    features = vectorizer.transform([cleaned])
    pred_label = model.predict(features)[0]

    probabilities = {}
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        proba_matrix = model.predict_proba(features)[0]
        classes = model.classes_
        for cls_name, prob in zip(classes, proba_matrix):
            probabilities[str(cls_name)] = round(float(prob) * 100.0, 2)
        probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        confidence = probabilities.get(str(pred_label), 0.0)
    else:
        confidence = 100.0
        probabilities = {str(pred_label): 100.0}

    return {
        "text": text,
        "cleaned_text": cleaned,
        "sentiment": str(pred_label),
        "confidence": confidence,
        "probabilities": probabilities,
    }


def predict_batch(
    df: pd.DataFrame,
    text_column: str,
    emotion_model,
    emotion_vectorizer,
    sentiment_model,
    sentiment_vectorizer,
) -> pd.DataFrame:
    """
    Process an entire DataFrame column in batch mode.
    Adds 'emotion', 'sentiment', and 'confidence' columns.

    Args:
        df: Input pandas DataFrame.
        text_column: Name of the column containing raw texts.
        emotion_model: Loaded emotion model.
        emotion_vectorizer: Loaded emotion TF-IDF vectorizer.
        sentiment_model: Loaded sentiment model.
        sentiment_vectorizer: Loaded sentiment TF-IDF vectorizer.

    Returns:
        pd.DataFrame: Copy of DataFrame with added emotion, sentiment, confidence columns.
    """
    result_df = df.copy()

    # Preprocess all rows
    cleaned_texts = [preprocess_text(str(val)) for val in result_df[text_column]]

    # Emotion predictions
    emotion_features = emotion_vectorizer.transform(cleaned_texts)
    emotion_preds = emotion_model.predict(emotion_features)

    # Sentiment predictions
    sentiment_features = sentiment_vectorizer.transform(cleaned_texts)
    sentiment_preds = sentiment_model.predict(sentiment_features)

    # Probabilities & confidences
    emotion_confs = []
    if hasattr(emotion_model, "predict_proba"):
        prob_matrix = emotion_model.predict_proba(emotion_features)
        classes = list(emotion_model.classes_)
        for i, pred in enumerate(emotion_preds):
            cls_idx = classes.index(pred)
            conf = round(float(prob_matrix[i, cls_idx]) * 100.0, 2)
            emotion_confs.append(conf)
    else:
        emotion_confs = [100.0] * len(emotion_preds)

    sentiment_confs = []
    if hasattr(sentiment_model, "predict_proba"):
        prob_matrix_sent = sentiment_model.predict_proba(sentiment_features)
        classes_sent = list(sentiment_model.classes_)
        for i, pred in enumerate(sentiment_preds):
            cls_idx = classes_sent.index(pred)
            conf = round(float(prob_matrix_sent[i, cls_idx]) * 100.0, 2)
            sentiment_confs.append(conf)
    else:
        sentiment_confs = [100.0] * len(sentiment_preds)

    # Assign columns
    result_df["emotion"] = [str(e) for e in emotion_preds]
    result_df["sentiment"] = [str(s) for s in sentiment_preds]
    result_df["confidence"] = [f"{c:.1f}%" for c in emotion_confs]
    result_df["sentiment_confidence"] = [f"{c:.1f}%" for c in sentiment_confs]

    return result_df
