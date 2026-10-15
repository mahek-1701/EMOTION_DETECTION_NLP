# Emotion Detection and Sentiment Analysis from Text

An end-to-end Natural Language Processing (NLP) and Machine Learning system built with **Python**, **Scikit-Learn**, **NLTK**, and **Streamlit**.

This application classifies raw English text into **7 fine-grained emotion classes** (*Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral*) and **3 sentiment polarities** (*Positive, Negative, Neutral*), complete with real-time probability distributions, bulk CSV batch processing, and an interactive analytics dashboard.

---

## 🌟 Key Features

1. **Dual NLP Classification Engine**:
   - **Emotion Detection**: Predicts 7 emotional states with class confidence percentages.
   - **Sentiment Analysis**: Predicts broad polarity (Positive, Negative, Neutral) with calibrated probabilities.
2. **Real-Time Interactive Prediction**:
   - Horizontal probability distribution chart (Plotly) sorting all 7 emotions.
   - Dynamic emoji badges and color-coded result cards.
   - Preprocessing inspection drawer revealing cleaned tokens and lemmatization.
3. **Batch CSV Processing**:
   - Upload any CSV file with text comments/tweets/reviews.
   - Auto-detects text columns and processes hundreds of rows in seconds.
   - Exports enriched CSV reports with predicted emotions, sentiments, and confidence values.
4. **Analytics Intelligence Dashboard**:
   - Session KPI metrics: Total texts analyzed, top emotion, top sentiment, average confidence.
   - Interactive Plotly visualizations: Emotion distribution, sentiment breakdown, and confidence benchmarks.
5. **Robust NLP Pipeline**:
   - Contraction expansion (`don't` $\rightarrow$ `do not`).
   - Sentiment-preserving stopword filtering (retains critical negation words such as `not`, `no`, `never`).
   - Morphological lemmatization using NLTK WordNet.
   - N-gram TF-IDF vectorization with sublinear scaling.
6. **Viva-Ready College Guide**:
   - Built-in "About Project" tab covering core theory, mathematical formulations, and top viva questions & answers.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Web Interface** | Streamlit |
| **Machine Learning** | Scikit-Learn (Logistic Regression, TF-IDF Vectorizer) |
| **NLP & Linguistics** | NLTK (WordNet Lemmatizer, Tokenizers, Stopwords) |
| **Data Processing** | Pandas, NumPy |
| **Visualizations** | Plotly Express & Graph Objects, Matplotlib |
| **Model Persistence** | Joblib |

---

## 📁 Project Structure

```text
emotion-detection/
│
├── app.py                      # Main Streamlit web application
├── train_model.py              # Model training and artifact serialization script
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
│
├── models/                     # Serialized model and vectorizer artifacts (.pkl)
│   ├── emotion_model.pkl
│   ├── emotion_vectorizer.pkl
│   ├── sentiment_model.pkl
│   └── sentiment_vectorizer.pkl
│
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── preprocessing.py        # Text cleaning, lemmatization & stopword pipeline
│   └── predict.py              # Single and batch prediction inference functions
│
├── data/                       # Datasets & batch testing files
│   ├── dataset.csv             # Balanced 7-class training dataset
│   └── sample_batch.csv        # Ready-to-use batch testing sample
│
└── assets/                     # Project screenshots & visual assets
```

---

## 🚀 Quick Start Guide

Follow these exact steps in your Windows terminal (PowerShell or Command Prompt):

### 1. Create and Navigate to Project Directory
```powershell
cd C:\Users\pc\.gemini\antigravity\scratch\emotion-detection
```

### 2. Create a Python Virtual Environment
```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment
On Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

On Windows (Command Prompt `cmd`):
```cmd
venv\Scripts\activate.bat
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Train the Machine Learning Models
Execute the training script to generate the TF-IDF vectorizers and trained models in the `models/` directory:
```powershell
python train_model.py
```
*This step trains both Emotion and Sentiment classifiers and verifies accuracy.*

### 6. Run the Streamlit Application
```powershell
streamlit run app.py
```

### 7. Open the Application
Streamlit will automatically open your default browser at:
```text
http://localhost:8501
```

---

## ⚙️ How the Pipeline Works

```text
User Input Text
      │
      ▼
[1] Preprocessing (src/preprocessing.py)
    • Lowercasing
    • Contraction Expansion ("can't" -> "cannot")
    • Special Character & URL Removal
    • Tokenization
    • Sentiment-Preserving Stopword Filtering (Keeps "not", "never")
    • WordNet Lemmatization
      │
      ▼
[2] Feature Extraction (TfidfVectorizer)
    • Unigram (1-gram) and Bigram (2-gram) TF-IDF extraction
    • Sublinear term-frequency scaling
      │
      ▼
[3] Model Inference (src/predict.py)
    • Multinomial Logistic Regression with Softmax
    • Probability distribution calculation across all 7 emotions
      │
      ▼
[4] Results & Visualization (app.py)
    • Emotion Class & Confidence (%)
    • Sentiment Polarity (Positive / Negative / Neutral)
    • Plotly Horizontal Probability Bar Chart
    • Batch Table Export (CSV)
```

---

## 📊 Supported Emotion Classes

| Emotion | Emoji | Typical Triggers |
|---|---|---|
| **Joy** | 😊 | Success, happiness, love, celebration, accomplishments |
| **Sadness** | 😢 | Grief, heartbreak, failure, loneliness, depression |
| **Anger** | 😠 | Frustration, unfairness, betrayal, insults, hostility |
| **Fear** | 😨 | Terror, anxiety, dread, phobia, impending doom |
| **Surprise** | 😲 | Astonishment, unexpected events, wonders, shocks |
| **Disgust** | 🤢 | Revulsion, foul odor/taste, immorality, uncleanliness |
| **Neutral** | 😐 | Factual statements, schedules, definitions, scientific facts |

---

## 🧪 Testing Batch Analysis

A pre-configured test file is included at `data/sample_batch.csv`.
1. Navigate to the **📊 Batch Analysis** tab in the sidebar.
2. Click **Browse files** and select `data/sample_batch.csv`.
3. Click **🚀 Run Batch Prediction**.
4. Review the generated emotion and sentiment predictions.
5. Click **⬇️ Download Results CSV** to export the annotated data.

---

## 🎓 College Viva Preparation Guide

### Q1: Why use TF-IDF instead of simple Bag-of-Words (CountVectorizer)?
> **Answer:** Bag-of-Words merely counts word frequencies, heavily favoring common words. TF-IDF penalizes words that appear ubiquitously across all documents, boosting unique, sentiment-rich words that distinguish one emotion from another.

### Q2: Why did you preserve negation words in stopword removal?
> **Answer:** Removing words like "not" or "no" causes negation blindness. For example, *"I am not happy"* would become *"happy"*, incorrectly predicting **Joy** instead of **Sadness**. Our custom stopword filter preserves negation and intensity terms.

### Q3: What is the benefit of Logistic Regression for this task?
> **Answer:** Logistic Regression produces well-calibrated posterior probabilities using the Softmax function, allowing us to display true confidence percentages for every class. It is lightweight, trains in seconds on standard CPU hardware, and handles high-dimensional sparse TF-IDF feature matrices effectively.

### Q4: How could this project be improved in future work?
> **Answer:** By fine-tuning pre-trained Transformer architectures such as **BERT** or **RoBERTa**, which use self-attention mechanisms to model contextual semantics and sarcasm. Additionally, expanding to multilingual models and multimodal emotion analysis (audio + video).

---

## 📄 License
This project is developed for academic and educational purposes under the MIT License.
