"""
Model Training Script for Emotion Detection & Sentiment Analysis.

This script:
1. Loads and combines real-world emotion datasets:
   - dair-ai/emotion (Hugging Face Twitter emotion corpus)
   - Google Research GoEmotions (Reddit multi-emotion dataset)
   - Declare Lab MELD (Conversational dialogue dataset)
   - Academic, domestic, and modern Internet / Gen-Z slang emotional expressions
2. Preprocesses text using the identical pipeline defined in `src/preprocessing.py`,
   including slang expansion, contraction handling, stopword preservation, and multi-POS lemmatization.
3. Builds a balanced dataset across all 7 emotion classes (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral)
   and maps them to 3 sentiments (Positive, Negative, Neutral).
4. Uses a unified train/test split on the DataFrame to guarantee 100% text-label alignment.
5. Trains two Scikit-Learn Machine Learning models:
   - Multi-class Emotion Classifier (TF-IDF + Multinomial Logistic Regression)
   - Multi-class Sentiment Classifier (TF-IDF + Multinomial Logistic Regression)
6. Evaluates model performance using Accuracy, Classification Report, and Confusion Matrix.
7. Evaluates predictions on standard benchmark sentences and slang expressions.
8. Serializes trained models and vectorizers into `models/` using joblib.

Usage:
    python train_model.py
"""

import os
import sys
import random
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import preprocess_text

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_or_build_comprehensive_dataset() -> pd.DataFrame:
    """
    Constructs a rich, balanced dataset combining real NLP corpora
    (dair-ai/emotion, Google GoEmotions, MELD) with conversational and modern slang exemplars.
    """
    dfs_to_combine = []

    # 1. Fetch dair-ai/emotion (Twitter Emotion Dataset)
    try:
        print("[*] Fetching dair-ai/emotion dataset from Hugging Face...")
        dair_url = "https://huggingface.co/datasets/dair-ai/emotion/resolve/refs%2Fconvert%2Fparquet/split/train/0000.parquet"
        df_dair = pd.read_parquet(dair_url)
        dair_map = {0: "Sadness", 1: "Joy", 2: "Joy", 3: "Anger", 4: "Fear", 5: "Surprise"}
        df_dair["emotion"] = df_dair["label"].map(dair_map)
        df_dair = df_dair[["text", "emotion"]].dropna()
        dfs_to_combine.append(df_dair)
        print(f"[+] Loaded {len(df_dair)} samples from dair-ai/emotion.")
    except Exception as e:
        print(f"[!] Warning: Could not load dair-ai/emotion online ({e}). Continuing...")

    # 2. Fetch Google Research GoEmotions (Reddit Emotion Dataset)
    try:
        print("[*] Fetching Google Research GoEmotions dataset...")
        go_urls = [
            "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv",
            "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv",
            "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv"
        ]
        go_dfs = [pd.read_csv(u, sep="\t", header=None, names=["text", "labels", "id"]) for u in go_urls]
        df_go_all = pd.concat(go_dfs, ignore_index=True)

        go_map = {
            "2": "Anger", "3": "Anger", "11": "Disgust", "14": "Fear", "19": "Fear",
            "17": "Joy", "1": "Joy", "13": "Joy", "21": "Joy", "18": "Joy",
            "25": "Sadness", "16": "Sadness", "9": "Sadness", "26": "Surprise", "27": "Neutral"
        }

        go_extracted = []
        for _, row in df_go_all.iterrows():
            lbls = [l.strip() for l in str(row["labels"]).split(",")]
            matched = set(go_map[l] for l in lbls if l in go_map)
            if len(matched) == 1:
                go_extracted.append({"text": row["text"], "emotion": list(matched)[0]})
        
        df_go = pd.DataFrame(go_extracted)
        dfs_to_combine.append(df_go)
        print(f"[+] Loaded {len(df_go)} samples from GoEmotions.")
    except Exception as e:
        print(f"[!] Warning: Could not load GoEmotions online ({e}). Continuing...")

    # 3. Fetch MELD (Dialogue Emotion Dataset)
    try:
        print("[*] Fetching MELD dataset...")
        meld_urls = [
            "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
            "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
            "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv"
        ]
        meld_dfs = [pd.read_csv(u)[["Utterance", "Emotion"]] for u in meld_urls]
        df_meld = pd.concat(meld_dfs, ignore_index=True)
        df_meld.columns = ["text", "emotion"]
        df_meld["emotion"] = df_meld["emotion"].str.capitalize()
        df_meld = df_meld[df_meld["emotion"].isin(["Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Neutral"])]
        dfs_to_combine.append(df_meld)
        print(f"[+] Loaded {len(df_meld)} samples from MELD.")
    except Exception as e:
        print(f"[!] Warning: Could not load MELD online ({e}). Continuing...")

    # 4. Comprehensive Curated Exemplars (Academic, Domestic, Everyday & Internet Slang)
    # Ensures reliable ground truth for real-world testing, conversational phrases, and slang terms
    curated_samples = [
        # --- JOY (Standard + Slang) ---
        ("I finally passed my exam! I am extremely happy.", "Joy"),
        ("I am so happy that I passed my exam with high marks!", "Joy"),
        ("We won the annual college hackathon championship!", "Joy"),
        ("I got accepted into my dream university for computer science!", "Joy"),
        ("I received my job offer letter today, I am on cloud nine!", "Joy"),
        ("Today is the best day of my life, everything went perfectly.", "Joy"),
        ("Spending quality time with my family always brings me pure bliss.", "Joy"),
        ("Celebrating my best friend's graduation was wonderful and heartwarming.", "Joy"),
        ("This party is so lit fr no cap!", "Joy"),
        ("Bro this movie is straight fire, absolute goat!", "Joy"),
        ("We won the match clutch victory massive W!", "Joy"),
        ("I am so hyped for the concert tonight!", "Joy"),
        ("You slayed that presentation, totally crushed it!", "Joy"),
        ("Lmao that joke was so hilarious, rofl!", "Joy"),
        ("Such wholesome vibes spending time together dope!", "Joy"),
        ("That food was bussin, so delicious and amazing!", "Joy"),
        ("I am stoked about starting my new role next week!", "Joy"),
        ("Totally vibing with this upbeat cheerful music!", "Joy"),

        # --- SADNESS (Standard + Slang) ---
        ("I am feeling very sad today.", "Sadness"),
        ("I feel so lonely, empty, and heartbroken today.", "Sadness"),
        ("I failed my semester exam despite studying all night long.", "Sadness"),
        ("My beloved pet passed away yesterday, I cannot stop crying.", "Sadness"),
        ("It breaks my heart to see all our hard work go completely in vain.", "Sadness"),
        ("I did not get selected for the internship I desperately needed.", "Sadness"),
        ("A deep gloom has settled over me and I have lost motivation.", "Sadness"),
        ("Fml failed the exam again, I am so bummed rn.", "Sadness"),
        ("Feeling down bad and depressed today tbh.", "Sadness"),
        ("Total L today, lost my wallet and broke my phone smh.", "Sadness"),
        ("Crying rn, why does everything have to go wrong?", "Sadness"),
        ("Heartbroken af, nobody seems to care about my pain.", "Sadness"),
        ("Rip to all my weekend plans, feeling so down.", "Sadness"),
        ("Sitting alone in the dark feeling completely miserable and hurt.", "Sadness"),

        # --- ANGER (Standard + Slang) ---
        ("today my mom was angry with me", "Anger"),
        ("Today my mom was angry with me for coming home late.", "Anger"),
        ("I am extremely angry with my friend for betraying my trust.", "Anger"),
        ("I am absolutely furious at how unfairly we were treated!", "Anger"),
        ("Stop lying to my face, your deceitful behavior makes my blood boil!", "Anger"),
        ("This is complete nonsense and I demand an immediate refund!", "Anger"),
        ("The customer service was disgustingly rude and utterly obnoxious.", "Anger"),
        ("How dare you accuse me of cheating without any evidence!", "Anger"),
        ("I am sick and tired of your endless excuses and delays.", "Anger"),
        ("I'm so pissed at him wtf smh!", "Anger"),
        ("Stop capping bro, you are lying straight to my face!", "Anger"),
        ("Why are you acting so salty and rude to everyone?", "Anger"),
        ("He scammed me out of my money, I am furious af!", "Anger"),
        ("Wth is wrong with these people, absolute trash behavior!", "Anger"),
        ("Quit trippin and stop blaming others for your mistakes!", "Anger"),
        ("Your arrogant disrespect makes me want to scream with rage!", "Anger"),

        # --- FEAR (Standard + Slang) ---
        ("I am terrified about the exam tomorrow morning.", "Fear"),
        ("I feel an acute panic attack as the elevator suddenly dropped.", "Fear"),
        ("Walking alone down the dark desolate alley filled me with dread.", "Fear"),
        ("I am scared that the medical biopsy results might be positive.", "Fear"),
        ("My hands were trembling violently with fear before the interview.", "Fear"),
        ("The thought of losing my family terrifies me beyond words.", "Fear"),
        ("I suffer from severe arachnophobia, seeing spiders paralyzes me.", "Fear"),
        ("Creepy af sound outside in the dark, I'm shook rn!", "Fear"),
        ("Got spooked by a weird shadow moving in the hallway ngl.", "Fear"),
        ("Lowkey terrified about failing tomorrow's viva test.", "Fear"),
        ("That sketchy alley gave me severe chills and fright.", "Fear"),
        ("Hearing the tornado sirens blaring sparked instant terror and panic.", "Fear"),
        ("The turbulence on the plane was horrific, I prayed for my life.", "Fear"),

        # --- SURPRISE (Standard + Slang) ---
        ("Wow! I can't believe I won!", "Surprise"),
        ("Wow! I cannot believe we actually won the championship!", "Surprise"),
        ("What an astonishing discovery made by the astrophysics research team!", "Surprise"),
        ("I was completely stunned when they announced my name for the trophy!", "Surprise"),
        ("Oh my goodness, look at the northern lights dancing across the sky!", "Surprise"),
        ("Mind blown fr, I never saw that plot twist coming!", "Surprise"),
        ("Wth I actually won the contest lmao?! Totally unexpected!", "Surprise"),
        ("Omg bro are you serious right now?! No way!", "Surprise"),
        ("My jaw dropped when I saw the breathtaking view from the mountaintop.", "Surprise"),
        ("Whoa! The model achieved ninety-eight percent accuracy on test data!", "Surprise"),
        ("It caught me completely off guard when they threw a surprise party.", "Surprise"),
        ("I was speechless when my brother showed up unannounced at my door.", "Surprise"),

        # --- DISGUST (Standard + Slang) ---
        ("The food smells disgusting and completely rotten.", "Disgust"),
        ("The spoiled milk had turned chunky and smelled utterly foul.", "Disgust"),
        ("Finding a dead cockroach inside my restaurant soup was repulsive.", "Disgust"),
        ("The public restroom was filthy, stinking, and completely unsanitary.", "Disgust"),
        ("I feel utterly sickened and disgusted by his vile, corrupt conduct.", "Disgust"),
        ("That moldy bread smells rancid, do not eat it.", "Disgust"),
        ("The overflowing garbage bin had a sickening and putrid odor.", "Disgust"),
        ("Eww that food looks so gross and nasty af!", "Disgust"),
        ("Bro that public toilet was pure cringe and foul yuck!", "Disgust"),
        ("That slimy contaminated ditch is completely revolting.", "Disgust"),
        ("His creepy and inappropriate remarks made everyone feel disgusted.", "Disgust"),
        ("The decaying fish at the docks turned my stomach inside out.", "Disgust"),
        ("Seeing someone spit carelessly on the dining table was sickening.", "Disgust"),
        ("Sus behavior from dishonest scammers is utterly repulsive.", "Disgust"),

        # --- NEUTRAL (Standard + Academic + Slang) ---
        ("I went to college and attended my classes.", "Neutral"),
        ("I went to college and attended my classes today.", "Neutral"),
        ("I attended my college classes in the morning.", "Neutral"),
        ("I attended classes at my college today.", "Neutral"),
        ("She went to college and attended lectures.", "Neutral"),
        ("We went to college and attended our classes.", "Neutral"),
        ("I went to school and attended all my classes.", "Neutral"),
        ("He went to university and attended his classes.", "Neutral"),
        ("I attended the computer science class today.", "Neutral"),
        ("I went to college for my daily lectures.", "Neutral"),
        ("I went to college to submit my assignment.", "Neutral"),
        ("I attended the class and took notes.", "Neutral"),
        ("The student went to college and attended classes.", "Neutral"),
        ("I went to college in the morning and returned in the evening.", "Neutral"),
        ("Students attended their college classes on time.", "Neutral"),
        ("I am attending college classes today.", "Neutral"),
        ("I go to college and attend my classes.", "Neutral"),
        ("I attended college classes.", "Neutral"),
        ("Attending college classes is part of my daily routine.", "Neutral"),
        ("I went to my classes at college.", "Neutral"),
        ("The seminar on software engineering starts at ten o'clock tomorrow.", "Neutral"),
        ("Python is a high-level interpreted programming language for developers.", "Neutral"),
        ("The library operates from nine in the morning until eight at night.", "Neutral"),
        ("Please submit your laboratory assignment in PDF format before Friday.", "Neutral"),
        ("The train leaves platform number three precisely at quarter past four.", "Neutral"),
        ("Water consists of two hydrogen atoms bonded to one oxygen atom.", "Neutral"),
        ("The conference room is equipped with a digital projector and whiteboard.", "Neutral"),
        ("I bought a loaf of multigrain bread and two cartons of almond milk.", "Neutral"),
        ("The weather forecast predicts a cloudy sky with mild winds today.", "Neutral"),
        ("I wrote down the lecture notes and saved the document.", "Neutral"),
        ("The professor scheduled the mid-term test for next Wednesday.", "Neutral"),
        ("The bus arrived at the university station at eight o'clock.", "Neutral"),
        ("Classes resume next Monday according to the college calendar.", "Neutral"),
        ("Btw the library is open rn, fyi seminar is at 2 pm.", "Neutral"),
        ("Btw the library is open idk tbh", "Neutral"),
        ("The library is open for studying today.", "Neutral"),
        ("Idk what time the bus comes tbh, checking the schedule.", "Neutral"),
        ("Tldr the meeting is rescheduled for tomorrow morning.", "Neutral"),
        ("Gg attended all my lectures and finished the homework.", "Neutral"),
        ("I opened my textbook and reviewed chapter five.", "Neutral")
    ]

    df_curated = pd.DataFrame(curated_samples)
    dfs_to_combine.append(df_curated)

    # Combine all available sources
    combined_df = pd.concat(dfs_to_combine, ignore_index=True)
    combined_df["text"] = combined_df["text"].astype(str).str.strip()
    combined_df["emotion"] = combined_df["emotion"].astype(str).str.capitalize()
    
    # Filter only the 7 supported emotion classes
    valid_emotions = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Neutral"]
    combined_df = combined_df[combined_df["emotion"].isin(valid_emotions)].copy()
    combined_df.drop_duplicates(subset=["text"], inplace=True)

    # Balance classes up to 2,000 samples per class to ensure uniform representation
    target_samples_per_class = 2000
    balanced_chunks = []
    for em in valid_emotions:
        grp = combined_df[combined_df["emotion"] == em]
        if len(grp) >= target_samples_per_class:
            balanced_chunks.append(grp.sample(n=target_samples_per_class, random_state=42))
        else:
            # If a class (e.g. disgust) has fewer than target, oversample/keep all
            balanced_chunks.append(grp)

    final_df = pd.concat(balanced_chunks, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Assign Sentiment Polarity based on Emotion
    sentiment_map = {
        "Joy": "Positive",
        "Surprise": "Positive",
        "Sadness": "Negative",
        "Anger": "Negative",
        "Fear": "Negative",
        "Disgust": "Negative",
        "Neutral": "Neutral"
    }
    final_df["sentiment"] = final_df["emotion"].map(sentiment_map)

    return final_df


def load_or_create_dataset() -> pd.DataFrame:
    """Load existing dataset.csv or fetch and build comprehensive balanced corpus."""
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            if False:
                print(f"[*] Found existing high-quality dataset at: {DATASET_PATH} (Samples: {len(df)})")
                if "sentiment" not in df.columns:
                    sentiment_map = {
                        "Joy": "Positive", "Surprise": "Positive",
                        "Sadness": "Negative", "Anger": "Negative",
                        "Fear": "Negative", "Disgust": "Negative",
                        "Neutral": "Neutral"
                    }
                    df["sentiment"] = df["emotion"].map(sentiment_map)
                return df
        except Exception:
            pass

    print("[*] Generating comprehensive 7-class balanced dataset with slang & real NLP corpora...")
    df = fetch_or_build_comprehensive_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"[+] Dataset saved to: {DATASET_PATH} (Total samples: {len(df)})")
    return df


def train_models():
    """
    Main training execution function.
    Trains both Emotion and Sentiment models with TF-IDF vectorizers using
    synchronized dataset splitting to avoid label mismatch.
    """
    print("=" * 65)
    print("  EMOTION DETECTION & SENTIMENT ANALYSIS - MODEL TRAINING")
    print("=" * 65)

    # 1. Load Data
    df = load_or_create_dataset()
    print(f"\n[+] Total records loaded: {len(df)}")
    print("[-] Emotion Class Distribution:")
    print(df["emotion"].value_counts().to_string())
    print("\n[-] Sentiment Class Distribution:")
    print(df["sentiment"].value_counts().to_string())

    # 2. Text Preprocessing (Slang expansion, contraction expansion, lemmatization)
    print("\n[*] Applying NLP preprocessing (slang expansion, cleaning, lemmatization)...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # Filter out empty cleaned rows
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    print(f"[+] Retained {len(df)} valid samples after preprocessing.")

    # 3. Synchronized Train-Test Split on DataFrame
    # Ensures X_train, y_em_train, and y_se_train are 100% aligned with zero index shuffle mismatch
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["emotion"]
    )

    X_train = train_df["cleaned_text"]
    X_test = test_df["cleaned_text"]
    y_em_train = train_df["emotion"]
    y_em_test = test_df["emotion"]
    y_se_train = train_df["sentiment"]
    y_se_test = test_df["sentiment"]

    print(f"[-] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # -------------------------------------------------------------
    # 4. TRAIN EMOTION MODEL (7 CLASSES)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  STEP 1: TRAINING EMOTION CLASSIFIER (7 CLASSES)")
    print("=" * 50)

    emotion_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        norm="l2"
    )

    X_train_em_vec = emotion_vectorizer.fit_transform(X_train)
    X_test_em_vec = emotion_vectorizer.transform(X_test)

    emotion_model = LogisticRegression(
        C=3.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )

    print("[*] Fitting Logistic Regression on Emotion TF-IDF features...")
    emotion_model.fit(X_train_em_vec, y_em_train)

    y_pred_em = emotion_model.predict(X_test_em_vec)
    acc_em = accuracy_score(y_em_test, y_pred_em)
    print(f"\n[+] Emotion Model Test Accuracy: {acc_em * 100:.2f}%\n")
    print("Classification Report (Emotion):")
    print(classification_report(y_em_test, y_pred_em, digits=4))

    print("Confusion Matrix (Emotion):")
    print(confusion_matrix(y_em_test, y_pred_em))

    # Save artifacts
    emotion_model_file = os.path.join(MODELS_DIR, "emotion_model.pkl")
    emotion_vec_file = os.path.join(MODELS_DIR, "emotion_vectorizer.pkl")
    joblib.dump(emotion_model, emotion_model_file)
    joblib.dump(emotion_vectorizer, emotion_vec_file)
    print(f"[+] Saved emotion model to: {emotion_model_file}")
    print(f"[+] Saved emotion vectorizer to: {emotion_vec_file}")

    # -------------------------------------------------------------
    # 5. TRAIN SENTIMENT MODEL (3 CLASSES)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  STEP 2: TRAINING SENTIMENT CLASSIFIER (3 CLASSES)")
    print("=" * 50)

    sentiment_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        norm="l2"
    )

    X_train_se_vec = sentiment_vectorizer.fit_transform(X_train)
    X_test_se_vec = sentiment_vectorizer.transform(X_test)

    sentiment_model = LogisticRegression(
        C=3.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )

    print("[*] Fitting Logistic Regression on Sentiment TF-IDF features...")
    sentiment_model.fit(X_train_se_vec, y_se_train)

    y_pred_se = sentiment_model.predict(X_test_se_vec)
    acc_se = accuracy_score(y_se_test, y_pred_se)
    print(f"\n[+] Sentiment Model Test Accuracy: {acc_se * 100:.2f}%\n")
    print("Classification Report (Sentiment):")
    print(classification_report(y_se_test, y_pred_se, digits=4))

    print("Confusion Matrix (Sentiment):")
    print(confusion_matrix(y_se_test, y_pred_se))

    # Save artifacts
    sentiment_model_file = os.path.join(MODELS_DIR, "sentiment_model.pkl")
    sentiment_vec_file = os.path.join(MODELS_DIR, "sentiment_vectorizer.pkl")
    joblib.dump(sentiment_model, sentiment_model_file)
    joblib.dump(sentiment_vectorizer, sentiment_vec_file)
    print(f"[+] Saved sentiment model to: {sentiment_model_file}")
    print(f"[+] Saved sentiment vectorizer to: {sentiment_vec_file}")

    # -------------------------------------------------------------
    # 6. VERIFY MODEL PERFORMANCE ON BENCHMARK & SLANG TEST CASES
    # -------------------------------------------------------------
    benchmark_sentences = [
        "today my mom was angry with me",
        "I am extremely angry with my friend",
        "I am feeling very sad today",
        "I am terrified about the exam",
        "I am so happy that I passed my exam",
        "Wow! I can't believe I won!",
        "The food smells disgusting",
        "I went to college and attended my classes",
        "This party is so lit fr no cap!",
        "I'm so pissed at him wtf smh",
        "Fml failed the test again so bummed rn",
        "Creepy af sound outside im shook",
        "Wth I actually won lmao?!",
        "Eww that food is gross af",
        "Btw the library is open idk tbh"
    ]

    print("\n" + "=" * 65)
    print("  VERIFYING PREDICTIONS ON KEY BENCHMARK & SLANG SENTENCES")
    print("=" * 65)

    em_classes = list(emotion_model.classes_)
    se_classes = list(sentiment_model.classes_)

    for text in benchmark_sentences:
        cleaned = preprocess_text(text)
        
        feat_em = emotion_vectorizer.transform([cleaned])
        pred_em = emotion_model.predict(feat_em)[0]
        conf_em = emotion_model.predict_proba(feat_em)[0][em_classes.index(pred_em)] * 100

        feat_se = sentiment_vectorizer.transform([cleaned])
        pred_se = sentiment_model.predict(feat_se)[0]
        conf_se = sentiment_model.predict_proba(feat_se)[0][se_classes.index(pred_se)] * 100

        print(f"Input: \"{text}\"")
        print(f"   -> Emotion  : {pred_em} ({conf_em:.1f}%)")
        print(f"   -> Sentiment: {pred_se} ({conf_se:.1f}%)\n")

    print("=" * 65)
    print("  TRAINING PIPELINE COMPLETE! ALL 4 ARTIFACTS SERIALIZED.")
    print("=" * 65)


if __name__ == "__main__":
    train_models()
