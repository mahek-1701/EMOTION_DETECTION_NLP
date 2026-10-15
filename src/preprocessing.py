"""
Text Preprocessing Module for Emotion Detection and Sentiment Analysis.

This module provides a robust, reusable NLP preprocessing pipeline:
1. Lowercasing
2. Contraction expansion (e.g., won't -> will not, don't -> do not)
3. Slang and Internet acronym expansion (e.g., lit, pissed, fml, tbh, fr, no cap)
4. Noise removal (URLs, HTML tags, special symbols, digits)
5. Tokenization
6. Sentiment-preserving stopword filtering (preserves negation words and victory words like 'won')
7. Multi-POS Lemmatization using WordNetLemmatizer (verbs, nouns, adjectives)
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data packages are downloaded quietly
def _ensure_nltk_resources():
    packages = ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]
    for pkg in packages:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass

_ensure_nltk_resources()

# Dictionary of English contractions for expansion
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "let's": "let us",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
}

# Comprehensive Slang & Internet Acronym Dictionary
# Maps colloquial online slang, Gen-Z terms, and chat abbreviations to emotional/semantic equivalents
SLANG_DICTIONARY = {
    # Joy / Excitement / Celebration / Laughter
    "lit": "awesome exciting",
    "fire": "amazing awesome",
    "goat": "greatest best",
    "slay": "amazing impressive",
    "slayed": "succeeded brilliantly",
    "hyped": "excited joyful",
    "stoked": "thrilled excited",
    "dope": "great awesome",
    "bussin": "delicious amazing",
    "wholesome": "heartwarming lovely",
    "vibing": "enjoying relaxing",
    "vibe": "feeling atmosphere",
    "w": "win victory success",
    "gg": "good game well done",
    "based": "admirable great",
    "clutch": "critical success victory",
    "lmao": "laughing hilarious",
    "lmfao": "laughing hilarious",
    "rofl": "laughing rolling",
    "lol": "laughing happy",
    "haha": "laughing joy",
    "hahaha": "laughing joy",

    # Sadness / Disappointment / Grief
    "fml": "terrible miserable day",
    "bummed": "sad disappointed",
    "down bad": "depressed desperate",
    "crying rn": "crying right now sad",
    "heartbroken": "heartbroken sorrowful",
    "rip": "rest in peace sad",
    "l": "loss defeat failure",

    # Anger / Outrage / Frustration
    "pissed": "furious angry",
    "pissed off": "furious enraged",
    "salty": "bitter angry irritated",
    "cap": "lie false",
    "capping": "lying deceitful",
    "no cap": "honestly truth",
    "fr": "for real honestly",
    "trash": "terrible worthless garbage",
    "wack": "terrible unfair bad",
    "trippin": "acting crazy irrational",
    "smh": "disappointed frustrated",
    "wth": "what the hell angry",
    "wtf": "furious disbelief outrage",

    # Fear / Dread / Anxiety
    "shook": "terrified shocked scared",
    "spooked": "frightened terrified",
    "creepy": "scary frightening eerie",
    "sketchy": "suspicious dangerous frightening",

    # Surprise / Astonishment
    "omg": "oh my god surprise",
    "mind blown": "astonished shocked amazed",
    "jaw dropped": "astonished surprised",
    "whoa": "amazed surprised",
    "shooketh": "utterly astonished",

    # Disgust / Revulsion
    "cringe": "embarrassing disgusted awkward",
    "cringey": "disgusting awkward",
    "gross": "disgusting repulsive foul",
    "eww": "disgusting revolting",
    "ew": "disgusting",
    "yuck": "disgusting foul",
    "nasty": "disgusting filthy repulsive",
    "sus": "suspicious shady untrustworthy",

    # Intensity / Conversational Context
    "af": "extremely very",
    "asf": "extremely very",
    "rn": "right now",
    "ngl": "not going to lie honestly",
    "tbh": "to be honest frankly",
    "idk": "i do not know",
    "btw": "by the way",
    "fyi": "for your information",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "lowkey": "secretly slightly",
    "highkey": "obviously strongly",
    "deadass": "seriously honestly",
    "tldr": "summary brief",
    "np": "no problem"
}

# Negation and intensity words essential for emotion & sentiment nuance
NEGATION_WORDS = {
    "not", "no", "nor", "neither", "never", "none", "hardly", "barely", 
    "scarcely", "cannot", "without", "against"
}

# Initialize lemmatizer and stopword set
lemmatizer = WordNetLemmatizer()
try:
    standard_stopwords = set(stopwords.words("english"))
except Exception:
    standard_stopwords = set()

# Preserve negation words and past-tense victory terms (NLTK includes 'won' due to 'won't')
CUSTOM_STOPWORDS = standard_stopwords - NEGATION_WORDS - {"won"}


def expand_contractions(text: str) -> str:
    """Expand common English contractions to preserve negative and grammatical meaning."""
    text_lower = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text_lower = text_lower.replace(contraction, expansion)
    return text_lower


def expand_slang(text: str) -> str:
    """
    Expand internet slang, Gen-Z terms, and acronyms into descriptive semantic words.
    Uses regex word boundaries to avoid false substring replacements inside regular words.
    """
    processed = text.lower()
    # Sort keys by descending length so multi-word phrases like 'no cap', 'crying rn' match first
    for slang in sorted(SLANG_DICTIONARY.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(slang) + r"\b"
        processed = re.sub(pattern, SLANG_DICTIONARY[slang], processed)
    return processed


def preprocess_text(text: str) -> str:
    """
    Standard preprocessing pipeline applied identically during training and inference.

    Steps:
    1. Validate input and handle non-string / empty inputs.
    2. Lowercase text and expand common contractions.
    3. Expand internet slang, acronyms, and conversational terms.
    4. Remove URLs, HTML tags, user handles, and special characters.
    5. Tokenize text into words.
    6. Filter punctuation and non-sentiment stopwords (preserves negations & 'won').
    7. Apply multi-POS WordNet lemmatization (verbs, nouns, adjectives).
    8. Reconstruct cleaned text.

    Args:
        text (str): Raw input text string.

    Returns:
        str: Cleaned, lemmatized, and normalized text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Lowercase and expand contractions
    text = expand_contractions(text)

    # Step 2: Expand internet slang & acronyms
    text = expand_slang(text)

    # Step 3: Remove URLs (http/https/ftp/www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Step 4: Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Step 5: Remove email addresses and user mentions
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    # Step 6: Remove digits and special characters (keep alphabetic characters)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Step 7: Tokenization
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback regex tokenization if punkt encounters an issue
        tokens = text.split()

    # Step 8: Stopword filtering & Lemmatization
    cleaned_tokens = []
    for token in tokens:
        token = token.strip().lower()
        if len(token) > 1:  # Filter single character artifacts
            if token not in CUSTOM_STOPWORDS:
                lemma = lemmatizer.lemmatize(token, pos="v")  # Lemmatize verbs
                lemma = lemmatizer.lemmatize(lemma, pos="n")  # Lemmatize nouns
                lemma = lemmatizer.lemmatize(lemma, pos="a")  # Lemmatize adjectives
                cleaned_tokens.append(lemma)

    # Step 9: Return reconstructed string with single spaces
    return " ".join(cleaned_tokens).strip()


if __name__ == "__main__":
    test_samples = [
        "I finally passed my exam! I am extremely happy.",
        "Today my mom was angry with me.",
        "This party is so lit fr no cap!",
        "I'm so pissed at him wtf smh",
        "Fml failed the test again so bummed rn",
        "Creepy af sound outside im shook",
        "Wth I actually won lmao?!",
        "Eww that food is gross af",
        "I went to college and attended my classes",
        "Btw the library is open idk tbh"
    ]
    for s in test_samples:
        print(f"Original: {s}")
        print(f"Cleaned : {preprocess_text(s)}\n")
