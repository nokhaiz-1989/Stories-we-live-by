# Advanced Eco-CDA Analyzer Project

import streamlit as st

st.set_page_config(
    page_title="Advanced Eco-CDA Analyzer",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from wordcloud import WordCloud
import spacy

# ─────────────────────────────────────────────────────────────────────────────
# NLTK
# ─────────────────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.corpus import stopwords
from nltk.util import ngrams

# ─────────────────────────────────────────────────────────────────────────────
# spaCy
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading spaCy model...")
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_nlp()

STOP = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────────────────────
# Dictionaries
# ─────────────────────────────────────────────────────────────────────────────
METAPHOR_WORDS = ["battle", "fight", "combat", "enemy", "tsunami"]

EVALUATION_WORDS = [
    "devastating", "catastrophic", "severe", "tragic",
    "dangerous", "massive", "horrific", "destructive"
]

IDENTITY_WORDS = [
    "government", "victims", "families", "citizens",
    "farmers", "authorities", "communities", "people",
    "minister", "official", "community", "citizen"
]

IDEOLOGY_WORDS = [
    "development", "progress", "responsibility",
    "policy", "national", "economic", "climate",
    "sustainability"
]

ERASURE_PATTERNS = [
    "were displaced", "was destroyed", "were affected",
    "was damaged", "lost their homes"
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str):
    text = re.sub(r"http\S+", "", str(text).lower())
    text = re.sub(r"[^a-z\s]", " ", text)

    doc = nlp(text)

    return [
        token.lemma_
        for token in doc
        if token.is_alpha
        and token.text not in STOP
        and not token.is_stop
        and len(token.text) > 1
    ]


def classify(sentence: str):
    doc = nlp(sentence.lower())

    words = [t.lemma_.lower() for t in doc if t.is_alpha]

    cats = []

    metaphor_patterns = ["battle against", "fight against", "war on", "wave of"]

    s = " ".join(words)

    if any(p in s for p in metaphor_patterns):
        cats.append("Metaphor")

    elif any(w in words for w in METAPHOR_WORDS):
        if any(c in words for c in ["climate", "flood", "disaster", "environment", "crisis"]):
            cats.append("Metaphor")

    if any(w in words for w in EVALUATION_WORDS):
        cats.append("Evaluation")

    if any(w in words for w in IDENTITY_WORDS):
        cats.append("Identity")

    if any(w in words for w in IDEOLOGY_WORDS):
        cats.append("Ideology")

    if any(p in sentence.lower() for p in ERASURE_PATTERNS):
        cats.append("Erasure")

    return cats


def word_freq(tokens):
    return Counter(tokens)


def concordance(text, keyword, width=60):
    out = []

    for m in re.finditer(re.escape(keyword), text, re.IGNORECASE):
        start = max(m.start() - width, 0)
        end = min(m.end() + width, len(text))

        out.append([
            text[start:m.start()],
            text[m.start():m.end()],
            text[m.end():end]
        ])

    return out


def count_words(text):
    return len(text.split())


def get_sentiment(text):
    p = TextBlob(text).sentiment.polarity

    if p > 0.1:
        return "Positive"
    elif p < -0.1:
        return "Negative"
    return "Neutral"


def generate_ngrams(tokens, n=2, top_k=15):
    grams = ngrams(tokens, n)
    freq = Counter(grams)

    return pd.DataFrame(
        [(" ".join(k), v) for k, v in freq.most_common(top_k)],
        columns=["Phrase", "Frequency"]
    )


def extract_entities(text):
    doc = nlp(text)

    ents = [(e.text, e.label_) for e in doc.ents]

    df = pd.DataFrame(ents, columns=["Entity", "Type"])

    if df.empty:
        return df

    return df.value_counts().reset_index(name="Frequency")


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🌍 Advanced Eco-CDA Analyzer")

st.subheader("Computational Critical Discourse Analysis Tool based on Stibbe")

st.sidebar.title("📂 Input Options")

input_method = st.sidebar.radio(
    "Choose Input Method",
    ["Upload CSV File", "Paste Text Directly"]
)

df = None
all_text = ""

# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────
if input_method == "Upload CSV File":

    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if file is None:
        st.info("Upload CSV file to begin.")
        st.stop()

    df = pd.read_csv(file)

    all_text = " ".join(df["article_text"].fillna(""))

else:

    pasted = st.text_area("Paste text here", height=300)

    if not pasted.strip():
        st.stop()

    df = pd.DataFrame({"article_text": [pasted]})
    all_text = pasted

# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
tokens = preprocess(all_text)

rows = []

for article in df["article_text"]:
    for sent in nlp(article).sents:

        cats = classify(sent.text)
        senti = get_sentiment(sent.text)

        if cats:
            rows.append({
                "Sentence": sent.text,
                "Category": ", ".join(cats),
                "Sentiment": senti
            })

class_df = pd.DataFrame(rows)
cat_counts = Counter()

for c in class_df["Category"]:
    cat_counts.update(c.split(", "))

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Corpus",
    "Frequency",
    "KWIC",
    "Classification",
    "Sentiment",
    "Entities",
    "CDA Insights",
    "Stories We Live By"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 - STORIES WE LIVE BY
# ─────────────────────────────────────────────────────────────────────────────

with tab8:

    st.header("📖 Stories We Live By")

    STORY_PATTERNS = {
        "Progress": ["development", "growth", "progress"],
        "Crisis": ["disaster", "crisis", "collapse"],
        "War": ["battle", "fight", "war"],
        "Apocalypse": ["end", "destroy", "ruin"],
        "Responsibility": ["policy", "sustainability", "responsibility"]
    }

    counts = Counter()

    for sent in nlp(all_text).sents:
        s = sent.text.lower()

        for story, words in STORY_PATTERNS.items():
            if any(w in s for w in words):
                counts[story] += 1

    if counts:

        df_story = pd.DataFrame(counts.items(),
                                columns=["Story", "Frequency"])

        st.dataframe(df_story)

        fig = px.bar(df_story, x="Story", y="Frequency")
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Dominant Story: {df_story.iloc[0]['Story']}")

    else:
        st.warning("No patterns found.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("Advanced Eco-CDA Analyzer • Streamlit + NLP + Ecolinguistics")
