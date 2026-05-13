import streamlit as st
st.set_page_config(page_title="Advanced Eco-CDA Analyzer", layout="wide")

import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.corpus import stopwords
from nltk.util import ngrams

import spacy

# ─────────────────────────────
# spaCy FIX (NO DOWNLOAD IN CLOUD)
# ─────────────────────────────
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

STOP = set(stopwords.words("english"))

# ─────────────────────────────
# WORD LISTS
# ─────────────────────────────
METAPHOR_WORDS = ["battle", "fight", "combat", "enemy", "tsunami"]

EVALUATION_WORDS = ["devastating", "catastrophic", "severe", "tragic",
                     "dangerous", "massive", "horrific", "destructive"]

IDENTITY_WORDS = ["government", "victims", "families", "citizens",
                   "farmers", "authorities", "communities", "people",
                   "minister", "official", "community"]

IDEOLOGY_WORDS = ["development", "progress", "responsibility",
                  "policy", "national", "economic", "climate", "sustainability"]

ERASURE_PATTERNS = ["were displaced", "was destroyed", "were affected",
                     "was damaged", "lost their homes"]

# ─────────────────────────────
# FUNCTIONS
# ─────────────────────────────
def preprocess(text):
    text = re.sub(r"http\S+", "", str(text).lower())
    text = re.sub(r"[^a-z\s]", " ", text)

    doc = nlp(text)

    return [
        t.lemma_
        for t in doc
        if t.is_alpha and t.text not in STOP and not t.is_stop
    ]


def classify(sentence):
    doc = nlp(sentence.lower())
    words = [t.lemma_ for t in doc if t.is_alpha]

    cats = []

    if any(p in sentence.lower() for p in ["battle against", "fight against", "war on"]):
        cats.append("Metaphor")

    if any(w in words for w in METAPHOR_WORDS):
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


def get_sentiment(text):
    p = TextBlob(text).sentiment.polarity
    if p > 0.1:
        return "Positive"
    elif p < -0.1:
        return "Negative"
    return "Neutral"


def word_freq(tokens):
    return Counter(tokens)


def generate_ngrams(tokens, n=2, top_k=15):
    grams = ngrams(tokens, n)
    freq = Counter(grams)

    return pd.DataFrame(
        [(" ".join(k), v) for k, v in freq.most_common(top_k)],
        columns=["Phrase", "Frequency"]
    )


# ─────────────────────────────
# UI
# ─────────────────────────────
st.title("🌍 Advanced Eco-CDA Analyzer")

input_method = st.sidebar.radio("Input Method", ["Upload CSV File", "Paste Text"])

df = None
all_text = ""

# ─────────────────────────────
# INPUT
# ─────────────────────────────
if input_method == "Upload CSV File":

    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if not file:
        st.info("Upload CSV with 'article_text' column")
        st.stop()

    df = pd.read_csv(file)

    if "article_text" not in df.columns:
        st.error("Missing article_text column")
        st.stop()

    all_text = " ".join(df["article_text"].astype(str))

else:

    all_text = st.text_area("Paste Text")

    if not all_text:
        st.stop()

    df = pd.DataFrame({"article_text": [all_text]})


# ─────────────────────────────
# PROCESSING
# ─────────────────────────────
tokens = preprocess(all_text)

rows = []

for article in df["article_text"]:
    doc = nlp(article)

    for sent in doc.sents:
        cats = classify(sent.text)

        if cats:
            rows.append({
                "Sentence": sent.text,
                "Category": ", ".join(cats),
                "Sentiment": get_sentiment(sent.text)
            })

class_df = pd.DataFrame(rows)

cat_counts = Counter()

for c in class_df.get("Category", []):
    cat_counts.update(c.split(", "))


# ─────────────────────────────
# TABS
# ─────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Corpus", "Frequency", "KWIC",
    "Classification", "Sentiment",
    "Entities", "CDA Insights"
])

# ─────────────────────────────
# TAB 1
# ─────────────────────────────
with tab1:
    st.header("Corpus Overview")

    st.metric("Words", len(tokens))
    st.metric("Unique Words", len(set(tokens)))

    st.text_area("Preview", all_text[:2000], height=200)

# ─────────────────────────────
# TAB 2
# ─────────────────────────────
with tab2:
    st.header("Frequency Analysis")

    freq_df = pd.DataFrame(word_freq(tokens).most_common(20),
                            columns=["Word", "Frequency"])

    st.dataframe(freq_df)

    st.plotly_chart(px.bar(freq_df, x="Word", y="Frequency"))

    st.write("Word Cloud")
    wc = WordCloud(width=900, height=400).generate(" ".join(tokens))
    st.image(wc.to_array())

    st.write("Bigrams")
    st.dataframe(generate_ngrams(tokens, 2))

    st.write("Trigrams")
    st.dataframe(generate_ngrams(tokens, 3))

# ─────────────────────────────
# TAB 3
# ─────────────────────────────
with tab3:
    st.header("KWIC")

    kw = st.text_input("Keyword")

    if kw:
        results = []
        for m in re.finditer(kw, all_text, re.I):
            results.append(all_text[max(0, m.start()-40):m.end()+40])

        st.write(results if results else "No matches")

# ─────────────────────────────
# TAB 4
# ─────────────────────────────
with tab4:
    st.header("Classification")

    st.dataframe(class_df)

# ─────────────────────────────
# TAB 5
# ─────────────────────────────
with tab5:
    st.header("Sentiment")

    if not class_df.empty:
        st.dataframe(class_df["Sentiment"].value_counts())

# ─────────────────────────────
# TAB 6
# ─────────────────────────────
with tab6:
    st.header("Named Entities")

    ents = [(e.text, e.label_) for e in nlp(all_text).ents]

    st.dataframe(pd.DataFrame(ents, columns=["Entity", "Type"]))

# ─────────────────────────────
# TAB 7
# ─────────────────────────────
with tab7:
    st.header("CDA Insights")

    if cat_counts:
        for k, v in cat_counts.items():
            st.write(f"**{k}:** {v}")
    else:
        st.info("No patterns detected")
