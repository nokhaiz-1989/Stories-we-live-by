import streamlit as st

st.set_page_config(page_title="Eco-CDA Analyzer", layout="wide")

# ── stdlib / third-party imports ──────────────────────────────────────────────
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

# ── NLTK data ─────────────────────────────────────────────────────────────────
for resource in ("stopwords", "punkt"):
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── spaCy ─────────────────────────────────────────────────────────────────────
import spacy


@st.cache_resource(show_spinner="Loading language model...")
def load_nlp():
    """Load en_core_web_sm; download on-the-fly if missing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


nlp = load_nlp()

# ── stopwords ─────────────────────────────────────────────────────────────────
STOP = set(stopwords.words("english"))

# ── Stibbe dictionaries ───────────────────────────────────────────────────────
METAPHOR_WORDS = [
    "battle",
    "fight",
    "war",
    "enemy",
    "attack",
    "combat",
    "storm",
    "wave",
    "drowning",
    "tsunami",
]

EVALUATION_WORDS = [
    "devastating",
    "catastrophic",
    "severe",
    "tragic",
    "dangerous",
    "massive",
    "horrific",
    "destructive",
]

IDENTITY_WORDS = [
    "government",
    "victims",
    "families",
    "citizens",
    "farmers",
    "authorities",
    "communities",
    "people",
]

IDEOLOGY_WORDS = [
    "development",
    "progress",
    "responsibility",
    "policy",
    "national",
    "economic",
    "climate",
    "sustainability",
]

ERASURE_PATTERNS = [
    "were displaced",
    "was destroyed",
    "were affected",
    "was damaged",
    "lost their homes",
]

# ── helper functions ──────────────────────────────────────────────────────────


def preprocess(text: str) -> list[str]:
    """Clean and preprocess text."""
    text = re.sub(r"http\S+", "", str(text).lower())
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = [
        w
        for w in word_tokenize(text)
        if w not in STOP and len(w.strip()) > 1
    ]

    doc = nlp(" ".join(tokens))

    return [
        token.lemma_
        for token in doc
        if token.lemma_.strip() != ""
    ]


def word_freq(tokens):
    return Counter(tokens)


def concordance(text: str, keyword: str, width: int = 60):
    """Find concordance lines."""
    out = []

    for m in re.finditer(re.escape(keyword), text, re.IGNORECASE):
        s = max(m.start() - width, 0)
        e = min(m.end() + width, len(text))
        out.append(text[s:e])

    return out


def classify(sentence: str) -> list[str]:
    """Classify sentence into Stibbe categories."""
    s = sentence.lower()
    cats = []

    if any(w in s for w in METAPHOR_WORDS):
        cats.append("Metaphor")

    if any(w in s for w in EVALUATION_WORDS):
        cats.append("Evaluation")

    if any(w in s for w in IDENTITY_WORDS):
        cats.append("Identity")

    if any(w in s for w in IDEOLOGY_WORDS):
        cats.append("Ideology")

    if any(p in s for p in ERASURE_PATTERNS):
        cats.append("Erasure")

    return cats


def count_words(text):
    return len(text.split())


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌍 Eco-CDA Analyzer")

st.subheader(
    "Computational CDA Tool based on Arran Stibbe's *Stories We Live By*"
)

st.markdown(
    """
This tool performs:

- Corpus Analysis
- Frequency Analysis
- Concordance Analysis
- Metaphor Detection
- Stibbe Category Classification
- Basic CDA Interpretation
"""
)

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📂 Input Options")

input_method = st.sidebar.radio(
    "Choose Input Method",
    ["Upload CSV File", "Paste Text Directly"],
)

# ── DATA INPUT ────────────────────────────────────────────────────────────────
df = None
all_text = ""

# OPTION 1 — CSV Upload
if input_method == "Upload CSV File":

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Please upload a CSV dataset to begin.")

        st.write("### Required CSV Format")

        st.dataframe(
            pd.DataFrame(
                {
                    "date": ["2024-06-01"],
                    "headline": ["Floods devastate Sindh"],
                    "article_text": [
                        "Pakistan is battling catastrophic floods causing devastation."
                    ],
                }
            )
        )

        st.stop()

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    if "article_text" not in df.columns:
        st.error("Dataset must contain an **article_text** column.")
        st.stop()

    df["article_text"] = df["article_text"].fillna("").astype(str)

    all_text = " ".join(df["article_text"])

    st.success("CSV dataset uploaded successfully!")

    st.write("## Dataset Preview")
    st.dataframe(df.head())

# OPTION 2 — Paste Text
elif input_method == "Paste Text Directly":

    pasted_text = st.text_area(
        "Paste your text here (Maximum 10,000 words)",
        height=300,
        placeholder="Paste article, discourse, essay, news report, or any text here...",
    )

    if not pasted_text.strip():
        st.info("Please paste text to begin analysis.")
        st.stop()

    word_count = count_words(pasted_text)

    if word_count > 10000:
        st.error(
            f"⚠️ Word limit exceeded. Current count: {word_count} words. "
            "Maximum allowed: 10,000 words."
        )
        st.stop()

    st.success(f"Text received successfully! ({word_count} words)")

    # Convert pasted text into dataframe
    df = pd.DataFrame(
        {
            "article_text": [pasted_text]
        }
    )

    all_text = pasted_text

# ── preprocessing ─────────────────────────────────────────────────────────────
with st.spinner("Preprocessing corpus..."):
    tokens = preprocess(all_text)

if not tokens:
    st.error(
        "⚠️ No valid text could be extracted. "
        "Please check your input data."
    )
    st.stop()

# ── sentence classification ───────────────────────────────────────────────────
rows = []

with st.spinner("Running Stibbe classification..."):

    for article in df["article_text"]:

        doc = nlp(article)

        for sent in doc.sents:

            sentence = sent.text.strip()

            if len(sentence) < 3:
                continue

            cats = classify(sentence)

            if cats:
                rows.append(
                    {
                        "Sentence": sentence,
                        "Category": ", ".join(cats),
                    }
                )

cat_counts = Counter()
class_df = None

if rows:
    class_df = pd.DataFrame(rows)

    for c in class_df["Category"]:
        cat_counts.update(c.split(", "))

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Corpus Overview",
        "📈 Frequency Analysis",
        "🔍 Concordance",
        "🧠 Stibbe Classification",
        "🌍 CDA Insights",
    ]
)

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:

    st.header("Corpus Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Documents", len(df))
    c2.metric("Total Words", len(tokens))
    c3.metric("Unique Words", len(set(tokens)))
    c4.metric("Sentences", len(list(nlp(all_text).sents)))

    st.write("### Text Sample")

    st.text_area(
        "Preview",
        all_text[:2000],
        height=200,
        disabled=True,
    )

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:

    st.header("Frequency Analysis")

    freq = word_freq(tokens)

    fdf = pd.DataFrame(
        freq.most_common(20),
        columns=["Word", "Frequency"],
    )

    st.write("### Top 20 Frequent Words")

    st.dataframe(
        fdf,
        use_container_width=True,
    )

    # BAR CHART
    fig_bar = px.bar(
        fdf,
        x="Word",
        y="Frequency",
        title="Top 20 Frequent Words",
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True,
    )

    # WORD CLOUD
    st.write("### Word Cloud")

    wc = WordCloud(
        width=1200,
        height=500,
        background_color="white",
    ).generate(" ".join(tokens))

    fig_wc, ax = plt.subplots(figsize=(14, 6))

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig_wc)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:

    st.header("Concordance Analysis")

    kw = st.text_input("Enter a keyword")

    if kw:

        hits = concordance(all_text, kw)

        st.write(f"### Results for '{kw}'")

        if hits:

            st.success(f"{len(hits)} occurrence(s) found.")

            for i, line in enumerate(hits[:50], 1):

                st.markdown(f"**{i}.** ...{line}...")

        else:
            st.warning("No occurrences found.")

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:

    st.header("Stibbe Category Classification")

    if class_df is not None and not class_df.empty:

        st.write("### Classified Sentences")

        st.dataframe(
            class_df,
            use_container_width=True,
        )

        cdf = pd.DataFrame(
            {
                "Category": list(cat_counts.keys()),
                "Count": list(cat_counts.values()),
            }
        )

        # PIE CHART
        fig_pie = px.pie(
            cdf,
            names="Category",
            values="Count",
            title="Distribution of Stibbe Categories",
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True,
        )

        # DOWNLOAD BUTTON
        st.download_button(
            "📥 Download Classification Results",
            data=class_df.to_csv(index=False),
            file_name="stibbe_classification.csv",
            mime="text/csv",
        )

    else:
        st.warning("No categories detected.")

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:

    st.header("CDA Insights")

    st.markdown("### Interpretation of Discourse Patterns")

    if cat_counts:

        INSIGHTS = {
            "Metaphor": (
                "Flood discourse frequently uses metaphorical framing, "
                "representing floods as conflict, warfare, or destructive force."
            ),

            "Evaluation": (
                "Evaluative language indicates emotional and ideological "
                "positioning within disaster reporting."
            ),

            "Identity": (
                "Identity constructions reveal how social actors "
                "(victims, authorities, communities) are represented."
            ),

            "Erasure": (
                "Erasure patterns suggest passive constructions that may "
                "hide responsibility or agency."
            ),

            "Ideology": (
                "Ideological patterns reflect assumptions about development, "
                "governance, climate, and responsibility."
            ),
        }

        for cat, msg in INSIGHTS.items():

            if cat in cat_counts:

                st.info(f"**{cat}:** {msg}")

    else:
        st.warning("No discourse patterns detected.")

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Eco-CDA Analyzer • Streamlit + NLP + Stibbe CDA Framework")
