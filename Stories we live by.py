import streamlit as st
import pandas as pd
import re
from collections import Counter
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from spacy.cli import download

# =====================================================
# DOWNLOAD NLTK RESOURCES
# =====================================================
nltk.download('punkt')
nltk.download('stopwords')

# =====================================================
# LOAD SPACY MODEL
# ====================================================

try:
    nlp = spacy.load("en_core_web_sm")

except:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Eco-CDA Analyzer",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================
st.title("🌍 Eco-CDA Analyzer")
st.subheader(
    "Computational CDA Tool based on Arran Stibbe's Stories We Live By"
)

st.markdown("""
This tool performs:
- Corpus Analysis
- Frequency Analysis
- Concordance Analysis
- Metaphor Detection
- Stibbe Category Classification
- Basic CDA Interpretation
""")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =====================================================
# STOPWORDS
# =====================================================
stop_words = set(stopwords.words('english'))

# =====================================================
# STIBBE CATEGORY DICTIONARIES
# =====================================================

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
    "tsunami"
]

EVALUATION_WORDS = [
    "devastating",
    "catastrophic",
    "severe",
    "tragic",
    "dangerous",
    "massive",
    "horrific",
    "destructive"
]

IDENTITY_WORDS = [
    "government",
    "victims",
    "families",
    "citizens",
    "farmers",
    "authorities",
    "communities",
    "people"
]

IDEOLOGY_WORDS = [
    "development",
    "progress",
    "responsibility",
    "policy",
    "national",
    "economic",
    "climate",
    "sustainability"
]

ERASURE_PATTERNS = [
    "were displaced",
    "was destroyed",
    "were affected",
    "was damaged",
    "lost their homes"
]

# =====================================================
# PREPROCESSING FUNCTION
# =====================================================
def preprocess_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    doc = nlp(" ".join(tokens))

    lemmas = [token.lemma_ for token in doc]

    return lemmas

# =====================================================
# WORD FREQUENCY
# =====================================================
def get_word_frequencies(tokens):
    return Counter(tokens)

# =====================================================
# CONCORDANCE FUNCTION
# =====================================================
def concordance(text, keyword, width=50):

    results = []

    text = str(text)

    for match in re.finditer(keyword, text, re.IGNORECASE):

        start = max(match.start() - width, 0)
        end = min(match.end() + width, len(text))

        context = text[start:end]

        results.append(context)

    return results

# =====================================================
# STIBBE CLASSIFICATION FUNCTION
# =====================================================
def classify_sentence(sentence):

    categories = []

    sent = sentence.lower()

    # METAPHOR
    if any(word in sent for word in METAPHOR_WORDS):
        categories.append("Metaphor")

    # EVALUATION
    if any(word in sent for word in EVALUATION_WORDS):
        categories.append("Evaluation")

    # IDENTITY
    if any(word in sent for word in IDENTITY_WORDS):
        categories.append("Identity")

    # IDEOLOGY
    if any(word in sent for word in IDEOLOGY_WORDS):
        categories.append("Ideology")

    # ERASURE
    if any(pattern in sent for pattern in ERASURE_PATTERNS):
        categories.append("Erasure")

    return categories

# =====================================================
# MAIN APP
# =====================================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully!")

    st.write("## Dataset Preview")
    st.dataframe(df.head())

    # =====================================================
    # CHECK REQUIRED COLUMN
    # =====================================================
    if "article_text" not in df.columns:
        st.error("Dataset must contain 'article_text' column.")
        st.stop()

    # =====================================================
    # COMBINE ALL TEXT
    # =====================================================
    all_text = " ".join(df["article_text"].astype(str))

    # =====================================================
    # PREPROCESS
    # =====================================================
    processed_tokens = preprocess_text(all_text)

    # =====================================================
    # CREATE TABS
    # =====================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Corpus Overview",
        "📈 Frequency Analysis",
        "🔍 Concordance",
        "🧠 Stibbe Classification",
        "🌍 CDA Insights"
    ])

    # =====================================================
    # TAB 1 — CORPUS OVERVIEW
    # =====================================================
    with tab1:

        st.header("Corpus Overview")

        total_articles = len(df)
        total_words = len(processed_tokens)
        unique_words = len(set(processed_tokens))

        col1, col2, col3 = st.columns(3)

        col1.metric("Articles", total_articles)
        col2.metric("Total Words", total_words)
        col3.metric("Unique Words", unique_words)

    # =====================================================
    # TAB 2 — FREQUENCY ANALYSIS
    # =====================================================
    with tab2:

        st.header("Frequency Analysis")

        freq = get_word_frequencies(processed_tokens)

        freq_df = pd.DataFrame(
            freq.most_common(20),
            columns=["Word", "Frequency"]
        )

        st.write("### Top 20 Frequent Words")
        st.dataframe(freq_df)

        # BAR CHART
        fig = px.bar(
            freq_df,
            x="Word",
            y="Frequency",
            title="Top 20 Frequent Words"
        )

        st.plotly_chart(fig, use_container_width=True)

        # WORD CLOUD
        st.write("### Word Cloud")

        wc = WordCloud(
            width=1000,
            height=500,
            background_color="white"
        ).generate(" ".join(processed_tokens))

        fig_wc, ax = plt.subplots(figsize=(12, 6))

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig_wc)

    # =====================================================
    # TAB 3 — CONCORDANCE
    # =====================================================
    with tab3:

        st.header("Concordance Analysis")

        keyword = st.text_input("Enter a keyword")

        if keyword:

            results = concordance(all_text, keyword)

            st.write(f"### Concordance Results for '{keyword}'")

            if results:

                for i, line in enumerate(results[:20]):
                    st.write(f"{i+1}. ...{line}...")

            else:
                st.warning("No occurrences found.")

    # =====================================================
    # TAB 4 — STIBBE CLASSIFICATION
    # =====================================================
    with tab4:

        st.header("Stibbe Category Classification")

        classification_results = []

        for article in df["article_text"].astype(str):

            doc = nlp(article)

            for sent in doc.sents:

                sentence = sent.text.strip()

                categories = classify_sentence(sentence)

                if categories:

                    classification_results.append({
                        "Sentence": sentence,
                        "Category": ", ".join(categories)
                    })

        if classification_results:

            classification_df = pd.DataFrame(classification_results)

            st.write("### Classified Sentences")
            st.dataframe(classification_df)

            # CATEGORY COUNTS
            all_categories = []

            for cats in classification_df["Category"]:
                split_cats = cats.split(", ")
                all_categories.extend(split_cats)

            category_counts = Counter(all_categories)

            category_df = pd.DataFrame({
                "Category": list(category_counts.keys()),
                "Count": list(category_counts.values())
            })

            # PIE CHART
            fig2 = px.pie(
                category_df,
                names="Category",
                values="Count",
                title="Distribution of Stibbe Categories"
            )

            st.plotly_chart(fig2, use_container_width=True)

            # DOWNLOAD BUTTON
            csv = classification_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Classification Results",
                data=csv,
                file_name="stibbe_classification.csv",
                mime="text/csv"
            )

        else:
            st.warning("No categories detected.")

    # =====================================================
    # TAB 5 — CDA INSIGHTS
    # =====================================================
    with tab5:

        st.header("CDA Insights")

        st.markdown("""
        ### Interpretation of Discourse Patterns

        This section provides basic computational interpretations
        inspired by Critical Discourse Analysis and Ecolinguistics.
        """)

        if 'category_counts' in locals():

            if "Metaphor" in category_counts:
                st.info("""
                Flood discourse frequently uses metaphorical framing,
                representing floods as conflict, warfare, or destructive force.
                """)

            if "Evaluation" in category_counts:
                st.info("""
                Evaluative language indicates emotional and ideological
                positioning within disaster reporting.
                """)

            if "Identity" in category_counts:
                st.info("""
                Identity constructions reveal how social actors such as
                victims, authorities, and communities are represented.
                """)

            if "Erasure" in category_counts:
                st.info("""
                Erasure patterns suggest passive constructions that may hide
                responsibility or agency.
                """)

            if "Ideology" in category_counts:
                st.info("""
                Ideological patterns reflect underlying assumptions about
                development, governance, climate, and responsibility.
                """)

else:

    st.info("Please upload a CSV dataset to begin.")

    st.write("### Required CSV Format")

    sample_df = pd.DataFrame({
        "date": ["2024-06-01"],
        "headline": ["Floods devastate Sindh"],
        "article_text": [
            "Pakistan is battling catastrophic floods causing devastation."
        ]
    })

    st.dataframe(sample_df)
