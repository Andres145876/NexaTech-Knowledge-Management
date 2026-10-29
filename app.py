
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="NexaTech | Knowledge Management",
    page_icon="N",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "DATA" / "documents.csv"

REQUIRED_COLUMNS = [
    "document_id",
    "title",
    "category",
    "content"
]


# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #10243A, #245B78);
        color: #FFFFFF;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: #FFFFFF !important;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #E1EDF5 !important;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background: #F3F6FA;
        border: 1px solid #E0E7EF;
        padding: 14px;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #34495E !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] div,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] p {
        color: #10243A !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #245B78 !important;
    }

    h1, h2, h3 {
        color: inherit;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #E0E7EF;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 8px;
        font-weight: 600;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load and validate dataset
# --------------------------------------------------
@st.cache_data
def load_documents(file_path, modified_time):
    df = pd.read_csv(file_path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    if df.empty:
        raise ValueError("The dataset contains no documents.")

    for column in REQUIRED_COLUMNS:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df = df[
        (df["document_id"] != "")
        & (df["title"] != "")
        & (df["category"] != "")
        & (df["content"] != "")
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid documents remain after data validation."
        )

    if df["document_id"].duplicated().any():
        raise ValueError(
            "Duplicate document IDs were found in the dataset."
        )

    df["search_text"] = (
        df["title"] + " " + df["content"]
    )

    return df.reset_index(drop=True)


# --------------------------------------------------
# Build search and classification models
# --------------------------------------------------
@st.cache_resource
def build_models(texts, labels):
    search_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    document_vectors = search_vectorizer.fit_transform(texts)

    classifier = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )
        ),
        ("model", MultinomialNB())
    ])

    classifier.fit(texts, labels)

    return search_vectorizer, document_vectors, classifier


# --------------------------------------------------
# Load current data and models
# --------------------------------------------------
def refresh_project_data():
    load_documents.clear()
    build_models.clear()


try:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    documents = load_documents(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime
    )

    search_vectorizer, document_vectors, classifier = build_models(
        tuple(documents["search_text"]),
        tuple(documents["category"])
    )

except Exception as error:
    st.error(f"Could not load the project dataset: {error}")
    st.info(
        "Check that DATA/documents.csv exists and contains "
        "document_id, title, category and content columns."
    )
    st.stop()


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
st.sidebar.title("NexaTech")
st.sidebar.caption("Knowledge Management System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Semantic Search",
        "Document Classifier",
        "Add Document",
        "Knowledge Base"
    ]
)

st.sidebar.divider()
st.sidebar.metric("Available Documents", len(documents))
st.sidebar.caption("Academic prototype | Version 1.2")


# --------------------------------------------------
# Dashboard
# --------------------------------------------------
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>NexaTech Solutions</h1>
        <p>
        Knowledge Management System — organize, classify and
        retrieve technical knowledge in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Knowledge Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Documents", len(documents))
    col2.metric(
        "Knowledge Categories",
        documents["category"].nunique()
    )
    col3.metric(
        "Average Documents per Category",
        f"{len(documents) / documents['category'].nunique():.1f}"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Documents by Category")
        category_counts = documents["category"].value_counts()
        st.bar_chart(category_counts)

    with right:
        st.subheader("Knowledge Base Summary")

        summary = (
            documents.groupby("category")
            .size()
            .reset_index(name="Documents")
            .sort_values("category")
        )

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

    st.subheader("System Capabilities")

    col1, col2, col3 = st.columns(3)

    col1.info(
        "**Semantic Search**\n\n"
        "Find documents using text similarity."
    )

    col2.info(
        "**Document Classification**\n\n"
        "Predict a knowledge category for new text."
    )

    col3.info(
        "**Knowledge Base**\n\n"
        "Browse, filter and add documents."
    )

    st.caption(
        "Prototype note: results depend on the size and quality "
        "of the available sample dataset."
    )


# --------------------------------------------------
# Semantic Search
# --------------------------------------------------
elif page == "Semantic Search":
    st.title("Semantic Document Search")

    st.write(
        "Describe the information you need. The system ranks "
        "documents using TF-IDF and cosine similarity."
    )

    query = st.text_input(
        "Search query",
        placeholder="Example: How can we protect employee accounts?"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        categories = sorted(documents["category"].unique())

        selected_category = st.selectbox(
            "Category filter",
            ["All Categories"] + categories
        )

    with col2:
        top_n = st.selectbox(
            "Maximum results",
            [3, 5, 10],
            index=1
        )

    with col3:
        min_similarity = st.slider(
            "Minimum similarity score",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.05,
            help=(
                "This filters results by cosine similarity. "
                "It is not a probability of correctness."
            )
        )

    search_clicked = st.button(
        "Search Knowledge Base",
        type="primary"
    )

    if search_clicked:
        if not query.strip():
            st.warning("Enter a search query first.")

        else:
            query_vector = search_vectorizer.transform([query])

            scores = cosine_similarity(
                query_vector,
                document_vectors
            ).flatten()

            results = documents.copy()
            results["Similarity"] = scores

            if selected_category != "All Categories":
                results = results[
                    results["category"] == selected_category
                ]

            results = results[
                results["Similarity"] >= min_similarity
            ]

            results = results[
                results["Similarity"] > 0
            ].sort_values(
                "Similarity",
                ascending=False
            ).head(top_n)

            if results.empty:
                st.warning(
                    "No matching documents were found with these "
                    "filters. Try different keywords or lower the "
                    "minimum similarity score."
                )

            else:
                st.success(
                    f"{len(results)} matching document(s) found."
                )

                for _, item in results.iterrows():
                    with st.container(border=True):
                        st.subheader(item["title"])

                        st.caption(
                            f"{item['document_id']} | "
                            f"{item['category']}"
                        )

                        st.progress(
                            min(float(item["Similarity"]), 1.0)
                        )

                        st.write(
                            f"Similarity score: "
                            f"{item['Similarity']:.4f}"
                        )

                        with st.expander("View document content"):
                            st.write(item["content"])

                st.caption(
                    "Method: TF-IDF vectorization and cosine "
                    "similarity. Scores measure textual similarity, "
                    "not factual correctness or probability."
                )


# --------------------------------------------------
# Document Classifier
# --------------------------------------------------
elif page == "Document Classifier":
    st.title("Document Classification")

    st.write(
        "Enter a title or content to predict the most likely "
        "knowledge category."
    )

    new_title = st.text_input(
        "Document title",
        placeholder="Example: Employee Password Guidelines"
    )

    new_content = st.text_area(
        "Document content",
        placeholder="Enter the technical document text here...",
        height=180
    )

    if st.button("Classify Document", type="primary"):
        combined_text = (
            f"{new_title} {new_content}"
        ).strip()

        if not combined_text:
            st.warning(
                "Enter a document title or document content."
            )

        else:
            prediction = classifier.predict([combined_text])[0]

            probabilities = classifier.predict_proba(
                [combined_text]
            )[0]

            classes = classifier.named_steps["model"].classes_

            confidence = float(probabilities.max())

            st.subheader("Predicted Category")
            st.success(prediction)

            st.metric(
                "Model Score for Predicted Category",
                f"{confidence:.1%}"
            )

            st.subheader("Scores by Category")

            probability_table = pd.DataFrame({
                "Category": classes,
                "Model Score": probabilities
            }).sort_values(
                "Model Score",
                ascending=False
            )

            st.bar_chart(
                probability_table.set_index("Category")
            )

            display_table = probability_table.copy()
            display_table["Model Score"] = display_table[
                "Model Score"
            ].map(lambda score: f"{score:.1%}")

            st.dataframe(
                display_table,
                hide_index=True,
                use_container_width=True
            )

            st.warning(
                "This is an automated suggestion, not a verified "
                "classification. The score is not a guarantee of "
                "correctness. Review the result before using it."
            )

            st.caption(
                "The prototype classifier is trained on the current "
                "sample dataset. Its reliability depends on the "
                "quantity and diversity of training documents."
            )


# --------------------------------------------------
# Add Document
# --------------------------------------------------
elif page == "Add Document":
    st.title("Add Knowledge Document")

    st.write(
        "Add a new document to the NexaTech knowledge base. "
        "The document will be stored in the project CSV file."
    )

    title = st.text_input(
        "Document title",
        placeholder="Example: Network Security Guidelines",
        key="add_document_title"
    )

    content = st.text_area(
        "Document content",
        placeholder="Enter the technical document content...",
        height=220,
        key="add_document_content"
    )

    combined_text = f"{title.strip()} {content.strip()}".strip()

    categories = sorted(documents["category"].unique().tolist())

    if combined_text:
        suggested_category = classifier.predict([combined_text])[0]
        probabilities = classifier.predict_proba([combined_text])[0]
        confidence = float(probabilities.max())

        st.caption(
            f"Suggested category: {suggested_category} "
            f"(model score: {confidence:.1%})"
        )

        st.caption(
            "The model score is an estimate from the classifier, "
            "not a guarantee of correctness."
        )
    else:
        suggested_category = categories[0]

    category_options = categories.copy()

    if suggested_category not in category_options:
        category_options.append(suggested_category)
        category_options.sort()

    default_index = category_options.index(suggested_category)

    category = st.selectbox(
        "Document category",
        category_options,
        index=default_index,
        key="add_document_category"
    )

    if st.button("Save Document", type="primary"):
        if not title.strip() or not content.strip():
            st.error(
                "Both the document title and content are required."
            )

        else:
            try:
                # Reload the current CSV before writing, so that
                # the operation uses the latest saved documents.
                current_documents = pd.read_csv(
                    DATA_FILE,
                    dtype=str
                ).fillna("")

                missing = (
                    set(REQUIRED_COLUMNS)
                    - set(current_documents.columns)
                )

                if missing:
                    raise ValueError(
                        "The dataset is missing required columns: "
                        + ", ".join(sorted(missing))
                    )

                current_documents["document_id"] = (
                    current_documents["document_id"]
                    .astype(str)
                    .str.strip()
                )

                current_documents["title"] = (
                    current_documents["title"]
                    .astype(str)
                    .str.strip()
                )

                current_documents["category"] = (
                    current_documents["category"]
                    .astype(str)
                    .str.strip()
                )

                current_documents["content"] = (
                    current_documents["content"]
                    .astype(str)
                    .str.strip()
                )

                duplicate_title = current_documents[
                    "title"
                ].str.casefold().eq(
                    title.strip().casefold()
                )

                duplicate_content = current_documents[
                    "content"
                ].str.casefold().eq(
                    content.strip().casefold()
                )

                if duplicate_title.any() and duplicate_content.any():
                    st.warning(
                        "A document with the same title and content "
                        "already exists."
                    )

                else:
                    existing_ids = set(
                        current_documents["document_id"]
                    )

                    new_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"

                    while new_id in existing_ids:
                        new_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"

                    new_document = pd.DataFrame([{
                        "document_id": new_id,
                        "title": title.strip(),
                        "category": category,
                        "content": content.strip()
                    }])

                    updated_documents = pd.concat(
                        [
                            current_documents[REQUIRED_COLUMNS],
                            new_document
                        ],
                        ignore_index=True
                    )

                    DATA_FILE.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    updated_documents.to_csv(
                        DATA_FILE,
                        index=False,
                        encoding="utf-8"
                    )

                    refresh_project_data()

                    st.session_state["document_saved"] = {
                        "id": new_id,
                        "title": title.strip()
                    }

                    st.rerun()

            except Exception as error:
                st.error(
                    f"Could not save the document: {error}"
                )

    saved_document = st.session_state.pop(
        "document_saved",
        None
    )

    if saved_document:
        st.success(
            f"Document '{saved_document['title']}' saved successfully."
        )

        st.write(f"Document ID: {saved_document['id']}")

        st.info(
            "The knowledge base, search index and classifier "
            "have been refreshed. The new document is now available."
        )


# --------------------------------------------------
# Knowledge Base
# --------------------------------------------------
elif page == "Knowledge Base":
    st.title("Knowledge Base")

    st.write(
        "Browse, filter and inspect the documents in the dataset."
    )

    col1, col2 = st.columns(2)

    categories = sorted(documents["category"].unique())

    with col1:
        selected_category = st.selectbox(
            "Filter by category",
            ["All Categories"] + categories,
            key="knowledge_category_filter"
        )

    with col2:
        search_term = st.text_input(
            "Search title or content",
            placeholder="Enter keywords...",
            key="knowledge_search_term"
        )

    filtered = documents.copy()

    if selected_category != "All Categories":
        filtered = filtered[
            filtered["category"] == selected_category
        ]

    if search_term.strip():
        mask = (
            filtered["title"].str.contains(
                search_term,
                case=False,
                na=False,
                regex=False
            )
            | filtered["content"].str.contains(
                search_term,
                case=False,
                na=False,
                regex=False
            )
            | filtered["document_id"].str.contains(
                search_term,
                case=False,
                na=False,
                regex=False
            )
        )

        filtered = filtered[mask]

    st.caption(
        f"{len(filtered)} of {len(documents)} document(s) displayed."
    )

    st.dataframe(
        filtered[REQUIRED_COLUMNS],
        hide_index=True,
        use_container_width=True
    )

    if filtered.empty:
        st.info(
            "No documents match these filters. Change the category "
            "or try another keyword."
        )
