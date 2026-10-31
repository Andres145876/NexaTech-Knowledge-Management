from pathlib import Path
from io import BytesIO
import uuid

from pypdf import PdfReader

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

/* ----------------------------------------
   MAIN CONTENT
---------------------------------------- */

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.2rem;
    padding-left: clamp(1rem, 4vw, 4rem);
    padding-right: clamp(1rem, 4vw, 4rem);
    max-width: 100% !important;
    width: 100%;
    margin-left: auto;
    margin-right: auto;
}

/* ----------------------------------------
   HERO
---------------------------------------- */

.hero {
    padding: 1.8rem 2rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #102B46 0%, #176B91 100%);
    color: #FFFFFF;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(16, 43, 70, 0.12);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 2.1rem;
    margin-bottom: 0.45rem;
}

.hero p {
    color: #E6F2F8 !important;
    font-size: 1.02rem;
    margin-bottom: 0;
}

/* ----------------------------------------
   METRICS
---------------------------------------- */

div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.25);
    padding: 16px 18px;
    border-radius: 14px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    color: var(--text-color) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] div,
div[data-testid="stMetric"] [data-testid="stMetricValue"] p {
    color: var(--text-color) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #3286B5 !important;
}

/* ----------------------------------------
   HEADINGS AND CONTAINERS
---------------------------------------- */

h1, h2, h3 {
    color: var(--text-color);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}

/* ----------------------------------------
   SIDEBAR ADAPTIVE THEME
---------------------------------------- */

section[data-testid="stSidebar"] {
    width: 330px !important;
    min-width: 330px !important;
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border-right: 1px solid rgba(128, 128, 128, 0.25);
}

section[data-testid="stSidebar"] > div {
    width: 330px !important;
}

section[data-testid="stSidebar"] .stRadio > label {
    color: var(--text-color) !important;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.55rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: var(--background-color) !important;
    border: 1px solid rgba(128, 128, 128, 0.28);
    border-radius: 11px;
    padding: 0.72rem 0.85rem;
    min-height: 48px;
    transition: all 0.15s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    border-color: #3286B5;
    background-color: var(--secondary-background-color) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-color) !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    color: var(--text-color) !important;
    font-size: 1.8rem;
    letter-spacing: -0.5px;
    margin-bottom: 0.1rem;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-color) !important;
}

/* NexaTech branding */

section[data-testid="stSidebar"] .stMarkdown div[style*="linear-gradient"] {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] .stMarkdown div[style*="linear-gradient"] div {
    color: #FFFFFF !important;
}

/* ----------------------------------------
   BUTTONS AND TABLES
---------------------------------------- */

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 42px;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ----------------------------------------
   RESPONSIVE MAIN CONTENT
---------------------------------------- */

section[data-testid="stMain"] {
    width: 100% !important;
    min-width: 0 !important;
}

section[data-testid="stMain"] > div {
    width: 100% !important;
    min-width: 0 !important;
}

.stMainBlockContainer {
    width: 100% !important;
    max-width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Distribute columns evenly */

div[data-testid="stHorizontalBlock"] {
    width: 100%;
    justify-content: center;
}

/* Prevent content overflow */

div[data-testid="stVerticalBlock"] {
    max-width: 100%;
}

/* ----------------------------------------
   SMALL SCREENS
---------------------------------------- */

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 1.3rem;
    }

    .hero h1 {
        font-size: 1.7rem;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Extract text from PDF and TXT files
# --------------------------------------------------
def extract_uploaded_text(uploaded_file):
    """Extract text from PDF/TXT, using OCR when a PDF has no text layer."""
    file_extension = Path(uploaded_file.name).suffix.lower()
    used_ocr = False

    if file_extension == ".txt":
        raw_content = uploaded_file.getvalue()

        try:
            extracted_text = raw_content.decode("utf-8-sig")
        except UnicodeDecodeError:
            extracted_text = raw_content.decode("latin-1")

    elif file_extension == ".pdf":
        pdf_bytes = uploaded_file.getvalue()
        reader = PdfReader(BytesIO(pdf_bytes))

        if reader.is_encrypted:
            raise ValueError(
                "This PDF is password-protected. Upload an unlocked PDF."
            )

        extracted_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).replace("\x00", "").strip()

        if not extracted_text:
            try:
                import fitz
                import pytesseract
                from PIL import Image
            except ImportError as error:
                raise RuntimeError(
                    "This PDF appears to be scanned. To read it, install the "
                    "OCR packages with: pip install pymupdf pytesseract pillow"
                ) from error

            try:
                pdf_document = fitz.open(
                    stream=pdf_bytes,
                    filetype="pdf"
                )

                ocr_pages = []

                for page in pdf_document:
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False
                    )

                    page_image = Image.open(
                        BytesIO(pixmap.tobytes("png"))
                    )

                    page_text = pytesseract.image_to_string(
                        page_image,
                        lang="eng"
                    ).strip()

                    if page_text:
                        ocr_pages.append(page_text)

                pdf_document.close()

                extracted_text = "\n".join(ocr_pages).strip()
                used_ocr = True

            except Exception as error:
                error_message = str(error)

                if (
                    "tesseract" in error_message.lower()
                    or "not installed" in error_message.lower()
                ):
                    raise RuntimeError(
                        "OCR could not start. Install Tesseract OCR on Windows "
                        "and make sure it is available in PATH."
                    ) from error

                raise RuntimeError(
                    f"OCR could not process this PDF: {error}"
                ) from error

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    extracted_text = extracted_text.replace("\x00", "").strip()

    if not extracted_text:
        if used_ocr:
            raise ValueError(
                "OCR did not detect readable text in this scanned PDF."
            )

        raise ValueError("No readable text was found in this file.")

    return extracted_text, used_ocr


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
st.sidebar.markdown("""
<div style="
    padding: 1.15rem 1rem 1.05rem 1rem;
    margin: 0.15rem 0 1rem 0;
    border-radius: 14px;
    background: linear-gradient(125deg, #102B46, #176B91);
    color: white;">
    <div style="font-size: 1.65rem; font-weight: 800; letter-spacing: -0.4px;">
        NexaTech
    </div>
    <div style="font-size: 0.88rem; color: #E6F2F8; margin-top: 0.25rem;">
        Knowledge Management System
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Semantic Search",
        "Document Classifier",
        "Add Document",
        "Knowledge Base"
    ],
    label_visibility="visible"
)

st.sidebar.divider()

st.sidebar.metric(
    "Available Documents",
    len(documents)
)

st.sidebar.caption("Academic prototype | Version 1.3")

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

    col1.metric(
        "Total Documents",
        len(documents)
    )

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

    col1, col2, col3, col4 = st.columns(4)

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

    col4.info(
        "**Add Documents**\n\n"
        "Upload and add new documents to the knowledge base."
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
        categories = sorted(
            documents["category"].unique()
        )

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
            prediction = classifier.predict([
                combined_text
            ])[0]

            probabilities = classifier.predict_proba([
                combined_text
            ])[0]

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
            ].map(
                lambda score: f"{score:.1%}"
            )

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

    # --------------------------------------------------
    # Import multiple documents from a CSV file
    # --------------------------------------------------
    st.subheader("Import CSV Dataset")

    csv_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="csv_dataset_upload",
        help=(
            "The CSV must contain document_id, title, category "
            "and content columns."
        )
    )

    if csv_file is not None:
        try:
            csv_documents = pd.read_csv(
                BytesIO(csv_file.getvalue()),
                dtype=str
            ).fillna("")

            missing_csv_columns = (
                set(REQUIRED_COLUMNS) - set(csv_documents.columns)
            )

            if missing_csv_columns:
                raise ValueError(
                    "The CSV is missing these required columns: "
                    + ", ".join(sorted(missing_csv_columns))
                )

            csv_documents = csv_documents[
                REQUIRED_COLUMNS
            ].copy()

            for column in REQUIRED_COLUMNS:
                csv_documents[column] = (
                    csv_documents[column]
                    .astype(str)
                    .str.strip()
                )

            if csv_documents.empty:
                raise ValueError(
                    "The uploaded CSV contains no documents."
                )

            if csv_documents.eq("").any().any():
                raise ValueError(
                    "Every imported row must have an ID, title, category and content."
                )

            if csv_documents["document_id"].duplicated().any():
                raise ValueError(
                    "The uploaded CSV contains duplicate document IDs."
                )

            st.write(
                f"Documents detected: {len(csv_documents)}"
            )

            st.dataframe(
                csv_documents.head(10),
                hide_index=True
            )

            if st.button(
                "Import CSV Documents",
                type="primary",
                key="import_csv_documents"
            ):
                current_documents = pd.read_csv(
                    DATA_FILE,
                    dtype=str
                ).fillna("")

                current_documents = current_documents[
                    REQUIRED_COLUMNS
                ].copy()

                existing_ids = set(
                    current_documents["document_id"]
                    .astype(str)
                    .str.strip()
                )

                imported_ids = set(
                    csv_documents["document_id"]
                )

                conflicting_ids = existing_ids.intersection(
                    imported_ids
                )

                if conflicting_ids:
                    st.error(
                        "Import cancelled. These document IDs already exist: "
                        + ", ".join(sorted(conflicting_ids)[:10])
                    )

                else:
                    existing_pairs = set(
                        zip(
                            current_documents["title"].str.casefold(),
                            current_documents["content"].str.casefold()
                        )
                    )

                    new_rows = []
                    skipped_duplicates = 0

                    for _, imported_row in csv_documents.iterrows():
                        pair = (
                            imported_row["title"].casefold(),
                            imported_row["content"].casefold()
                        )

                        if pair in existing_pairs:
                            skipped_duplicates += 1
                            continue

                        new_rows.append(
                            imported_row.to_dict()
                        )

                        existing_pairs.add(pair)

                    if not new_rows:
                        st.warning(
                            "No new documents were imported. "
                            "All uploaded documents already exist."
                        )

                    else:
                        updated_documents = pd.concat(
                            [
                                current_documents,
                                pd.DataFrame(
                                    new_rows,
                                    columns=REQUIRED_COLUMNS
                                )
                            ],
                            ignore_index=True
                        )

                        updated_documents.to_csv(
                            DATA_FILE,
                            index=False,
                            encoding="utf-8"
                        )

                        refresh_project_data()

                        st.session_state["csv_import_result"] = {
                            "imported": len(new_rows),
                            "skipped": skipped_duplicates
                        }

                        st.rerun()

        except Exception as error:
            st.error(
                f"Could not process the CSV file: {error}"
            )

    csv_import_result = st.session_state.pop(
        "csv_import_result",
        None
    )

    if csv_import_result:
        st.success(
            f"{csv_import_result['imported']} document(s) imported successfully."
        )

        if csv_import_result["skipped"]:
            st.info(
                f"{csv_import_result['skipped']} duplicate document(s) were skipped."
            )

        st.info(
            "The knowledge base, search index and classifier have been refreshed."
        )

    st.divider()

    # --------------------------------------------------
    # Import multiple documents from PDF or TXT
    # --------------------------------------------------
    st.subheader("Import Multiple Documents from PDF or TXT")

    st.write(
        "Select several PDF or TXT files. NexaTech will extract their text, "
        "suggest a category and import the documents together."
    )

    uploaded_files = st.file_uploader(
        "Select PDF and TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="multiple_document_upload",
        help=(
            "You can select multiple PDFs and TXT files. Scanned PDFs are "
            "processed with OCR when Tesseract is installed."
        )
    )

    if uploaded_files:
        st.caption(
            f"{len(uploaded_files)} file(s) selected."
        )

        st.dataframe(
            pd.DataFrame({
                "File name": [
                    file.name for file in uploaded_files
                ]
            }),
            hide_index=True,
            use_container_width=True
        )

        if st.button(
            "Process and Import Files",
            type="primary",
            key="process_multiple_documents"
        ):
            try:
                current_documents = pd.read_csv(
                    DATA_FILE,
                    dtype=str
                ).fillna("")

                current_documents = current_documents[
                    REQUIRED_COLUMNS
                ].copy()

                existing_ids = set(
                    current_documents["document_id"].astype(str)
                )

                existing_pairs = set(
                    zip(
                        current_documents["title"].str.casefold(),
                        current_documents["content"].str.casefold()
                    )
                )

                new_rows = []
                imported_names = []
                skipped_names = []
                failed_files = []
                ocr_names = []

                for uploaded_file in uploaded_files:
                    try:
                        extracted_text, used_ocr = extract_uploaded_text(
                            uploaded_file
                        )

                        if used_ocr:
                            ocr_names.append(
                                uploaded_file.name
                            )

                        document_title = (
                            Path(uploaded_file.name).stem.strip()
                        )

                        duplicate_pair = (
                            document_title.casefold(),
                            extracted_text.casefold()
                        )

                        if duplicate_pair in existing_pairs:
                            skipped_names.append(
                                f"{uploaded_file.name}: duplicate document"
                            )
                            continue

                        predicted_category = classifier.predict([
                            f"{document_title} {extracted_text}"
                        ])[0]

                        new_id = (
                            f"DOC-{uuid.uuid4().hex[:8].upper()}"
                        )

                        while new_id in existing_ids:
                            new_id = (
                                f"DOC-{uuid.uuid4().hex[:8].upper()}"
                            )

                        existing_ids.add(new_id)
                        existing_pairs.add(duplicate_pair)

                        new_rows.append({
                            "document_id": new_id,
                            "title": document_title,
                            "category": predicted_category,
                            "content": extracted_text
                        })

                        imported_names.append(
                            uploaded_file.name
                        )

                    except Exception as file_error:
                        failed_files.append(
                            f"{uploaded_file.name}: {file_error}"
                        )

                if new_rows:
                    updated_documents = pd.concat(
                        [
                            current_documents,
                            pd.DataFrame(
                                new_rows,
                                columns=REQUIRED_COLUMNS
                            )
                        ],
                        ignore_index=True
                    )

                    updated_documents.to_csv(
                        DATA_FILE,
                        index=False,
                        encoding="utf-8"
                    )

                    refresh_project_data()

                st.session_state["multiple_import_result"] = {
                    "imported": imported_names,
                    "skipped": skipped_names,
                    "failed": failed_files,
                    "ocr": ocr_names
                }

                st.rerun()

            except Exception as error:
                st.error(
                    f"Could not import the selected files: {error}"
                )

    multiple_import_result = st.session_state.pop(
        "multiple_import_result",
        None
    )

    if multiple_import_result:
        imported_names = multiple_import_result["imported"]
        skipped_names = multiple_import_result["skipped"]
        failed_files = multiple_import_result["failed"]
        ocr_names = multiple_import_result.get("ocr", [])

        if imported_names:
            st.success(
                f"{len(imported_names)} document(s) imported successfully."
            )

            with st.expander("View imported files"):
                for file_name in imported_names:
                    st.write(f"- {file_name}")

        if ocr_names:
            st.info(
                f"OCR was used on {len(ocr_names)} scanned PDF file(s)."
            )

            with st.expander("View files processed with OCR"):
                for file_name in ocr_names:
                    st.write(f"- {file_name}")

        if skipped_names:
            st.warning(
                f"{len(skipped_names)} duplicate file(s) were skipped."
            )

            with st.expander("View skipped files"):
                for message in skipped_names:
                    st.write(f"- {message}")

        if failed_files:
            st.error(
                f"{len(failed_files)} file(s) could not be processed."
            )

            with st.expander("View files with errors"):
                for message in failed_files:
                    st.write(f"- {message}")

        if imported_names:
            st.info(
                "The knowledge base, search index and classifier have been refreshed."
            )

    st.divider()

    # --------------------------------------------------
    # Add a document manually
    # --------------------------------------------------
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

    combined_text = (
        f"{title.strip()} {content.strip()}"
    ).strip()

    categories = sorted(
        documents["category"].unique().tolist()
    )

    if combined_text:
        suggested_category = classifier.predict([
            combined_text
        ])[0]

        probabilities = classifier.predict_proba([
            combined_text
        ])[0]

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

    default_index = category_options.index(
        suggested_category
    )

    category = st.selectbox(
        "Document category",
        category_options,
        index=default_index,
        key="add_document_category"
    )

    if st.button(
        "Save Document",
        type="primary"
    ):
        if not title.strip() or not content.strip():
            st.error(
                "Both the document title and content are required."
            )

        else:
            try:
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

                for column in REQUIRED_COLUMNS:
                    current_documents[column] = (
                        current_documents[column]
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

                    new_id = (
                        f"DOC-{uuid.uuid4().hex[:8].upper()}"
                    )

                    while new_id in existing_ids:
                        new_id = (
                            f"DOC-{uuid.uuid4().hex[:8].upper()}"
                        )

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

        st.write(
            f"Document ID: {saved_document['id']}"
        )

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

    categories = sorted(
        documents["category"].unique()
    )

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
        