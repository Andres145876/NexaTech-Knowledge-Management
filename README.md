# NexaTech Knowledge Management System

A Python-based Knowledge Management System prototype developed for **NexaTech Solutions**, a fictional technology company. The application uses machine learning and text similarity techniques to organize, classify, and retrieve internal technical knowledge.

## Live Application

**Streamlit App:**  
https://nexatech-knowledge-management-vmi3bjmvscaqeyuurj9eha.streamlit.app/

## Project Overview

NexaTech Knowledge Management System is designed to centralize technical documentation and make organizational knowledge easier to access, classify, and manage.

The application provides an interactive interface where users can search for information, classify documents, add new knowledge resources, and explore the existing document collection.

## Main Features

### 1. Dashboard
Provides an overview of the knowledge base, including document statistics and category distribution.

### 2. Semantic Search
Allows users to search for information using natural-language queries. The system uses TF-IDF vectorization and cosine similarity to identify and rank the most relevant documents.

### 3. Document Classifier
Automatically predicts the category of a document using a machine learning model based on TF-IDF and Multinomial Naive Bayes.

### 4. Add Document
Allows users to add new documents to the knowledge base, including their title, category, and content.

### 5. Knowledge Base
Displays the stored documents and their associated information, making it easier to explore the organization's technical knowledge.

## Document Categories

The system organizes documents into four predefined categories:

- **Infrastructure**
- **Technical Support**
- **Cybersecurity**
- **Software Development**

The sample dataset contains **100 documents**, distributed evenly across the four categories.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| Pandas | Dataset loading and data management |
| Scikit-learn | Machine learning and text processing |
| TF-IDF | Text vectorization |
| Multinomial Naive Bayes | Document classification |
| Cosine Similarity | Semantic document retrieval |
| CSV | Knowledge base and results storage |
| pypdf | PDF text extraction |

The application also includes optional OCR support for scanned PDF documents through PyMuPDF, pytesseract, and Pillow. The Tesseract OCR engine must be installed separately to use this functionality.

## Repository Structure

```text
NexaTech-Knowledge-Management/
│
├── app.py
├── requirements.txt
├── README.md
│
├── DATA/
│   └── documents.csv
│
├── RESULTS/
│   └── classification_results.csv
│
└── SRC/
    ├── Classify_documents.py
    ├── check_documents.py
    ├── evaluate_classifier.py
    └── semantic_search.py
```

### Important Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit application containing all system modules. |
| `DATA/documents.csv` | Dataset containing the sample knowledge documents. |
| `SRC/Classify_documents.py` | Trains and evaluates the document classification model. |
| `SRC/semantic_search.py` | Implements semantic search using TF-IDF and cosine similarity. |
| `SRC/evaluate_classifier.py` | Evaluates classifier performance using stratified cross-validation. |
| `SRC/check_documents.py` | Utility for checking the document dataset. |
| `RESULTS/classification_results.csv` | Stores actual and predicted categories from the classifier test split. |
| `requirements.txt` | Lists the Python dependencies required by the project. |

## Installation and Local Execution

### 1. Clone the repository

```bash
git clone https://github.com/Andres145876/NexaTech-Knowledge-Management.git
```

### 2. Navigate to the project folder

```bash
cd NexaTech-Knowledge-Management
```

### 3. Create a virtual environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Once the application starts, Streamlit will provide a local URL. Normally, it is:

```text
http://localhost:8501
```

## Running the Machine Learning Scripts

Run these commands from the root directory of the repository.

### Document Classification

```bash
python SRC/Classify_documents.py
```

This script:

1. Loads the document dataset.
2. Combines document titles and content.
3. Splits the data into training and testing sets.
4. Converts text into TF-IDF features.
5. Trains a Multinomial Naive Bayes classifier.
6. Evaluates the model using accuracy and classification metrics.
7. Exports the test predictions to `RESULTS/classification_results.csv`.

### Semantic Search

```bash
python SRC/semantic_search.py
```

This script demonstrates natural-language searches and ranks documents according to their cosine similarity scores.

### Classifier Cross-Validation

```bash
python SRC/evaluate_classifier.py
```

This script evaluates the classification model using stratified 3-fold cross-validation and reports the accuracy for each fold, the mean accuracy, and the standard deviation.

## How Semantic Search Works

The semantic search module follows these steps:

1. Document titles and content are combined into a single text field.
2. TF-IDF converts the documents into numerical vectors.
3. The user's natural-language query is transformed using the same vectorizer.
4. Cosine similarity measures the similarity between the query and each document.
5. The system ranks the results and displays the most relevant documents.

## How Document Classification Works

The classification module follows these steps:

1. The document dataset is loaded from the CSV file.
2. Document titles and content are combined.
3. TF-IDF transforms the text into numerical features.
4. A Multinomial Naive Bayes model is trained using labeled examples.
5. The trained model predicts the category of new documents.

## Notes and Limitations

- The sample knowledge base is stored in a CSV file.
- Documents added while running a local copy are saved to the local dataset.
- The hosted Streamlit application uses its deployed dataset. Changes made to the hosted instance should not be considered permanent storage across deployments or restarts.
- For a production environment, a persistent database or external storage service would be recommended.
- The application supports text extraction from PDF and TXT files.
- OCR is optional and requires the Tesseract OCR engine to be installed separately.

## Project Purpose

This project was developed as an academic prototype to demonstrate how artificial intelligence and machine learning techniques can support organizational knowledge management.

By combining automated document classification, semantic information retrieval, and a centralized knowledge base, NexaTech aims to improve how technical information is organized, accessed, and shared within an organization.

---

**Project:** NexaTech Knowledge Management System  
**Repository:** https://github.com/Andres145876/NexaTech-Knowledge-Management  
**Application:** https://nexatech-knowledge-management-vmi3bjmvscaqeyuurj9eha.streamlit.app/
