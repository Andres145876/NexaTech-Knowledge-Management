
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "DATA" / "documents.csv"

documents = pd.read_csv(DATA_FILE)

# Combine document titles and content
documents["search_text"] = (
    documents["title"].fillna("") + " " +
    documents["content"].fillna("")
)

# Convert documents into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(documents["search_text"])

def search_documents(query, top_n=5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    results = documents.copy()
    results["similarity_score"] = similarities

    results = results.sort_values(
        by="similarity_score",
        ascending=False
    ).head(top_n)

    results = results[results["similarity_score"] > 0]

    return results[
        ["document_id", "title", "category", "similarity_score"]
    ]

if __name__ == "__main__":
    print("NEXATECH SEMANTIC DOCUMENT SEARCH")
    print("=" * 45)

    example_queries = [
        "How can we protect employee accounts and passwords?",
        "How do developers test and fix software bugs?"
    ]

    for query in example_queries:
        print(f"\nQuery: {query}")
        results = search_documents(query)

        if results.empty:
            print("No relevant documents found.")
        else:
            print(results.to_string(
                index=False,
                formatters={
                    "similarity_score": "{:.4f}".format
                }
            ))
