
print("TEST: The Python script is running!")

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "DATA" / "documents.csv"

print(f"Dataset path: {DATA_FILE}")

if not DATA_FILE.exists():
    print("ERROR: documents.csv was not found.")
else:
    documents = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully!")
    print(f"Number of documents: {len(documents)}")

    print("\nDocuments per category:")
    print(documents["category"].value_counts())

    print("\nFirst five documents:")
    print(documents[["document_id", "title", "category"]].head())
