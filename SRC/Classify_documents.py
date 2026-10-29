
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "DATA" / "documents.csv"
RESULTS_DIR = PROJECT_DIR / "RESULTS"
RESULTS_DIR.mkdir(exist_ok=True)

# Load the document dataset
documents = pd.read_csv(DATA_FILE)

X = documents["title"] + " " + documents["content"]
y = documents["category"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create and train the classification pipeline
classifier = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("model", MultinomialNB())
])

classifier.fit(X_train, y_train)

# Evaluate the model
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("DOCUMENT CLASSIFICATION RESULTS")
print("=" * 40)
print(f"Total documents: {len(documents)}")
print(f"Training documents: {len(X_train)}")
print(f"Testing documents: {len(X_test)}")
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save predictions for the test documents
test_results = pd.DataFrame({
    "document_text": X_test,
    "actual_category": y_test,
    "predicted_category": predictions
})

test_results.to_csv(
    RESULTS_DIR / "classification_results.csv",
    index=False
)

# Classify a new example
new_document = (
    "Employees must enable multi-factor authentication, "
    "protect passwords, and report suspicious emails."
)

predicted_category = classifier.predict([new_document])[0]

print("\nNEW DOCUMENT TEST")
print(f"Document: {new_document}")
print(f"Predicted category: {predicted_category}")
print("\nResults saved to RESULTS/classification_results.csv")
