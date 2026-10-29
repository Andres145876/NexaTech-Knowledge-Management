
from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "DATA" / "documents.csv"

documents = pd.read_csv(DATA_FILE)

X = documents["title"].fillna("") + " " + documents["content"].fillna("")
y = documents["category"]

classifier = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("model", MultinomialNB())
])

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    classifier,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("CROSS-VALIDATION RESULTS")
print("=" * 35)
print(f"Total documents: {len(documents)}")
print(f"Categories: {y.nunique()}")
print(f"Accuracy per fold: {scores}")
print(f"Mean accuracy: {scores.mean():.2%}")
print(f"Standard deviation: {scores.std():.2%}")
