"""
Topic 5 - Text classification tests
Test 1: compare features (Bag-of-Words vs TF-IDF, unigrams vs bigrams) and
        models (Naive Bayes, Logistic Regression, Linear SVM) with 5-fold CV.
Test 2: effect of the Naive Bayes smoothing parameter alpha.
Test 3: short informal texts like the ones employees type.

Contribution: Test 2 found why the first classifier favoured Infrastructure
(alpha=1.0) and led to alpha=0.1 in app.py and SRC/Classify_documents.py.
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from common import TOKENS, Report, load_documents

r = Report("topic5_classification_tests")
docs = load_documents()
X, y = docs["text"], docs["category"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

features = {
    "BoW unigrams": lambda: CountVectorizer(stop_words="english", token_pattern=TOKENS),
    "TF-IDF unigrams": lambda: TfidfVectorizer(stop_words="english", token_pattern=TOKENS),
    "TF-IDF uni+bigrams": lambda: TfidfVectorizer(stop_words="english", token_pattern=TOKENS,
                                                  ngram_range=(1, 2)),
}
models = {
    "Naive Bayes a=0.1": lambda: MultinomialNB(alpha=0.1),
    "Logistic Regr.": lambda: LogisticRegression(max_iter=3000, C=10, class_weight="balanced"),
    "Linear SVM": lambda: LinearSVC(),
}
r(f"Test 1 - 5-fold cross-validation accuracy ({len(docs)} documents)")
r(f"{'Features':<22}" + "".join(f"{m:>19}" for m in models))
r("-" * 79)
for fname, make_features in features.items():
    row = f"{fname:<22}"
    for make_model in models.values():
        row += f"{cross_val_score(make_pipeline(make_features(), make_model()), X, y, cv=cv).mean():>19.3f}"
    r(row)

r("\nTest 2 - Naive Bayes smoothing (TF-IDF uni+bigrams, 5-fold CV)")
r(f"{'alpha':>8}{'accuracy':>10}   recall per category (Cyber / Infra / SoftDev / Support)")
for alpha in [1.0, 0.5, 0.3, 0.1, 0.05]:
    model = make_pipeline(features["TF-IDF uni+bigrams"](), MultinomialNB(alpha=alpha))
    predictions = cross_val_predict(model, X, y, cv=cv)
    accuracy = (predictions == y).mean()
    recalls = recall_score(y, predictions, average=None,
                           labels=["Cybersecurity", "Infrastructure", "Software Development", "Technical Support"])
    infra_share = (predictions == "Infrastructure").mean()
    r(f"{alpha:>8}{accuracy:>10.3f}   " + " / ".join(f"{v:.2f}" for v in recalls)
      + f"   (predicted Infrastructure: {infra_share:.0%} of docs, real: {(y == 'Infrastructure').mean():.0%})")

r("\nTest 3 - short informal texts typed by employees (TF-IDF + NB alpha=0.1)")
model = make_pipeline(features["TF-IDF uni+bigrams"](), MultinomialNB(alpha=0.1))
model.fit(X, y)
for text in ["my laptop does not turn on and the printer is not working",
             "someone tried to log in to my account from another country",
             "the deployment pipeline failed after the last commit",
             "the server room temperature is too high",
             "employees must not share confidential files"]:
    r(f"  {model.predict([text])[0]:<22} <- \"{text}\"")
r.save()
