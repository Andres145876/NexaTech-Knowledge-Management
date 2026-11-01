"""
Topic 2 - Text representation: preprocessing, Bag-of-Words and TF-IDF
Follow how raw text becomes a vector: tokenization -> lowercase -> stop words
-> simple stemming -> Bag-of-Words -> TF-IDF (by hand and with scikit-learn).

Contribution: justified the TF-IDF settings (stop words, bigrams) used in
Classify_documents.py, semantic_search.py and app.py.
"""
import math
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from common import TOKENS, Report, load_documents


def simple_stem(word):
    for suffix in ("ations", "ation", "ments", "ment", "ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


r = Report("topic2_text_preprocessing")
docs = load_documents().set_index("document_id")
sentence = docs.loc["DOC059", "content"].split(". ")[0] + "."
r(f"Original sentence (DOC059 Multi-Factor Authentication Policy):\n  {sentence}\n")
tokens = re.findall(r"[A-Za-z]+", sentence)
r(f"1) Tokens ({len(tokens)}): {tokens}")
lower = [t.lower() for t in tokens]
no_stop = [t for t in lower if t not in ENGLISH_STOP_WORDS]
r(f"2) Without stop words ({len(no_stop)}): {no_stop}")
r(f"3) Stemmed: {[simple_stem(t) for t in no_stop]}\n")

ids = ["DOC021", "DOC059", "DOC071"]   # password reset, MFA policy, password management
mini = [docs.loc[i, "content"] for i in ids]
cv = CountVectorizer(stop_words="english", token_pattern=TOKENS)
bow = cv.fit_transform(mini)
r(f"4) Bag-of-Words for {ids}: matrix {bow.shape}")
for term in [t for t in ["password", "passwords", "accounts", "authentication", "employees"] if t in cv.vocabulary_]:
    r(f"     count('{term}') = {bow[:, cv.vocabulary_[term]].toarray().ravel().tolist()}")

r("\n5) TF-IDF by hand  (idf = ln((1+N)/(1+df)) + 1, N=3) for document DOC059:")
for term in [t for t in ["accounts", "password", "authentication", "mfa"] if t in cv.vocabulary_]:
    column = bow[:, cv.vocabulary_[term]].toarray().ravel()
    df = int((column > 0).sum())
    idf = math.log(4 / (1 + df)) + 1
    r(f"     {term:<15} tf(DOC059)={column[1]} df={df} idf={idf:.3f} tf-idf={column[1] * idf:.3f}")
r("   -> terms present in fewer documents get a higher idf (more discriminative).")

tfidf = TfidfVectorizer(stop_words="english", token_pattern=TOKENS)
matrix = tfidf.fit_transform(mini)
row = matrix[1].toarray().ravel()
names = tfidf.get_feature_names_out()
r("\n6) scikit-learn TF-IDF (L2-normalized), top terms of DOC059:")
for index in row.argsort()[::-1][:6]:
    r(f"     {names[index]:<14} {row[index]:.3f}")
r.save()
