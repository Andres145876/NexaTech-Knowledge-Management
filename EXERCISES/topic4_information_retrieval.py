"""
Topic 4 - Information retrieval: Boolean model vs vector space model
Build an inverted index, run Boolean AND queries, compare them with ranked
TF-IDF + cosine retrieval and evaluate with manual relevance judgments
(Precision@3, Recall@3, Average Precision).

Contribution: showed that ranked TF-IDF retrieval works for natural-language
questions, so it was chosen for semantic_search.py and app.py.
"""
import re
from collections import defaultdict

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from common import Report, load_documents

JUDGMENTS = {  # relevant documents chosen manually by the team
    "password and account security for employees": {"DOC059", "DOC071", "DOC021", "DOC068", "DOC075"},
    "server backup and data recovery": {"DOC011", "DOC006", "DOC051", "DOC077"},
    "software testing and bug fixing": {"DOC084", "DOC092", "DOC094"},
    "phishing malware security incident": {"DOC055", "DOC056", "DOC058", "DOC070", "DOC076"},
}


def tokenize(text):
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in ENGLISH_STOP_WORDS]


def average_precision(ranked, relevant):
    hits, total = 0, 0.0
    for k, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            hits += 1
            total += hits / k
    return total / len(relevant)


r = Report("topic4_information_retrieval")
docs = load_documents()
inverted = defaultdict(set)
for _, d in docs.iterrows():
    for token in tokenize(d["text"]):
        inverted[token].add(d["document_id"])
r(f"Inverted index: {len(inverted)} terms over {len(docs)} documents")
r(f"  postings('backup')   = {sorted(inverted['backup'])}")
r(f"  postings('phishing') = {sorted(inverted['phishing'])}\n")

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
matrix = vectorizer.fit_transform(docs["text"])
ids = docs["document_id"].tolist()

r(f"{'Query':<46}{'Boolean AND':>12}{'P@3':>7}{'R@3':>7}{'AP':>7}")
r("-" * 79)
aps = []
for query, relevant in JUDGMENTS.items():
    boolean = set.intersection(*(inverted.get(t, set()) for t in tokenize(query)))
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    ranked = [ids[i] for i in scores.argsort()[::-1][:10]]
    top3 = ranked[:3]
    p3 = len(set(top3) & relevant) / 3
    r3 = len(set(top3) & relevant) / len(relevant)
    ap = average_precision(ranked, relevant)
    aps.append(ap)
    r(f"{query:<46}{len(boolean):>12}{p3:>7.2f}{r3:>7.2f}{ap:>7.2f}")
    r(f"    Boolean AND: {sorted(boolean) or 'EMPTY'} | TF-IDF top-3: {top3}")
r(f"\nMean Average Precision (TF-IDF): {sum(aps) / len(aps):.3f}")
r("Boolean AND needs every word in the document, so natural-language queries")
r("often return nothing; ranked TF-IDF still finds the relevant documents.")
r.save()
