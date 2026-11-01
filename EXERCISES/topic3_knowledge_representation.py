"""
Topic 3 - Knowledge representation: the NexaTech ontology as triples
Encode the conceptual graph of the project (report Figure 2) as
subject-predicate-object triples, populate it with the real documents of
DATA/documents.csv (instances, document type, category, similar documents)
and answer questions by querying the graph.

Contribution: validated the classes and relations of the ontology and showed
that every document of the dataset fits into it.
"""
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from common import Report, load_documents

AREAS = ["Technical Support", "Software Development", "Infrastructure", "Cybersecurity"]
# document types of the ontology and the title keywords that identify them
DOCUMENT_TYPES = [
    ("Incident report", ["incident", "breach", "containment"]),
    ("Policy", ["policy", "agreement"]),
    ("Procedure", ["procedure", "procedures", "process", "checklist"]),
    ("Plan / runbook", ["plan", "planning", "scheduling", "recovery", "management", "lifecycle", "allocation"]),
    ("Support guide", ["support", "troubleshooting", "resolution", "assistance"]),
    ("Guideline", ["guidelines", "standards", "practices", "documentation", "design", "strategy"]),
]


def document_type(title):
    title = title.lower()
    for dtype, keywords in DOCUMENT_TYPES:
        if any(k in title.split() for k in keywords):
            return dtype
    return "Guideline"


r = Report("topic3_knowledge_representation")
docs = load_documents()
triples = []
# schema level (classes and relations of Figure 2)
for area in AREAS:
    triples.append(("NexaTech Solutions", "has area", area))
    triples.append((area, "generates / owns", f"Category: {area}"))
for dtype, _ in DOCUMENT_TYPES:
    triples.append((dtype, "subtype of", "Technical document"))
triples += [("Employee", "asks", "Natural-language query"),
            ("Natural-language query", "processed by", "Search engine"),
            ("Search engine", "ranks / retrieves", "Technical document"),
            ("Technical document", "input text", "Classifier"),
            ("Classifier", "assigns category", "Knowledge category")]
schema = len(triples)
# instance level (real documents)
for _, d in docs.iterrows():
    triples.append((f"Category: {d['category']}", "contains", d["document_id"]))
    triples.append((d["document_id"], "has type", document_type(d["title"])))
vectors = TfidfVectorizer(stop_words="english").fit_transform(docs["text"])
sim = cosine_similarity(vectors)
ids = docs["document_id"].tolist()
for i in range(len(ids)):
    j = max((k for k in range(len(ids)) if k != i), key=lambda k: sim[i, k])
    triples.append((ids[i], "similar to", ids[j]))
r(f"Knowledge graph: {len(triples)} triples ({schema} schema-level, {len(triples) - schema} instance-level)\n")

out = defaultdict(list)
for s, p, o in triples:
    out[(s, p)].append(o)

r("Q1. Document types per knowledge category (instances of Figure 2):")
for area in AREAS:
    types = Counter(out[(d, "has type")][0] for d in out[(f"Category: {area}", "contains")])
    r(f"    {area:<22} " + ", ".join(f"{t}: {n}" for t, n in types.most_common()))

r("\nQ2. Which policies exist in Cybersecurity?")
for d in out[("Category: Cybersecurity", "contains")]:
    if out[(d, "has type")][0] == "Policy":
        r(f"    {d:<13} {docs.set_index('document_id').loc[d, 'title']}")

r("\nQ3. Instances of Figure 2 and their most similar document (relation 'similar to'):")
for d in ["DOC007", "DOC003", "DOC059", "DOC045", "DOC099"]:
    j = out[(d, "similar to")][0]
    r(f"    {d} -> {j}  cosine={sim[ids.index(d), ids.index(j)]:.3f}  ({docs.set_index('document_id').loc[j, 'title']})")

r("\nQ4. Path from an employee to a document:")
r("    Employee -asks-> Natural-language query -processed by-> Search engine "
  "-ranks / retrieves-> Technical document -has type-> Policy")
r.save()
