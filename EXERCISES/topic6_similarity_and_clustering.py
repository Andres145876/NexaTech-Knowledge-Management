"""
Topic 6 - Semantic similarity and document clustering
(a) cosine similarity between documents and "related documents" recommendations,
(b) K-Means clustering (k=4) without labels compared with the human categories,
(c) similarity heatmap.

Contribution: confirmed that cosine similarity on TF-IDF vectors captures
topical closeness (basis of the semantic search) and that the 4 categories
match the natural structure of the documents.
"""
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from common import OUTPUT_DIR, TOKENS, Report, load_documents

r = Report("topic6_similarity_and_clustering")
docs = load_documents().sort_values(["category", "document_id"]).reset_index(drop=True)
vectorizer = TfidfVectorizer(stop_words="english", token_pattern=TOKENS, sublinear_tf=True)
X = vectorizer.fit_transform(docs["text"])
sim = cosine_similarity(X)
ids = docs["document_id"].tolist()

for target in ["DOC007", "DOC059", "DOC094"]:
    i = ids.index(target)
    r(f"Related to {target} ({docs.loc[i, 'title']}):")
    for j in [j for j in sim[i].argsort()[::-1] if j != i][:3]:
        r(f"    {sim[i, j]:.3f}  {ids[j]:<13} {docs.loc[j, 'title']}")

cats = docs["category"].to_numpy()
same = sim[(cats[:, None] == cats[None, :]) & ~np.eye(len(docs), dtype=bool)].mean()
diff = sim[cats[:, None] != cats[None, :]].mean()
r(f"\nMean similarity, same category:      {same:.3f}")
r(f"Mean similarity, different category: {diff:.3f}")

km = KMeans(n_clusters=4, n_init=20, random_state=42).fit(X)
terms = vectorizer.get_feature_names_out()
r("\nK-Means clusters (k=4, no labels used):")
hits = 0
for c in range(4):
    members = cats[km.labels_ == c]
    major, n = Counter(members).most_common(1)[0]
    hits += n
    top = [terms[t] for t in km.cluster_centers_[c].argsort()[::-1][:5]]
    r(f"  cluster {c}: {len(members):>2} docs | majority = {major:<21} | {', '.join(top)}")
r(f"Purity: {hits / len(docs):.3f} | Adjusted Rand Index: {adjusted_rand_score(cats, km.labels_):.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
shown = sim.copy()
np.fill_diagonal(shown, np.nan)
fig, ax = plt.subplots(figsize=(8.5, 7.5))
image = ax.imshow(shown, cmap="viridis", vmin=0, vmax=0.25)
ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=5)
ax.set_yticks(range(len(ids)), ids, fontsize=5)
ax.set_title("Cosine similarity between NexaTech documents (sorted by category)")
fig.colorbar(image, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "topic6_similarity_heatmap.png", dpi=150)
r("\nHeatmap saved: EXERCISES/outputs/topic6_similarity_heatmap.png")
r.save()
