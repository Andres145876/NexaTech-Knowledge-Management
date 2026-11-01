"""
lifecycle_map.py - draws Figure 1 of the report: the knowledge life cycle of
NexaTech Solutions (6 stages) with tacit/explicit knowledge, SECI conversion,
the prototype component and the data involved in each stage.

Usage: python DIAGRAMS/lifecycle_map.py   -> DIAGRAMS/knowledge_lifecycle.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "knowledge_lifecycle.png"

STAGES = [
    ("1 Creation\nand capture", "Debugging judgment:\nsensing where the\nbug hides", "Externalization",
     "Ticket notes,\ncommits, incident logs", "Add Document\nPDF / TXT / CSV\nupload (+ OCR)", "New rows:\nID, title, category,\ncontent"),
    ("2 Storage and\ndocumentation", "Deciding what is\nworth writing down", "Externalization",
     "Articles, runbooks,\npolicies in one\nrepository", "documents.csv\n(validated: no empty\nfields or duplicate IDs)", "102 documents\nin one CSV file"),
    ("3 Organization and\nclassification", "Curation sense:\nwhere a document\nbelongs", "Combination",
     "Category labels,\ndocument types", "Document Classifier\n(TF-IDF +\nNaive Bayes)", "4 categories,\n6 document types"),
    ("4 Sharing\nand transfer", "Mentoring, pair\nprogramming, shift\nhandovers", "Socialization",
     "Runbooks, release\nnotes, API docs", "Knowledge Base\n+ Dashboard", "Browse, filter,\nsearch by keyword"),
    ("5 Application\nin daily work", "Judging if a fix is\nsafe on a live\nclient system", "Internalization",
     "Procedures,\nchecklists, SLAs", "Semantic Search\n(TF-IDF + cosine)", "Ranked results\nwith similarity score"),
    ("6 Update and\nimprovement", "Lessons the team\nabsorbed but never\nwrote down", "Externalization + combination",
     "Postmortems, revised\narticles, trend reports", "Re-import data\nand retrain models", "Dataset grows;\nmodels rebuilt"),
]

NAVY, RED, GREEN, BLUE, GREY = "#13263F", "#C0504D", "#3A8A5B", "#2E5AA8", "#999999"
fig, ax = plt.subplots(figsize=(24, 14.1), dpi=110)
ax.set_xlim(0, 24)
ax.set_ylim(0, 14.1)
ax.axis("off")

ax.text(0.15, 13.75, "NexaTech Solutions: knowledge life cycle and prototype",
        fontsize=26, fontweight="bold", color=NAVY, va="center")
ax.text(0.15, 13.2, "Six stages, tacit and explicit knowledge, SECI conversion and the prototype component "
        "that supports each stage (case: a bank's VPN drops after a firmware update)",
        fontsize=14.5, color="#555555", va="center")

x0, w, gap = 2.9, 3.35, 0.12
rows = {  # name: (y, height, edge, face, label color, label)
    "stage": (10.45, 1.45, NAVY, "white", NAVY, "STAGE"),
    "tacit": (8.25, 1.75, RED, "#F6DAD7", RED, "TACIT\n(PEOPLE)"),
    "explicit": (5.55, 1.55, GREEN, "#D5EADD", GREEN, "EXPLICIT\n(RECORDS)"),
    "proto": (2.95, 2.1, BLUE, "#DAE5F4", BLUE, "PROTOTYPE\nCOMPONENT"),
    "data": (0.9, 1.55, GREY, "#F3F4F6", "#555555", "DATA IN\nTHE SYSTEM"),
}
for key, (y, h, edge, face, color, label) in rows.items():
    ax.text(0.15, y + h / 2, label, fontsize=13.5, fontweight="bold", color=color, va="center")
ax.text(0.15, 7.6, "SECI", fontsize=13.5, fontweight="bold", color="#555555", va="center")

for i, (stage, tacit, seci, explicit, proto, data) in enumerate(STAGES):
    x = x0 + i * (w + gap)
    cx = x + w / 2
    for key, text in [("stage", stage), ("tacit", tacit), ("explicit", explicit),
                      ("proto", proto), ("data", data)]:
        y, h, edge, face, _c, _l = rows[key]
        style = "--" if key == "tacit" else "-"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                    linewidth=2.2, edgecolor=edge, facecolor=face, linestyle=style))
        weight = "bold" if key in ("stage", "proto") else "normal"
        size = 15 if key == "stage" else (14 if key == "proto" else 13)
        color = NAVY if key == "stage" else ("#1F3B73" if key == "proto" else "#222222")
        ax.text(cx, y + h / 2, text, ha="center", va="center", fontsize=size,
                fontweight=weight, color=color, linespacing=1.25)
    ax.text(cx, 7.6, seci, ha="center", va="center", fontsize=12.5, style="italic", color="#444444")
    if i < len(STAGES) - 1:
        ax.add_patch(FancyArrowPatch((x + w - 0.02, 11.17), (x + w + gap + 0.04, 11.17),
                                     arrowstyle="-|>", mutation_scale=16, color="#333333", lw=1.5))

# feedback arrow: stage 6 -> stage 1
first_c, last_c = x0 + w / 2, x0 + 5 * (w + gap) + w / 2
ax.add_patch(FancyArrowPatch((last_c, 12.15), (first_c, 12.15), arrowstyle="-|>",
                             mutation_scale=18, color="#666666", lw=2, linestyle=(0, (5, 3))))
ax.text((first_c + last_c) / 2, 12.33, "Feedback: lessons learned restart the cycle",
        ha="center", fontsize=13, color="#555555")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT}")
