"""
Topic 1 - Knowledge management fundamentals: knowledge audit of NexaTech
Identify the knowledge assets of each team, classify them as tacit or
explicit, map the SECI conversion (Socialization, Externalization,
Combination, Internalization) and check which ones are already documented
in DATA/documents.csv.

Contribution: defined the 4 knowledge categories of the classifier and the
teams of the ontology.
"""
from collections import Counter
from common import Report, load_documents

AUDIT = [
    # team, knowledge asset, type, SECI conversion, category, keyword to find it in the dataset
    ("IT Infrastructure", "Server maintenance steps", "explicit", "Internalization", "Infrastructure", "maintenance"),
    ("IT Infrastructure", "Experience recovering from incidents", "tacit", "Externalization", "Infrastructure", "incident"),
    ("IT Infrastructure", "Network configuration standards", "explicit", "Combination", "Infrastructure", "network"),
    ("Help Desk", "How senior technicians diagnose problems", "tacit", "Socialization", "Technical Support", "diagnose"),
    ("Help Desk", "Solutions to frequent user problems", "tacit", "Externalization", "Technical Support", "troubleshoot"),
    ("Help Desk", "Ticket priority rules", "explicit", "Internalization", "Technical Support", "priority"),
    ("Cybersecurity", "Security policies (passwords, encryption)", "explicit", "Combination", "Cybersecurity", "policy"),
    ("Cybersecurity", "Intuition to detect attacks", "tacit", "Socialization", "Cybersecurity", "suspicious"),
    ("Cybersecurity", "Lessons from security incidents", "tacit", "Externalization", "Cybersecurity", "incident"),
    ("Software Development", "Coding and review standards", "explicit", "Combination", "Software Development", "code"),
    ("Software Development", "Debugging know-how of senior developers", "tacit", "Socialization", "Software Development", "defects"),
    ("Software Development", "Deployment steps", "explicit", "Internalization", "Software Development", "deployment"),
]

r = Report("topic1_knowledge_audit")
docs = load_documents()
r(f"{'Team':<22}{'Knowledge asset':<42}{'Type':<10}{'SECI':<17}{'Docs found':>10}")
r("-" * 101)
for team, asset, ktype, seci, category, keyword in AUDIT:
    subset = docs[docs["category"] == category]
    found = subset["text"].str.lower().str.contains(keyword).sum()
    r(f"{team:<22}{asset:<42}{ktype:<10}{seci:<17}{found:>10}")
r()
r(f"Tacit vs explicit: {dict(Counter(a[2] for a in AUDIT))}")
r(f"SECI conversions:  {dict(Counter(a[3] for a in AUDIT))}")
r(f"Knowledge categories needed: {sorted(set(a[4] for a in AUDIT))}")
r("\nConclusion: half of the critical knowledge is tacit (diagnosis experience,")
r("lessons from incidents). The KMS must externalize it as documents and make")
r("it easy to classify and find.")
r.save()
