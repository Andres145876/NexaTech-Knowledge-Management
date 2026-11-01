"""Shared helpers for the practical exercises."""
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "DATA" / "documents.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TOKENS = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"


def load_documents():
    documents = pd.read_csv(DATA_FILE).fillna("")
    documents["text"] = documents["title"] + " " + documents["content"]
    return documents


class Report:
    """Prints lines and saves them to EXERCISES/outputs/<name>.txt"""

    def __init__(self, name):
        self.name = name
        self.lines = []

    def __call__(self, line=""):
        print(line)
        self.lines.append(str(line))

    def save(self):
        (OUTPUT_DIR / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n", encoding="utf-8")
