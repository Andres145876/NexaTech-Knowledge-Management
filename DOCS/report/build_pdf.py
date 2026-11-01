"""Builds DOCS/Project_Progress.pdf = cover.pdf + Project_Progress.html.

Two passes: the first one finds the page of every heading, the second one
writes those page numbers in the table of contents. Then the cover is added
in front.

    pip install playwright pypdf && playwright install chromium
    python DOCS/report/build_pdf.py

The cover comes from cover.docx (exported to cover.pdf with Word or LibreOffice).
"""
import io
import re
from pathlib import Path

from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Project_Progress.html"
COVER = HERE / "cover.pdf"
TARGET = HERE.parent / "Project_Progress.pdf"

HEADER = """
<div style="width:100%; font-family:'Liberation Sans',Helvetica,Arial,sans-serif; -webkit-print-color-adjust:exact;">
  <div style="position:absolute; top:0; left:0; right:0; height:7mm; background:#4aa83c;"></div>
  <div style="position:absolute; top:10mm; left:18mm; right:18mm; font-size:7.5pt; color:#666; display:flex; justify-content:space-between;">
    <span>NexaTech Solutions &nbsp;|&nbsp; Knowledge Management System</span><span>Project Progress Documentation</span>
  </div>
</div>"""
FOOTER = """
<div style="width:100%; font-family:'Liberation Sans',Helvetica,Arial,sans-serif; font-size:8pt; color:#124f4c; text-align:center; font-weight:bold;">
  <span class="pageNumber"></span>
</div>"""
PDF_OPTIONS = dict(format="Letter", print_background=True, prefer_css_page_size=True,
                   display_header_footer=True, header_template=HEADER, footer_template=FOOTER)


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(SOURCE.as_uri())
    page.wait_for_load_state("networkidle")

    # pass 1: render and locate every heading listed in the table of contents
    targets = page.eval_on_selector_all(
        ".toc-pg", "els => els.map(e => [e.dataset.target, document.getElementById(e.dataset.target).innerText])")
    first = PdfReader(io.BytesIO(page.pdf(**PDF_OPTIONS)))
    texts = [normalize(pg.extract_text() or "") for pg in first.pages]
    numbers, start = {}, 1                      # page index 0 is the table of contents
    for target, heading in targets:
        key = normalize(heading)
        for index in range(start, len(texts)):
            if key in texts[index]:
                numbers[target] = index + 1
                start = index
                break
        else:
            print(f"WARNING: heading not found: {heading}")
    page.evaluate("nums => document.querySelectorAll('.toc-pg').forEach("
                  "e => { e.textContent = nums[e.dataset.target] ?? '' })", numbers)

    # pass 2: final render
    body = PdfReader(io.BytesIO(page.pdf(**PDF_OPTIONS)))
    browser.close()

writer = PdfWriter()
if COVER.exists():
    writer.append(PdfReader(str(COVER)))
writer.append(body)
writer.add_metadata({"/Title": "Project Progress Documentation - NexaTech Knowledge Management System"})
with open(TARGET, "wb") as fh:
    writer.write(fh)
print(f"Saved {TARGET} ({len(writer.pages)} pages, {len(numbers)} TOC entries)")
