import json
import re
from pathlib import Path

import pytest

from rfp_intel.config import Settings

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.index_dir = tmp_path / "index"
    s.output_dir = tmp_path / "out"
    s.trace_dir = tmp_path / "out" / "traces"
    s.cache_dir = tmp_path / "cache"
    s.embedding_backend = "hash"
    s.reranker_backend = "none"
    s.ocr_enabled = False
    s.llm_cache = False
    s.llm_max_retries = 2
    s.max_field_retries = 1
    return s


def make_pdf(path: Path, pages: list[str]) -> Path:
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        y = 72
        for line in text.split("\n"):
            page.insert_text((72, y), line, fontsize=10)
            y += 14
    doc.save(path)
    return path


@pytest.fixture
def mini_bid(tmp_path) -> Path:
    """A small synthetic bid folder: portal page, RFP, addendum, affidavit."""
    folder = tmp_path / "BidX"
    folder.mkdir()
    (folder / "portal.html").write_text("""
    <html><head><title>Widget Bid - Bid Information</title></head><body>
    <nav class="main-nav"><a>Home</a><a>Search</a></nav>
    <div class="content"><h3>Basic Information</h3>
      <div class="field"><span class="field-label">Solicitation Number</span><div class="field-body"><p>WX-1001</p></div></div>
      <div class="field"><span class="field-label">Title</span><div class="field-body"><p>Widget Procurement</p></div></div>
      <div class="field"><span class="field-label">Closing Date</span><div class="field-body"><p>03/15/2025 02:00 PM EST</p></div></div>
      <div class="field"><span class="field-label">Publication</span><div class="field-body"><p>02/01/2025 09:00 AM EST</p></div></div>
    </div><footer class="site-footer">Copyright</footer></body></html>""")
    header = "City of Example - RFP WX-1001"
    make_pdf(folder / "RFP WX-1001 Widgets.pdf", [
        f"{header}\nREQUEST FOR PROPOSAL\nProposals are due February 28, 2025 at 2:00 PM EST.\n"
        "Submit proposals through the city e-procurement portal only.\nPage 1 of 3",
        f"{header}\nBID BOND\nA bid bond of five percent (5%) of the bid amount is required.\n"
        "Payment terms are Net 30 after receipt of a correct invoice.\nPage 2 of 3",
        f"{header}\nSPECIFICATIONS\nWidget model WGT-500, part number 900-AB12, quantity 250.\n"
        "Delivery within 30 days of award.\nPage 3 of 3",
    ])
    make_pdf(folder / "Addendum 1 WX-1001.pdf", [
        "ADDENDUM No. 1\nThe due date is extended to March 15, 2025 at 2:00 PM EST.\nAll other terms remain unchanged."])
    make_pdf(folder / "Vendor_Affidavit.pdf", ["VENDOR AFFIDAVIT\nI affirm that the business is in good standing."])
    return folder


def field_responder(values: dict, critic_reject: set | None = None, first_uncited: set | None = None):
    """Scripted LLM: answers extraction by finding the passage containing the expected value."""
    critic_reject = critic_reject or set()
    first_uncited = set(first_uncited or set())

    def respond(schema, system, user):
        if schema == "ExtractionLLM":
            names = json.loads(re.search(r"exactly these names: (\[.*\])", user).group(1))
            passages = re.findall(r"\[(E\d+)\] \([^)]*\)\n(?:Section:[^\n]*\n)?(.*?)(?=\n\[E\d+\] \(|\n\nReturn one entry)", user, re.S)
            out = []
            for n in names:
                want = values.get(n)
                eid = next((e for e, t in passages if want and want.split()[0].lower() in t.lower()), None)
                if n in first_uncited:
                    first_uncited.discard(n)
                    out.append({"field": n, "value": want, "evidence_ids": [], "confidence": 0.9, "notes": ""})
                elif want and eid:
                    out.append({"field": n, "value": want, "evidence_ids": [eid], "confidence": 0.9, "notes": ""})
                else:
                    out.append({"field": n, "value": None, "evidence_ids": [], "confidence": 0.7,
                                "notes": "Not found in documents"})
            return {"fields": out}
        if schema == "AddendumAnalysisLLM":
            return {"addendum_number": 1, "summary": "Extends due date", "changes": [
                {"field": "Due Date", "change_type": "modified", "description": "Due date extended",
                 "old_value": "2025-02-28 14:00 EST", "new_value": "2025-03-15 14:00 EST", "evidence_ids": ["E1"]}]}
        if schema == "CriticLLM":
            fields = re.findall(r"^### (.+)$", user, re.M)
            return {"verdicts": [{"field": f, "supported": f not in critic_reject, "issue": "" if f not in critic_reject else "unsupported",
                                  "retry_query": "better query"} for f in fields]}
        if schema == "SummaryLLM":
            return {"summary": "The City of Example is buying widgets."}
        if schema == "QueryRewrite":
            return {"queries": [], "keywords": []}
        if schema == "QueryPlanLLM":
            return {"intent": "lookup", "bids": ["BidX"], "sub_queries": [{"bid_id": "BidX", "query": user.split("QUESTION: ")[-1]}]}
        if schema == "AnswerLLM":
            return {"found": True, "answer": "5% bid bond [E1]", "evidence_ids": ["E1"], "confidence": 0.9}
        if schema == "_Relevance":
            return {"scores": [5] * 50}
        raise ValueError(schema)

    return respond
