from pathlib import Path

from rfp_intel.ingestion.cleaning import clean_lines, find_repeated_lines, is_page_number
from rfp_intel.ingestion.html_parser import parse_html
from rfp_intel.ingestion.loader import load_bid_folder, parse_file
from rfp_intel.ingestion.metadata import addendum_number, classify_doc_type, parse_date

from .conftest import DATA, make_pdf


def test_clean_lines_joins_wrapped_text_and_dehyphenates():
    out = clean_lines(["The submis-", "sion must be made", "in writing.", "• Bullet one", "• Bullet two"])
    assert out == ["The submission must be made in writing.", "• Bullet one", "• Bullet two"]


def test_repeated_header_detection():
    pages = [["ACME RFP 123", f"body {i}", f"Page {i} of 5"] for i in range(1, 6)]
    rep = find_repeated_lines(pages)
    assert "acme rfp #" in rep and "page # of #" in rep
    assert is_page_number("Page 3 of 40") and not is_page_number("Section 3")


def test_metadata_rules():
    assert classify_doc_type("Addendum 2 RFP.pdf", "", False) == "addendum"
    assert classify_doc_type("Mercury_Affidavit.pdf", "", False) == "affidavit"
    assert classify_doc_type("Dell_Laptop_Specs.pdf", "", False) == "specs"
    assert classify_doc_type("x.pdf", "REQUEST FOR PROPOSAL 55", False) == "rfp"
    assert classify_doc_type("page.html", "", True) == "bid_page"
    assert addendum_number("Addendum 2 RFP.pdf", "") == 2
    assert addendum_number("amend.pdf", "ADDENDUM No. 3\nText") == 3
    assert parse_date("Due 26-MAY-2024 08:00") == "2024-05-26"
    assert parse_date("new due date July 9, 2024 at 2:00 PM") == "2024-07-09"
    assert parse_date("on 05/24/2024") == "2024-05-24"


def test_html_parser_extracts_key_values_and_drops_nav(mini_bid):
    blocks, kv, title, warnings = parse_html(mini_bid / "portal.html")
    assert kv["Solicitation Number"] == "WX-1001"
    assert kv["Closing Date"].startswith("03/15/2025")
    text = "\n".join(b.text for b in blocks)
    assert "Home" not in text and "Copyright" not in text
    assert title.startswith("Widget Bid")


def test_pdf_parsing_removes_running_headers_and_keeps_pages(mini_bid):
    doc = parse_file(mini_bid / "RFP WX-1001 Widgets.pdf", "BidX", ocr=False)
    assert doc.doc_type == "rfp" and doc.num_pages == 3
    texts = [b.text for b in doc.blocks]
    assert sum("City of Example - RFP WX-1001" in t for t in texts) == 1  # header kept once
    assert not any("Page 2 of 3" in t for t in texts)
    assert any(b.page == 2 and "bid bond" in b.text.lower() for b in doc.blocks)
    assert any(b.kind == "heading" and b.text == "BID BOND" for b in doc.blocks)


def test_addendum_metadata(mini_bid):
    doc = parse_file(mini_bid / "Addendum 1 WX-1001.pdf", "BidX", ocr=False)
    assert doc.doc_type == "addendum" and doc.addendum_number == 1


def test_empty_and_corrupt_files_are_logged_not_fatal(tmp_path):
    folder = tmp_path / "BidBad"
    folder.mkdir()
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    make_pdf(folder / "blank.pdf", [""])
    docs = load_bid_folder(folder, ocr=False)
    by_name = {d.file_name: d for d in docs}
    assert any("unreadable" in w for w in by_name["broken.pdf"].warnings)
    assert any("empty page" in w for w in by_name["blank.pdf"].warnings)


def test_real_bid2_parsing():
    import pytest

    if not (DATA / "Bid2").exists():
        pytest.skip("sample data not present")
    docs = {d.file_name: d for d in load_bid_folder(DATA / "Bid2", ocr=False)}
    porfp = docs["PORFP_-_Dell_Laptop_Final.pdf"]
    assert porfp.doc_type == "rfp"
    assert any(b.kind == "table" and "WD22TB4" in b.text for b in porfp.blocks)  # product table kept intact
    specs = docs["Dell_Laptop_Specs.pdf"]
    assert specs.doc_type == "specs"
    assert "Dell Latitude 5550 XCTO Base 210-BLYZ" in specs.full_text  # description + SKU on one line
    page = next(d for d in docs.values() if d.doc_type == "bid_page")
    assert page.key_values["Solicitation Number"] == "BPM044557"
