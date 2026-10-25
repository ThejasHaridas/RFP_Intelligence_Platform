from rfp_intel.evaluation import Config, evaluate
from rfp_intel.ingestion.models import Block, ParsedDocument
from rfp_intel.search.chunker import chunk_document
from rfp_intel.search.embeddings import HashEmbedder, tokenize
from rfp_intel.search.engine import SearchEngine
from rfp_intel.search.index import SearchIndex
from rfp_intel.search.query import expand_query


def test_tokenizer_keeps_identifiers():
    toks = tokenize("SKU 210-BLYZ for RFP JA-207652")
    assert "210-blyz" in toks and "ja-207652" in toks and "207652" in toks


def test_query_expansion():
    v = expand_query("What is the submission deadline?")
    assert v[0] == "What is the submission deadline?"
    assert any("closing date" in x for x in v[1:])


def test_hash_embedder_is_deterministic_and_normalised():
    e1, e2 = HashEmbedder(), HashEmbedder()
    a, b = e1.encode(["bid bond"]), e2.encode(["bid bond"])
    assert (a == b).all() and abs(float((a ** 2).sum()) - 1) < 1e-4


def test_chunker_respects_pages_sections_and_tables():
    blocks = [Block(text="SCOPE", kind="heading", page=1, section="SCOPE"),
              Block(text=("Sentence about scope. " * 120).strip(), page=1, section="SCOPE"),
              Block(text="Item | SKU\nLaptop | 210-BLYZ\nDock | WD22TB4", kind="table", page=2, section="SCOPE")]
    doc = ParsedDocument(bid_id="B", file_name="f.pdf", file_path="f.pdf", doc_type="rfp", blocks=blocks, sha1="x")
    chunks = chunk_document(doc, target=600, max_chars=900, overlap=100)
    assert len(chunks) >= 4
    assert all(len(c.text) <= 1000 for c in chunks)
    table = [c for c in chunks if c.kind == "table"]
    assert len(table) == 1 and "WD22TB4" in table[0].text and table[0].page_number == 2
    assert all(c.section == "SCOPE" for c in chunks)
    # overlap: consecutive prose chunks share text
    prose = [c for c in chunks if c.kind == "text"]
    assert prose[1].text[:30] in prose[0].text


def test_index_search_filters_citations_and_incremental(settings, mini_bid):
    idx = SearchIndex(settings)
    rep = idx.index_folder(mini_bid)
    assert rep.status == "indexed" and rep.chunks > 0
    assert rep.doc_types["Addendum 1 WX-1001.pdf"] == "addendum"
    # unchanged folder is not re-embedded
    assert idx.index_folder(mini_bid).status == "unchanged"
    eng = SearchEngine(idx, settings)
    hits = eng.search("part number 900-AB12", top_k=3)
    assert hits and "900-AB12" in hits[0].chunk.text
    c = hits[0].citation
    assert c["file"] == "RFP WX-1001 Widgets.pdf" and c["page"] == 3
    only_add = eng.search("due date", filters={"bid_id": "BidX", "doc_type": "addendum"})
    assert only_add and all(h.chunk.doc_type == "addendum" for h in only_add)
    assert eng.search("due date", filters={"bid_id": "Nope"}) == []
    # persistence: a new index object loads shards from disk
    assert len(SearchIndex(settings).chunks) == len(idx.chunks)


def test_adding_a_bid_does_not_reindex_others(settings, mini_bid, tmp_path):
    import shutil

    idx = SearchIndex(settings)
    idx.index_folder(mini_bid)
    before = (settings.index_dir / "bids" / "BidX" / "vectors.npy").stat().st_mtime_ns
    other = tmp_path / "BidY"
    shutil.copytree(mini_bid, other)
    idx.index_folder(other)
    after = (settings.index_dir / "bids" / "BidX" / "vectors.npy").stat().st_mtime_ns
    assert before == after and set(idx.bid_ids()) == {"BidX", "BidY"}
    assert idx.remove_bid("BidY") and idx.bid_ids() == ["BidX"]


def test_evaluation_metrics(settings, mini_bid):
    idx = SearchIndex(settings)
    idx.index_folder(mini_bid)
    qs = [{"id": "q1", "bid_id": "BidX", "question": "bid bond amount", "file": "RFP", "phrases": ["five percent"]},
          {"id": "q2", "bid_id": "BidX", "question": "SKU 900-AB12", "file": "", "phrases": ["900-AB12"]}]
    res = evaluate(SearchEngine(idx, settings), qs, [Config("bm25", mode="bm25"), Config("hybrid")])
    row = res["configs"][0]
    assert row["Recall@5"] == 1.0 and 0 < row["MRR@10"] <= 1.0
