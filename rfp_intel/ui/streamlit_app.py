"""Streamlit UI: search, chat (cited Q&A) and extraction viewer.

    streamlit run rfp_intel/ui/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st

from rfp_intel.agents.orchestrator import Orchestrator
from rfp_intel.config import get_settings
from rfp_intel.search.engine import SearchEngine
from rfp_intel.search.index import SearchIndex

st.set_page_config(page_title="RFP Intelligence", layout="wide")

LLM_OPTIONS = {"GPT (OpenAI)": "gpt", "OpenRouter": "open"}


@st.cache_resource
def engine() -> SearchEngine:
    """Index + embedding/reranker models are loaded once and shared by every LLM choice."""
    s = get_settings()
    s.ensure_dirs()
    return SearchEngine(SearchIndex(s), s)


st.sidebar.title("RFP Intelligence")
base = get_settings()
labels = list(LLM_OPTIONS)
default = 0 if base.llm_provider == "openai" else 1
choice = LLM_OPTIONS[st.sidebar.radio("LLM", labels, index=default, horizontal=True)]
# keys live only in this browser session (never written to disk); empty -> key from .env
api_key = st.sidebar.text_input(f"{'OpenAI' if choice == 'gpt' else 'OpenRouter'} API key", type="password",
                                key=f"api_key_{choice}", placeholder="optional: uses the key from .env")
settings = base.with_llm(choice, api_key.strip())
o = Orchestrator(settings=settings, engine=engine())
if o.llm is None:
    st.sidebar.warning("No API key for this LLM: search works, chat and extraction need a key.")

bids = o.engine.index.bid_ids()
st.sidebar.write(f"**Indexed bids:** {', '.join(bids) or 'none'}")
st.sidebar.caption(f"embedder `{o.engine.index.embedder.name}` · reranker `{o.engine.reranker.name}` · "
                   f"LLM `{o.llm.provider if o.llm else 'none'}` · model `{settings.llm_model}`")
folder = st.sidebar.text_input("Index a bid folder", placeholder="./data/Bid3")
if st.sidebar.button("Index") and folder:
    with st.spinner("Parsing and indexing..."):
        rep = o.engine.index.index_folder(Path(folder))
    st.sidebar.success(f"{rep.bid_id}: {rep.status}, {rep.chunks} chunks")

tab_search, tab_chat, tab_extract = st.tabs(["Search", "Chat", "Extraction"])

with tab_search:
    q = st.text_input("Query", placeholder="bid bond requirement")
    c1, c2, c3, c4 = st.columns(4)
    bid = c1.selectbox("Bid", ["(all)"] + bids)
    dtypes = c2.multiselect("Doc type", ["bid_page", "rfp", "addendum", "specs", "affidavit", "form", "other"])
    mode = c3.selectbox("Mode", ["hybrid", "bm25", "dense"])
    k = c4.slider("Top k", 1, 20, 5)
    if q:
        filters = {"bid_id": None if bid == "(all)" else bid, "doc_type": dtypes or None}
        for h in o.engine.search(q, top_k=k, filters={k2: v for k2, v in filters.items() if v}, mode=mode):
            c = h.citation
            with st.expander(f"#{h.rank} · {c['file']} p.{c['page']} · {c['doc_type']} · score {h.score:.3f}"):
                st.caption(h.chunk.section or "")
                st.text(h.chunk.text)

with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []
    for role, msg in st.session_state.history:
        st.chat_message(role).markdown(msg)
    question = st.chat_input("Ask about the bids", disabled=o.llm is None)
    if question:
        st.chat_message("user").markdown(question)
        with st.spinner("Agents working..."):
            res = asyncio.run(o.answer(question))
        cites = "\n".join(f"- `{c['id']}` {c['bid_id']} / {c['file']} p.{c['page']}" for c in res.get("citations", []))
        msg = f"{res['answer']}\n\n{cites}"
        st.chat_message("assistant").markdown(msg)
        st.session_state.history += [("user", question), ("assistant", msg)]

with tab_extract:
    out_dir = Path(get_settings().output_dir)
    files = sorted(out_dir.glob("*.json"))
    choice = st.selectbox("Extraction output", [f.name for f in files]) if files else None
    target = st.selectbox("Run extraction for", bids) if bids else None
    if target and st.button("Run extraction", disabled=o.llm is None):
        with st.spinner("Running multi-agent extraction..."):
            asyncio.run(o.extract_bid(bid_id=target))
        st.rerun()
    if choice:
        rec = json.loads((out_dir / choice).read_text())
        st.metric("Validation", f"{rec['validation'].get('passed', 0)} passed / {rec['validation'].get('failed', 0)} failed "
                                f"/ {rec['validation'].get('not_found', 0)} not found")
        rows = [{"field": k, "value": json.dumps(v["value"]) if isinstance(v["value"], list) else v["value"],
                 "confidence": v["confidence"], "sources": "; ".join(f"{s['file']} p.{s['page']}" for s in v["sources"]),
                 "notes": v["notes"]} for k, v in rec["fields"].items()]
        st.dataframe(rows, width="stretch")
        if rec.get("addendum_changes"):
            st.subheader("Addendum changes")
            st.dataframe(rec["addendum_changes"], width="stretch")
