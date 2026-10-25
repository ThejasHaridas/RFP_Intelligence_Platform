import asyncio
import json

from rfp_intel.agents.orchestrator import Orchestrator
from rfp_intel.agents.schemas import FieldResult, Source
from rfp_intel.agents.validator import find_dates, rule_checks
from rfp_intel.llm.client import ScriptedLLM, extract_json

from .conftest import field_responder

VALUES = {
    "Bid Number": "WX-1001",
    "Due Date": "2025-02-28 14:00 EST",
    "Bid Bond Requirement": "five percent (5%) of the bid amount",
    "Payment Terms": "Net 30 after receipt of a correct invoice",
    "Part_no": "900-AB12",
    "Model_no": "WGT-500",
    "Delivery Date": "within 30 days of award",
}


def _orch(settings, responder):
    return Orchestrator(settings=settings, llm=ScriptedLLM(responder, settings))


def test_extract_json_repairs_fences_and_trailing_commas():
    assert extract_json('```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert extract_json('Here you go: {"x": 1} thanks') == {"x": 1}


def test_extraction_end_to_end_with_addendum_override(settings, mini_bid):
    o = _orch(settings, field_responder(VALUES))
    rec = asyncio.run(o.extract_bid(folder=mini_bid))
    f = rec["fields"]
    assert len(f) == 20
    assert f["Bid Number"]["value"] == "WX-1001" and f["Bid Number"]["sources"]
    # addendum agent overrides the base RFP due date and records the original
    assert f["Due Date"]["value"] == "2025-03-15 14:00 EST"
    assert "Addendum 1" in f["Due Date"]["notes"] and "2025-02-28" in f["Due Date"]["notes"]
    assert f["Due Date"]["sources"][0]["file"] == "Addendum 1 WX-1001.pdf"
    assert rec["addendum_changes"][0]["applied"] is True
    # fields not in the docs are null with a reason, never invented
    assert f["Pre Bid Meeting"]["value"] is None and "Not found" in f["Pre Bid Meeting"]["notes"]
    v = rec["validation"]
    assert v["passed"] + v["failed"] + v["not_found"] == 20
    assert (settings.output_dir / "BidX.json").exists() and (settings.output_dir / "BidX_report.md").exists()
    # every found value has a citation
    for name, field in f.items():
        if field["value"] is not None:
            assert field["sources"], name
    # trace written with agent steps, tool calls and llm calls
    events = [json.loads(l) for l in open(rec["trace_file"])]
    kinds = {(e["agent"], e["step"]) for e in events}
    assert ("retrieval", "tool_call") in kinds and ("orchestrator", "plan") in kinds
    assert any(e["step"] == "llm_call" and "tokens_in" in e for e in events)


def test_feedback_loop_retries_uncited_and_rejected_fields(settings, mini_bid):
    responder = field_responder(VALUES, critic_reject={"Payment Terms"}, first_uncited={"Part_no"})
    llm = ScriptedLLM(responder, settings)
    rec = asyncio.run(Orchestrator(settings=settings, llm=llm).extract_bid(folder=mini_bid))
    retries = rec["run"]["retries"]
    assert retries.get("Part_no") == 1          # uncited value rejected by guardrail, then fixed on retry
    assert rec["fields"]["Part_no"]["value"] == "900-AB12"
    assert retries.get("Payment Terms") == settings.max_field_retries  # critic keeps rejecting: bounded retries
    assert "VALIDATION FAILED" in rec["fields"]["Payment Terms"]["notes"]
    assert rec["validation"]["failed"] >= 1


def test_llm_failures_do_not_crash_pipeline(settings, mini_bid):
    def broken(schema, system, user):
        if schema == "ExtractionLLM":
            return "this is not json"
        return field_responder(VALUES)(schema, system, user)

    rec = asyncio.run(_orch(settings, broken).extract_bid(folder=mini_bid))
    assert rec["validation"]["failed"] >= 17
    assert "extraction failed" in rec["fields"]["Bid Number"]["notes"]
    # the addendum agent still works independently and supplies the due date
    assert rec["fields"]["Due Date"]["value"] == "2025-03-15 14:00 EST"


def test_qa_answer_has_citations_and_guardrail(settings, mini_bid):
    o = _orch(settings, field_responder(VALUES))
    o.engine.index.index_folder(mini_bid)
    res = asyncio.run(o.answer("Is a bid bond required?"))
    assert res["found"] and res["citations"] and res["citations"][0]["file"]

    def uncited(schema, system, user):
        if schema == "AnswerLLM":
            return {"found": True, "answer": "Probably yes", "evidence_ids": [], "confidence": 0.9}
        return field_responder(VALUES)(schema, system, user)

    res2 = asyncio.run(_orch(settings, uncited).answer("Is a bid bond required?"))
    assert res2["answer"] == "Not found in documents" and not res2["found"]


def test_validator_rules():
    src = [Source(file="f", page=1, chunk_id="c1")]
    ok = FieldResult(value="2025-03-15 14:00 EST", sources=src, confidence=0.9, status="found")
    issues = rule_checks("Due Date", ok, "extended to March 15, 2025 at 2:00 PM EST", {"c1"})
    assert all(i.passed for i in issues), issues
    bad = FieldResult(value="2025-04-01 14:00 EST", sources=src, confidence=0.9, status="found")
    assert not all(i.passed for i in rule_checks("Due Date", bad, "March 15, 2025", {"c1"}))
    ident = FieldResult(value="WX-9999", sources=src, confidence=0.9, status="found")
    assert not all(i.passed for i in rule_checks("Bid Number", ident, "RFP WX-1001", {"c1"}))
    nosrc = FieldResult(value="Net 30", sources=[], confidence=0.9, status="found")
    assert not all(i.passed for i in rule_checks("Payment Terms", nosrc, "", {"c1"}))
    assert find_dates("26-MAY-2024 and 07/09/2024") == {"2024-05-26", "2024-07-09"}
