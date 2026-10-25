"""Query understanding: domain synonym expansion + optional LLM rewriting.

The synonym map is generic procurement vocabulary (not bid-specific). Each
expansion becomes an extra query variant; variants are searched separately and
fused with RRF, so expansion widens recall without diluting the original query.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

# concept -> alternative phrasings used across RFPs / RFQs / IFBs
SYNONYMS: dict[str, list[str]] = {
    "deadline": ["due date", "submission deadline", "closing date", "proposal due date and time", "solicitation due"],
    "due date": ["submission deadline", "closing date", "proposals due", "solicitation due"],
    "submission": ["submittal", "how to submit proposal", "electronic submission portal", "sealed envelope"],
    "pre-bid": ["pre-proposal meeting", "prebid conference", "pre-bid conference", "site visit"],
    "pre bid": ["pre-proposal meeting", "prebid conference", "pre-bid conference"],
    "bond": ["bid bond", "bid security", "surety", "performance bond", "bid guarantee"],
    "payment": ["payment terms", "net 30", "invoice", "invoicing instructions", "prompt payment"],
    "warranty": ["extended warranty", "manufacturer warranty", "warranty years", "OEM warranty"],
    "term": ["length of contract", "contract term", "initial term", "renewal options", "extension"],
    "contract duration": ["length of contract", "initial term", "renewal"],
    "delivery": ["delivery within days of award", "ship date", "lead time", "delivery date"],
    "install": ["installation", "deployment services", "imaging", "white glove", "setup", "asset tagging"],
    "affidavit": ["affidavit", "required forms", "certification", "W-9", "attachments to submit"],
    "documents required": ["required forms", "attachments", "affidavit", "certificate of insurance", "W-9"],
    "contact": ["procurement officer", "buyer", "point of contact", "email", "phone"],
    "manufacturer": ["OEM", "authorized reseller", "letter of authorization", "manufacturer name"],
    "cooperative": ["cooperative purchasing", "interlocal agreement", "master contract", "piggyback", "state contract"],
    "model": ["model number", "model #", "product name", "SKU"],
    "part": ["part number", "SKU", "item number", "manufacturer part"],
    "spec": ["specifications", "minimum requirements", "processor memory storage display", "configuration"],
    "quantity": ["qty", "target quantity", "number of units"],
    "issuer": ["issuing organization", "agency", "district", "department"],
    "changed": ["addendum", "revised", "amended", "extended", "updated"],
    "addendum": ["addendum changes", "amendment", "revised due date", "questions and answers"],
}


def expand_query(query: str, max_variants: int = 4) -> list[str]:
    """Return [original, *expansions]. Each expansion = original + synonym phrase(s)."""
    q = query.strip()
    ql = q.lower()
    variants = [q]
    for key, alts in SYNONYMS.items():
        if re.search(rf"\b{re.escape(key)}", ql):
            missing = [a for a in alts if a.lower() not in ql]
            if missing:
                variants.append(f"{q} {' '.join(missing[:3])}")
        if len(variants) > max_variants:
            break
    return variants[: max_variants + 1]


class QueryRewrite(BaseModel):
    """LLM output: alternative search queries for the same information need."""

    queries: list[str] = Field(default_factory=list, description="2-4 rephrasings using RFP vocabulary")
    keywords: list[str] = Field(default_factory=list, description="exact identifiers / terms to keyword-match")


REWRITE_SYSTEM = (
    "You rewrite search queries for a retrieval engine over government RFP / bid documents. "
    "Produce 2-4 short alternative queries using procurement vocabulary (e.g. 'due date' vs 'closing date', "
    "'bid bond' vs 'bid security') and list any exact identifiers worth keyword matching. "
    "Do not answer the question."
)
