"""Specification of the 20 bid fields: what they mean, how to search for them,
which specialist extracts them, and how the validator checks their format."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Group = Literal["dates_logistics", "commercial_legal", "product_specs", "report"]
Fmt = Literal["text", "datetime", "identifier", "contact", "list", "summary"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    group: Group
    description: str
    queries: tuple[str, ...]
    fmt: Fmt = "text"
    doc_types: Optional[tuple[str, ...]] = None  # preferred doc types (soft: used as an extra filtered search)
    affected_by_addenda: bool = True
    allow_none_value: Optional[str] = None  # canonical value when the docs explicitly say it doesn't apply
    instructions: str = ""


FIELDS: list[FieldSpec] = [
    FieldSpec("Bid Number", "commercial_legal", "Official solicitation / RFP identifier (e.g. RFP number, solicitation number, PORFP number).",
              ("solicitation number", "RFP number bid number", "reference number PORFP number project number"),
              fmt="identifier", affected_by_addenda=False,
              instructions="Return the identifier exactly as written. If several identifiers exist (e.g. RFP number and portal/project number), give the primary solicitation number first and mention the others in notes."),
    FieldSpec("Title", "commercial_legal", "Full title of the bid or RFP.",
              ("title of the solicitation", "request for proposal title", "purchase order request for proposals"),
              affected_by_addenda=False),
    FieldSpec("Due Date", "dates_logistics", "Final submission deadline including time and time zone, after all addendum changes.",
              ("proposal due date and time", "solicitation due closing date", "submission deadline", "new due date addendum extension"),
              fmt="datetime",
              instructions="Format as 'YYYY-MM-DD HH:MM TZ' (24h clock, time zone abbreviation as written). If sources disagree, prefer the latest addendum, then the RFP, then the portal page, and explain in notes."),
    FieldSpec("Bid Submission Type", "dates_logistics", "How the bid is submitted (electronic portal, sealed paper, email, etc.).",
              ("how to submit proposal electronic submission", "responses will only be accepted through", "sealed envelope hard copy manual submission", "e-procurement portal registration"),
              affected_by_addenda=False),
    FieldSpec("Term of Bid", "dates_logistics", "Contract duration and any renewal options.",
              ("length of contract term of agreement", "initial term renewal options extension", "contract period years"),
              allow_none_value="Not specified"),
    FieldSpec("Pre Bid Meeting", "dates_logistics", "Pre-bid / pre-proposal meeting date, time, location and whether attendance is mandatory, or 'None'.",
              ("pre-proposal meeting", "pre-bid conference date time location", "mandatory site visit attendance"),
              allow_none_value="None",
              instructions="Include date, time with time zone, location / format, and whether attendance is mandatory. If the documents describe no such meeting, value must be 'None' and cite nothing only if truly absent (status not_found)."),
    FieldSpec("Installation", "dates_logistics", "Whether installation / deployment / imaging / setup services are required.",
              ("installation services required", "deployment services white glove imaging asset tagging", "setup configuration delivery to locations"),
              instructions="Answer 'Required: ...' or 'Not required' with a short description of the services."),
    FieldSpec("Bid Bond Requirement", "commercial_legal", "Required bid bond or security (amount or percentage), or 'Not required'.",
              ("bid bond", "bid security surety bond guarantee", "insurance and bonds requirements", "performance bond percentage"),
              allow_none_value="Not required",
              instructions="Only report a bond if the documents state one is required. Generic boilerplate ('comply with any bid bond requirements noted elsewhere') with no actual requirement means 'Not required' — say so in notes."),
    FieldSpec("Delivery Date", "dates_logistics", "Required delivery date or delivery window after award.",
              ("delivery within days of award", "delivery date required", "when do you need these delivered", "ship date lead time"),
              allow_none_value="Not specified"),
    FieldSpec("Payment Terms", "commercial_legal", "Payment terms (e.g. Net 30) and invoicing conditions.",
              ("payment terms net 30 days", "invoicing instructions invoice submitted", "payment will be made within", "prompt payment"),
              allow_none_value="Not specified"),
    FieldSpec("Any Additional Documentation Required", "commercial_legal", "Affidavits, forms, certificates, insurance, W-9, etc. that must be submitted.",
              ("affidavit must provide", "forms to be completed and attached W9", "required attachments certificate of insurance", "signed addendum acknowledgement", "warranty certificate"),
              fmt="list", doc_types=("rfp", "addendum", "affidavit", "bid_page"),
              instructions="Return a list of distinct documents/forms. Include affidavit files that are part of the bid package."),
    FieldSpec("MFG for Registration", "commercial_legal", "Manufacturer(s) the bidder must be registered / authorized with.",
              ("authorized reseller for manufacturer", "manufacturer name", "factory-authorized OEM certification", "letter of authorization from manufacturer"),
              allow_none_value="Not specified"),
    FieldSpec("Contract or Cooperative to use", "commercial_legal", "Referenced cooperative / state contract vehicle, if any.",
              ("cooperative purchasing interlocal agreement", "master contract", "piggyback contract", "state contract vehicle"),
              allow_none_value="None", affected_by_addenda=False),
    FieldSpec("Model_no", "product_specs", "Model number(s) of the requested product(s).",
              ("model number", "model # product description", "proposed device make and model"),
              fmt="list", doc_types=("rfp", "specs", "bid_page"), allow_none_value="Not specified",
              instructions="List model numbers exactly as written in the documents. If the RFP only defines tiers/specs and leaves the model to the vendor, say so ('Not specified - vendor to propose') with evidence."),
    FieldSpec("Part_no", "product_specs", "Part / SKU number(s) of the requested product(s).",
              ("SKU part number", "manufacturer part number", "item number"),
              fmt="list", doc_types=("specs", "rfp", "bid_page"), allow_none_value="Not specified",
              instructions="List SKUs/part numbers exactly as written; for long configuration lists include the base/system SKU and the key component SKUs."),
    FieldSpec("Product", "product_specs", "Product name(s) and quantity.",
              ("product name quantity qty", "line items target quantity", "devices requested laptops desktops"),
              fmt="list", doc_types=("rfp", "bid_page", "specs"),
              instructions="One entry per product line: '<name> - qty <n>' (use the quantity as written; mention if quantities are estimates)."),
    FieldSpec("contact_info", "commercial_legal", "Procurement contact: name, email, phone.",
              ("buyer email phone", "procurement officer point of contact", "contact information agency POC"),
              fmt="contact", affected_by_addenda=False),
    FieldSpec("company_name", "commercial_legal", "Issuing organization / agency name.",
              ("issuing organization", "agency name district department", "purchasing department"),
              affected_by_addenda=False),
    FieldSpec("Product Specification", "product_specs", "Key technical specifications (CPU, RAM, storage, display, OS, warranty, etc.).",
              ("minimum requirements processor memory storage display", "specifications warranty operating system", "configuration processor RAM SSD"),
              doc_types=("rfp", "specs", "addendum"),
              instructions="Summarise key specs per product/tier on separate lines ('<product>: CPU ..., RAM ..., storage ..., display ..., OS ..., warranty ...'). Include spec changes from addenda."),
    FieldSpec("Bid Summary", "report", "Concise 3-6 sentence summary of the bid, generated by the system.",
              (), fmt="summary", affected_by_addenda=False),
]

FIELD_BY_NAME = {f.name: f for f in FIELDS}
EXTRACTABLE = [f for f in FIELDS if f.group != "report"]
GROUPS: dict[str, list[FieldSpec]] = {}
for _f in EXTRACTABLE:
    GROUPS.setdefault(_f.group, []).append(_f)

GROUP_TITLES = {
    "dates_logistics": "Dates & Logistics specialist",
    "commercial_legal": "Commercial & Legal specialist",
    "product_specs": "Product & Specs specialist",
}
