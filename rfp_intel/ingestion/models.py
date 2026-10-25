from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DocType = Literal["bid_page", "rfp", "addendum", "specs", "affidavit", "form", "other"]
BlockKind = Literal["text", "heading", "table", "kv"]


class Block(BaseModel):
    """A structural unit of a document page (paragraph run, heading, table, key/value)."""

    text: str
    kind: BlockKind = "text"
    page: int = 1
    section: Optional[str] = None


class ParsedDocument(BaseModel):
    bid_id: str
    file_name: str
    file_path: str
    doc_type: DocType
    addendum_number: Optional[int] = None
    doc_date: Optional[str] = None
    title: Optional[str] = None
    num_pages: int = 0
    blocks: list[Block] = Field(default_factory=list)
    key_values: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    sha1: str = ""

    def page_text(self, page: int) -> str:
        return "\n".join(b.text for b in self.blocks if b.page == page)

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


class Chunk(BaseModel):
    chunk_id: str
    bid_id: str
    file_name: str
    doc_type: DocType
    addendum_number: Optional[int] = None
    doc_date: Optional[str] = None
    page_number: int
    section: Optional[str] = None
    kind: BlockKind = "text"
    text: str

    def citation(self) -> dict:
        return {"file": self.file_name, "page": self.page_number, "chunk_id": self.chunk_id,
                "doc_type": self.doc_type, "addendum_number": self.addendum_number}

    def embed_text(self) -> str:
        """Text used for embeddings/BM25: the section heading adds context to short chunks.
        Only addenda get a doc-type tag (queries often say "addendum 2")."""
        prefix = f"Addendum {self.addendum_number or ''}".strip() + ". " if self.doc_type == "addendum" else ""
        if self.section:
            prefix += f"{self.section}: "
        return f"{prefix}{self.text}"
