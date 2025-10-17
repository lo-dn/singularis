"""Parsers package: contains helpers for extracting document structures."""

from .pdf_parser import parse_pdf_document, extract_title_and_authors
# from .latex_parser import parse_latex_sources
# from .docling_parser import parse_pdf_with_docling, extract_title_and_authors_from_docling
from .docling_direct_parser import parse_pdf_with_docling_direct

__all__ = [
    "parse_pdf_document",
    "extract_title_and_authors",
    # "parse_latex_sources",
    # "parse_pdf_with_docling",
    # "extract_title_and_authors_from_docling",
    "parse_pdf_with_docling_direct",
]
