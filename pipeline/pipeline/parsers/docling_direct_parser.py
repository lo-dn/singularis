"""Direct parser that converts Docling output to s0.json format without additional logic."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

from .utils import normalize_text


def parse_pdf_with_docling_direct(pdf_path: str) -> Dict[str, Any]:
    """
    Parse PDF using Docling and directly convert to s0.json format.

    This approach uses Docling's structured output directly without additional
    processing, avoiding issues with duplicate detection and caption extraction.
    """
    pdf_path = Path(pdf_path)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # Disable OCR
    pipeline_options.do_table_structure = False  # Disable table structure recognition
    pipeline_options.table_structure_options.do_cell_matching = False  # Disable cell matching
    pipeline_options.images_scale = 1.0  # Don't upscale images
    pipeline_options.generate_picture_images = False  # Don't generate picture images
    pipeline_options.do_picture_description = False  # Disable picture description

    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"🔄 Converting PDF with Docling...")
    result = converter.convert(str(pdf_path))
    doc = result.document

    print(f"✅ Docling conversion complete. Extracting structure...")

    # Extract sections by processing body items
    sections = _extract_sections_direct(doc)

    # Extract figures and tables with caption-based reclassification
    figures, tables, all_captions = _extract_figures_and_tables_with_reclassification(doc)

    # Extract metadata
    metadata = _extract_metadata(doc)
    doc_id = _generate_doc_id(doc, pdf_path)

    print(f"📊 Extracted: {len(sections)} sections, {len(all_captions)} captions, {len(figures)} figures, {len(tables)} tables")

    return {
        "sections": sections,
        "captions": all_captions,
        "figures": figures,
        "tables": tables,
        "suggested_doc_id": doc_id,
        "metadata": metadata
    }


def _extract_sections_direct(doc) -> List[Dict[str, Any]]:
    """Extract sections by iterating through document items."""
    sections = []
    section_counter = 1
    current_section = None

    # Section keywords for detection
    section_keywords = {
        'abstract', 'introduction', 'background', 'related work',
        'methods', 'methodology', 'materials and methods', 'experimental',
        'results', 'discussion', 'results and discussion',
        'conclusion', 'conclusions', 'references', 'bibliography',
        'acknowledgments', 'acknowledgements', 'appendix', 'supplementary',
        'nomenclature'
    }

    try:
        for item_tuple in doc.iterate_items():
            if len(item_tuple) < 1:
                continue

            item = item_tuple[0]

            # Get item properties
            label = item.label.lower() if hasattr(item, 'label') and item.label else ''
            text = str(item.text).strip() if hasattr(item, 'text') and item.text else ""

            if not text:
                continue

            # Skip captions, pictures, tables
            if label in {'caption', 'picture', 'table', 'figure'}:
                continue

            # Get page number
            page_no = None
            if hasattr(item, 'prov') and item.prov:
                page_no = item.prov[0].page_no if hasattr(item.prov[0], 'page_no') else None

            # Check if this is a section header
            is_section_header = (label == 'section_header')

            if is_section_header:
                # Save previous section
                if current_section and current_section['text']:
                    sections.append(current_section)

                # Start new section
                # section_name = _normalize_section_name(text)
                section_name = text
                current_section = {
                    "name": section_name,
                    "text": "",
                    "page_start": page_no if page_no else 1,
                    "page_end": page_no if page_no else 1,
                    "id": f"sec-{section_counter:02d}"
                }
                section_counter += 1
            else:
                # Add text to current section
                if current_section is None:
                    current_section = {
                        "name": "FrontMatter",
                        "text": "",
                        "page_start": page_no if page_no else 1,
                        "page_end": page_no if page_no else 1,
                        "id": f"sec-{section_counter:02d}"
                    }
                    section_counter += 1

                if current_section['text']:
                    current_section['text'] += ' '
                current_section['text'] += normalize_text(text)

                # Update page_end
                if page_no and page_no > current_section['page_end']:
                    current_section['page_end'] = page_no

        # Save last section
        if current_section and current_section['text']:
            sections.append(current_section)

    except Exception as e:
        print(f"⚠️  Warning: Could not extract sections: {e}")
        # Fallback
        try:
            full_text = doc.export_to_text()
            if full_text:
                sections.append({
                    "name": "Body",
                    "text": normalize_text(full_text),
                    "page_start": 1,
                    "page_end": doc.num_pages if hasattr(doc, 'num_pages') else None,
                    "id": "sec-01"
                })
        except Exception:
            pass

    return sections


def _get_page_bbox_from_prov(item) -> Tuple[int | None, List[float] | None]:
    page_no = None
    bbox = None
    try:
        if hasattr(item, 'prov') and item.prov:
            prov = item.prov[0]
            if hasattr(prov, 'page_no'):
                page_no = prov.page_no
            if hasattr(prov, 'bbox') and prov.bbox:
                b = prov.bbox
                bbox = [float(b.l), float(b.t), float(b.r), float(b.b)]
    except Exception:
        pass
    return page_no, bbox


def _collect_caption_items(doc) -> List[Dict[str, Any]]:
    """Collect caption-labeled items for optional fallbacks; only serializable fields."""
    items: List[Dict[str, Any]] = []
    try:
        for tup in doc.iterate_items():
            if not tup:
                continue
            it = tup[0]
            label = (getattr(it, 'label', None) or '').lower()
            if label != 'caption':
                continue
            text = getattr(it, 'text', None)
            if not text:
                continue
            text = normalize_text(str(text))
            page_no, bbox = _get_page_bbox_from_prov(it)
            items.append({
                'text': text,
                'page': page_no,
                'bbox': bbox,
            })
    except Exception:
        pass
    return items


def _bbox_distance(a: List[float] | None, b: List[float] | None) -> float:
    """Rough distance between two bboxes (vertical priority)."""
    if not a or not b:
        return float('inf')
    # center points
    ay = (a[1] + a[3]) / 2.0
    by = (b[1] + b[3]) / 2.0
    ax = (a[0] + a[2]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    # prioritize vertical closeness
    return abs(ay - by) * 10.0 + abs(ax - bx)


def _extract_figures_and_tables_with_reclassification(doc) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract figures and tables using Docling object's caption_text(doc).
    Falls back to first caption ref for bbox if present.
    """
    figures: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    captions: List[Dict[str, Any]] = []

    # Figures
    if hasattr(doc, 'pictures') and doc.pictures:
        print(f"  Found {len(doc.pictures)} pictures in document")
        for idx, picture in enumerate(doc.pictures, start=1):
            page_no, bbox = _get_page_bbox_from_prov(picture)

            # Prefer doc API for caption text
            cap_text = ""
            try:
                txt = picture.caption_text(doc)
                if txt:
                    cap_text = normalize_text(str(txt))
            except Exception:
                cap_text = ""

            # Try to get caption bbox from first caption ref if available
            cap_bbox = bbox
            try:
                if hasattr(picture, 'captions') and picture.captions:
                    cap_ref = picture.captions[0]
                    if hasattr(cap_ref, 'resolve'):
                        cap_item = cap_ref.resolve(doc)
                        c_page, c_bbox = _get_page_bbox_from_prov(cap_item)
                        if c_bbox:
                            cap_bbox = c_bbox
            except Exception:
                pass

            fig_id = f"Figure{idx}"
            prov_data = {"source": "pdf", "page": page_no, "bbox": cap_bbox}

            figures.append({
                "id": fig_id,
                "page": page_no,
                "caption": cap_text,
                "caption_provenance": [prov_data],
            })
            captions.append({
                "id": fig_id,
                "kind": "Figure",
                "text": cap_text,
                "provenance": [prov_data],
            })

    # Tables
    if hasattr(doc, 'tables') and doc.tables:
        print(f"  Found {len(doc.tables)} tables in document")
        for idx, table in enumerate(doc.tables, start=1):
            page_no, bbox = _get_page_bbox_from_prov(table)

            cap_text = ""
            try:
                txt = table.caption_text(doc)
                if txt:
                    cap_text = normalize_text(str(txt))
            except Exception:
                cap_text = ""

            cap_bbox = bbox
            try:
                if hasattr(table, 'captions') and table.captions:
                    cap_ref = table.captions[0]
                    if hasattr(cap_ref, 'resolve'):
                        cap_item = cap_ref.resolve(doc)
                        c_page, c_bbox = _get_page_bbox_from_prov(cap_item)
                        if c_bbox:
                            cap_bbox = c_bbox
            except Exception:
                pass

            tab_id = f"Table{idx}"
            prov_data = {"source": "pdf", "page": page_no, "bbox": cap_bbox}

            tables.append({
                "id": tab_id,
                "page": page_no,
                "caption": cap_text,
                "caption_provenance": [prov_data],
            })
            captions.append({
                "id": tab_id,
                "kind": "Table",
                "text": cap_text,
                "provenance": [prov_data],
            })

    print(f"  Result: {len(figures)} figures, {len(tables)} tables")

    return figures, tables, captions


def _extract_metadata(doc) -> Dict[str, Any]:
    """Extract metadata from Docling document."""
    metadata = {"source_format": "pdf"}

    try:
        if hasattr(doc, 'name') and doc.name:
            metadata['title'] = doc.name

        if hasattr(doc, 'document_meta'):
            doc_meta = doc.document_meta
            if hasattr(doc_meta, 'title') and doc_meta.title:
                metadata['title'] = doc_meta.title
            if hasattr(doc_meta, 'authors') and doc_meta.authors:
                metadata['author'] = ', '.join(str(a) for a in doc_meta.authors) if isinstance(doc_meta.authors, list) else str(doc_meta.authors)
    except Exception:
        pass

    return metadata


def _generate_doc_id(doc, pdf_path: Path) -> str:
    """Generate document ID from document or filename."""
    from .utils import slugify

    try:
        if hasattr(doc, 'name') and doc.name:
            return slugify(doc.name)
    except Exception:
        pass

    return slugify(pdf_path.stem)


def _is_section_header_text(text: str, keywords: set) -> bool:
    """Check if text looks like a section header."""
    text_lower = text.lower().strip()
    text_clean = re.sub(r'^\d+\.?\s*', '', text_lower)  # remove leading numbers
    # Heuristic: if it contains a keyword and doesn't look like a caption, it's a section header
    return any(keyword in text_clean for keyword in keywords) and not re.search(r'^(fig|tab)', text_clean)


def _normalize_section_name(header_text: str) -> str:
    """Normalize section header to standard name (lightweight)."""
    # strip leading numbering like '1.2 '
    header_text = re.sub(r'^\d+\.?\d*\s+', '', header_text or '')
    return (header_text or '').strip().title()
