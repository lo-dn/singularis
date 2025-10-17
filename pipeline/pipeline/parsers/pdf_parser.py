from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import List, Dict, Any, Optional

import pymupdf

from .utils import normalize_text

SECTION_PAT = re.compile(
    r"""(?ix) ^
        (?: \d+\.?\s+ | [ivxl]+\.\s+ )?
        (abstract|introduction|background|methodology|
         materials?\s+and\s+methods?|materials?\s*&\s*methods?|materials?|
         methods?|method|
         results?\s+and\s+discussion|results?\s+and\s+analysis|results?|discussion|
         analysis|analysis\s+and\s+discussion|findings|
         case\s+stud(?:y|ies)|study\s+area|experimental|experiments|evaluation|
         conclusions?|conclusion|discussion\s+and\s+conclusions?|future\s+work|
         references|acknowledg(e)?ments|related\s+work|literature\s+review|limitations)
        \s* $
    """
)
ALLCAPS_SECTION_PAT = re.compile(r"^[A-Z][A-Z\s\-]{3,}$")

CAPTION_PAT = re.compile(
    r"""
    ^\s*
    (?P<label>fig(?:\.|ure)?|table|tab\.)
    \s*
    (?P<num>[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*)?
    \s*
    (?P<punct>[:.\-–—])
    \s*
    (?P<body>.*)
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

INLINE_HEADING_RE = re.compile(
    r'^\s*(abstract|introduction|conclusion|conclusions|discussion|results|'
    r'materials\s+and\s+methods|materials\s*&\s*methods|results?\s+and\s+discussion)'
    r'\s*[-:—–]\s*(.+)$',
    re.I
)

SECTION_NUMBER_RE = re.compile(
    r'''
    ^\s*
    (?:
        [\(\[]?\s*
        (?:[0-9]+|[ivxlcdm]+)       # leading number / roman numeral
        (?:[\.\-][0-9a-z]+)*        # hierarchical suffixes like .1 or -a
        [\)\]]?
        |
        [A-Z]\.                     # enumerated bullet like A.
    )
    [\.\)\-:]?\s+
    ''',
    re.IGNORECASE | re.VERBOSE
)

HEADING_CANONICAL = {
    "abstract", "introduction", "background", "related work", "literature review",
    "methodology", "materials and methods", "materials and method", "materials", "methods",
    "method", "data and methods", "experimental", "experiments", "study area",
    "case study", "case studies", "results", "results and discussion", "discussion",
    "analysis", "analysis and discussion", "findings", "evaluation",
    "conclusion", "conclusions", "discussion and conclusions",
    "future work", "limitations", "acknowledgements", "acknowledgments", "references"
}
TITLECASE_STOPWORDS = {
    "and", "or", "the", "in", "of", "with", "on", "to", "for", "by", "as", "at"
}
ACRONYM_WHITELIST = {
    "NASA", "NATO", "OECD", "OPEC", "ASEAN", "UNESCO", "UNICEF", "UNDP", "UNEP", "IEEE",
    "FAO", "WHO", "IMF", "OCHA", "NAFTA", "FBI", "CIA", "UN", "EU", "UK", "USA", "UAE"
}


def _looks_like_heading(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if len(stripped) > 80:
        return False
    if re.search(r'[.!?;:,]$', stripped):
        return False
    if re.match(r'^[A-Z]\.$', stripped):
        return True
    words = stripped.split()
    if not words:
        return False
    if len(words) <= 6 and all(w.isupper() or (w[0].isupper() and w[1:].islower()) for w in words):
        return True
    return False


@dataclass
class CaptionData:
    id: str
    kind: str
    page: int
    text: str
    bbox: Optional[List[float]] = None
    image_bbox: Optional[List[float]] = None


def extract_title_and_authors(doc: pymupdf.Document) -> tuple[Optional[str], Optional[str]]:
    """Heuristic detection of title/authors from the first page."""
    if len(doc) == 0:
        return None, None
    page = doc[0]
    data = page.get_text("dict") or {}
    spans = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                spans.append({
                    "text": text,
                    "size": float(span.get("size", 0.0)),
                    "y": float(line.get("bbox", [0, 0, 0, 0])[1]),
                })
    if not spans:
        return None, None

    def _clean_len(s: dict) -> int:
        return len((s.get("text") or "").strip())

    candidates = [s for s in spans if _clean_len(s) >= 4]
    if not candidates:
        candidates = spans[:]

    max_size = max(span["size"] for span in candidates)
    title_lines = [s for s in candidates if s["size"] >= max_size - 0.5 and s["y"] < spans[0]["y"] + 300]
    title = " ".join(s["text"] for s in title_lines).strip()

    below_title = [s for s in spans if s["y"] > (title_lines[0]["y"] if title_lines else 0)]
    sizes = sorted({s["size"] for s in below_title}, reverse=True)
    second = sizes[0] if sizes else 0
    author_lines = [s for s in below_title if abs(s["size"] - second) < 0.6]
    authors = " ".join(s["text"] for s in author_lines).strip()
    return (title or None), (authors or None)


def _match_caption(line: str) -> Optional[tuple[str, str, str]]:
    m = CAPTION_PAT.match(line or "")
    if not m:
        return None
    label = m.group("label").lower()
    kind = "Figure" if label.startswith("fig") else "Table"
    num = (m.group("num") or "").strip()
    tail = (m.group("body") or "").strip()
    return kind, num, tail


def _extract_blocks_sample(page: pymupdf.Page, limit: int = 5) -> List[Dict[str, Any]]:
    blocks = []
    for b in page.get_text("blocks")[:limit]:
        x0, y0, x1, y1, text, *_ = b
        if not (text or "").strip():
            continue
        blocks.append({
            "bbox": [round(float(x0), 1), round(float(y0), 1), round(float(x1), 1), round(float(y1), 1)],
            "text_preview": text.strip().replace("\n", " ")[:160]
        })
    return blocks


def _split_inline_heading(text: str) -> Optional[tuple[str, str]]:
    m = INLINE_HEADING_RE.match(text or "")
    if not m:
        return None
    title = (m.group(1) or "").strip().title()
    remainder = (m.group(2) or "").strip()
    return title, remainder


def _assign_nearest_image_bbox(caption_bbox: List[float],
                               image_blocks: List[Dict[str, Any]]) -> Optional[List[float]]:
    if not caption_bbox or not image_blocks:
        return None
    best_block: Optional[Dict[str, Any]] = None
    best_distance = float("inf")
    cap_top = caption_bbox[1]
    cap_left = caption_bbox[0]
    cap_right = caption_bbox[2]

    for block in image_blocks:
        if block.get("used"):
            continue
        bbox = block.get("bbox")
        if not bbox:
            continue
        bottom = bbox[3]
        if bottom > cap_top + 10:
            continue
        horiz_overlap = not (bbox[2] < cap_left or bbox[0] > cap_right)
        vertical_distance = max(0.0, cap_top - bottom)
        if not horiz_overlap and vertical_distance > 200:
            continue
        if vertical_distance < best_distance:
            best_distance = vertical_distance
            best_block = block

    if best_block:
        best_block["used"] = True
        bbox = best_block.get("bbox")
        if bbox:
            return [round(float(x), 2) for x in bbox]
    return None


def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    cleaned_chars = []
    for ch in text:
        if ch in "\n\r\t":
            cleaned_chars.append(" ")
        elif ch.isprintable():
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    cleaned = "".join(cleaned_chars)
    return re.sub(r'\s+', ' ', cleaned.strip())


def _strip_section_numbering(text: str) -> str:
    value = (text or "").strip()
    prev = None
    while value and value != prev:
        prev = value
        match = SECTION_NUMBER_RE.match(value)
        if not match:
            break
        value = value[match.end():]
        value = value.lstrip()
    return value.strip(" -–—:").strip()


def _clean_heading_text(text: str) -> str:
    stripped = _strip_section_numbering(text)
    stripped = _normalize_whitespace(stripped)
    return stripped


def _uppercase_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    uppers = sum(1 for ch in letters if ch.isupper())
    return uppers / len(letters)


def _titlecase_ratio(text: str) -> float:
    words = [w for w in re.split(r'[^A-Za-z0-9]+', text) if w]
    if not words:
        return 0.0
    title_like = 0
    for w in words:
        if len(w) == 1:
            if w.isupper():
                title_like += 1
            continue
        if w[0].isupper() and (w[1:].islower() or w[1:].isdigit()):
            title_like += 1
    return title_like / len(words) if words else 0.0


def _heading_score(line: Dict[str, Any], body_font_size: float) -> float:
    clean = line.get("clean_text") or ""
    if not clean:
        return 0.0
    text = line.get("text") or ""
    word_count = line.get("word_count") or 0
    if word_count == 0:
        return 0.0

    score = 0.0
    font_size = line.get("font_size") or 0.0
    gap_above = line.get("gap_above") or 0.0
    upper_ratio = line.get("upper_ratio") or 0.0
    title_ratio = line.get("title_ratio") or 0.0
    is_hint = bool(line.get("is_heading_hint"))

    if body_font_size:
        delta = font_size - body_font_size
        if delta >= 2.0:
            score += 0.45
        elif delta >= 1.0:
            score += 0.35
        elif delta >= 0.5:
            score += 0.2
    elif font_size:
        score += 0.2

    if upper_ratio > 0.85 and word_count <= 12:
        score += 0.4
    elif upper_ratio > 0.6 and word_count <= 10:
        score += 0.28

    if title_ratio > 0.8 and word_count <= 12:
        score += 0.25
    elif title_ratio > 0.6 and word_count <= 14:
        score += 0.15

    if gap_above >= 8:
        score += 0.25
    elif gap_above >= 5:
        score += 0.15
    elif gap_above <= 1:
        score -= 0.2
    elif gap_above <= 3:
        score -= 0.05

    if "," in clean:
        score -= 0.25

    lowered = clean.lower()
    if lowered in HEADING_CANONICAL:
        score += 0.35

    if is_hint:
        score += 0.15

    if word_count > 18:
        score -= 0.5
    elif word_count > 14:
        score -= 0.25

    if re.search(r'[.!?;:,]$', text):
        score -= 0.35

    return score


def _merge_heading_lines(lines: List[Dict[str, Any]], start_idx: int, threshold: float) -> tuple[Optional[str], int]:
    if start_idx >= len(lines):
        return None, 0
    first = lines[start_idx]
    if first.get("heading_score", 0.0) < threshold:
        return None, 0

    parts = [first.get("clean_text") or first.get("text") or ""]
    consumed = 1
    base_size = first.get("font_size") or 0.0

    for idx in range(start_idx + 1, len(lines)):
        line = lines[idx]
        if not line.get("text"):
            break
        candidate_score = line.get("heading_score", 0.0)
        if candidate_score < (threshold - 0.1):
            break
        gap = line.get("gap_above", 0.0)
        font_size = line.get("font_size") or 0.0
        if abs(font_size - base_size) > 1.0:
            break
        if gap > 3.0:
            break
        parts.append(line.get("clean_text") or line.get("text") or "")
        consumed += 1
    name = " ".join(part.strip(" -–—") for part in parts if part).strip()
    name = _normalize_whitespace(name)
    if not name:
        return None, 0
    return name, consumed


def _format_section_name(name: str) -> str:
    cleaned = _clean_heading_text(name or "")
    if not cleaned:
        cleaned = (name or "").strip()
    cleaned = cleaned.strip(" .:-–—")
    if not cleaned:
        return "Body"
    if cleaned.isupper():
        words = []
        for word in cleaned.split():
            lower = word.lower()
            if not word:
                continue
            if not word.isalpha():
                words.append(word.capitalize())
                continue
            if lower in TITLECASE_STOPWORDS:
                words.append(lower.capitalize())
                continue
            if word in ACRONYM_WHITELIST or len(word) <= 3:
                words.append(word)
                continue
            has_digit = any(ch.isdigit() for ch in word)
            vowels = sum(1 for ch in word if ch in "AEIOU")
            if has_digit or vowels == 0:
                words.append(word)
                continue
            words.append(word.capitalize())
        return " ".join(words)
    return cleaned


def _extract_pdf_page_struct(page: pymupdf.Page, page_no: int) -> Dict[str, Any]:
    raw_blocks = page.get_text("blocks") or []
    dict_data = page.get_text("dict") or {}
    lines: List[Dict[str, Any]] = []
    image_blocks: List[Dict[str, Any]] = []
    block_index_map: Dict[tuple, int] = {}

    for block_idx, block in enumerate(raw_blocks):
        if not block or len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        block_type = int(block[6]) if len(block) > 6 else 0
        bbox = [round(float(x0), 2), round(float(y0), 2), round(float(x1), 2), round(float(y1), 2)]
        block_index_map.setdefault(tuple(bbox), block_idx)

        if block_type == 1:
            image_blocks.append({
                "block_index": block_idx,
                "bbox": bbox,
                "used": False,
                "page": page_no
            })
            continue

    prev_bottom = None
    for block in dict_data.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_bbox = block.get("bbox") or []
        rounded_block = tuple(round(float(coord), 2) for coord in (block_bbox or []))
        block_index = block_index_map.get(rounded_block)
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            text = "".join(span.get("text") or "" for span in spans).strip()
            if not text:
                continue
            bbox = line.get("bbox") or block_bbox or [0, 0, 0, 0]
            bbox = [round(float(coord), 2) for coord in bbox]
            font_size = 0.0
            if spans:
                font_size = sum(float(span.get("size", 0.0)) for span in spans) / len(spans)
            gap_above = 0.0
            top = float(bbox[1])
            bottom = float(bbox[3])
            if prev_bottom is not None:
                gap_above = max(0.0, top - prev_bottom)

            dir_vec = line.get("dir") or (1.0, 0.0)
            dir_x, dir_y = dir_vec if isinstance(dir_vec, (list, tuple)) and len(dir_vec) == 2 else (1.0, 0.0)
            abs_dx, abs_dy = abs(float(dir_x)), abs(float(dir_y))
            width = abs(bbox[2] - bbox[0]) if len(bbox) == 4 else 0.0
            height = abs(bbox[3] - bbox[1]) if len(bbox) == 4 else 0.0
            is_vertical = False
            if abs_dx < 1e-3 and abs_dy > 0:
                is_vertical = True
            elif abs_dy > abs_dx * 1.5:
                is_vertical = True
            if not is_vertical and width and height:
                if height > width * 4 and width < 100:
                    is_vertical = True
            if is_vertical:
                continue

            clean_text = _clean_heading_text(text)
            entry = {
                "text": _normalize_whitespace(text),
                "clean_text": clean_text,
                "bbox": bbox,
                "block_index": block_index,
                "page": page_no,
                "font_size": font_size,
                "gap_above": gap_above,
                "is_heading_hint": _looks_like_heading(text)
            }
            entry["upper_ratio"] = _uppercase_ratio(entry["text"])
            entry["title_ratio"] = _titlecase_ratio(entry["text"])
            entry["word_count"] = len([w for w in entry["text"].split() if w])
            entry["is_heading"] = entry["is_heading_hint"]
            lines.append(entry)
            prev_bottom = bottom

    return {
        "text_blocks": raw_blocks,
        "lines": lines,
        "image_blocks": image_blocks
    }


def _is_section_heading(line: str) -> Optional[str]:
    raw = _normalize_whitespace(line or "")
    if not raw:
        return None
    # Strip leading bullets/symbols like '■' and whitespace
    raw = re.sub(r'^[^A-Za-z0-9]+', '', raw)
    cleaned = _clean_heading_text(raw)
    if not cleaned:
        return None
    m = SECTION_PAT.match(cleaned)
    if m:
        return m.group(1).title()
    if cleaned and cleaned == cleaned.upper() and len(cleaned.split()) <= 12 and ALLCAPS_SECTION_PAT.match(cleaned):
        word = cleaned.lower()
        for candidate in [
            "abstract", "introduction", "methods", "materials and methods",
            "results", "discussion", "conclusions", "conclusion",
            "supplementary", "appendix", "acknowledgements", "acknowledgments", "references", "biographies", "biography"
        ]:
            if candidate.replace(" ", "") in word.replace(" ", ""):
                return candidate.title()
    return None


def _build_pdf_sections(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []

    def finalize(sec: Optional[Dict[str, Any]]):
        if not sec or not sec.get("chunks"):
            return
        chunks = [ch for ch in sec["chunks"] if ch.get("text")]
        if not chunks:
            return

        text_parts: List[str] = []
        page_numbers: List[int] = []

        for chunk in chunks:
            chunk_text = chunk.get("text", "").strip()
            if not chunk_text:
                continue
            if text_parts:
                text_parts.append(" ")
            text_parts.append(chunk_text)
            page = chunk.get("page")
            if page is not None:
                page_numbers.append(page)

        text = "".join(text_parts).strip()
        if not text:
            return

        sections.append({
            "name": sec.get("name") or "Body",
            "text": text,
            "page_start": min(page_numbers) if page_numbers else None,
            "page_end": max(page_numbers) if page_numbers else None,
        })

    def _is_biographies(name: Optional[str]) -> bool:
        n = (name or "").strip().lower()
        return bool(n) and ("biograph" in n or n in {"biography", "biographies", "author biographies", "authors biographies", "authors' biographies"})

    current: Optional[Dict[str, Any]] = None
    body_font_size = None
    font_values = [line.get("font_size") for line in lines if line.get("font_size")]
    if font_values:
        body_font_size = median(font_values)
    heading_threshold = 0.6

    effective_body_font = body_font_size or 0.0
    for entry in lines:
        if "heading_score" not in entry:
            entry["heading_score"] = _heading_score(entry, effective_body_font)

    idx = 0
    total = len(lines)
    while idx < total:
        entry = lines[idx]
        text = entry.get("text", "")
        if not text:
            idx += 1
            continue

        heading_name: Optional[str] = None
        consumed_lines = 1
        heading = _is_section_heading(text)
        if heading:
            heading_name = heading
        elif entry.get("heading_score", 0.0) >= heading_threshold:
            dominance = max(entry.get("upper_ratio", 0.0), entry.get("title_ratio", 0.0))
            if dominance >= 0.2:
                merged_name, consumed = _merge_heading_lines(lines, idx, heading_threshold)
                if merged_name:
                    heading_name = merged_name
                    consumed_lines = max(consumed, 1)

        if heading_name:
            normalized = _clean_heading_text(heading_name) or heading_name
            lowered = normalized.lower()
            use_heading = any(ch.isalpha() for ch in normalized) and len(normalized) >= 3 and bool(re.search(r'[A-Za-z]', normalized))
            # Do not create sub-sections inside Biographies unless it's a canonical heading
            if _is_biographies(current.get("name") if current else None):
                # Only allow canonical headings to break out of Biographies
                if lowered not in HEADING_CANONICAL and lowered not in {"biography", "biographies"}:
                    use_heading = False
            if use_heading:
                starts_with_number = bool(re.match(r'^\s*[\(\[]?\d', entry.get("text", "")))
                starts_with_roman = bool(re.match(r'^\s*[\(\[]?[ivxlcdm]+[\.\)]\s', entry.get("text", ""), re.I))
                if (
                    not sections
                    and (current is None or current.get("name") == "FrontMatter")
                    and lowered not in HEADING_CANONICAL
                    and not starts_with_number
                    and not starts_with_roman
                ):
                    use_heading = False
                if use_heading and lowered not in HEADING_CANONICAL and not starts_with_number and not starts_with_roman:
                    bbox = entry.get("bbox") or []
                    width = (bbox[2] - bbox[0]) if len(bbox) == 4 else None
                    font_size = entry.get("font_size") or 0.0
                    if width is not None and width < 80 and font_size <= effective_body_font + 0.8:
                        use_heading = False
            if use_heading:
                finalize(current)
                current = {"name": _format_section_name(heading_name), "chunks": []}
                idx += consumed_lines
                continue
            heading_name = None

        # Inline headings: suppress inside Biographies except canonical headings
        inline = _split_inline_heading(text)
        if inline:
            heading_inline, remainder = inline
            heading_l = (heading_inline or "").strip().lower()
            is_canonical_inline = heading_l in HEADING_CANONICAL or heading_l in {"references", "acknowledgments", "acknowledgements"}
            if _is_biographies(current.get("name") if current else None) and not is_canonical_inline:
                # treat as normal text within Biographies
                pass
            else:
                finalize(current)
                current = {"name": heading_inline, "chunks": []}
                if remainder:
                    current["chunks"].append({**entry, "text": remainder})
                idx += 1
                continue

        # Lines that look like headings: treat them as normal text within Biographies
        if entry.get("is_heading") and not _is_biographies(current.get("name") if current else None):
            idx += 1
            continue

        if current is None:
            current = {"name": "FrontMatter", "chunks": []}
        current.setdefault("chunks", []).append(entry)
        idx += 1

    finalize(current)

    if not sections:
        text_parts: List[str] = []
        page_numbers: List[int] = []
        for entry in lines:
            chunk_text = entry.get("text", "").strip()
            if not chunk_text:
                continue
            if text_parts:
                text_parts.append(" ")
            text_parts.append(chunk_text)
            page = entry.get("page")
            if page is not None:
                page_numbers.append(page)
        combined_text = "".join(text_parts).strip()
        if combined_text:
            sections.append({
                "name": "Body",
                "text": combined_text,
                "page_start": min(page_numbers) if page_numbers else None,
                "page_end": max(page_numbers) if page_numbers else None,
            })
    return sections


def _collect_captions_from_blocks(blocks: List[Any],
                                  page_no: int,
                                  image_blocks: Optional[List[Dict[str, Any]]] = None) -> List[CaptionData]:
    caps: List[CaptionData] = []
    image_blocks = image_blocks or []

    for block in blocks:
        if not block or len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        block_type = int(block[6]) if len(block) > 6 else 0
        if block_type != 0:
            continue
        if not (text or "").strip():
            continue
        bbox = [round(float(x0), 2), round(float(y0), 2), round(float(x1), 2), round(float(y1), 2)]
        raw_lines = []
        for line in (text or "").splitlines():
            line_text = line.strip()
            if not line_text:
                continue
            raw_lines.append({
                "text": line_text,
                "bbox": bbox
            })
        if not raw_lines:
            continue

        lines = [row["text"] for row in raw_lines]
        i = 0
        while i < len(lines):
            line = lines[i]
            cap_head = _match_caption(line)
            if not cap_head:
                i += 1
                continue

            kind, num, tail = cap_head
            cap_id = f"{kind}{num}" if num else kind
            collected_lines: List[str] = []
            used_indices: List[int] = []
            if tail:
                collected_lines.append(tail)
                used_indices.append(i)

            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt:
                    break
                if _match_caption(nxt) or _is_section_heading(nxt):
                    break
                collected_lines.append(nxt)
                used_indices.append(j)
                j += 1

            caption_text = " ".join(collected_lines).strip()
            if not caption_text:
                i = j
                continue

            indices = used_indices or [i]
            xs0 = min(raw_lines[idx]["bbox"][0] for idx in indices)
            ys0 = min(raw_lines[idx]["bbox"][1] for idx in indices)
            xs1 = max(raw_lines[idx]["bbox"][2] for idx in indices)
            ys1 = max(raw_lines[idx]["bbox"][3] for idx in indices)
            caption_bbox = [xs0, ys0, xs1, ys1]
            image_bbox = _assign_nearest_image_bbox(caption_bbox, image_blocks)

            caps.append(CaptionData(
                kind=kind,
                id=cap_id,
                page=page_no,
                text=caption_text,
                bbox=caption_bbox,
                image_bbox=image_bbox
            ))
            i = j
    return caps


def parse_pdf_document(doc: pymupdf.Document) -> Dict[str, Any]:
    page_count = len(doc)
    pages_sample: List[Dict[str, Any]] = []
    all_lines: List[Dict[str, Any]] = []
    all_captions: List[CaptionData] = []

    for idx in range(page_count):
        page = doc[idx]
        page_no = idx + 1
        page_struct = _extract_pdf_page_struct(page, page_no)
        pages_sample.append({
            "page": page_no,
            "blocks_sample": _extract_blocks_sample(page)
        })
        all_lines.extend(page_struct["lines"])
        page_caps = _collect_captions_from_blocks(
            page_struct["text_blocks"],
            page_no,
            page_struct["image_blocks"]
        )
        all_captions.extend(page_caps)

    sections = _build_pdf_sections(all_lines)
    for idx, sec in enumerate(sections):
        sec["id"] = f"sec-{idx + 1:02d}"

    captions_list: List[Dict[str, Any]] = []
    seen_caps: set[tuple] = set()
    for cap in all_captions:
        if not cap.text:
            continue
        norm_text = normalize_text(cap.text)
        key = (cap.id, norm_text, cap.page)
        if key in seen_caps:
            continue
        seen_caps.add(key)
        entry: Dict[str, Any] = {
            "id": cap.id,
            "kind": cap.kind,
            "text": norm_text,
            "provenance": [{
                "source": "pdf",
                "page": cap.page,
                "bbox": cap.bbox
            }]
        }
        if cap.image_bbox:
            entry["content_bbox"] = cap.image_bbox
        captions_list.append(entry)

    figures = []
    tables = []
    for cap in captions_list:
        prov = cap.get("provenance") or []
        page_ref = prov[0]["page"] if prov else None
        base_entry = {
            "id": cap.get("id"),
            "page": page_ref,
            "caption": cap.get("text"),
            "caption_provenance": prov
        }
        if cap.get("content_bbox"):
            base_entry["content_bbox"] = cap["content_bbox"]
        if cap.get("kind") == "Figure":
            figures.append(base_entry)
        elif cap.get("kind") == "Table":
            tables.append(base_entry)

    return {
        "sections": sections,
        "captions": captions_list,
        "figures": figures,
        "tables": tables,
        "pages_sample": pages_sample
    }
