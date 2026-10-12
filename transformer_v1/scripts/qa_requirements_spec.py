from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "software_delivery" / "端到端协同跟踪原型软件_需求规格说明书.docx"
PDF = ROOT / ".codex_work" / "final_req_review_render" / "requirements_spec_final.pdf"


def _paragraph_text(element) -> str:
    return "".join(node.text or "" for node in element.iter(qn("w:t")))


def _run_text(run) -> str:
    return "".join(node.text or "" for node in run.iter(qn("w:t")))


def _effective_style_value(style, name: str):
    current = style
    while current is not None:
        if name == "eastAsia":
            rpr = current._element.rPr
            fonts = None if rpr is None else rpr.rFonts
            value = None if fonts is None else fonts.get(qn("w:eastAsia"))
        else:
            value = getattr(current.font, name)
        if value is not None:
            return value
        current = current.base_style
    return None


def main() -> None:
    doc = Document(DOCX)
    body = list(doc._element.body)
    boundary = next(
        index
        for index, child in enumerate(body)
        if child.tag == qn("w:p") and _paragraph_text(child).strip() == "1 范围"
    )
    body_elements = body[boundary:]

    font_errors: list[str] = []
    heading_errors: list[str] = []
    style_errors: list[str] = []
    header_fill_errors: list[str] = []
    header_bold_errors: list[str] = []
    body_fill_errors: list[str] = []
    body_text_parts: list[str] = []
    table_count = 0

    for child in body_elements:
        body_text_parts.append(_paragraph_text(child))
        is_heading = False
        if child.tag == qn("w:p"):
            ppr = child.find(qn("w:pPr"))
            pstyle = None if ppr is None else ppr.find(qn("w:pStyle"))
            style_value = None if pstyle is None else pstyle.get(qn("w:val"))
            is_heading = style_value in {"1", "2", "3", "4", "Heading1", "Heading2", "Heading3", "Heading4"}
        for run in child.iter(qn("w:r")):
            text = _run_text(run)
            if not text.strip():
                continue
            rpr = run.find(qn("w:rPr"))
            fonts = None if rpr is None else rpr.find(qn("w:rFonts"))
            size = None if rpr is None else rpr.find(qn("w:sz"))
            east_asia = None if fonts is None else fonts.get(qn("w:eastAsia"))
            ascii_font = None if fonts is None else fonts.get(qn("w:ascii"))
            hansi_font = None if fonts is None else fonts.get(qn("w:hAnsi"))
            size_value = None if size is None else size.get(qn("w:val"))
            explicit_values = (east_asia, ascii_font, hansi_font, size_value)
            if any(value is not None for value in explicit_values) and explicit_values != (
                "楷体", "Times New Roman", "Times New Roman", "28"
            ):
                font_errors.append(
                    repr((text[:20], east_asia, ascii_font, hansi_font, size_value)).encode("unicode_escape").decode("ascii")
                )
            if is_heading:
                bold = None if rpr is None else rpr.find(qn("w:b"))
                if bold is not None and bold.get(qn("w:val"), "1") in {"0", "false", "False"}:
                    heading_errors.append(_paragraph_text(child).strip())

        if child.tag != qn("w:tbl"):
            continue
        table_count += 1
        for row_index, row in enumerate(child.findall(qn("w:tr"))):
            for cell in row.findall(qn("w:tc")):
                tcpr = cell.find(qn("w:tcPr"))
                shading = None if tcpr is None else tcpr.find(qn("w:shd"))
                fill = None if shading is None else shading.get(qn("w:fill"))
                if row_index == 0:
                    if fill != "D9D9D9":
                        header_fill_errors.append(str(fill))
                    for run in cell.iter(qn("w:r")):
                        if not _run_text(run).strip():
                            continue
                        rpr = run.find(qn("w:rPr"))
                        bold = None if rpr is None else rpr.find(qn("w:b"))
                        if bold is None or bold.get(qn("w:val"), "1") in {"0", "false", "False"}:
                            header_bold_errors.append(_run_text(run)[:30])
                elif fill not in {None, "auto"}:
                    body_fill_errors.append(str(fill))

    body_text = "".join(body_text_parts)
    all_text = "\n".join(paragraph.text for paragraph in doc.paragraphs) + body_text
    for style_name in ("Normal", "Body Text", "List Bullet", "List Number", "List Paragraph"):
        if style_name not in doc.styles:
            continue
        style = doc.styles[style_name]
        east_asia = _effective_style_value(style, "eastAsia")
        font_name = _effective_style_value(style, "name")
        font_size = _effective_style_value(style, "size")
        if font_name != "Times New Roman" or east_asia != "楷体" or font_size.pt != 14:
            style_errors.append(style_name)
    for style_name in ("Heading 1", "Heading 2", "Heading 3", "Heading 4"):
        style = doc.styles[style_name]
        east_asia = _effective_style_value(style, "eastAsia")
        font_name = _effective_style_value(style, "name")
        font_size = _effective_style_value(style, "size")
        bold = _effective_style_value(style, "bold")
        if (
            font_name != "Times New Roman"
            or east_asia != "楷体"
            or font_size.pt != 14
            or bold is not True
        ):
            style_errors.append(style_name)
    pages = len(PdfReader(str(PDF)).pages)
    toc_fields = [
        node.text or ""
        for node in doc._element.xpath(".//w:instrText")
        if "TOC" in (node.text or "").upper()
    ]
    required_terms = (
        "CoopTrackerModel",
        "d_model",
        "target_valid_mask",
        "Learned-K",
        "GAT",
        "V8",
        "40 Hz",
        "10 Hz",
        "IMM-JPDA",
        "AssociationStats",
        "strict加载",
    )
    stale_patterns = (r"版本记录", r"修订记录", r"更新时间", r"V1\.0", r"共XX页")
    checks = {
        "pages": pages,
        "cover_page_count": f"共{pages}页" in all_text,
        "toc_present": bool(toc_fields),
        "toc_levels_1_to_2": any('\\o "1-2"' in field for field in toc_fields),
        "font_errors": len(font_errors),
        "heading_bold_errors": len(heading_errors),
        "style_errors": style_errors,
        "tables": table_count,
        "table_header_fill_errors": len(header_fill_errors),
        "table_header_bold_errors": len(header_bold_errors),
        "table_body_fill_errors": len(body_fill_errors),
        "required_terms_present": all(term in body_text for term in required_terms),
        "stale_version_hits": [pattern for pattern in stale_patterns if re.search(pattern, all_text, re.I)],
    }
    for key, value in checks.items():
        print(f"{key}={value}")
    if font_errors:
        print("font_error_sample=" + repr(font_errors[:10]))
    if heading_errors:
        print("heading_error_sample=" + repr(heading_errors[:10]))

    failed = (
        not 25 <= pages <= 55
        or not checks["cover_page_count"]
        or not checks["toc_present"]
        or not checks["toc_levels_1_to_2"]
        or font_errors
        or heading_errors
        or style_errors
        or table_count < 20
        or header_fill_errors
        or header_bold_errors
        or body_fill_errors
        or not checks["required_terms_present"]
        or bool(checks["stale_version_hits"])
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
