from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "software_delivery" / "端到端协同跟踪原型软件_设计说明书.docx"
PDF = ROOT / ".codex_work" / "final_design_toc_render" / "design_spec_final.pdf"


def _attr(element, tag: str, attr: str = "val") -> str | None:
    child = element.find(qn(tag))
    return None if child is None else child.get(qn(f"w:{attr}"))


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
    toc_fields = []
    for node in doc._element.iter(qn("w:instrText")):
        if "TOC" in (node.text or ""):
            toc_fields.append(node.text or "")

    first_heading_index = None
    first_heading_text = None
    for index, child in enumerate(body):
        if child.tag != qn("w:p"):
            continue
        ppr = child.find(qn("w:pPr"))
        pstyle = None if ppr is None else _attr(ppr, "w:pStyle")
        text = "".join(node.text or "" for node in child.iter(qn("w:t"))).strip()
        if pstyle in {"1", "Heading1"} and re.match(r"^1\s+前言", text):
            first_heading_index = index
            first_heading_text = text
            break

    if first_heading_index is None:
        raise SystemExit("未找到正文首个一级标题。")

    font_errors = []
    heading_errors = []
    body_text_parts = []
    table_count = 0
    header_fill_errors = []
    header_bold_errors = []
    body_fill_errors = []

    for child_index, child in enumerate(body[first_heading_index:], start=first_heading_index):
        for run in child.iter(qn("w:r")):
            text = "".join(node.text or "" for node in run.iter(qn("w:t")))
            if not text.strip():
                continue
            rpr = run.find(qn("w:rPr"))
            if rpr is not None and rpr.find(qn("w:vanish")) is not None:
                continue
            body_text_parts.append(text)
            fonts = None if rpr is None else rpr.find(qn("w:rFonts"))
            size = None if rpr is None else _attr(rpr, "w:sz")
            east_asia = None if fonts is None else fonts.get(qn("w:eastAsia"))
            ascii_font = None if fonts is None else fonts.get(qn("w:ascii"))
            hansi_font = None if fonts is None else fonts.get(qn("w:hAnsi"))
            explicit_values = (east_asia, ascii_font, hansi_font, size)
            expected_values = ("楷体", "Times New Roman", "Times New Roman", "28")
            if any(
                value is not None and value != expected
                for value, expected in zip(explicit_values, expected_values)
            ):
                font_errors.append(
                    f"child={child_index}: sz={size}, eastAsia={east_asia}, ascii={ascii_font}, hAnsi={hansi_font}: {text[:30]}"
                )

        if child.tag == qn("w:p"):
            ppr = child.find(qn("w:pPr"))
            pstyle = None if ppr is None else _attr(ppr, "w:pStyle")
            if pstyle in {"1", "2", "3", "21", "31", "Heading1", "Heading2", "Heading3"}:
                heading_text = "".join(node.text or "" for node in child.iter(qn("w:t")))
                for run in child.iter(qn("w:r")):
                    text = "".join(node.text or "" for node in run.iter(qn("w:t")))
                    if not text.strip():
                        continue
                    rpr = run.find(qn("w:rPr"))
                    bold = None if rpr is None else rpr.find(qn("w:b"))
                    if bold is not None and bold.get(qn("w:val"), "1") in {"0", "false", "False"}:
                        heading_errors.append(heading_text)
                        break

        if child.tag != qn("w:tbl"):
            continue
        table_count += 1
        rows = child.findall(qn("w:tr"))
        for row_index, row in enumerate(rows):
            for cell_index, cell in enumerate(row.findall(qn("w:tc"))):
                tcpr = cell.find(qn("w:tcPr"))
                shading = None if tcpr is None else tcpr.find(qn("w:shd"))
                fill = None if shading is None else shading.get(qn("w:fill"))
                location = f"table={table_count},row={row_index},cell={cell_index}"
                if row_index == 0:
                    if fill != "D9D9D9":
                        header_fill_errors.append(f"{location},fill={fill}")
                    for run in cell.iter(qn("w:r")):
                        text = "".join(node.text or "" for node in run.iter(qn("w:t")))
                        if not text.strip():
                            continue
                        rpr = run.find(qn("w:rPr"))
                        bold = None if rpr is None else rpr.find(qn("w:b"))
                        if bold is None or bold.get(qn("w:val"), "1") in {"0", "false", "False"}:
                            header_bold_errors.append(location)
                            break
                elif fill not in {None, "auto"}:
                    body_fill_errors.append(f"{location},fill={fill}")

    body_text = "".join(body_text_parts)
    style_errors = []
    for style_name in ("Normal", "Body Text", "List Bullet", "List Number", "List Paragraph"):
        if style_name not in doc.styles:
            continue
        style = doc.styles[style_name]
        size = _effective_style_value(style, "size")
        if (
            _effective_style_value(style, "name") != "Times New Roman"
            or _effective_style_value(style, "eastAsia") != "楷体"
            or size is None
            or size.pt != 14
        ):
            style_errors.append(style_name)
    for style_name in ("Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[style_name]
        size = _effective_style_value(style, "size")
        if (
            _effective_style_value(style, "name") != "Times New Roman"
            or _effective_style_value(style, "eastAsia") != "楷体"
            or size is None
            or size.pt != 14
            or _effective_style_value(style, "bold") is not True
        ):
            style_errors.append(style_name)
    stale_hits = [
        pattern
        for pattern in (r"版本记录", r"修订记录", r"更新时间", r"V1\.0")
        if re.search(pattern, body_text, re.I)
    ]
    page_count = len(PdfReader(str(PDF)).pages)
    cover_text = "\n".join(p.text for p in doc.paragraphs[:20])
    checks = {
        "pages": page_count,
        "cover_page_count": f"共{page_count}页" in cover_text,
        "toc_present": bool(toc_fields),
        "toc_levels_1_to_2": any('\\o "1-2"' in field for field in toc_fields),
        "first_body_heading": first_heading_text,
        "font_errors": len(font_errors),
        "heading_bold_errors": len(heading_errors),
        "style_errors": style_errors,
        "tables": table_count,
        "table_header_fill_errors": len(header_fill_errors),
        "table_header_bold_errors": len(header_bold_errors),
        "table_body_fill_errors": len(body_fill_errors),
        "required_model_terms": all(
            term in body_text
            for term in (
                "CoopTrackerModel",
                "LearnedKFrontEnd",
                "CrossAttentionBlock",
                "TemporalTransformer",
                "CollaborativeGAT",
                "target_valid_mask",
                "40 Hz",
                "10 Hz",
            )
        ),
        "stale_version_hits": stale_hits,
    }
    for key, value in checks.items():
        print(f"{key}={value}")
    if font_errors:
        print("FONT_ERROR_SAMPLE")
        print("\n".join(font_errors[:10]))
    if heading_errors:
        print("HEADING_ERROR_SAMPLE")
        print("\n".join(heading_errors[:10]))
    if header_fill_errors or header_bold_errors or body_fill_errors:
        print("TABLE_ERROR_SAMPLE")
        print("\n".join((header_fill_errors + header_bold_errors + body_fill_errors)[:10]))

    failed = (
        not checks["cover_page_count"]
        or not checks["toc_present"]
        or not checks["toc_levels_1_to_2"]
        or first_heading_text != "1 前言"
        or font_errors
        or heading_errors
        or style_errors
        or table_count < 30
        or header_fill_errors
        or header_bold_errors
        or body_fill_errors
        or stale_hits
        or not 75 <= page_count <= 82
        or not checks["required_model_terms"]
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
