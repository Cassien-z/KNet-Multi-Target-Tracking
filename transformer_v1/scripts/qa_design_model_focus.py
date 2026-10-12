from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from pypdf import PdfReader

from rewrite_design_model_focus import _front_boundary, _prefix_digest


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "software_delivery" / "端到端协同跟踪原型软件_设计说明书.docx"
PDF = ROOT / ".codex_work" / "final80_render" / "final80.pdf"
EXPECTED_FRONT_SHA256 = "e7638f8dc8262b49ca003f1aebce87df4670cad86782fdcb9ec6d4eed52760b5"


def _attr(element, tag: str, attr: str = "val") -> str | None:
    child = element.find(qn(tag))
    return None if child is None else child.get(qn(f"w:{attr}"))


def main() -> None:
    doc = Document(DOCX)
    boundary = _front_boundary(doc)
    front_equal = _prefix_digest(doc, boundary) == EXPECTED_FRONT_SHA256

    body_children = list(doc._element.body)[boundary:]
    font_errors: list[str] = []
    heading_errors: list[str] = []
    body_text_parts: list[str] = []
    nonempty_run_count = 0
    heading3_count = 0
    table_count = 0
    table_header_fill_errors: list[str] = []
    table_header_bold_errors: list[str] = []
    table_body_fill_errors: list[str] = []
    section_text: dict[str, list[str]] = {"data_flow": [], "control_flow": []}
    active_section: str | None = None

    for child_index, child in enumerate(body_children, start=boundary):
        for run in child.iter(qn("w:r")):
            text = "".join(node.text or "" for node in run.iter(qn("w:t")))
            if not text.strip():
                continue
            rpr = run.find(qn("w:rPr"))
            if rpr is not None and rpr.find(qn("w:vanish")) is not None:
                continue
            nonempty_run_count += 1
            body_text_parts.append(text)
            if rpr is None:
                font_errors.append(f"child={child_index}: missing rPr: {text[:30]}")
                continue
            fonts = rpr.find(qn("w:rFonts"))
            size = _attr(rpr, "w:sz")
            east_asia = None if fonts is None else fonts.get(qn("w:eastAsia"))
            ascii_font = None if fonts is None else fonts.get(qn("w:ascii"))
            hansi_font = None if fonts is None else fonts.get(qn("w:hAnsi"))
            if size != "28" or east_asia != "楷体" or ascii_font != "Times New Roman" or hansi_font != "Times New Roman":
                font_errors.append(
                    f"child={child_index}: sz={size}, eastAsia={east_asia}, ascii={ascii_font}, hAnsi={hansi_font}: {text[:30]}"
                )

        if child.tag != qn("w:p"):
            continue
        ppr = child.find(qn("w:pPr"))
        pstyle = None if ppr is None else _attr(ppr, "w:pStyle")
        paragraph_text = "".join(node.text or "" for node in child.iter(qn("w:t")))
        if pstyle in {"2", "21", "Heading2"}:
            if re.match(r"^2\.5\s+数据流", paragraph_text):
                active_section = "data_flow"
            elif re.match(r"^2\.6\s+控制流", paragraph_text):
                active_section = "control_flow"
            elif active_section is not None:
                active_section = None
        elif active_section is not None and paragraph_text.strip():
            section_text[active_section].append(paragraph_text)
        if pstyle not in {"1", "2", "3", "21", "31", "Heading1", "Heading2", "Heading3"}:
            continue
        if pstyle in {"3", "31", "Heading3"}:
            heading3_count += 1
        heading_text = "".join(node.text or "" for node in child.iter(qn("w:t")))
        for run in child.iter(qn("w:r")):
            text = "".join(node.text or "" for node in run.iter(qn("w:t")))
            if not text.strip():
                continue
            rpr = run.find(qn("w:rPr"))
            bold = None if rpr is None else rpr.find(qn("w:b"))
            if bold is None or bold.get(qn("w:val"), "1") in {"0", "false", "False"}:
                heading_errors.append(heading_text)
                break

    for child_index, child in enumerate(body_children, start=boundary):
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
                        table_header_fill_errors.append(f"{location},fill={fill}")
                    for run in cell.iter(qn("w:r")):
                        text = "".join(node.text or "" for node in run.iter(qn("w:t")))
                        if not text.strip():
                            continue
                        rpr = run.find(qn("w:rPr"))
                        bold = None if rpr is None else rpr.find(qn("w:b"))
                        if bold is None or bold.get(qn("w:val"), "1") in {"0", "false", "False"}:
                            table_header_bold_errors.append(location)
                            break
                elif fill not in {None, "auto"}:
                    table_body_fill_errors.append(f"{location},fill={fill}")

    body_text = "".join(body_text_parts)
    stale_patterns = [r"版本记录", r"修订记录", r"更新时间", r"V1\.0"]
    stale_hits = [pattern for pattern in stale_patterns if re.search(pattern, body_text, re.I)]

    hyperlinks = doc._element.xpath("//w:hyperlink[@w:anchor]")
    anchors = [h.get(qn("w:anchor")) for h in hyperlinks]
    bookmarks = {
        b.get(qn("w:name")) for b in doc._element.xpath("//w:bookmarkStart[@w:name]")
    }
    missing_anchors = sorted({anchor for anchor in anchors if anchor not in bookmarks})

    page_count = len(PdfReader(str(PDF)).pages)
    checks = {
        "front_boundary": boundary,
        "front_unchanged": front_equal,
        "body_nonempty_runs": nonempty_run_count,
        "font_errors": len(font_errors),
        "heading_bold_errors": len(heading_errors),
        "heading3_count": heading3_count,
        "data_flow_characters": len("".join(section_text["data_flow"])),
        "control_flow_characters": len("".join(section_text["control_flow"])),
        "table_count": table_count,
        "table_header_fill_errors": len(table_header_fill_errors),
        "table_header_bold_errors": len(table_header_bold_errors),
        "table_body_fill_errors": len(table_body_fill_errors),
        "inline_images": len(doc.inline_shapes),
        "body_characters": len(body_text),
        "toc_hyperlink_anchors": len(anchors),
        "toc_bookmarks": len(bookmarks),
        "missing_toc_anchors": len(missing_anchors),
        "toc_update_deferred": True,
        "stale_version_hits": stale_hits,
        "rendered_pages": page_count,
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
    }
    for key, value in checks.items():
        print(f"{key}={value}")
    if font_errors:
        print("FONT_ERROR_SAMPLE")
        print("\n".join(font_errors[:10]))
    if heading_errors:
        print("HEADING_ERROR_SAMPLE")
        print("\n".join(heading_errors[:10]))
    if table_header_fill_errors or table_header_bold_errors or table_body_fill_errors:
        print("TABLE_ERROR_SAMPLE")
        print("\n".join((table_header_fill_errors + table_header_bold_errors + table_body_fill_errors)[:10]))

    failed = (
        not front_equal
        or font_errors
        or heading_errors
        or not 25 <= heading3_count <= 50
        or checks["data_flow_characters"] < 1000
        or checks["control_flow_characters"] < 1000
        or table_count < 15
        or table_header_fill_errors
        or table_header_bold_errors
        or table_body_fill_errors
        or len(doc.inline_shapes) != 1
        or len(body_text) < 45000
        or stale_hits
        or not 79 <= page_count <= 81
        or not checks["required_model_terms"]
        or not all(
            phrase in body_text
            for phrase in (
                "训练与部署掩码差异",
                "首帧不存在上一状态",
                "三类权重的区别",
                "有效标签前提",
            )
        )
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
