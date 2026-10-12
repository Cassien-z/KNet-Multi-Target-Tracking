from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "software_delivery" / "端到端协同跟踪原型软件_设计说明书.docx"


def main() -> None:
    doc = Document(DOCX)
    body = doc._element.body
    children = list(body)
    toc_index = None

    for index, child in enumerate(children):
        fields = list(child.iter(qn("w:instrText")))
        if any("TOC" in (field.text or "").upper() for field in fields):
            toc_index = index
            for field in fields:
                if "TOC" in (field.text or "").upper():
                    field.text = ' TOC \\o "1-2" \\h \\z \\u '
            break

    if toc_index is None:
        raise RuntimeError("未找到设计说明书目录域")

    # 需求规格说明书让正文自然接续在目录之后。设计说明书原有的额外分页符
    # 会在目录末尾产生一张空白页，因此只移除紧随目录的空分页段落。
    children = list(body)
    if toc_index + 1 < len(children):
        candidate = children[toc_index + 1]
        text = "".join(node.text or "" for node in candidate.iter(qn("w:t"))).strip()
        page_breaks = [
            node for node in candidate.iter(qn("w:br"))
            if node.get(qn("w:type")) == "page"
        ]
        if candidate.tag == qn("w:p") and not text and page_breaks:
            body.remove(candidate)

    doc.save(DOCX)
    print(DOCX)


if __name__ == "__main__":
    main()
