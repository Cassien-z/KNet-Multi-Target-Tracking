from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


root = Path("docs/software_delivery")
outline_path = root / "端到端协同跟踪算法软件_测试大纲.docx"
report_path = root / "端到端协同跟踪算法软件_测试报告.docx"
outline = Document(outline_path)
report = Document(report_path)

assert len(outline.tables) == 16
assert len(report.tables) == 16
outline_table = outline.tables[11]
report_table = report.tables[11]

for row in (1, 2, 3, 4):
    assert outline_table.cell(row, 1).text == report_table.cell(row, 1).text
    assert "Python" in outline_table.cell(row, 1).text or row == 4

assert report_table.cell(5, 0).text.strip() == "测试结果"
result_cell = report_table.cell(5, 1)
result_text = result_cell.text
assert "46项检查，46项通过" in result_text
assert "本用例通过" in result_text
assert "System Error: File system inconsistency" in result_text
assert len(result_cell._tc.xpath(".//a:blip")) == 4

text_runs = [
    run
    for paragraph in result_cell.paragraphs
    for run in paragraph.runs
    if run.text.strip()
]
font_issues = []
for run in text_runs:
    east_asia = (
        run._element.rPr.rFonts.get(qn("w:eastAsia"))
        if run._element.rPr is not None and run._element.rPr.rFonts is not None
        else None
    )
    size = run.font.size.pt if run.font.size is not None else None
    if east_asia != "楷体" or size is None or abs(size - 14) > 0.01:
        font_issues.append((run.text[:30], east_asia, size))

assert not font_issues, font_issues
assert len(report.inline_shapes) == 38
assert len(report_table.rows[5]._tr.tc_lst) == 2
spans = [
    tc.tcPr.gridSpan.val if tc.tcPr is not None and tc.tcPr.gridSpan is not None else 1
    for tc in report_table.rows[5]._tr.tc_lst
]
assert spans == [1, 3], spans

print({
    "outline_tables": len(outline.tables),
    "report_tables": len(report.tables),
    "report_inline_shapes": len(report.inline_shapes),
    "result_figures": len(result_cell._tc.xpath(".//a:blip")),
    "result_text_runs": len(text_runs),
    "font_issues": len(font_issues),
    "result_row_spans": spans,
    "outline_bytes": outline_path.stat().st_size,
    "report_bytes": report_path.stat().st_size,
})
