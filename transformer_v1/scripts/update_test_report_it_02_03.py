from __future__ import annotations

import json
import shutil
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "software_delivery" / "端到端协同跟踪算法软件_测试报告.docx"
RUN_DIR = ROOT / "artifacts" / "reports" / "software_test" / "IT-02-03" / "20260923_run01"
FIG_DIR = RUN_DIR / "figures"
RESULT = json.loads((RUN_DIR / "result.json").read_text(encoding="utf-8"))
BACKUP = RUN_DIR / "端到端协同跟踪算法软件_测试报告_写入IT-02-03前备份.docx"


def set_run_font(run, bold: bool | None = None, size: float = 14.0) -> None:
    run.font.name = "楷体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "楷体")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def add_paragraph(cell, text: str = "", *, bold: bool = False, center: bool = False, after: float = 5.0):
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.15
    if text:
        set_run_font(p.add_run(text), bold=bold)
    return p


def add_compact_table(cell, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = cell.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    header_pr = table.rows[0]._tr.get_or_add_trPr()
    header_repeat = OxmlElement("w:tblHeader")
    header_repeat.set(qn("w:val"), "true")
    header_pr.append(header_repeat)
    for col, text in enumerate(headers):
        target = table.rows[0].cells[col]
        target.text = text
        set_cell_shading(target, "D9EAF7")
        target.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        if widths:
            target.width = Cm(widths[col])
    for row in rows:
        cells = table.add_row().cells
        for col, text in enumerate(row):
            cells[col].text = text
            cells[col].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths:
                cells[col].width = Cm(widths[col])
    for row_index, row in enumerate(table.rows):
        row_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        row_pr.append(cant_split)
        for target in row.cells:
            for p in target.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                for run in p.runs:
                    set_run_font(run, bold=(row_index == 0), size=14.0)
    add_paragraph(cell, "", after=2)


def add_figure(cell, filename: str, caption: str) -> None:
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    # The report template reserves the rightmost column for signatures/status.
    # Keep evidence figures inside the merged result area to avoid clipping.
    p.add_run().add_picture(str(FIG_DIR / filename), width=Cm(10.2))
    cp = add_paragraph(cell, caption, center=True, after=7)
    cp.paragraph_format.keep_with_next = False


doc = Document(REPORT)
target_table = doc.tables[9]
cell = target_table.cell(5, 0)
cell._element.clear_content()
cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP

add_paragraph(cell, "测试结果", bold=True, center=True, after=8)
add_paragraph(
    cell,
    "2026年9月23日使用Python 3.11.5、PyTorch 2.1.1在CPU环境执行IT-02-03。加载冻结V7模型best_model.pt，"
    "SHA-256为d22427274a8b680c477f2e7074d91efd03a89a5347f33862a6c6de7ce35dc51a；"
    "预热帧固定为前5帧，误差仅统计target_valid_mask为真的其余帧。共执行44项检查，44项通过。",
)

add_paragraph(cell, "1. 网络结构、序列接口与代表场景", bold=True)
add_paragraph(
    cell,
    "代码和运行时钩子均确认时序编码器为2层、每层4个注意力头、batch_first=True。CV、CA、CT、CSSW及V8复合干扰补充样例均按100帧、5平台输入，"
    "时序模块输入和输出长度均为100，最终状态形状均为[1,100,6]，全部数值有限。四种use_temporal/use_gat开关组合也均保持[1,100,6]有限输出。",
)

scenario_rows = []
for row in RESULT["scenario_metrics"]:
    scenario_rows.append([
        row["scenario"], str(row["valid_frames"]), f"{row['position_rmse_m']:.3f}",
        f"{row['position_p95_m']:.3f}", f"{row['velocity_rmse_mps']:.3f}",
    ])
add_compact_table(
    cell,
    ["场景", "有效帧", "位置RMSE/m", "位置P95/m", "速度RMSE/(m/s)"],
    scenario_rows[:2],
    [3.3, 1.7, 2.6, 2.5, 3.4],
)
add_compact_table(
    cell,
    ["场景", "有效帧", "位置RMSE/m", "位置P95/m", "速度RMSE/(m/s)"],
    scenario_rows[2:],
    [3.3, 1.7, 2.6, 2.5, 3.4],
)
add_paragraph(
    cell,
    "V7四类代表样例的位置RMSE为5.623～11.531 m。V8复合干扰样例的位置RMSE为185.102 m、速度RMSE为111.829 m，"
    "只作为强干扰边界补充观察，不作为本用例精度达标证据；合同精度指标按测试用例IT-02-06统一评价。",
)
add_figure(cell, "figure_1_scenario_metrics.png", "图4-8-1  代表场景位置与速度误差")
add_figure(cell, "figure_2_framewise_position_error.png", "图4-8-2  代表场景逐帧位置误差")

add_paragraph(cell, "2. 短时缺失、状态保持与恢复记录", bold=True)
add_paragraph(
    cell,
    "在CA样例中分别设置单帧、连续5帧和连续10帧全源缺失。缺失窗口内雷达、光电和电子侦察掩码均为0，快慢链路同步置为无更新，"
    "三组输出均为有限值；Learned-K观测位置在缺失窗口内保持，最大绝对变化均为0 m。",
)
missing_rows = []
for row in RESULT["missing_recovery"]:
    recovery = row["diagnostic_recovery_to_1m_from_baseline_frame"]
    missing_rows.append([
        row["condition"], f"{row['gap_position_rmse_m']:.3f}", f"{row['first_post_gap_error_m']:.3f}",
        f"{row['post_10_frame_rmse_m']:.3f}", str(recovery) if recovery is not None else "本段内未达到",
    ])
add_compact_table(
    cell,
    ["缺失条件", "窗口RMSE/m", "恢复首帧误差/m", "后10帧RMSE/m", "与基线差≤1m帧"],
    missing_rows[:2],
    [3.0, 2.4, 3.1, 2.7, 3.2],
)
add_compact_table(
    cell,
    ["缺失条件", "窗口RMSE/m", "恢复首帧误差/m", "后10帧RMSE/m", "与基线差≤1m帧"],
    missing_rows[2:],
    [3.0, 2.4, 3.1, 2.7, 3.2],
)
add_paragraph(
    cell,
    "“与基线差≤1 m帧”只用于描述恢复过程，不是合同判据。单帧缺失在第78帧回到与基线位置输出差不超过1 m；连续5帧和10帧缺失在该100帧片段结束前未达到该诊断条件。"
    "本用例未擅自增加恢复时限或恢复误差阈值。",
)
add_figure(cell, "figure_3_missing_and_recovery.png", "图4-8-3  缺失窗口与恢复过程")

add_paragraph(cell, "3. 完整窗口上下文、填充边界和运行稳定性", bold=True)
add_paragraph(
    cell,
    "模型源码未传入causal mask或padding attention mask。仅扰动第60～99帧观测后，第0～59帧输出最大绝对变化为46.342453；"
    "同一60帧直接推理与尾部补零至100帧推理相比，前60帧最大绝对变化为16.000000。结果确认模型使用完整窗口上下文，并说明简单补零不能视为不影响预测的有效填充。"
    "60帧Python研究路径可输出[1,60,6]有限状态，但该结果不扩展为固定长度部署接口的可变长承诺。",
)
add_paragraph(
    cell,
    "同一代表样例重复推理3次的最大绝对差为0，CV/CA组成二样例批次后交换批次顺序的对应输出最大绝对差为0。所有样例dt均为有限正数；全无效样例的有效指标帧数为0，不产生误差统计。",
)
add_figure(cell, "figure_4_temporal_scope_and_repeatability.png", "图4-8-4  时序结构、完整窗口作用与重复性")

add_paragraph(cell, "4. 判定", bold=True)
add_paragraph(
    cell,
    "本用例通过。2层4头结构、100帧时序保持、六维状态输出、标准缺失处理、有效帧与5帧预热统计、模块开关接口以及重复运行稳定性均符合测试大纲。"
    "当前模型按完整片段进行非因果推理，后帧会影响前帧，且没有填充注意力掩码；因此本结果不证明在线输出与未来帧独立，也不支持把任意补零序列直接作为正式输入。",
)

for paragraph in cell.paragraphs:
    for run in paragraph.runs:
        if run._element.xpath(".//a:blip"):
            continue
        set_run_font(run)

if not BACKUP.exists():
    shutil.copy2(REPORT, BACKUP)
doc.save(REPORT)
print(REPORT)
