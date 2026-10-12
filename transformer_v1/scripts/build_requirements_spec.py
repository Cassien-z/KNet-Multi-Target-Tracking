"""Build the model-focused software requirements specification."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_software_docs import (
    add_numbered_steps,
    add_table,
    prevent_row_split,
    set_repeat_table_header,
    set_table_borders,
)

TEMPLATE = ROOT / "docs" / "software_delivery" / "资料模板" / "OODA-3-需求规格说明.docx"
OUTPUT = ROOT / "docs" / "software_delivery" / "端到端协同跟踪原型软件_需求规格说明书.docx"
WORK_DIR = ROOT / ".codex_work" / "requirements_spec"

EAST_ASIA_FONT = "楷体"
LATIN_FONT = "Times New Roman"
GRAY = "D9D9D9"


def _set_run_font(run, *, bold: bool | None = None) -> None:
    run.font.name = LATIN_FONT
    fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn("w:eastAsia"), EAST_ASIA_FONT)
    fonts.set(qn("w:ascii"), LATIN_FONT)
    fonts.set(qn("w:hAnsi"), LATIN_FONT)
    fonts.set(qn("w:cs"), LATIN_FONT)
    run.font.size = Pt(14)
    if bold is not None:
        run.bold = bold


def _remove_body_from(doc: Document, heading_text: str) -> int:
    body = doc._element.body
    boundary = None
    for index, child in enumerate(body):
        if child.tag != qn("w:p"):
            continue
        paragraph = Paragraph(child, doc._body)
        if paragraph.text.strip() == heading_text:
            boundary = index
            break
    if boundary is None:
        raise RuntimeError(f"Cannot locate template heading: {heading_text}")
    for child in list(body)[boundary:]:
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    return boundary


def _replace_cover_text(doc: Document) -> None:
    replacements = {
        "面向OODA闭环的“协同感知-决策”误差分析与联合优化项目": "基于多源异构数据融合的端对端适变跟踪方法研究",
        "需 求 规 格 说 明": "需 求 规 格 说 明 书",
        "共26页": "共XX页",
        "承担单位：上海交通大学": "承担单位：华东理工大学",
    }
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text not in replacements:
            continue
        replacement = replacements[text]
        if paragraph.runs:
            paragraph.runs[0].text = replacement
            for run in paragraph.runs[1:]:
                run.text = ""
        else:
            paragraph.add_run(replacement)


def _set_toc_levels(doc: Document) -> None:
    """Keep the template TOC compact enough for a short requirements document."""
    for node in doc._element.xpath(".//w:instrText"):
        if node.text and "TOC" in node.text.upper():
            node.text = ' TOC \\o "1-2" \\h \\z \\u '


def _add_toc_field(doc: Document) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("目录")
    _set_run_font(run, bold=True)
    title.paragraph_format.space_after = Pt(12)

    paragraph = doc.add_paragraph()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "目录将在打开文档时更新"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run = paragraph.add_run()
    run._r.extend([begin, instr, separate, text, end])
    doc.add_page_break()


def _lead(doc: Document, lead: str, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.add_run(f"{lead}：").bold = True
    paragraph.add_run(text)


def add_bullets(doc: Document, items: list[str]) -> None:
    """Add portable bullet paragraphs without relying on template style names."""
    for item in items:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.28)
        paragraph.paragraph_format.first_line_indent = Inches(-0.22)
        paragraph.add_run("● ")
        paragraph.add_run(item)


def _requirement_table(doc: Document, rows: list[list[str]]) -> None:
    compact_rows = [[row[0], f"{row[1]}：{row[2]}", row[3]] for row in rows]
    add_table(
        doc,
        ["需求标识", "需求规定", "优先级"],
        compact_rows,
        widths=[1.35, 4.9, 0.6],
        center_cols=(0, 2),
        small=False,
    )


def _build_body(doc: Document) -> None:
    doc.add_heading("1 范围", level=1)
    doc.add_heading("1.1 标识", level=2)
    doc.add_paragraph("a) 文档标识号：TS-DOC-SRS-001。")
    doc.add_paragraph("b) 文档名称：端到端协同跟踪原型软件需求规格说明书。")
    doc.add_paragraph("c) 适用软件：端到端协同跟踪原型软件。")
    doc.add_paragraph("d) 软件形态：用于模型部署、序列推理、批量验证、性能评估和报告归档的本地原型验证软件。")

    doc.add_heading("1.2 软件概述", level=2)
    doc.add_paragraph(
        "本软件以端到端协同跟踪模型为核心。模型接收已完成目标注册和时间对齐的多无人机雷达、光电、电子侦察、平台状态与协同链路张量，在固定窗口内输出单目标六维状态。软件提供权重加载、场景解析、完整序列推理、批测评估、受控消融、结果回放和报告归档。前后端只负责受控调用与展示，不承担原始点迹检测、在线目标关联、飞行控制或任务规划。"
    )
    doc.add_paragraph(
        "输入场景必须按目标注册并映射到统一时间轴。当前权重使用100帧、约10 Hz序列，且注意力结构为固定窗口离线推理。V8保留400帧、40 Hz源数据；在不重新训练条件下，软件以同一组索引同步生成100帧兼容视图，不得据此宣称模型具备40 Hz在线推理能力。"
    )

    doc.add_heading("1.3 文档概述", level=2)
    doc.add_paragraph(
        "本文档规定软件必须具备的状态、能力、功能、接口、数据、环境、质量和验收要求。第三章给出可实现和可验证的软件需求；第四章规定测试、检查、分析和演示方法；第五章建立需求与设计实现及验证活动之间的追踪关系；第六章统一术语。需求使用唯一标识管理，后续设计、测试用例和交付报告应引用同一标识，避免只依据章节名称描述覆盖情况。"
    )

    doc.add_heading("2 引用文档", level=1)
    doc.add_paragraph("本说明书依据合同任务要求、现有软件源码、设计说明书和可执行测试形成。甲方文件编号与发布日期应在正式交付前按最终合同资料统一补录，但不得改变本文需求标识及其技术含义。")
    add_table(
        doc,
        ["序号", "文档或资料", "用途", "来源"],
        [
            ["1", "项目合同书及任务要求", "确定交付范围和技术指标", "项目交付资料"],
            ["2", "端到端协同跟踪原型软件设计说明书", "确定模型结构、数据口径和运行边界", "docs/software_delivery"],
            ["3", "模型与数据处理源码", "核对张量、模型结构、训练约束和V8兼容处理", "src、train.py、scripts/data_generation"],
            ["4", "后端运行与批测源码", "核对模型加载、场景解析、批量任务、指标和报告", "tracking_system/backend"],
            ["5", "前端源码", "核对页面、操作流程、状态显示和下载入口", "tracking_system/frontend"],
            ["6", "自动化测试", "核对安全求解、V8兼容、评估与接口行为", "tests"],
            ["7", "跨语言接口与交付说明", "核对ONNX固定输入、MATLAB/C入口和黄金样例", "portable/README.md、INTERFACE.md"],
        ],
        widths=[0.55, 2.3, 2.15, 1.85],
        center_cols=(0,),
    )
    doc.add_paragraph("合同或任务书的强制指标高于界面示例值和历史结果。模型输入输出含义以可执行源码和有效权重结构为准；正式性能结论必须来自完整批测和独立证据，单场景回放仅用于链路检查。未取得的甲方文件编号、单位信息和签署日期应留待正式资料补录。")

    doc.add_heading("3 需求", level=1)
    doc.add_heading("3.1 要求的状态和方式", level=2)
    doc.add_paragraph(
        "软件具有未就绪、就绪、单场景推理、批测和结果可评估五种运行状态。服务或目录检查失败时仅提供诊断；候选权重严格加载后进入就绪状态。单场景推理在场景和目标有效时执行一次完整序列前向；批测任务经校验后依次经历排队、运行以及完成、失败或取消；已完成任务进入结果评估和制品导出。"
    )
    doc.add_paragraph(
        "系统不提供训练状态，训练与权重搜索不由交付前端发起。模型切换采用候选引擎先加载、校验成功后再替换活动引擎的方式，失败时保留旧模型。批测运行中允许查询进度和请求取消，服务重启后不得把中断任务标记为正常完成。"
    )

    doc.add_heading("3.2 软件能力需求", level=2)
    doc.add_paragraph(
        "软件能力由模型资产管理、注册数据解析、端到端序列推理、V8兼容验证、单场景回放、批量测试、技术指标判定和报告追溯组成。资产管理保证权重与设备可用；数据解析把场景按目标转换为统一张量；模型推理生成B×T×6状态；验证支撑层负责形成航迹、误差、时延、独立证据及JSON和CSV制品。八项能力共同构成从受控输入到可复核结论的验证链路。"
    )

    doc.add_paragraph(
        "模型推理必须使用与权重一致的d_model结构。一次调用接收100帧、5个平台槽位的多源张量并输出逐时刻六维状态。物理支路使用雷达和快慢链路形成粗状态，深度支路学习多模态、时序与跨平台关系，最终输出为深度残差与Learned-K状态之和。支撑软件不得绕过模型拼接轨迹，也不得把回放刷新率作为模型推理频率。"
    )
    doc.add_paragraph(
        "当前注意力未配置因果掩码，结果属于固定窗口离线序列推理。当前统一训练入口与部署入口均保留各平台实际原始雷达、EO和ESM掩码，僚机同时通过原始观测和有效链路参与；部署应保持这一输入口径。解析器从t_grid正差分中取中位数作为样本dt，明显变步长数据应先重采样。V8兼容程序在0.1秒网格上选择最近源帧，并对传感器、平台、链路、掩码、真值和时间戳同步使用同一索引，不执行未说明的均值或插值。"
    )
    add_table(
        doc,
        ["能力标识", "能力名称", "能力说明", "主要输出"],
        [
            ["CAP-001", "模型资产管理", "发现V5至V7登记权重，校验文件、网络宽度和计算设备，并以候选加载方式切换活动模型。", "模型目录、活动模型和加载记录"],
            ["CAP-002", "注册数据解析", "把按场景保存的注册数据拆分为目标级固定窗口样本，统一字段、维度、单位、坐标、时间与掩码。", "模型输入张量和输入摘要"],
            ["CAP-003", "端到端序列推理", "执行物理先验、Learned-K、多模态编码、时序Transformer和图协同组成的完整模型前向。", "B×T×6目标状态序列"],
            ["CAP-004", "V8兼容验证", "保存40 Hz原始数据，同时生成与现有权重兼容的约10 Hz固定窗口视图并保留同步索引。", "兼容样本、索引清单和实际dt"],
            ["CAP-005", "单场景跟踪验证", "按场景和目标执行完整推理，形成真值、预测、逐帧误差、三维回放和序列导出。", "TrackingSequence和CSV"],
            ["CAP-006", "批量测试", "在单执行槽内处理可复核的场景集合，统计有效点、位置误差、时延和异常。", "批测任务记录和原始报告"],
            ["CAP-007", "指标评估", "按统一统计口径判定RMSE、关联率、相对基线降幅和单帧折算时延，并识别证据不足。", "指标结论和独立证据"],
            ["CAP-008", "展示与归档", "通过本地前后端提供模型选择、结果回放、任务查询和报告下载，不参与模型内部计算。", "页面状态、JSON和CSV制品"],
            ["CAP-009", "跨语言固定接口", "以同一ONNX权重和18项float32输入向MATLAB与C调用方提供固定尺寸推理入口和黄金样例。", "ONNX模型、测试向量和一致性结果"],
        ],
        widths=[1.0, 1.25, 3.85, 1.05],
        center_cols=(0,),
    )

    doc.add_heading("3.3 功能需求", level=2)
    doc.add_heading("3.3.1 数据与场景功能需求", level=3)
    _requirement_table(doc, [
        ["FR-DAT-001", "数据与场景发现", "列出注册数据集、可用路径、时间口径和稳定排序的场景；拒绝根目录之外的路径。", "高"],
        ["FR-DAT-002", "目标拆分与张量校验", "按目标索引构造独立样本，校验多源字段的B、T、N维、单位、掩码和有限性。", "高"],
        ["FR-DAT-003", "可选字段补齐", "缺失IMU、慢链、龄期或更新标志时按固定形状补齐并同步标记无效，零值不得冒充观测。", "高"],
        ["FR-DAT-004", "坐标与状态顺序", "平台、链路和真值使用同一ref_pos；输出仅对位置加回参考值，状态顺序固定。", "高"],
        ["FR-DAT-005", "时间步长", "优先从有效t_grid推导dt，无有效差分时使用兼容默认值并记录实际取值。", "高"],
        ["FR-DAT-006", "V8兼容视图", "从40 Hz、400帧源数据同步生成约10 Hz、100帧样本，并保存源路径和原始帧索引。", "高"],
        ["FR-DAT-007", "场景覆盖统计", "数据目录应能够汇总传感器模态、典型想定、干扰类型、机动类别、场景数和目标航迹数，覆盖不足时不得以目录名称推定满足。", "高"],
        ["FR-DAT-008", "源数据保护", "注册数据和V8原始数据只读使用；兼容样本、缓存与报告写入独立目录，不得覆盖原文件。", "高"],
    ])

    doc.add_heading("3.3.2 模型管理与推理功能需求", level=3)
    _requirement_table(doc, [
        ["FR-MOD-001", "模型发现与严格加载", "列出登记权重及可用状态，以d_model为96构造同构模型并strict加载；键或形状不匹配必须失败。", "高"],
        ["FR-MOD-002", "设备与原子切换", "支持CPU和可用CUDA；候选引擎校验成功后才替换活动模型，失败时旧模型继续服务。", "高"],
        ["FR-MOD-003", "完整序列前向", "在inference_mode下执行一次完整100帧前向，CUDA计时边界前后同步设备。", "高"],
        ["FR-MOD-004", "有效帧与数值检查", "使用target_valid_mask同步过滤时间、真值和预测；空结果、NaN或Inf应终止并给出原因。", "高"],
        ["FR-MOD-005", "训练掩码复现", "部署预处理应保留各平台实际原始传感器掩码，与当前训练一致；输入策略差异必须记录为新的验证配置。", "高"],
        ["FR-MOD-006", "推理结果缓存", "缓存键应至少包含活动模型、场景和目标；模型切换、数据路径变化或输入契约变化后不得复用旧缓存。", "中"],
        ["FR-MOD-007", "模型边界声明", "接口、页面和报告应明确当前模型为100帧非因果固定窗口推理，禁止把回放刷新或V8源采样率表述为在线输出频率。", "高"],
    ])

    doc.add_heading("3.3.3 单场景跟踪验证功能需求", level=3)
    _requirement_table(doc, [
        ["FR-TRK-001", "场景目标选择", "选择活动模型数据目录中的场景和目标，并在推理前显示场景、目标和帧数摘要。", "中"],
        ["FR-TRK-002", "结果与误差", "返回有效时间轴、预测和真值六维状态、逐帧位置误差、运行RMSE及模型前向耗时。", "高"],
        ["FR-TRK-003", "回放与显示", "支持播放、暂停、跳转及三维视角；控制只改变已计算结果的发送或显示位置。", "中"],
        ["FR-TRK-004", "序列导出", "导出当前序列的时间、真值坐标、预测坐标和位置误差CSV。", "中"],
        ["FR-TRK-005", "场景异常处理", "场景不存在、目标索引越界、无注册目标或有效帧为空时停止当前请求，并返回可定位的场景标识和原因。", "高"],
    ])

    doc.add_heading("3.3.4 批量测试与评估功能需求", level=3)
    _requirement_table(doc, [
        ["FR-BAT-001", "任务配置与校验", "配置模型、数据集、设备、预热帧、批大小、数据线程和范围；非法值在提交前拒绝。", "高"],
        ["FR-BAT-002", "任务队列和状态", "提交后返回唯一任务ID；单执行槽依次处理任务，并记录状态、进度和异常原因。", "高"],
        ["FR-BAT-003", "任务恢复", "重启后保留已完成历史；无法续跑的排队或运行任务标记失败并记录中断。", "中"],
        ["FR-BAT-004", "位置与时延指标", "计算稳定帧RMSE、MAE、中位数、P95、最大误差、逐航迹指标及模型前向耗时。", "高"],
        ["FR-BAT-005", "指标判定与证据", "四项指标形成通过、失败或数据不足；关联率和基线降幅须绑定独立证据。", "高"],
        ["FR-BAT-006", "消融比较", "完整模型、无图协同和仅雷达主流模式复用同一权重与样本，仅用于结构贡献分析。", "中"],
        ["FR-BAT-007", "可复现实验范围", "报告应保存筛选规则、实际文件清单、样本数量、随机种子或确定性顺序、批大小、线程数和预热帧。", "高"],
        ["FR-BAT-008", "结果有限性", "进入统计的预测、真值和误差必须为有限数；发生非有限值时记录对应任务、批次或航迹并中止正式判定。", "高"],
    ])

    doc.add_heading("3.3.5 报告和支撑界面功能需求", level=3)
    _requirement_table(doc, [
        ["FR-RPT-001", "报告与来源", "保存任务配置、模型与数据摘要、场景范围、有效点、指标、异常、环境和来源签名。", "高"],
        ["FR-RPT-002", "查询与下载", "按状态、模型、数据集、名称和ID查询，并下载批测JSON、评估JSON及CSV制品。", "中"],
        ["FR-UI-001", "系统总览", "显示服务、活动模型、设备、数据集、任务队列和四项技术指标。", "中"],
        ["FR-UI-002", "错误提示", "连接、加载、参数、任务和报告错误应给出中文提示及请求ID或任务ID。", "中"],
        ["FR-UI-003", "状态一致性", "页面显示应来源于后端状态和任务记录；刷新页面不得伪造任务完成、模型切换或指标通过状态。", "中"],
        ["FR-RPT-003", "报告完整性", "报告写入采用临时文件和原子替换；下载前应校验任务终态、文件存在性和任务标识一致性。", "高"],
    ])

    doc.add_heading("3.3.6 跨语言部署功能需求", level=3)
    _requirement_table(doc, [
        ["FR-POR-001", "固定推理尺寸", "MATLAB与C交付接口固定使用B为1、T为100、N为5，输出为1×100×6；其他尺寸必须重新导出并重新验证ONNX模型。", "高"],
        ["FR-POR-002", "输入缓冲契约", "接口接收18项连续float32张量；字段名称、行优先展开顺序、尺寸、单位和掩码含义应与portable/INTERFACE.md一致。", "高"],
        ["FR-POR-003", "模型与测试向量", "交付同一ONNX模型、黄金样例、输入清单和参考输出；C与MATLAB入口必须使用该模型和样例完成独立验证。", "高"],
        ["FR-POR-004", "跨实现一致性", "MATLAB或C相对Python参考输出的最大位置分量差不超过1 m，最大速度分量差不超过0.2 m/s。", "高"],
        ["FR-POR-005", "指标边界", "黄金样例仅证明跨语言数值一致性，不替代完整验证集上的合同指标测试，也不承担原始MAT解析、数据生成或训练。", "高"],
    ])

    doc.add_heading("3.4 外部接口需求", level=2)
    doc.add_paragraph(
        "软件的外部边界包括本地REST接口、单场景WebSocket接口、模型与数据文件接口、报告下载接口以及跨语言ONNX接口。前端是这些接口的一个调用方，而不是模型功能的组成部分。外部调用必须通过稳定标识选择模型、数据集、场景、目标和任务，不得依据文件枚举顺序或页面显示顺序推断资源身份。"
    )
    doc.add_heading("3.4.1 接口标识", level=3)
    add_table(
        doc,
        ["接口标识", "接口名称", "接口类型", "方向", "接口内容"],
        [
            ["EXT-REST", "本地服务接口", "HTTP/JSON", "双向", "模型、设备、场景、批测、评估和报告资源"],
            ["EXT-WS", "跟踪回放接口", "WebSocket/JSON", "双向", "单场景请求、回放控制和逐帧结果"],
            ["EXT-MODEL", "PyTorch权重接口", "PT文件", "输入", "登记模型权重和可选指标摘要"],
            ["EXT-DATA", "注册数据接口", "PKL文件", "输入", "按场景组织的多目标多源注册序列"],
            ["EXT-REPORT", "报告制品接口", "JSON/CSV", "输出", "任务配置、指标、逐航迹结果和比较结果"],
            ["EXT-ONNX", "固定模型接口", "ONNX文件", "输入", "MATLAB与C共用的固定尺寸推理图"],
            ["EXT-GOLDEN", "黄金样例接口", "NPZ/BIN/JSON", "输入输出", "18项输入、参考输出、尺寸清单和一致性阈值"],
        ],
        widths=[1.0, 1.45, 1.15, 0.75, 2.6],
        center_cols=(0, 2, 3),
    )

    doc.add_heading("3.4.2 服务接口", level=3)
    add_table(
        doc,
        ["方法", "资源", "主要输入", "主要输出"],
        [
            ["GET", "/api/health", "无", "服务状态、活动模型和任务概况"],
            ["GET", "/api/system、/api/devices", "无", "运行环境、依赖和CPU/CUDA设备"],
            ["GET", "/api/config/acceptance-criteria", "无", "四项技术指标名称、阈值和口径"],
            ["GET/POST", "/api/models系列", "模型标识、设备", "模型目录、活动模型、刷新和切换结果"],
            ["GET", "/api/scenarios、/api/datasets", "模型或数据集标识", "场景、目标和数据集摘要"],
            ["POST/GET", "/api/batch-tests系列", "任务配置、筛选、任务ID", "任务创建、列表、详情、取消和进度"],
            ["GET/POST", "/assessment、/evidence系列", "任务ID、指标标识和证据", "指标判定、证据登记和校验结果"],
            ["GET", "/api/evaluations系列", "参考或比较条件", "参考摘要、跨数据集矩阵和任务比较"],
            ["GET", "/api/reports及下载接口", "任务ID或任务集合", "报告目录、JSON和UTF-8 CSV"],
            ["WS", "/ws/tracking", "场景、目标、播放控制", "序列元数据和逐时刻跟踪结果"],
        ],
        widths=[0.75, 2.65, 1.8, 1.8],
        center_cols=(0,),
    )

    doc.add_heading("3.4.3 软件接收的消息", level=3)
    add_table(
        doc,
        ["消息标识", "信源", "信息内容", "接收校验"],
        [
            ["MSG-MODEL-ACTIVATE", "模型管理页面或REST调用方", "模型标识和目标设备", "模型必须登记且文件存在；CUDA索引必须可用"],
            ["MSG-TRACK-START", "跟踪验证页面或WebSocket调用方", "场景标识、目标索引和可选播放参数", "场景属于活动数据目录，目标索引在注册目标范围内"],
            ["MSG-PLAYBACK", "跟踪验证页面", "play、pause或seek及目标时间位置", "命令属于允许集合，跳转位置位于已计算序列范围内"],
            ["MSG-BATCH-CREATE", "批量测试页面或REST调用方", "模型、数据集、设备、范围、批大小、线程数和预热帧", "字段类型、取值范围和组合关系均有效"],
            ["MSG-BATCH-CANCEL", "批量测试页面或REST调用方", "任务ID", "任务存在且尚未处于不可取消的终态"],
            ["MSG-EVIDENCE", "测试验收人员", "任务、指标、统计值、来源文件和证据摘要", "指标标识有效，来源签名与当前任务相符"],
            ["MSG-FILE-INPUT", "配置目录", "PT权重、PKL注册场景、ONNX模型或黄金样例", "路径位于配置根目录，类型、尺寸和必要字段完整"],
        ],
        widths=[1.25, 1.8, 2.35, 1.75],
        center_cols=(0,),
    )

    doc.add_heading("3.4.4 软件发送的消息", level=3)
    add_table(
        doc,
        ["消息标识", "信宿", "信息内容", "完整性要求"],
        [
            ["MSG-STATUS", "页面或REST调用方", "服务、模型、设备、数据集和任务状态", "包含稳定资源标识和状态发生时间"],
            ["MSG-TRACK-FRAME", "跟踪验证页面", "时间、真值六维状态、预测六维状态和位置误差", "仅发送已完成模型前向且通过有限性检查的数据"],
            ["MSG-BATCH-RESULT", "批测与评估页面", "配置、进度、样本量、误差统计、时延和异常", "包含任务ID、来源签名和终态"],
            ["MSG-ASSESSMENT", "评估页面和报告", "四项指标的阈值、实测值、结论和证据状态", "通过、失败和证据不足必须明确区分"],
            ["MSG-REPORT", "文件接收方", "批测JSON、评估JSON、逐航迹CSV和比较CSV", "文件内容与任务ID一致且写入完成后方可下载"],
            ["MSG-ERROR", "所有调用方", "状态码、中文原因、请求ID或任务ID", "不得泄露凭据、完整环境变量和根目录外文件内容"],
        ],
        widths=[1.3, 1.55, 2.55, 1.75],
        center_cols=(0,),
    )

    doc.add_heading("3.4.5 协议和错误处理", level=3)
    doc.add_paragraph(
        "REST接口使用JSON交换结构化数据，文件下载使用JSON或UTF-8 CSV。HTTP响应应包含X-Request-ID和X-Process-Time-Ms响应头；资源不存在、状态冲突、参数非法和未处理异常分别使用404、409、422和500状态码。WebSocket连接建立后先校验活动模型、场景和目标，再执行一次完整序列推理；后续play、pause和seek仅控制已计算结果的发送进度，不触发逐帧重算。"
    )
    doc.add_paragraph(
        "权重、数据和报告文件接口使用本地文件系统。调用方只能提交登记标识或根目录内的安全文件名，服务端必须解析并校验最终路径。输出文件在临时路径写入完成后再原子替换为正式文件，任何异常均不得留下可被识别为已完成报告的半成品。"
    )

    doc.add_heading("3.5 内部接口需求", level=2)
    doc.add_paragraph(
        "内部接口按照数据解析、物理状态估计、深度特征处理、运行时封装、批量统计和报告归档划分。各接口以显式张量或结构化对象传递数据，不以全局可变变量隐式交换模型输入。调用方必须在进入下一层前完成本层负责的尺寸、设备、类型和有限性检查。"
    )
    add_table(
        doc,
        ["接口标识", "信源", "信宿", "传输对象", "主要约束"],
        [
            ["INT-DAT-RUN", "注册PKL解析器", "TransformerEngine", "目标级标准样本", "字段、单位、T/N维、掩码和dt完整"],
            ["INT-RUN-MOD", "TransformerEngine", "CoopTrackerModel", "18项批张量", "统一device与dtype，批次维一致"],
            ["INT-PHY-DEEP", "LearnedKFrontEnd", "多模态深度支路", "粗状态和30维质量特征", "参考坐标、更新标志和缺失信息含义固定"],
            ["INT-MOD-SEQ", "CoopTrackerModel", "TransformerEngine", "B×T×6绝对状态", "顺序为px、vx、py、vy、pz、vz且数值有限"],
            ["INT-SEQ-WS", "TransformerEngine", "WebSocket发送器", "TrackingSequence", "时间、真值、预测和误差使用同一有效帧索引"],
            ["INT-MGR-BAT", "ModelManager", "BatchTestManager", "活动模型快照和任务配置", "任务运行期间模型与数据来源可追溯"],
            ["INT-BAT-EVAL", "BatchTestManager", "EvaluationService", "任务报告、指标和来源签名", "只评估已完成或明确失败的正式记录"],
            ["INT-EVAL-RPT", "EvaluationService", "报告目录", "评估JSON和比较CSV", "文件名、任务ID和签名一致"],
            ["INT-ONNX-LANG", "ONNX Runtime", "MATLAB/C封装", "输入缓冲区和prediction", "连续float32、固定尺寸和行列顺序转换正确"],
        ],
        widths=[1.15, 1.35, 1.45, 1.65, 2.0],
        center_cols=(0,),
    )
    _lead(doc, "数据到模型", "解析器先按场景读取注册PKL，再按目标索引构造100帧、5平台样本；运行时增加批次维并将全部张量移动到活动模型设备。解析层负责字段和时间口径，运行时负责设备和批次，模型内部不再猜测缺失字段。")
    _lead(doc, "模型到结果", "CoopTrackerModel返回绝对ECEF六维状态。运行时使用同一target_valid_mask过滤时间、预测和真值，计算逐帧三维位置误差并封装TrackingSequence；若任一数组为空或存在NaN、Inf，结果不得进入回放和统计。")
    _lead(doc, "任务到报告", "BatchTestManager固定一个高负载执行槽，按确定性文件集合运行任务并持久化状态。EvaluationService读取任务报告与外部证据，形成技术指标结论；报告层只归档已生成数据，不修改模型预测或重新解释样本范围。")

    doc.add_heading("3.6 内部数据需求", level=2)
    doc.add_heading("3.6.1 模型张量", level=3)
    add_table(
        doc,
        ["数据对象", "形状或结构", "单位或取值", "用途"],
        [
            ["radar", "B×T×N×3", "az、el为rad，r为m", "几何反演、信息融合和雷达主流编码"],
            ["eo、esm", "B×T×N×2", "az、el为rad", "光电和电子支援方向特征编码"],
            ["radar/eo/esm_mask", "B×T×N", "0或1", "通道替代、观测权重和有效节点判断"],
            ["uav_pos", "B×T×N×3", "相对ref_pos，m", "平台位置、邻接距离和协同节点特征"],
            ["rotmat_b2e", "B×T×N×3×3", "正交旋转矩阵", "机体系方向向ECEF转换"],
            ["fast_link、slow_link", "B×T×N×3", "相对ref_pos，m", "协同位置观测和无雷达时的物理补充"],
            ["fast/slow_update、age", "B×T×N", "0/1及秒", "链路更新状态和新鲜度特征"],
            ["imu、esm_mode", "B×T×N×6及B×T×N", "工程定义值", "姿态门控、平台运动和ESM模式特征"],
            ["ref_pos、dt", "B×3及B", "ECEF m及s", "局部坐标基准和状态转移步长"],
            ["label_abs、target_valid_mask", "B×T×6及B×T", "m、m/s及0/1", "真值监督和推理评估有效帧"],
            ["prediction", "B×T×6", "m和m/s", "最终绝对位置与速度序列"],
        ],
        widths=[1.55, 1.45, 1.65, 2.15],
    )
    _lead(doc, "有效标签前提", "当前训练循环对固定窗口内全部时刻计算损失，没有使用target_valid_mask从训练损失中删除无效标签。因此用于形成交付权重的训练样本应保证100帧标签完整有效。若以后引入变长或局部缺标数据，应同步修改损失掩码、训练统计和部署验证，不能只在推理输出后删除无效帧。")

    doc.add_heading("3.6.2 文件与记录数据", level=3)
    add_table(
        doc,
        ["数据类别", "载体", "内容", "完整性和生命周期"],
        [
            ["注册场景", "PKL", "RegisteredTrack、多源量测、平台状态、时间网格和可选AssociationStats", "原始文件只读；解析失败时记录场景和原因"],
            ["PyTorch模型", "PT", "CoopTrackerModel参数或model_state_dict", "以strict方式加载；模型目录刷新不修改文件"],
            ["模型指标摘要", "JSON", "训练轮数、d_model、误差和时延摘要", "仅作模型目录说明，不替代正式批测"],
            ["批测任务", "JSON", "配置、状态、进度、来源签名、统计、异常和逐航迹结果", "排队、运行与终态可区分；重启后保留历史"],
            ["评估证据", "JSON", "指标标识、数值、来源、适用范围和任务签名", "来源不一致时不得用于当前任务判定"],
            ["报告导出", "JSON/CSV", "综合评估、逐航迹数据和任务比较", "UTF-8编码；仅在写入完成后提供下载"],
            ["固定模型", "ONNX", "B为1、T为100、N为5的推理图", "与黄金样例和接口清单成套交付"],
            ["黄金样例", "NPZ/BIN/JSON", "18项输入、参考prediction和manifest", "跨语言验证期间只读，不作为合同性能样本"],
        ],
        widths=[1.25, 1.0, 3.05, 2.1],
    )
    doc.add_paragraph(
        "正式任务必须保存能够重建样本集合的标识，包括模型、数据集、场景选择、源文件摘要、V8抽样规则、预热帧、有效点数量、计算设备和代码摘要。缓存可以删除并重新生成，原始数据、正式任务记录和证据文件不得因页面刷新或服务重启被覆盖。"
    )

    doc.add_heading("3.7 适应性需求", level=2)
    _requirement_table(doc, [
        ["NFR-ADP-001", "设备适应", "同一PyTorch权重应能在CPU或可用CUDA设备上运行，张量、模型参数和临时矩阵必须位于同一设备。", "高"],
        ["NFR-ADP-002", "数据集适应", "支持登记的V5至V8数据目录；数据集差异必须通过显式目录和元数据表达，不得根据文件名猜测字段含义。", "高"],
        ["NFR-ADP-003", "缺失通道适应", "传感器缺失通过mask token、缺失向量、链路龄期和图节点过滤处理，补零必须与无效掩码同步。", "高"],
        ["NFR-ADP-004", "固定窗口适应", "不足100帧的数据不得在无padding mask条件下直接补零，应拒绝或构造经过训练与测试验证的完整窗口。", "高"],
        ["NFR-ADP-005", "跨语言适应", "固定ONNX模型应由MATLAB和Windows x64 C调用入口使用同一张量契约执行，并以黄金样例验证数值一致性。", "高"],
    ])

    doc.add_heading("3.8 安全性需求", level=2)
    _requirement_table(doc, [
        ["NFR-SAF-001", "本机服务边界", "后端默认监听127.0.0.1，CORS仅允许明确配置的本机前端地址；改变监听范围前必须单独评审访问控制。", "高"],
        ["NFR-SAF-002", "路径安全", "场景、权重、证据和报告路径必须解析到各自配置根目录，拒绝目录穿越、绝对外部路径和不受控扩展名。", "高"],
        ["NFR-SAF-003", "源数据保护", "服务不得修改模型权重、原始注册数据、ONNX模型和黄金样例；派生制品写入独立目录。", "高"],
        ["NFR-SAF-004", "故障隔离", "单请求、单场景或单任务异常不得把未完成结果标记为成功；模型切换失败时保留原活动引擎。", "高"],
    ])

    doc.add_heading("3.9 保密性需求", level=2)
    doc.add_paragraph(
        "软件不在文档中自行设定项目密级。数据、模型、源码、报告和交付介质应按合同、需方制度及实际标识执行访问、复制、携带和销毁管理。应用日志和错误响应不得输出凭据、完整系统环境变量、用户目录信息或配置根目录之外的文件内容。"
    )
    doc.add_paragraph(
        "原型软件默认不提供账号、远程鉴权和互联网发布能力，因此本地运行边界不能等同于完整的网络安全防护。如果后续要求多用户、远程访问或跨网络部署，应新增身份认证、权限分级、传输保护、审计留痕和安全测试需求，并在变更评审后实施。"
    )

    doc.add_heading("3.10 环境需求", level=2)
    doc.add_paragraph(
        "主要验证环境为Windows 64位工作站。后端、模型和批测组件在Python环境中运行，前端通过本机HTTP和WebSocket访问服务；系统不依赖互联网连接。模型权重、注册数据和黄金样例目录按只读方式使用，批测报告目录必须可写，并预留与完整验证范围相适应的空间。"
    )
    doc.add_paragraph(
        "运行环境的系统时间应保持稳定，以便记录任务时间、模型激活时间和报告生成时间。正式时延结果必须同时记录操作系统、处理器、内存、GPU与显存、驱动、CUDA运行时、Python、PyTorch、模型权重、批大小、预热和测量次数；开发工作站结果不得直接替代其他目标平台结果。"
    )

    doc.add_heading("3.11 计算机资源需求", level=2)
    doc.add_heading("3.11.1 计算机硬件需求", level=3)
    add_table(
        doc,
        ["资源", "最低功能要求", "推荐配置或验收记录"],
        [
            ["处理器", "支持Windows 64位和当前Python/PyTorch运行环境，能够完成100帧、5平台CPU推理", "推荐4个及以上逻辑核心；记录型号和核心数"],
            ["内存", "能够同时容纳活动模型、一个批次样本、服务进程和报告缓存", "推荐16 GB及以上；增加批大小时重新评估"],
            ["GPU", "非强制，CPU模式必须可用", "推荐使用与PyTorch兼容的NVIDIA CUDA GPU；记录型号、显存和驱动"],
            ["存储", "能够保存源码、权重、注册数据和正式报告", "推荐使用固态存储并为完整批测保留冗余空间"],
            ["显示与网络", "能够运行现代浏览器并访问本机服务", "显示器分辨率满足三维回放和表格查看；本机TCP/IP可用"],
        ],
        widths=[1.15, 3.15, 2.85],
    )

    doc.add_heading("3.11.2 计算机软件需求", level=3)
    add_table(
        doc,
        ["用途", "软件或运行库", "要求"],
        [
            ["Python模型与后端", "Python、PyTorch、NumPy、SciPy、h5py、FastAPI、Uvicorn", "使用项目依赖清单中兼容的版本组合，正式环境记录实际版本"],
            ["前端展示", "Vue、TypeScript、Vite、ECharts及ECharts GL", "依照frontend/package.json构建；使用现代Chrome或Edge访问"],
            ["ONNX导出和验证", "onnx、onnxruntime", "Python交付环境使用portable/python/requirements.txt规定的依赖"],
            ["MATLAB调用", "MATLAB R2025a或更新版本及ONNX转换相关工具箱", "能够载入固定ONNX模型并运行黄金样例"],
            ["C调用", "CMake、C/C++编译器、ONNX Runtime 1.18.1 Windows x64 CPU包", "能够生成调用库和黄金样例验证程序"],
            ["文档与交付", "Microsoft Word或兼容办公软件", "能够显示自动目录、表格和中文字体，并导出PDF"],
        ],
        widths=[1.4, 3.25, 2.5],
    )

    doc.add_heading("3.11.3 计算机通信需求", level=3)
    doc.add_paragraph(
        "前后端通过本机TCP/IP通信：HTTP服务默认使用127.0.0.1:8000，开发前端通常使用127.0.0.1或localhost的5173端口；单场景回放使用WebSocket。软件不要求访问互联网，也不要求连接外部数据库。若端口被占用或防火墙阻断，应在启动阶段给出可诊断错误。"
    )
    doc.add_paragraph(
        "模型、注册数据、报告和跨语言制品通过本地文件系统交换。通过移动介质或共享目录交付时，应保持目录结构、文件名、manifest及相对路径关系，并在复制后核对文件大小或摘要；不得只复制ONNX模型而遗漏黄金样例、接口说明或运行库。"
    )

    doc.add_heading("3.12 软件质量因素", level=2)
    _requirement_table(doc, [
        ["NFR-QLT-001", "功能正确性", "模型输入字段、状态顺序、坐标恢复、掩码传播和指标公式应与设计说明书及可执行源码一致。", "高"],
        ["NFR-QLT-002", "数值可靠性", "三阶安全求解、模型输出和误差统计应保持有限；非有限值、空样本和病态输入必须被发现并记录。", "高"],
        ["NFR-QLT-003", "运行可靠性", "模型切换采用候选加载，任务异常保存原因，报告写入采用临时文件和原子替换，服务重启后终态可恢复。", "高"],
        ["NFR-QLT-004", "可追踪与复现", "正式任务保存模型、数据、场景、配置、环境、抽样规则、有效帧、代码摘要和证据签名。", "高"],
        ["NFR-QLT-005", "性能可比性", "区分序列前向、单帧折算、批量吞吐和端到端响应时间，正式报告给出设备与测量条件。", "高"],
        ["NFR-QLT-006", "可维护性", "数据、模型、运行时、批测、评估、接口和页面职责分离；需求标识能够映射到设计对象和测试。", "中"],
        ["NFR-QLT-007", "可测试性", "关键数据契约、模型加载、V8兼容、任务状态、指标和跨语言一致性均应具有可重复测试入口。", "高"],
        ["NFR-QLT-008", "易用性", "主要操作和异常使用清晰中文说明，危险操作二次确认，并显示活动模型、数据集、设备和任务状态。", "中"],
    ])

    doc.add_heading("3.13 设计和实现约束", level=2)
    add_bullets(doc, [
        "活动PyTorch模型使用d_model等于96的CoopTrackerModel，并以strict方式加载权重。",
        "序列长度T为100，平台槽位N为5，输出状态顺序固定为px、vx、py、vy、pz、vz。",
        "当前时序Transformer和双向跨注意力为非因果结构，正式用途限于完整窗口离线推理。",
        "图邻接距离阈值为20000 m；传感器有效性与距离邻接逐元素相乘后加入自环。",
        "V8兼容数据采用最近时间帧同步抽样，传感器、平台、链路、真值、掩码和时间戳必须使用同一索引。",
        "交付推理必须保留与当前训练一致的各平台实际原始雷达、EO和ESM掩码，不额外屏蔽僚机；改变输入策略须重新验证。",
        "支撑界面不得修改模型权重、注意力矩阵或中间状态，只能选择受控配置并展示、导出模型结果。",
        "ONNX、MATLAB和C固定接口采用B为1、T为100、N为5及18项float32输入，不支持运行时动态形状。",
    ])
    doc.add_paragraph(
        "修改序列长度、采样频率、注意力因果性、僚机掩码、状态顺序、d_model、图邻接规则、输入字段或跨语言缓冲区顺序时，应同时复核数据解析、训练入口、模型前向、ONNX导出、运行时、批测统计、测试向量和文档。仅修改一个调用点不能证明原权重继续适用；涉及学习分布或网络结构时，应评估是否需要重新训练。"
    )

    doc.add_heading("3.14 人员需求", level=2)
    doc.add_paragraph(
        "模型研发人员应理解多源量测、坐标变换、Learned-K状态递推、跨注意力、时序Transformer、图协同和损失函数，能够判断输入契约变化是否影响权重适用性。后端与测试人员应能核对模型加载、场景解析、有效帧、批测状态、指标口径和证据签名；前端维护人员只需掌握接口契约和展示状态，不应修改模型计算逻辑。"
    )
    doc.add_paragraph(
        "部署维护人员应掌握Windows环境启动、目录配置、CPU/CUDA设备检查、端口占用处理、日志查看和报告备份。使用MATLAB或C接口的人员应理解固定尺寸、连续float32缓冲区、行列存储顺序和黄金样例判据。验收人员应能够依据需求标识选择测试方法，并区分模型性能证据、跨语言一致性证据和界面演示记录。"
    )

    doc.add_heading("3.15 培训需求", level=2)
    doc.add_paragraph(
        "交付培训应至少覆盖软件启动与关闭、活动模型和设备选择、数据集及场景识别、单场景推理与回放、批量任务提交与取消、四项技术指标解释、报告下载和常见故障定位。培训示例应使用可公开交付的样例数据，不应把历史截图或页面示例值作为正式性能结论。"
    )
    doc.add_paragraph(
        "面向模型研发和二次集成人员的技术交流还应覆盖18项输入、参考坐标、dt、V8兼容抽样、僚机掩码、非因果固定窗口边界、ONNX导出以及MATLAB/C缓冲区契约。培训完成后，使用人员应能够独立完成一次单场景验证、一次小范围批测、一次报告导出和一次黄金样例验证。"
    )

    doc.add_heading("3.16 软件保障需求", level=2)
    doc.add_paragraph(
        "软件保障活动应覆盖配置管理、问题登记、缺陷修复、回归测试、交付物核对和环境恢复。每个影响模型输入、权重、采样、指标或报告字段的问题应记录复现条件、影响范围、修复内容和验证证据；关闭问题前应运行对应的自动测试或人工验证。"
    )
    doc.add_paragraph(
        "应保留能够重新建立运行环境的依赖文件、启动说明和目录示例。模型权重、ONNX模型、第三方运行库及黄金样例应作为成套资产管理；模型重新发现、活动模型切换、任务历史读取、报告重新下载和跨语言一致性验证应纳入交付检查。"
    )

    doc.add_heading("3.17 其它需求", level=2)
    doc.add_paragraph(
        "软件不承担原始点迹检测、未注册多目标在线关联、飞行控制、传感器调度、任务规划或模型训练服务。数据中的AssociationStats和IMM-JPDA结果属于外部或离线证据，除非其来源、样本范围和任务签名可核对，否则只能显示为参考，不能生成验收通过结论。"
    )
    doc.add_paragraph(
        "后续新增实时流输入、动态平台数、变长序列、因果推理、远程多用户访问或其他传感器类型时，应建立新的需求标识并完成变更评审。新增功能不得以兼容名义改变现有固定窗口模型的输入输出含义。"
    )

    doc.add_heading("3.18 验收交付和包装", level=2)
    doc.add_heading("3.18.1 验收", level=3)
    doc.add_paragraph(
        "验收应在双方约定的工作站或目标环境中，依据批准的测试方案和本说明书逐项实施。开始前应冻结活动权重、数据范围、代码摘要、计算设备、预热帧和指标证据；验收过程中产生的任务记录、日志、JSON和CSV应归档。发现问题时应记录对应需求标识、复现条件和影响结论，整改后对相关需求及其回归范围重新验证。"
    )
    add_numbered_steps(doc, [
        "核对活动权重、数据集路径、计算设备、软件依赖、代码摘要和报告目录。",
        "执行固定三阶安全求解、V8兼容、权重严格加载、场景解析、有效帧过滤和非有限值拒绝测试。",
        "选择代表性场景完成单场景推理、WebSocket回放、暂停、跳转和CSV导出。",
        "使用冻结的数据范围执行正式批测，记录实际场景、航迹、有效点、异常和测量条件。",
        "生成关联正确率和IMM-JPDA相对降幅的独立证据，核对适用范围和任务签名。",
        "使用黄金样例分别验证ONNX、MATLAB和C入口，并确认误差不超过跨语言一致性阈值。",
        "导出原始报告、综合评估、逐航迹指标和任务比较文件，核对四项合同指标结论。",
    ])

    doc.add_heading("3.18.2 交付", level=3)
    add_table(
        doc,
        ["交付类别", "交付内容", "检查要求"],
        [
            ["软件源码", "模型、数据解析、训练入口、后端、前端、测试和V8兼容程序", "目录完整，能够定位需求对应实现"],
            ["模型资产", "登记的PyTorch权重、ONNX固定模型和必要指标摘要", "文件存在、可加载并记录摘要"],
            ["数据与样例", "注册验证数据说明、V8兼容说明、黄金样例和manifest", "来源、尺寸、索引和用途明确"],
            ["跨语言组件", "MATLAB入口、C头文件与实现、构建文件和ONNX Runtime运行包", "能够完成黄金样例验证"],
            ["配套文档", "需求规格说明书、设计说明书、使用说明书、接口说明和依赖说明", "名称统一，内容与交付软件一致"],
            ["验证制品", "测试记录、批测报告、指标证据、逐航迹CSV和问题关闭记录", "能够追踪到需求和任务标识"],
        ],
        widths=[1.25, 3.7, 2.2],
    )

    doc.add_heading("3.18.3 包装", level=3)
    doc.add_paragraph(
        "软件采用电子介质或双方约定的安全传输方式交付。交付目录应把源码、权重、数据说明、portable跨语言组件、文档和验证制品分区保存，并提供顶层清单。第三方ONNX Runtime运行包应保留许可证和第三方声明。复制完成后应核对目录结构、关键文件数量、文件大小或摘要，确认没有把临时缓存、未完成报告或开发机绝对路径作为正式交付内容。"
    )

    doc.add_heading("3.19 需求的优先顺序和关键程度", level=2)
    doc.add_paragraph(
        "优先级高的需求直接影响模型输入含义、权重可加载性、状态输出、合同指标、跨语言一致性和正式证据，未满足时不得形成验收通过结论。优先级中的需求主要影响操作效率、展示完整性和维护便利性，可在不改变模型结果和正式证据的前提下安排整改。软件总体关键性等级应以合同或需方正式文件为准；在该信息未补录前，本文只对具体需求标注高或中优先级，不擅自设定项目关键性等级。"
    )

    doc.add_heading("4 合格性规定", level=1)
    doc.add_heading("4.1 合格性方法", level=2)
    doc.add_paragraph(
        "每项可验收需求至少对应一种可执行、可重复的合格性方法。验证活动应在冻结的模型、数据、代码和环境条件下执行，并形成能够追踪到需求标识的记录。模型性能、跨语言一致性和界面演示属于不同证据类型，不得互相替代。"
    )
    add_table(
        doc,
        ["方法标识", "方法", "适用内容", "基本输出"],
        [
            ["Q-TST", "测试", "功能、接口、张量、数值、异常、性能、回归和跨语言一致性", "测试输入、步骤、期望、实测和结论"],
            ["Q-INS", "检查", "配置、文件、目录、日志、源码结构、依赖、许可证和交付物", "检查清单、文件摘要和问题记录"],
            ["Q-ANA", "分析", "误差、时延、证据适用性、V8抽样、基线比较和资源使用", "统计结果、计算口径和来源说明"],
            ["Q-DEM", "演示", "模型切换、场景选择、回放、批测、评估、下载和黄金样例运行", "操作记录、页面结果和导出制品"],
        ],
        widths=[1.0, 0.85, 3.45, 1.85],
        center_cols=(0, 1),
    )
    doc.add_paragraph(
        "测试失败时应保留失败输入、错误信息和环境条件。因样本、关联证据或可比基线不足而无法计算的指标，应判为证据不足，不得按零值、历史页面示例或其他任务结果推定通过。检查和演示只能证明配置、流程或可操作性，不能单独证明RMSE、关联率、相对降幅或推理时延满足阈值。"
    )
    doc.add_heading("4.2 需求合格性矩阵", level=2)
    add_table(
        doc,
        ["需求范围", "主要方法", "合格判据", "证据"],
        [
            ["CAP-001至CAP-009", "Q-INS、Q-DEM", "九类能力均有对应入口、输出和责任边界", "能力演示记录与交付清单"],
            ["FR-DAT系列", "测试、检查", "张量、坐标、时间和V8索引同步且可追溯", "解析测试与兼容清单"],
            ["FR-MOD系列", "测试、分析", "权重strict加载，完整前向有限，过滤与掩码符合配置", "加载日志与运行时测试"],
            ["FR-TRK系列", "演示、测试", "可选择、回放、控制并导出一致序列", "演示记录与CSV"],
            ["FR-BAT系列", "测试、分析", "队列、指标、判定、证据和消融范围正确", "任务报告与独立证据"],
            ["FR-RPT和FR-UI", "检查、演示", "报告完整可下载，状态和错误可定位", "JSON、CSV与页面记录"],
            ["FR-POR系列", "测试、演示", "固定尺寸、18项输入和跨语言误差均符合契约", "黄金样例与一致性报告"],
            ["NFR-ADP系列", "测试、分析", "CPU/CUDA、数据集、缺失通道和固定窗口行为符合约束", "适应性回归报告"],
            ["NFR-SAF系列", "测试、检查", "本机边界、路径限制、源数据保护和故障隔离有效", "安全检查与异常测试"],
            ["NFR-QLT系列", "检查、测试", "正确性、可靠性、追踪、复现、维护、可用和性能口径满足要求", "测试报告与交付检查表"],
            ["3.9至3.17", "检查、测试", "保密边界、环境、资源、人员、培训和保障要求落实", "环境记录、培训和保障记录"],
            ["3.18", "检查、演示", "验收步骤完成，交付内容齐全，电子包装结构和声明完整", "验收记录与交付清单"],
        ],
        widths=[1.45, 1.25, 3.05, 1.15],
    )

    doc.add_heading("4.3 技术指标判定", level=2)
    add_table(
        doc,
        ["指标", "要求", "统计口径", "缺失或异常处理"],
        [
            ["稳定阶段位置RMSE", "不超过8 m", "按target_valid_mask保留有效帧，再按每条航迹排除配置的预热帧，使用三维位置欧氏误差汇总", "无有效点时任务失败"],
            ["目标正确关联率", "不低于85%", "使用独立关联测试或数据中可验证的AssociationStats，并绑定当前任务签名", "无独立证据时判为证据不足"],
            ["相对IMM-JPDA降幅", "不低于15%", "同数据范围、场景子集、有效帧规则、序列长度和预热设置下比较RMSE", "缺少可比基线时判为证据不足"],
            ["单帧跟踪结果生成时间", "不超过100 ms", "已构造模型张量进入CoopTrackerModel至B×T×6输出的完整序列耗时除以有效帧数", "记录设备；不得用回放刷新率替代"],
        ],
        widths=[1.4, 1.15, 3.05, 1.3],
    )
    doc.add_paragraph(
        "单帧折算耗时用于比较当前完整序列模型的计算成本，不等同于每25毫秒或100毫秒产生一次在线可用输出。正式报告应同时列出序列总耗时、单帧折算值、批大小、设备和测量次数。V8兼容任务应另外记录40 Hz源帧数、10 Hz目标帧数、抽样索引和实际dt。"
    )

    doc.add_heading("4.4 合格性实施条件", level=2)
    add_table(
        doc,
        ["阶段", "实施条件", "通过条件", "异常处理"],
        [
            ["环境确认", "记录硬件、系统、依赖、设备、端口和目录权限", "服务、模型和数据检查均正常", "停止正式测试并修复环境"],
            ["输入确认", "冻结权重、数据集、场景范围、V8规则和预热帧", "输入摘要与任务配置一致", "重新生成任务，不覆盖旧记录"],
            ["功能验证", "执行解析、加载、推理、回放、批测、评估和导出", "输出结构正确且异常路径可诊断", "记录需求标识和复现步骤"],
            ["指标验证", "完整范围、有效帧、独立关联证据和可比基线齐备", "按4.3口径形成明确结论", "证据缺失时标记证据不足"],
            ["跨语言验证", "ONNX、黄金样例、MATLAB/C环境和18项输入齐备", "位置与速度差不超过规定阈值", "核对存储顺序、模型和运行库"],
            ["交付核对", "正式文档、源码、资产、数据说明和验证制品齐备", "目录清单与实际文件一致", "补齐后重新执行交付检查"],
        ],
        widths=[1.1, 2.55, 2.3, 1.2],
        center_cols=(0,),
    )
    doc.add_paragraph(
        "正式测试应优先使用自动化测试和批测程序生成机器可读结果，同时保留足够的人工检查记录。重复测试时不得删除原任务；新的权重、数据范围、代码摘要、设备或统计规则应产生新的任务标识，以便比较和复核。"
    )

    doc.add_heading("5 需求可追踪性", level=1)
    doc.add_paragraph(
        "需求追踪采用正向和反向两条路径。正向追踪从本文需求定位到来源、设计实现和验证活动，用于检查是否遗漏实现；反向追踪从合同指标、源码组件和交付物定位到本文需求，用于确认没有无依据功能或未受控变更。追踪表使用需求系列进行汇总，具体测试用例仍应引用完整需求标识。"
    )
    doc.add_heading("5.1 需求正向追踪", level=2)
    add_table(
        doc,
        ["需求范围", "来源依据", "设计与实现对象", "验证活动"],
        [
            ["CAP系列", "合同任务和软件范围", "模型、数据、运行时、验证和跨语言组件", "能力演示与交付检查"],
            ["FR-DAT系列", "多源数据和场景要求", "src/dataset.py、scenario_catalog.py、V8兼容程序", "形状、坐标、时间、缺失字段和抽样测试"],
            ["FR-MOD系列", "端到端模型部署要求", "model_manager.py、runtime.py、src/model.py、train.py", "strict加载、完整前向、有效帧和掩码审计"],
            ["FR-TRK系列", "单场景验证要求", "TrackingSequence、WebSocket和跟踪页面", "回放、控制、误差和CSV演示"],
            ["FR-BAT系列", "批量测试和指标要求", "batch_testing.py、EvaluationService和批测页面", "队列、指标、证据和消融比较"],
            ["FR-RPT、FR-UI", "展示与性能评估要求", "报告目录、服务接口和前端页面", "字段签名、页面构建和下载检查"],
            ["FR-POR系列", "跨语言交付需求", "portable目录、ONNX模型、MATLAB和C入口", "黄金样例和误差阈值验证"],
            ["NFR-ADP系列", "数据与部署适应需求", "设备选择、缺失处理、固定窗口和ONNX接口", "跨设备、缺失场景和跨语言回归"],
            ["NFR-SAF系列", "本地原型安全边界", "服务监听、路径解析、只读资产和任务状态", "路径、异常和故障隔离测试"],
            ["NFR-QLT系列", "软件工程与验收要求", "全部模型与支撑模块", "代码复核、自动测试、批测和交付检查"],
            ["3.9至3.19", "甲方模板规定项", "环境、依赖、文档、培训、保障和交付目录", "环境、培训、保障和交付记录"],
        ],
        widths=[1.25, 1.55, 2.65, 1.65],
    )

    doc.add_heading("5.2 来源反向追踪", level=2)
    add_table(
        doc,
        ["来源或实现对象", "对应需求", "追踪说明"],
        [
            ["合同四项技术指标", "FR-BAT-004至008、NFR-QLT-001至005、4.3", "对应误差、关联、基线和时延的输入、统计与证据"],
            ["不少于三种传感器模态", "CAP-002、FR-DAT-002和007、3.6.1", "对应雷达、EO和ESM输入及覆盖统计"],
            ["不少于两种典型想定和三种干扰模式", "FR-DAT-001、007、FR-BAT-001和007", "以数据目录实际统计和批测文件清单证明"],
            ["CoopTrackerModel", "CAP-003、FR-MOD系列、3.13", "对应模型结构、固定尺寸、推理和输入约束"],
            ["V8 40 Hz源数据", "CAP-004、FR-DAT-005至008、4.3", "对应兼容视图、同步索引和结果解释"],
            ["前后端原型", "CAP-005至008、FR-TRK、FR-BAT、FR-RPT和FR-UI", "对应模型验证、展示和报告，不扩大模型边界"],
            ["portable跨语言组件", "CAP-009、FR-POR系列、NFR-ADP-005", "对应ONNX、MATLAB、C和黄金样例"],
            ["需求与设计说明书", "全部需求、5.1至5.4", "作为设计、测试和交付检查的统一基线"],
        ],
        widths=[2.1, 2.65, 2.35],
    )

    doc.add_heading("5.3 合同指标追踪", level=2)
    add_table(
        doc,
        ["合同或任务要求", "对应需求", "输出证据"],
        [
            ["模型部署、完整序列推理及场景验证", "FR-MOD、FR-TRK和FR-RPT系列", "加载记录、六维序列、回放与逐航迹报告"],
            ["稳定阶段位置RMSE不超过8 m", "FR-BAT-004至005、NFR-QLT-001", "完整任务稳定帧误差报告"],
            ["关联率不低于85%且基线降幅不低于15%", "FR-BAT-005", "签名一致的关联与IMM-JPDA证据"],
            ["单帧结果生成时间不超过100 ms", "FR-MOD-003、FR-BAT-004、NFR-QLT-005", "序列前向与单帧折算报告"],
            ["多源量测、批量验证和真实40 Hz场景", "FR-DAT、FR-BAT及NFR-ADP-002", "批测结果、输入摘要、V8源数据与兼容清单"],
        ],
        widths=[2.45, 2.15, 2.45],
    )

    doc.add_heading("5.4 变更影响追踪", level=2)
    doc.add_paragraph(
        "修改输入字段、单位、ref_pos、dt、状态顺序、序列长度、d_model、注意力因果性、图邻接距离、僚机掩码或V8采样规则时，应同时复核数据解析、训练入口、PyTorch模型、ONNX导出、MATLAB/C接口、运行时、批测统计、接口返回、黄金样例、文档和权重适用性。只修改一个调用点不能证明需求仍被满足。"
    )
    doc.add_paragraph(
        "修改技术指标阈值、预热帧、证据签名、场景范围或基线数据时，应同步更新验收矩阵、测试用例和报告字段，并保留变更前后任务配置。修改服务监听、访问方式或交付介质时，应重新评审安全、保密和环境需求。所有变更均应建立新的需求标识或明确修订现有标识，避免用章节标题代替可追踪条款。"
    )

    doc.add_heading("6 注释和缩略语", level=1)
    add_table(
        doc,
        ["缩略语或术语", "英文或符号", "定义"],
        [
            ["RMSE", "Root Mean Square Error", "三维位置欧氏误差平方均值的平方根"],
            ["MAE", "Mean Absolute Error", "三维位置欧氏误差的算术平均值"],
            ["P95", "95th Percentile", "位置误差样本的第95百分位数"],
            ["EO", "Electro-Optical", "光电方向量测通道"],
            ["ESM", "Electronic Support Measures", "电子支援方向量测通道"],
            ["ECEF", "Earth-Centered Earth-Fixed", "地心地固坐标系"],
            ["Learned-K", "Learned Gain", "使用学习得到的对角增益执行状态更新的物理前端"],
            ["GAT", "Graph Attention Network", "对跨平台节点特征进行图注意力聚合的模块"],
            ["Transformer", "Transformer", "用于时序关系和跨模态关系建模的注意力网络"],
            ["IMM-JPDA", "Interacting Multiple Model and Joint Probabilistic Data Association", "用于合同基线比较的经典模块化跟踪方法"],
            ["ONNX", "Open Neural Network Exchange", "用于跨语言部署的模型交换格式"],
            ["ref_pos", "Reference Position", "样本局部坐标平移使用的ECEF参考位置"],
            ["dt", "Time Step", "模型样本的标量时间步长，单位为秒"],
            ["mask", "Validity Mask", "表示传感器、链路、节点或目标帧是否有效的0/1张量"],
            ["AssociationStats", "Association Statistics", "注册数据中可选的目标关联统计证据"],
            ["黄金样例", "Golden Case", "用于比较Python、ONNX、MATLAB和C输出的一组固定输入与参考输出"],
            ["单帧折算时延", "Per-frame normalized latency", "完整序列前向时间除以帧数，仅为比较口径，不等于在线逐帧响应时间"],
            ["来源签名", "Source Signature", "由模型、数据、代码或任务范围摘要形成的证据适用性标识"],
        ],
        widths=[1.45, 2.55, 3.05],
    )
    doc.add_paragraph("需求标识规则：CAP为能力需求；FR-DAT、FR-MOD、FR-TRK、FR-BAT、FR-RPT、FR-UI、FR-POR依次表示数据、模型、跟踪验证、批测评估、报告、界面、跨语言部署功能；NFR-ADP、NFR-SAF、NFR-QLT依次表示适应性、安全性、软件质量。设计、测试和报告均引用完整标识。")


def _format_document(doc: Document, body_boundary: int) -> None:
    for style_name in ("Normal", "Body Text", "List Bullet", "List Number", "List Paragraph"):
        if style_name not in doc.styles:
            continue
        style = doc.styles[style_name]
        style.font.name = LATIN_FONT
        fonts = style._element.get_or_add_rPr().get_or_add_rFonts()
        fonts.set(qn("w:eastAsia"), EAST_ASIA_FONT)
        fonts.set(qn("w:ascii"), LATIN_FONT)
        fonts.set(qn("w:hAnsi"), LATIN_FONT)
        style.font.size = Pt(14)

    for style_name in ("Heading 1", "Heading 2", "Heading 3", "Heading 4"):
        style = doc.styles[style_name]
        style.font.name = LATIN_FONT
        fonts = style._element.get_or_add_rPr().get_or_add_rFonts()
        fonts.set(qn("w:eastAsia"), EAST_ASIA_FONT)
        fonts.set(qn("w:ascii"), LATIN_FONT)
        fonts.set(qn("w:hAnsi"), LATIN_FONT)
        style.font.size = Pt(14)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.italic = False
        style.paragraph_format.keep_with_next = True

    body = doc._element.body
    for child in list(body)[body_boundary:]:
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, doc._body)
            style_name = paragraph.style.name if paragraph.style is not None else ""
            is_heading = style_name in {"Heading 1", "Heading 2", "Heading 3", "Heading 4"}
            if is_heading:
                paragraph.paragraph_format.keep_with_next = True
            else:
                paragraph.paragraph_format.line_spacing = 1.5
                paragraph.paragraph_format.space_after = Pt(6)
                paragraph.paragraph_format.widow_control = True
                if paragraph.text.startswith("需求标识规则："):
                    paragraph.paragraph_format.line_spacing = Pt(18)
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.widow_control = False
            for run in paragraph.runs:
                _set_run_font(run, bold=True if is_heading else None)
                if is_heading:
                    run.font.color.rgb = RGBColor(0, 0, 0)
                    run.italic = False

    for table in doc.tables:
        set_table_borders(table)
        if table.rows:
            set_repeat_table_header(table.rows[0])
        for row_index, row in enumerate(table.rows):
            prevent_row_split(row)
            for cell in row.cells:
                tc_pr = cell._tc.get_or_add_tcPr()
                for shading in list(tc_pr.findall(qn("w:shd"))):
                    tc_pr.remove(shading)
                if row_index == 0:
                    shading = OxmlElement("w:shd")
                    shading.set(qn("w:fill"), GRAY)
                    shading.set(qn("w:val"), "clear")
                    tc_pr.append(shading)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.line_spacing = 1.2
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.space_after = Pt(0)
                    for run in paragraph.runs:
                        _set_run_font(run, bold=True if row_index == 0 else None)
                        run.font.color.rgb = RGBColor(0, 0, 0)

    settings = doc.settings._element
    update_fields = settings.find(qn("w:updateFields"))
    if update_fields is None:
        update_fields = OxmlElement("w:updateFields")
        settings.append(update_fields)
    update_fields.set(qn("w:val"), "true")
    doc.core_properties.title = "端到端协同跟踪原型软件需求规格说明书"
    doc.core_properties.subject = "软件配套交付文档"
    doc.core_properties.author = "项目组"


def build() -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    working = WORK_DIR / "requirements_working.docx"
    shutil.copy2(TEMPLATE, working)
    doc = Document(working)
    _replace_cover_text(doc)
    _set_toc_levels(doc)
    boundary = _remove_body_from(doc, "范围")
    _build_body(doc)
    _format_document(doc, boundary)
    doc.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
