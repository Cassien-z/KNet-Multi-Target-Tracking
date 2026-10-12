from pathlib import Path

from pypdf import PdfReader


root = Path("artifacts/reports/software_test/IT-02-05/20260924_run01")
needles = ["4.10", "IT-02-05", "图4-10-1", "图4-10-4", "本用例通过"]
for name in ["测试大纲_渲染校验.pdf", "测试报告_渲染校验.pdf"]:
    reader = PdfReader(root / name)
    print(name, len(reader.pages))
    for index, page in enumerate(reader.pages, start=1):
        value = page.extract_text() or ""
        hits = [needle for needle in needles if needle in value]
        if hits:
            print(index, hits, value[:160].replace("\n", " / "))
