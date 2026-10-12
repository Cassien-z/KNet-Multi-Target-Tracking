from pathlib import Path

from PIL import Image, ImageDraw


root = Path("artifacts/reports/software_test/IT-02-05/20260924_run01")


def make_sheet(paths: list[Path], output: Path, columns: int = 2) -> None:
    opened = [Image.open(path).convert("RGB") for path in paths]
    thumb_width = 720
    thumbs = []
    for path, image in zip(paths, opened):
        height = round(image.height * thumb_width / image.width)
        thumb = image.resize((thumb_width, height), Image.Resampling.LANCZOS)
        thumbs.append((path.name, thumb))
    label_height = 36
    cell_height = max(image.height for _, image in thumbs) + label_height
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(thumbs):
        x = (index % columns) * thumb_width
        y = (index // columns) * cell_height
        draw.text((x + 10, y + 8), name, fill="black")
        sheet.paste(image, (x, y + label_height))
    sheet.save(output)


make_sheet(sorted((root / "rendered_outline").glob("*.png")), root / "outline_contact_sheet.png")
report_paths = sorted((root / "rendered_report_final").glob("*.png"))
make_sheet(report_paths[:5], root / "report_final_contact_sheet_1.png")
make_sheet(report_paths[5:], root / "report_final_contact_sheet_2.png")
