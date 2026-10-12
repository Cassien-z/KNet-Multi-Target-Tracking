from pathlib import Path

from PIL import Image, ImageDraw


directory = Path(__file__).resolve().parents[1] / "artifacts" / "reports" / "software_test" / "IT-03-04" / "20260925_run01" / "rendered_report_word"
files = sorted(directory.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[-1]))
output = directory / "contact_sheets"
output.mkdir(exist_ok=True)
thumb_width, thumb_height = 310, 438
columns, rows = 4, 3
for start in range(0, len(files), columns * rows):
    batch = files[start : start + columns * rows]
    canvas = Image.new("RGB", (columns * (thumb_width + 12) + 12, rows * (thumb_height + 32) + 12), "#d0d0d0")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(batch):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width, thumb_height))
        x = 12 + (index % columns) * (thumb_width + 12)
        y = 12 + (index // columns) * (thumb_height + 32)
        canvas.paste(image, (x, y))
        draw.text((x, y + thumb_height + 4), f"page {int(path.stem.split('-')[-1])}", fill="black")
    canvas.save(output / f"contact_{start + 1:03d}_{start + len(batch):03d}.png")
print({"pages": len(files), "contact_sheets": len(list(output.glob('*.png')))})
