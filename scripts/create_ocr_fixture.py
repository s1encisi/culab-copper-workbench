"""Create the deterministic synthetic scan used by verify_knowledge_ocr.py."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True, help="Locally available Chinese TrueType font")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "runs/dependencies/knowledge-v1"))
    import pypdfium2 as pdfium
    import reportlab
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(TTFont("TestChinese", str(args.font)))
    vector = args.output / "vector_source.pdf"
    c = canvas.Canvas(str(vector), pagesize=A4, invariant=1)
    c.setFont("TestChinese", 21)
    c.drawString(48, 785, "合成 OCR 验证页")
    c.setFont("TestChinese", 12)
    c.drawString(48, 752, "仅用于软件测试，数值不代表工厂设置。")
    c.setFont("TestChinese", 15)
    c.drawString(48, 710, "一、浓度记录")
    xs, ys = [48, 232, 380, 540], [680, 640, 600, 560]
    for x in xs:
        c.line(x, ys[-1], x, ys[0])
    for y in ys:
        c.line(xs[0], y, xs[-1], y)
    rows = [["字段", "合成值", "单位"], ["铜浓度", "12.5", "g/L"], ["砷浓度", "24.0", "mg/L"]]
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            c.drawString(xs[j] + 12, ys[i] - 26, value)
    c.setFont("TestChinese", 11)
    c.drawString(48, 540, "表注：所有数值均为合成测试样例。")
    c.setFont("TestChinese", 15)
    c.drawString(48, 495, "二、表达式及符号")
    c.setFont("TestChinese", 18)
    c.drawString(62, 457, "c = m / V")
    c.setFont("TestChinese", 12)
    c.drawString(48, 425, "c 为质量浓度，m 为溶质质量，V 为溶液体积。")
    c.drawString(48, 385, "识别结果需要核对表头、单位与符号后使用。")
    c.showPage()
    c.save()
    document = pdfium.PdfDocument(str(vector))
    page = document[0]
    bitmap = page.render(scale=2.5)
    try:
        image = bitmap.to_pil().convert("RGB")
        image.save(args.output / "source_page.png")
        image.save(args.output / "scan.pdf", resolution=180)
        image.close()
    finally:
        bitmap.close()
        page.close()
        document.close()
    print(
        json.dumps(
            {
                "fixture": str(args.output / "scan.pdf"),
                "reportlab": reportlab.Version,
                "scope": "synthetic fixture; no industrial setpoints",
            }
        )
    )


if __name__ == "__main__":
    main()
