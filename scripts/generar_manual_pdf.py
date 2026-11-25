"""Genera manual/manual_calidad.pdf a partir de manual/manual_calidad.md.

Solo entiende el Markdown que usa el manual: '# título', '## sección' y
párrafos. Uso: uv run python scripts/generar_manual_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

RAIZ = Path(__file__).resolve().parent.parent
ORIGEN = RAIZ / "manual" / "manual_calidad.md"
DESTINO = RAIZ / "manual" / "manual_calidad.pdf"


def main() -> None:
  estilos = getSampleStyleSheet()
  elementos = []
  for bloque in ORIGEN.read_text(encoding="utf-8").split("\n\n"):
    bloque = bloque.strip()
    if bloque.startswith("## "):
      elementos += [Spacer(1, 0.3 * cm), Paragraph(bloque[3:], estilos["Heading2"])]
    elif bloque.startswith("# "):
      elementos.append(Paragraph(bloque[2:], estilos["Title"]))
    elif bloque:
      elementos.append(Paragraph(bloque, estilos["BodyText"]))

  SimpleDocTemplate(
      str(DESTINO), pagesize=A4, title="Manual de calidad (ejemplo)",
      leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
  ).build(elementos)
  print(f"PDF generado: {DESTINO}")


if __name__ == "__main__":
  main()
