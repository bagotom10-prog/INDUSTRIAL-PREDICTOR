"""Inspector industrial: detección de defectos en acero con YOLO.

Uso: uv run inspector-industrial <comando> [opciones]
  preparar | entrenar | inspeccionar | informe | buscar | servir
"""

import argparse

from .diagnostico import MODOS
from .inspeccion import (
    FUENTE_POR_DEFECTO,
    MANUAL_PDF,
    MANUAL_POR_DEFECTO,
    PESOS_POR_DEFECTO,
    SALIDA_POR_DEFECTO,
)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="inspector-industrial",
      description="Detección de defectos superficiales en piezas de acero.",
  )
  sub = parser.add_subparsers(dest="comando", required=True)

  sub.add_parser("preparar", help="Descarga NEU-DET y lo convierte a YOLO")

  p = sub.add_parser("entrenar", help="Entrena el modelo YOLO")
  p.add_argument("--modelo", default="yolo11n.pt",
                 help="Modelo base (por defecto: yolo11n.pt)")
  p.add_argument("--epocas", type=int, default=50)
  p.add_argument("--imgsz", type=int, default=640)
  p.add_argument("--nombre", default="n_640",
                 help="Carpeta del resultado en runs/detect/")
  p.add_argument("--workers", type=int, default=2,
                 help="Procesos de carga; súbelo solo si sobra RAM")
  p.add_argument("--batch", type=int, default=8)

  p = sub.add_parser("inspeccionar",
                     help="Inspecciona imágenes y genera JSON + informe HTML")
  p.add_argument("--pesos", default=PESOS_POR_DEFECTO,
                 help=f"Modelo .pt (por defecto: {PESOS_POR_DEFECTO})")
  p.add_argument("--fuente", default=FUENTE_POR_DEFECTO,
                 help="Imagen o carpeta a inspeccionar")
  p.add_argument("--salida", default=SALIDA_POR_DEFECTO,
                 help=f"Carpeta de resultados (por defecto: {SALIDA_POR_DEFECTO})")
  p.add_argument("--manual", default=MANUAL_POR_DEFECTO,
                 help="Manual de criterios YAML")
  p.add_argument("--manual-pdf", default=MANUAL_PDF,
                 help="Manual de calidad (PDF o Markdown) para el diagnóstico")
  p.add_argument("--diagnostico", choices=MODOS, default="auto",
                 help="auto: LLM si hay ANTHROPIC_API_KEY, si no reglas")
  p.add_argument("--conf", type=float, default=None,
                 help="Umbral único para todas las clases (ignora los del manual)")

  p = sub.add_parser("informe", help="Regenera el informe HTML desde un JSON")
  p.add_argument("--reporte",
                 default=f"{SALIDA_POR_DEFECTO}/reporte_inspeccion.json")

  p = sub.add_parser("buscar", help="Busca en el manual de calidad")
  p.add_argument("consulta")
  p.add_argument("--manual-pdf", default=MANUAL_PDF)
  p.add_argument("-k", type=int, default=3)

  p = sub.add_parser("servir", help="Arranca la API web (FastAPI)")
  p.add_argument("--host", default="0.0.0.0")
  p.add_argument("--puerto", type=int, default=8000)
  return parser


def main(argv=None) -> None:
  args = _parser().parse_args(argv)

  if args.comando == "preparar":
    from .datos import preparar
    preparar()
  elif args.comando == "entrenar":
    from .entrenamiento import entrenar
    entrenar(args.modelo, args.epocas, args.imgsz, args.nombre,
             args.workers, args.batch)
  elif args.comando == "inspeccionar":
    from .inspeccion import inspeccionar
    inspeccionar(args.pesos, args.fuente, args.salida, args.manual, args.conf,
                 args.manual_pdf, args.diagnostico)
  elif args.comando == "informe":
    from .informe import generar_desde_json
    print(f"Informe HTML: {generar_desde_json(args.reporte)}")
  elif args.comando == "buscar":
    from .manual import BuscadorManual
    for seccion, puntos in BuscadorManual.desde_archivo(
        args.manual_pdf).buscar(args.consulta, args.k):
      print(f"[{puntos:.2f}] {seccion.referencia}\n{seccion.texto}\n")
  elif args.comando == "servir":
    from .api import servir
    servir(args.host, args.puerto)


if __name__ == "__main__":
  main()
