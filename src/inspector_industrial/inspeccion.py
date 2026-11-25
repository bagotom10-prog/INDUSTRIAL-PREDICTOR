"""Estación 1 y orquestación: detecta defectos con YOLO, aplica el manual de
criterios, busca en el manual de calidad y diagnostica cada pieza.

`Inspector` carga todo una vez y analiza imágenes sueltas (lo usa la API).
`inspeccionar` recorre una carpeta y escribe el reporte JSON y el informe HTML.

Las rutas por defecto se pueden cambiar con variables de entorno (útil en
Docker): INSPECTOR_PESOS, INSPECTOR_CRITERIOS, INSPECTOR_MANUAL_PDF.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from . import criterios
from .diagnostico import Diagnosticador
from .informe import generar_html
from .manual import MANUAL_PDF_POR_DEFECTO, BuscadorManual

# Modelo aprobado para producción. Cámbialo solo tras comparar métricas.
PESOS_POR_DEFECTO = os.environ.get(
    "INSPECTOR_PESOS", "runs/detect/n_640/weights/best.pt")
MANUAL_POR_DEFECTO = os.environ.get(
    "INSPECTOR_CRITERIOS", "manual_criterios.yaml")
MANUAL_PDF = os.environ.get("INSPECTOR_MANUAL_PDF", MANUAL_PDF_POR_DEFECTO)
FUENTE_POR_DEFECTO = "datasets/NEU-DET/images/val"
SALIDA_POR_DEFECTO = "runs/detect/inspeccion_real"


class Inspector:
  def __init__(self, pesos=PESOS_POR_DEFECTO, manual=MANUAL_POR_DEFECTO,
               manual_pdf=MANUAL_PDF, conf=None, diagnostico="auto"):
    from ultralytics import YOLO

    self.ruta_pesos = Path(pesos)
    if not self.ruta_pesos.is_file():
      raise FileNotFoundError(
          f"No existe el modelo {self.ruta_pesos}. Indica otro con --pesos"
          " (o INSPECTOR_PESOS) o entrena uno primero."
      )
    self.ruta_manual = str(manual)
    self.reglas = criterios.cargar_manual(manual)
    if conf is not None:
      for regla in self.reglas["clases"].values():
        regla["conf_deteccion"] = conf

    self.modelo = YOLO(self.ruta_pesos)
    criterios.comprobar_clases(self.reglas, self.modelo.names.values())
    self.umbral = {n: self.reglas["clases"][n]["conf_deteccion"]
                   for n in self.modelo.names.values()}

    self.ruta_manual_pdf = manual_pdf
    buscador = (BuscadorManual.desde_archivo(manual_pdf)
                if manual_pdf and Path(manual_pdf).is_file() else None)
    self.diagnosticador = Diagnosticador(self.reglas, buscador, diagnostico)

  def predecir(self, fuente, **kwargs):
    """Predice con el umbral más bajo del manual; `analizar` filtra después."""
    return self.modelo.predict(
        source=fuente, conf=min(self.umbral.values()), verbose=False, **kwargs
    )

  def analizar(self, res, nombre: str | None = None):
    """Convierte un resultado de YOLO en una pieza evaluada y diagnosticada.

    Devuelve (pieza, resultado_filtrado); el segundo sirve para dibujar.
    """
    validas = [i for i, c in enumerate(res.boxes.cls.tolist())
               if res.boxes.conf[i] >= self.umbral[res.names[int(c)]]]
    res = res[validas]
    alto, ancho = res.orig_shape

    defectos = []
    for caja in res.boxes:
      x1, y1, x2, y2 = caja.xyxy[0].tolist()
      defectos.append({
          "clase": res.names[int(caja.cls.item())],
          "confianza": round(float(caja.conf.item()), 4),
          "area_relativa": round((x2 - x1) * (y2 - y1) / (ancho * alto), 4),
          "caja_xyxy": [round(v, 1) for v in (x1, y1, x2, y2)],
      })
    defectos.sort(key=lambda d: d["confianza"], reverse=True)

    pieza = {
        "imagen": nombre or Path(res.path).name,
        **criterios.evaluar_pieza(defectos, self.reglas),
        "num_defectos": len(defectos),
        "defectos": defectos,
    }
    pieza["diagnostico"] = self.diagnosticador.diagnosticar(pieza)
    return pieza, res

  def info(self) -> dict:
    return {
        "modelo": str(self.ruta_pesos),
        "manual_criterios": {"ruta": self.ruta_manual, **self.reglas},
        "manual_pdf": str(self.ruta_manual_pdf),
        "diagnostico_modo": self.diagnosticador.modo,
    }


def inspeccionar(pesos=PESOS_POR_DEFECTO, fuente=FUENTE_POR_DEFECTO,
                 salida=SALIDA_POR_DEFECTO, manual=MANUAL_POR_DEFECTO,
                 conf=None, manual_pdf=MANUAL_PDF,
                 diagnostico="auto") -> Path:
  """Inspecciona `fuente` y devuelve la ruta del reporte JSON."""
  inspector = Inspector(pesos, manual, manual_pdf, conf, diagnostico)

  # Ruta absoluta: con una relativa, Ultralytics anidaría la salida en
  # runs/detect/runs/detect/...
  dir_salida = Path(salida).resolve()
  dir_salida.mkdir(parents=True, exist_ok=True)

  piezas = []
  # stream=True procesa imagen a imagen sin acumular todo en memoria.
  for res in inspector.predecir(fuente, stream=True):
    pieza, filtrado = inspector.analizar(res)
    filtrado.save(filename=str(dir_salida / pieza["imagen"]))
    piezas.append(pieza)

  info = inspector.info()
  estados = [p["estado"] for p in piezas]
  reporte = {
      "fecha": datetime.now().isoformat(timespec="seconds"),
      "modelo": info["modelo"],
      "fuente": str(fuente),
      "manual": info["manual_criterios"],
      "manual_pdf": info["manual_pdf"],
      "diagnostico_modo": info["diagnostico_modo"],
      "resumen": {
          "total_piezas": len(piezas),
          "aprobadas": estados.count(criterios.APROBADA),
          "revision": estados.count(criterios.REVISION),
          "rechazadas": estados.count(criterios.RECHAZADA),
      },
      "piezas": piezas,
  }

  ruta_json = dir_salida / "reporte_inspeccion.json"
  ruta_json.write_text(
      json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  ruta_html = generar_html(reporte, dir_salida / "informe_inspeccion.html")

  r = reporte["resumen"]
  print(f"Reporte JSON: {ruta_json}")
  print(f"Informe HTML: {ruta_html}")
  print(f"Diagnóstico: {reporte['diagnostico_modo']} "
        f"({inspector.diagnosticador.llamadas_llm} llamadas al LLM)")
  print(f"Piezas: {r['total_piezas']} | Aprobadas: {r['aprobadas']} | "
        f"Revisión: {r['revision']} | Rechazadas: {r['rechazadas']}")
  return ruta_json
