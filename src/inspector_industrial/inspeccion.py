"""Detecta defectos con YOLO, aplica el manual y genera reporte JSON + HTML."""

import json
from datetime import datetime
from pathlib import Path

from . import criterios
from .informe import generar_html

# Modelo aprobado para producción. Cámbialo solo tras comparar métricas.
PESOS_POR_DEFECTO = "runs/detect/n_640/weights/best.pt"
FUENTE_POR_DEFECTO = "datasets/NEU-DET/images/val"
SALIDA_POR_DEFECTO = "runs/detect/inspeccion_real"
MANUAL_POR_DEFECTO = "manual_criterios.yaml"


def inspeccionar(pesos=PESOS_POR_DEFECTO, fuente=FUENTE_POR_DEFECTO,
                 salida=SALIDA_POR_DEFECTO, manual=MANUAL_POR_DEFECTO,
                 conf=None) -> Path:
  """Inspecciona `fuente` y devuelve la ruta del reporte JSON.

  conf: si se indica, sustituye el conf_deteccion de todas las clases.
  """
  from ultralytics import YOLO

  ruta_pesos = Path(pesos)
  if not ruta_pesos.is_file():
    raise FileNotFoundError(
        f"No existe el modelo {ruta_pesos}. Indica otro con --pesos o"
        " entrena uno primero."
    )
  reglas = criterios.cargar_manual(manual)
  if conf is not None:
    for regla in reglas["clases"].values():
      regla["conf_deteccion"] = conf

  modelo = YOLO(ruta_pesos)
  criterios.comprobar_clases(reglas, modelo.names.values())
  umbral = {n: reglas["clases"][n]["conf_deteccion"]
            for n in modelo.names.values()}

  # Ruta absoluta: con una relativa, Ultralytics anidaría la salida en
  # runs/detect/runs/detect/...
  dir_salida = Path(salida).resolve()
  dir_salida.mkdir(parents=True, exist_ok=True)

  # Predice con el umbral más bajo del manual y filtra después por clase.
  # stream=True procesa imagen a imagen sin acumular todo en memoria.
  resultados = modelo.predict(
      source=fuente, conf=min(umbral.values()), stream=True, verbose=False
  )

  piezas = []
  for res in resultados:
    validas = [i for i, c in enumerate(res.boxes.cls.tolist())
               if res.boxes.conf[i] >= umbral[res.names[int(c)]]]
    res = res[validas] if validas else res[[]]
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

    nombre = Path(res.path).name
    res.save(filename=str(dir_salida / nombre))  # imagen con cajas dibujadas
    piezas.append({
        "imagen": nombre,
        **criterios.evaluar_pieza(defectos, reglas),
        "num_defectos": len(defectos),
        "defectos": defectos,
    })

  estados = [p["estado"] for p in piezas]
  reporte = {
      "fecha": datetime.now().isoformat(timespec="seconds"),
      "modelo": str(ruta_pesos),
      "fuente": str(fuente),
      "manual": {"ruta": str(manual), **reglas},
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
  print(f"Piezas: {r['total_piezas']} | Aprobadas: {r['aprobadas']} | "
        f"Revisión: {r['revision']} | Rechazadas: {r['rechazadas']}")
  return ruta_json
