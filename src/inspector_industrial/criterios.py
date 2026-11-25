"""Carga el manual de criterios y decide el estado de cada pieza.

Este módulo no depende de YOLO: recibe defectos ya detectados y aplica reglas
deterministas, para que cada decisión sea reproducible y explicable.
"""

from pathlib import Path

import yaml

CRITICO = "CRITICO"
REVISAR = "REVISAR"
NO_CRITICO = "NO CRITICO"
GRAVEDADES = (CRITICO, REVISAR, NO_CRITICO)

APROBADA = "APROBADA"
REVISION = "REVISION"
RECHAZADA = "RECHAZADA"


class ManualInvalido(ValueError):
  pass


def cargar_manual(ruta: str | Path) -> dict:
  manual = yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))
  validar_manual(manual)
  return manual


def validar_manual(manual: dict) -> None:
  for clave in ("max_box_size", "min_confidence", "clases"):
    if clave not in manual:
      raise ManualInvalido(f"Falta la clave '{clave}' en el manual.")
  for clave in ("max_box_size", "min_confidence"):
    if not 0 < manual[clave] <= 1:
      raise ManualInvalido(f"'{clave}' debe estar entre 0 y 1.")
  for nombre, regla in manual["clases"].items():
    if regla.get("gravedad") not in GRAVEDADES:
      raise ManualInvalido(
          f"Clase '{nombre}': gravedad debe ser una de {GRAVEDADES}."
      )
    if not 0 < regla.get("conf_deteccion", -1) <= 1:
      raise ManualInvalido(
          f"Clase '{nombre}': conf_deteccion debe estar entre 0 y 1."
      )


def comprobar_clases(manual: dict, clases_modelo) -> None:
  """Exige que el manual tenga una regla para cada clase del modelo."""
  faltan = sorted(set(clases_modelo) - set(manual["clases"]))
  if faltan:
    raise ManualInvalido(f"El manual no tiene reglas para: {faltan}")


def dictaminar_defecto(defecto: dict, manual: dict) -> tuple[str, str]:
  """Devuelve (dictamen, motivo). Dictamen: rechazo, revision o tolerado."""
  clase = defecto["clase"]
  gravedad = manual["clases"][clase]["gravedad"]
  area = defecto["area_relativa"]
  max_area = manual["max_box_size"]

  if gravedad == CRITICO:
    dictamen, motivo = "rechazo", f"{clase} es un defecto crítico"
  elif gravedad == REVISAR:
    return "revision", f"{clase} requiere revisión manual"
  elif area > max_area:
    dictamen, motivo = (
        "rechazo", f"{clase} ocupa {area:.0%} de la pieza (máx. {max_area:.0%})"
    )
  else:
    return "tolerado", f"{clase} dentro de tolerancia ({area:.0%})"

  if defecto["confianza"] < manual["min_confidence"]:
    return "revision", (
        f"{motivo}, pero con confianza baja ({defecto['confianza']:.2f})"
    )
  return dictamen, motivo


def evaluar_pieza(defectos: list[dict], manual: dict) -> dict:
  """Añade 'dictamen' a cada defecto y devuelve estado y motivos de la pieza."""
  for d in defectos:
    d["dictamen"], d["motivo"] = dictaminar_defecto(d, manual)

  dictamenes = {d["dictamen"] for d in defectos}
  if "rechazo" in dictamenes:
    estado = RECHAZADA
  elif "revision" in dictamenes:
    estado = REVISION
  else:
    estado = APROBADA

  # Los motivos de la pieza son los de los defectos que fijan su estado.
  decisivo = {RECHAZADA: "rechazo", REVISION: "revision",
              APROBADA: "tolerado"}[estado]
  motivos = list(dict.fromkeys(
      d["motivo"] for d in defectos if d["dictamen"] == decisivo
  ))
  if not defectos:
    motivos = ["Sin defectos detectados"]
  return {"estado": estado, "motivos": motivos}
