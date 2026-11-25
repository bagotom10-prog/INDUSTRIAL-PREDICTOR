"""Estación 3: diagnóstico estructurado de cada pieza.

Recibe la pieza ya evaluada por `criterios` (el estado APROBADA / REVISION /
RECHAZADA es definitivo) y las secciones del manual que recupera
`BuscadorManual`, y redacta: defecto, gravedad, acción, plazo, justificación y
las secciones del manual en que se basa.

Dos modos:
  - "llm": Claude redacta el diagnóstico. Necesita credenciales de Anthropic
    (variable ANTHROPIC_API_KEY). El LLM no puede cambiar el estado de la pieza.
  - "reglas": copia la acción y el plazo de las secciones del manual. No
    necesita red ni clave; es el respaldo si el LLM falla.
"""

import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from .criterios import APROBADA, CRITICO, NO_CRITICO, REVISAR
from .manual import BuscadorManual, Seccion

MODELO_LLM = os.environ.get("INSPECTOR_MODELO_LLM", "claude-opus-5-5")
MODOS = ("auto", "llm", "reglas", "ninguno")

GRAVEDAD_TEXTO = {CRITICO: "Alta", REVISAR: "Media", NO_CRITICO: "Baja"}
_ORDEN_GRAVEDAD = {CRITICO: 0, REVISAR: 1, NO_CRITICO: 2}


class Diagnostico(BaseModel):
  defecto: str = Field(description="Nombre del defecto principal en español")
  gravedad: Literal["Alta", "Media", "Baja"]
  accion: str = Field(description="Qué hacer con la pieza y con la línea")
  plazo: str
  justificacion: str = Field(description="Por qué, citando el manual")
  referencias: list[str] = Field(description="Secciones del manual usadas")


ESQUEMA = {
    "type": "object",
    "properties": {
        "defecto": {"type": "string"},
        "gravedad": {"type": "string", "enum": ["Alta", "Media", "Baja"]},
        "accion": {"type": "string"},
        "plazo": {"type": "string"},
        "justificacion": {"type": "string"},
        "referencias": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["defecto", "gravedad", "accion", "plazo", "justificacion",
                 "referencias"],
    "additionalProperties": False,
}

SISTEMA = """Eres el asistente de calidad de una línea de laminado de acero.
Recibes una pieza ya inspeccionada por visión artificial y las secciones
relevantes del manual de calidad de la planta. Redacta su diagnóstico en
español.

Reglas:
- El estado de la pieza (APROBADA, REVISION o RECHAZADA) ya está decidido por
  el sistema de criterios. No lo cambies ni lo cuestiones; explica la acción
  coherente con ese estado.
- Basa la acción y el plazo solo en el manual proporcionado. Si el manual no
  cubre algo, dilo en la justificación en lugar de inventarlo.
- La gravedad es Alta para defectos CRITICO, Media para REVISAR y Baja para
  NO CRITICO; usa la del defecto más grave de la pieza.
- En referencias, copia exactamente el encabezado de cada sección usada
  (por ejemplo "Sección 4. Grietas finas en red (crazing)").
- Sé conciso: una o dos frases por campo."""


def modo_efectivo(modo: str) -> str:
  if modo not in MODOS:
    raise ValueError(f"Modo de diagnóstico desconocido: {modo}. Usa {MODOS}.")
  if modo == "auto":
    hay_clave = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "ANTHROPIC_AUTH_TOKEN")
    return "llm" if hay_clave else "reglas"
  return modo


def _clases_por_gravedad(pieza: dict, manual: dict) -> list[str]:
  clases = {d["clase"] for d in pieza["defectos"]}
  return sorted(clases, key=lambda c: _ORDEN_GRAVEDAD[
      manual["clases"][c]["gravedad"]])


def recuperar_secciones(pieza: dict, manual: dict,
                        buscador: BuscadorManual) -> list[Seccion]:
  """Busca en el manual la sección de cada defecto y, si la pieza va a
  revisión, la del procedimiento de revisión manual."""
  consultas = [f"{c} {manual['clases'][c].get('descripcion', '')}"
               for c in _clases_por_gravedad(pieza, manual)]
  if pieza["estado"] == "REVISION":
    consultas.append("revisión manual inspector cuarentena")
  secciones = []
  for consulta in consultas:
    for seccion, _ in buscador.buscar(consulta, k=1):
      if seccion not in secciones:
        secciones.append(seccion)
  return secciones


_DICTAMEN_DECISIVO = {"RECHAZADA": "rechazo", "REVISION": "revision",
                      "APROBADA": "tolerado"}


def defecto_principal(pieza: dict, manual: dict) -> str:
  """Clase más grave entre los defectos que fijan el estado de la pieza."""
  decisivo = _DICTAMEN_DECISIVO[pieza["estado"]]
  clases = {d["clase"] for d in pieza["defectos"] if d["dictamen"] == decisivo}
  return min(clases, key=lambda c: _ORDEN_GRAVEDAD[
      manual["clases"][c]["gravedad"]])


def diagnostico_por_reglas(pieza: dict, manual: dict,
                           secciones: list[Seccion]) -> Diagnostico:
  principal = defecto_principal(pieza, manual)
  gravedad_manual = manual["clases"][principal]["gravedad"]
  gravedad = GRAVEDAD_TEXTO[gravedad_manual]
  seccion = next((s for s in secciones if f"({principal})" in s.titulo), None)
  revision = next((s for s in secciones if "Revisión manual" in s.titulo), None)
  referencias = [s.referencia for s in secciones]
  justificacion = f"Pieza {pieza['estado']}: " + "; ".join(pieza["motivos"]) + "."
  if seccion is None:
    return Diagnostico(
        defecto=principal, gravedad=gravedad,
        accion="Consultar al responsable de calidad.",
        plazo="Sin plazo definido en el manual.",
        justificacion=f"{justificacion} El manual no tiene una sección para "
                      f"{principal}.",
        referencias=referencias,
    )

  accion = seccion.campo("Acción") or "Ver manual."
  plazo = seccion.campo("Plazo") or "Ver manual."
  if pieza["estado"] == "REVISION" and gravedad_manual == CRITICO and revision:
    # Defecto crítico detectado con poca confianza: primero, que lo confirme
    # un inspector; la acción de rechazo solo aplica si se confirma.
    accion = (f"{revision.campo('Acción')} Si el inspector confirma el "
              f"defecto: {accion}")
    plazo = revision.campo("Plazo") or plazo
  elif pieza["estado"] == "RECHAZADA" and gravedad_manual == NO_CRITICO:
    # Defecto tolerable en general, pero fuera del tamaño máximo admitido.
    accion = f"Rechazar la pieza: {'; '.join(pieza['motivos'])}."
    plazo = "El manual no fija plazo para este caso."

  criterio = seccion.campo("Criterio de aceptación")
  return Diagnostico(
      defecto=seccion.titulo,
      gravedad=gravedad,
      accion=accion,
      plazo=plazo,
      justificacion=f"{justificacion} {criterio}" if criterio else justificacion,
      referencias=referencias,
  )


def _mensaje_usuario(pieza: dict, secciones: list[Seccion]) -> str:
  defectos = [
      {k: d[k] for k in ("clase", "confianza", "area_relativa", "dictamen",
                         "motivo")}
      for d in pieza["defectos"]
  ]
  extractos = "\n\n".join(f"{s.referencia}\n{s.texto}" for s in secciones)
  return (
      f"<pieza>\nImagen: {pieza['imagen']}\n"
      f"Estado decidido: {pieza['estado']}\n"
      f"Motivos: {'; '.join(pieza['motivos'])}\n"
      f"Defectos detectados: {json.dumps(defectos, ensure_ascii=False)}\n"
      f"</pieza>\n\n<manual>\n{extractos}\n</manual>"
  )


def diagnostico_por_llm(pieza: dict, secciones: list[Seccion],
                        cliente=None) -> Diagnostico:
  import anthropic

  cliente = cliente or anthropic.Anthropic()
  respuesta = cliente.beta.messages.create(
      model=MODELO_LLM,
      max_tokens=16000,
      system=SISTEMA,
      messages=[{"role": "user", "content": _mensaje_usuario(pieza, secciones)}],
      output_config={
          "effort": "low",
          "format": {"type": "json_schema", "schema": ESQUEMA},
      },
      # Si el modelo rechaza la petición, el servidor la reintenta con otro.
      betas=["server-side-fallback-2026-07-01"],
      fallbacks="default",
  )
  if respuesta.stop_reason == "refusal":
    raise RuntimeError("El modelo rechazó la petición.")
  if respuesta.stop_reason == "max_tokens":
    raise RuntimeError("Respuesta truncada por max_tokens.")
  texto = next(b.text for b in respuesta.content if b.type == "text")
  return Diagnostico.model_validate_json(texto)


class Diagnosticador:
  """Diagnostica piezas reutilizando resultados de piezas equivalentes.

  Dos piezas con las mismas clases, dictámenes y estado reciben el mismo
  diagnóstico, así que en un lote de cientos de piezas el LLM solo se llama
  una vez por combinación distinta.
  """

  def __init__(self, manual: dict, buscador: BuscadorManual | None,
               modo: str = "auto", cliente=None):
    self.manual = manual
    self.buscador = buscador
    self.modo = modo_efectivo(modo)
    self.cliente = cliente
    self._cache: dict[tuple, dict] = {}
    self.llamadas_llm = 0

  def diagnosticar(self, pieza: dict) -> dict | None:
    if self.modo == "ninguno" or self.buscador is None:
      return None
    if pieza["estado"] == APROBADA and not pieza["defectos"]:
      return None  # Pieza limpia: no hay nada que diagnosticar.

    clave = (pieza["estado"], tuple(sorted(
        {(d["clase"], d["dictamen"]) for d in pieza["defectos"]})))
    if clave in self._cache:
      return self._cache[clave]

    secciones = recuperar_secciones(pieza, self.manual, self.buscador)
    origen = "reglas"
    if self.modo == "llm":
      try:
        self.llamadas_llm += 1
        diag = diagnostico_por_llm(pieza, secciones, self.cliente)
        origen = f"llm ({MODELO_LLM})"
      except Exception as error:  # noqa: BLE001
        # Cualquier fallo del LLM (red, credenciales, rechazo, JSON inválido)
        # no debe parar la inspección: se usa el diagnóstico por reglas.
        diag = diagnostico_por_reglas(pieza, self.manual, secciones)
        origen = f"reglas (el LLM falló: {type(error).__name__})"
    else:
      diag = diagnostico_por_reglas(pieza, self.manual, secciones)

    resultado = {"origen": origen, **diag.model_dump()}
    self._cache[clave] = resultado
    return resultado
