"""Estación 4: servicio web con FastAPI.

Arranque: uv run inspector-industrial servir   (documentación en /docs)

Configuración por variables de entorno (pensado para Docker):
  INSPECTOR_PESOS, INSPECTOR_CRITERIOS, INSPECTOR_MANUAL_PDF,
  INSPECTOR_DIAGNOSTICO (auto | llm | reglas | ninguno),
  ANTHROPIC_API_KEY (opcional, activa el diagnóstico con LLM).
"""

import base64
import io
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .diagnostico import Diagnostico
from .inspeccion import Inspector

MAX_BYTES = 10 * 1024 * 1024
TIPOS_IMAGEN = {"image/jpeg", "image/png", "image/bmp", "image/tiff",
                "image/webp"}


class Defecto(BaseModel):
  clase: str
  confianza: float
  area_relativa: float
  caja_xyxy: list[float]
  dictamen: Literal["rechazo", "revision", "tolerado"]
  motivo: str


class DiagnosticoRespuesta(Diagnostico):
  origen: str


class Inspeccion(BaseModel):
  imagen: str
  estado: Literal["APROBADA", "REVISION", "RECHAZADA"]
  motivos: list[str]
  num_defectos: int
  defectos: list[Defecto]
  diagnostico: DiagnosticoRespuesta | None
  imagen_marcada_png_base64: str | None = None


def crear_app(inspector: Inspector | None = None) -> FastAPI:
  """Crea la aplicación. Los tests pueden inyectar su propio `inspector`."""

  @asynccontextmanager
  async def ciclo_de_vida(app: FastAPI):
    # El modelo se carga una vez al arrancar, no en cada petición.
    app.state.inspector = inspector or Inspector(
        diagnostico=os.environ.get("INSPECTOR_DIAGNOSTICO", "auto"))
    yield

  app = FastAPI(
      title="Inspector industrial",
      description=(
          "Detecta defectos superficiales en piezas de acero (YOLO11), decide "
          "APROBADA / REVISION / RECHAZADA según el manual de criterios y "
          "redacta un diagnóstico apoyado en el manual de calidad."
      ),
      version="0.2.0",
      lifespan=ciclo_de_vida,
  )
  cerrojo = threading.Lock()

  @app.get("/", include_in_schema=False)
  def raiz():
    return RedirectResponse("/docs")

  @app.get("/salud", summary="Estado del servicio y configuración cargada")
  def salud():
    return {"estado": "ok", **app.state.inspector.info()}

  @app.post("/inspecciones", response_model=Inspeccion,
            summary="Inspecciona la imagen de una pieza")
  def inspeccionar_imagen(
      imagen: UploadFile = File(description="Foto de la pieza (JPG, PNG...)"),
      devolver_imagen: bool = Query(
          False, description="Incluir la imagen con los defectos marcados"),
  ):
    # Función síncrona a propósito: FastAPI la ejecuta en un hilo aparte y
    # la inferencia no bloquea el servidor.
    if imagen.content_type not in TIPOS_IMAGEN:
      raise HTTPException(415, f"Tipo no admitido: {imagen.content_type}")
    datos = imagen.file.read(MAX_BYTES + 1)
    if len(datos) > MAX_BYTES:
      raise HTTPException(413, "La imagen supera los 10 MB.")
    try:
      foto = Image.open(io.BytesIO(datos)).convert("RGB")
    except UnidentifiedImageError:
      raise HTTPException(422, "El archivo no es una imagen válida.")

    insp = app.state.inspector
    # El predictor de Ultralytics no es seguro entre hilos: una a una.
    with cerrojo:
      res = insp.predecir(foto)[0]
      pieza, filtrado = insp.analizar(res, nombre=imagen.filename or "imagen")

    if devolver_imagen:
      marcada = Image.fromarray(filtrado.plot()[..., ::-1])  # BGR -> RGB
      buffer = io.BytesIO()
      marcada.save(buffer, format="PNG")
      pieza["imagen_marcada_png_base64"] = base64.b64encode(
          buffer.getvalue()).decode()
    return pieza

  @app.get("/manual/buscar", summary="Busca en el manual de calidad (RAG)")
  def buscar_en_manual(q: str = Query(min_length=2), k: int = Query(3, ge=1, le=10)):
    buscador = app.state.inspector.diagnosticador.buscador
    if buscador is None:
      raise HTTPException(503, "No hay manual PDF cargado.")
    return [{"seccion": s.referencia, "puntuacion": p, "texto": s.texto}
            for s, p in buscador.buscar(q, k)]

  return app


def servir(host: str = "0.0.0.0", puerto: int = 8000) -> None:
  import uvicorn
  uvicorn.run(crear_app(), host=host, port=puerto)
