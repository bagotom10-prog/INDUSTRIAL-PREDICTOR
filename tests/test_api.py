"""Tests de la API con el modelo real. Se saltan si no hay pesos entrenados."""

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inspector_industrial.api import crear_app
from inspector_industrial.inspeccion import PESOS_POR_DEFECTO, Inspector

RAIZ = Path(__file__).resolve().parent.parent
PESOS = RAIZ / PESOS_POR_DEFECTO
VAL = RAIZ / "datasets" / "NEU-DET" / "images" / "val"

pytestmark = pytest.mark.skipif(
    not PESOS.is_file() or not VAL.is_dir(),
    reason="Faltan los pesos entrenados o el dataset",
)


@pytest.fixture(scope="module")
def cliente():
  inspector = Inspector(
      pesos=PESOS, manual=RAIZ / "manual_criterios.yaml",
      manual_pdf=RAIZ / "manual" / "manual_calidad.pdf", diagnostico="reglas",
  )
  with TestClient(crear_app(inspector)) as c:
    yield c


def enviar(cliente, ruta, **params):
  with open(ruta, "rb") as f:
    return cliente.post("/inspecciones", params=params,
                        files={"imagen": (ruta.name, f, "image/jpeg")})


def test_salud(cliente):
  r = cliente.get("/salud")
  assert r.status_code == 200
  assert r.json()["diagnostico_modo"] == "reglas"


def test_inclusion_se_rechaza_con_diagnostico(cliente):
  r = enviar(cliente, VAL / "inclusion_241.jpg")
  assert r.status_code == 200
  cuerpo = r.json()
  assert cuerpo["estado"] == "RECHAZADA"
  assert any(d["clase"] == "inclusion" for d in cuerpo["defectos"])
  assert cuerpo["diagnostico"]["gravedad"] == "Alta"
  assert cuerpo["imagen_marcada_png_base64"] is None


def test_scratches_se_aprueba(cliente):
  cuerpo = enviar(cliente, VAL / "scratches_241.jpg").json()
  assert cuerpo["estado"] == "APROBADA"
  assert cuerpo["diagnostico"]["gravedad"] == "Baja"


def test_devuelve_imagen_marcada(cliente):
  cuerpo = enviar(cliente, VAL / "patches_241.jpg", devolver_imagen=True).json()
  png = base64.b64decode(cuerpo["imagen_marcada_png_base64"])
  assert png.startswith(b"\x89PNG")


def test_rechaza_archivos_que_no_son_imagen(cliente):
  r = cliente.post("/inspecciones",
                   files={"imagen": ("a.txt", b"hola", "text/plain")})
  assert r.status_code == 415


def test_rechaza_imagen_corrupta(cliente):
  r = cliente.post("/inspecciones",
                   files={"imagen": ("a.jpg", b"no soy un jpg", "image/jpeg")})
  assert r.status_code == 422


def test_buscar_en_manual(cliente):
  r = cliente.get("/manual/buscar", params={"q": "cascarilla", "k": 1})
  assert r.json()[0]["seccion"].startswith("Sección 7.")
