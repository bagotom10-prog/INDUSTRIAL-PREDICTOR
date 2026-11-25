from pathlib import Path

import pytest

from inspector_industrial import criterios
from inspector_industrial.datos import CLASES
from inspector_industrial.manual import BuscadorManual, dividir_secciones

RAIZ = Path(__file__).resolve().parent.parent
PDF = RAIZ / "manual" / "manual_calidad.pdf"
MD = RAIZ / "manual" / "manual_calidad.md"


@pytest.fixture(scope="module")
def buscador():
  return BuscadorManual.desde_archivo(PDF)


def test_divide_en_secciones_numeradas():
  secciones = dividir_secciones("1. Alcance\nTexto uno.\n2. Otra\nTexto dos.")
  assert [(s.numero, s.titulo, s.texto) for s in secciones] == [
      (1, "Alcance", "Texto uno."), (2, "Otra", "Texto dos.")]


def test_pdf_y_markdown_dan_las_mismas_secciones():
  pdf = BuscadorManual.desde_archivo(PDF).secciones
  md = BuscadorManual.desde_archivo(MD).secciones
  assert [s.referencia for s in pdf] == [s.referencia for s in md]


@pytest.mark.parametrize("clase", CLASES)
def test_cada_clase_encuentra_su_seccion(buscador, clase):
  seccion, _ = buscador.buscar(clase, k=1)[0]
  assert f"({clase})" in seccion.titulo


def test_consulta_en_lenguaje_natural(buscador):
  seccion, _ = buscador.buscar("grietas en la superficie", k=1)[0]
  assert "crazing" in seccion.titulo


def test_extrae_campos_de_una_seccion(buscador):
  seccion, _ = buscador.buscar("inclusion", k=1)[0]
  assert seccion.campo("Gravedad") == "CRITICO."
  assert "24 horas" in seccion.campo("Plazo")


def test_sin_coincidencias_devuelve_lista_vacia(buscador):
  assert buscador.buscar("xyzzy") == []


def test_manual_pdf_coincide_con_criterios_yaml(buscador):
  """La gravedad escrita en el PDF debe ser la misma que aplica el YAML."""
  reglas = criterios.cargar_manual(RAIZ / "manual_criterios.yaml")
  for clase, regla in reglas["clases"].items():
    seccion, _ = buscador.buscar(clase, k=1)[0]
    assert seccion.campo("Gravedad").rstrip(".") == regla["gravedad"], clase
