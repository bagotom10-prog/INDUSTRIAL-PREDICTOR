import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inspector_industrial import criterios
from inspector_industrial.diagnostico import Diagnosticador, modo_efectivo
from inspector_industrial.manual import BuscadorManual

RAIZ = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def manual():
  return criterios.cargar_manual(RAIZ / "manual_criterios.yaml")


@pytest.fixture(scope="module")
def buscador():
  return BuscadorManual.desde_archivo(RAIZ / "manual" / "manual_calidad.pdf")


def pieza(manual, *defectos):
  defectos = [{"clase": c, "confianza": conf, "area_relativa": 0.1}
              for c, conf in defectos]
  return {"imagen": "x.jpg", **criterios.evaluar_pieza(defectos, manual),
          "defectos": defectos}


class ClienteFalso:
  """Imita anthropic.Anthropic().beta.messages.create sin llamar a la red."""

  def __init__(self, texto=None, stop_reason="end_turn", error=None):
    self.llamadas = []
    self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
    self._texto, self._stop, self._error = texto, stop_reason, error

  def _create(self, **kwargs):
    self.llamadas.append(kwargs)
    if self._error:
      raise self._error
    return SimpleNamespace(
        stop_reason=self._stop,
        content=[SimpleNamespace(type="text", text=self._texto)],
    )


RESPUESTA_LLM = json.dumps({
    "defecto": "Grieta superficial", "gravedad": "Alta",
    "accion": "Parar máquina y sustituir pieza",
    "plazo": "Menos de 24 horas", "justificacion": "Manual sección 4",
    "referencias": ["Sección 4. Grietas finas en red (crazing)"],
})


def test_modo_auto_sin_clave_usa_reglas(monkeypatch):
  monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
  assert modo_efectivo("auto") == "reglas"


def test_modo_auto_con_clave_usa_llm(monkeypatch):
  monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-prueba")
  assert modo_efectivo("auto") == "llm"


def test_reglas_copian_accion_y_plazo_del_manual(manual, buscador):
  d = Diagnosticador(manual, buscador, "reglas").diagnosticar(
      pieza(manual, ("crazing", 0.9)))
  assert d["gravedad"] == "Alta"
  assert "rechazar la pieza" in d["accion"]
  assert "24 horas" in d["plazo"]
  assert d["referencias"] == ["Sección 4. Grietas finas en red (crazing)"]
  assert d["origen"] == "reglas"


def test_revision_incluye_procedimiento_de_revision(manual, buscador):
  d = Diagnosticador(manual, buscador, "reglas").diagnosticar(
      pieza(manual, ("rolled-in_scale", 0.9)))
  assert d["gravedad"] == "Media"
  assert "Sección 3. Revisión manual" in d["referencias"]


def test_pieza_limpia_no_se_diagnostica(manual, buscador):
  assert Diagnosticador(manual, buscador, "reglas").diagnosticar(
      pieza(manual)) is None


def test_defecto_mas_grave_manda(manual, buscador):
  d = Diagnosticador(manual, buscador, "reglas").diagnosticar(
      pieza(manual, ("scratches", 0.9), ("inclusion", 0.8)))
  assert d["gravedad"] == "Alta"
  assert "inclusion" in d["defecto"]


def test_llm_recibe_manual_y_estado(manual, buscador):
  cliente = ClienteFalso(RESPUESTA_LLM)
  d = Diagnosticador(manual, buscador, "llm", cliente).diagnosticar(
      pieza(manual, ("crazing", 0.9)))
  assert d["origen"].startswith("llm")
  assert d["accion"] == "Parar máquina y sustituir pieza"
  mensaje = cliente.llamadas[0]["messages"][0]["content"]
  assert "Estado decidido: RECHAZADA" in mensaje
  assert "Sección 4. Grietas finas en red (crazing)" in mensaje
  assert cliente.llamadas[0]["output_config"]["format"]["type"] == "json_schema"


def test_piezas_equivalentes_reutilizan_el_diagnostico(manual, buscador):
  cliente = ClienteFalso(RESPUESTA_LLM)
  diag = Diagnosticador(manual, buscador, "llm", cliente)
  for conf in (0.9, 0.8, 0.95):
    diag.diagnosticar(pieza(manual, ("crazing", conf)))
  assert len(cliente.llamadas) == 1


@pytest.mark.parametrize("cliente", [
    ClienteFalso(error=ConnectionError("sin red")),
    ClienteFalso("esto no es JSON"),
    ClienteFalso(RESPUESTA_LLM, stop_reason="refusal"),
    ClienteFalso(json.dumps({"defecto": "x"})),  # faltan campos
])
def test_fallo_del_llm_cae_a_reglas(manual, buscador, cliente):
  d = Diagnosticador(manual, buscador, "llm", cliente).diagnosticar(
      pieza(manual, ("crazing", 0.9)))
  assert d["origen"].startswith("reglas (el LLM falló")
  assert "24 horas" in d["plazo"]


def test_critico_con_confianza_baja_pide_revision_antes_de_rechazar(
    manual, buscador):
  d = Diagnosticador(manual, buscador, "reglas").diagnosticar(
      pieza(manual, ("crazing", 0.3)))
  assert d["accion"].startswith("segregar la pieza en la zona de cuarentena")
  assert "Si el inspector confirma el defecto: rechazar" in d["accion"]
  assert "8 horas" in d["plazo"]


def test_no_critico_demasiado_grande_se_rechaza(manual, buscador):
  defectos = [{"clase": "patches", "confianza": 0.9, "area_relativa": 0.95}]
  p = {"imagen": "x.jpg", **criterios.evaluar_pieza(defectos, manual),
       "defectos": defectos}
  d = Diagnosticador(manual, buscador, "reglas").diagnosticar(p)
  assert p["estado"] == "RECHAZADA"
  assert d["accion"].startswith("Rechazar la pieza")
  assert d["gravedad"] == "Baja"
