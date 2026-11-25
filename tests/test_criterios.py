from pathlib import Path

import pytest

from inspector_industrial import criterios
from inspector_industrial.datos import CLASES, voc_a_yolo

RAIZ = Path(__file__).resolve().parent.parent

MANUAL = {
    "max_box_size": 0.8,
    "min_confidence": 0.5,
    "clases": {
        "crazing": {"gravedad": "CRITICO", "conf_deteccion": 0.15},
        "pitted_surface": {"gravedad": "REVISAR", "conf_deteccion": 0.25},
        "scratches": {"gravedad": "NO CRITICO", "conf_deteccion": 0.30},
    },
}


def defecto(clase, confianza=0.9, area=0.1):
  return {"clase": clase, "confianza": confianza, "area_relativa": area}


def estado(*defectos):
  return criterios.evaluar_pieza(list(defectos), MANUAL)["estado"]


def test_sin_defectos_aprueba():
  assert estado() == "APROBADA"


def test_no_critico_pequeno_se_tolera():
  assert estado(defecto("scratches", area=0.1)) == "APROBADA"


def test_no_critico_demasiado_grande_rechaza():
  assert estado(defecto("scratches", area=0.9)) == "RECHAZADA"


def test_critico_rechaza():
  assert estado(defecto("crazing")) == "RECHAZADA"


def test_critico_con_confianza_baja_va_a_revision():
  assert estado(defecto("crazing", confianza=0.3)) == "REVISION"


def test_revisar_va_a_revision():
  assert estado(defecto("pitted_surface")) == "REVISION"


def test_rechazo_tiene_prioridad_sobre_revision():
  assert estado(defecto("pitted_surface"), defecto("crazing")) == "RECHAZADA"


def test_motivos_explican_el_estado():
  r = criterios.evaluar_pieza(
      [defecto("crazing"), defecto("scratches")], MANUAL
  )
  assert r["motivos"] == ["crazing es un defecto crítico"]


def test_manual_rechaza_gravedad_desconocida():
  malo = {**MANUAL, "clases": {"x": {"gravedad": "GRAVE",
                                     "conf_deteccion": 0.5}}}
  with pytest.raises(criterios.ManualInvalido):
    criterios.validar_manual(malo)


def test_comprobar_clases_detecta_clase_sin_regla():
  with pytest.raises(criterios.ManualInvalido, match="inclusion"):
    criterios.comprobar_clases(MANUAL, ["crazing", "inclusion"])


def test_manual_del_proyecto_cubre_todas_las_clases():
  manual = criterios.cargar_manual(RAIZ / "manual_criterios.yaml")
  criterios.comprobar_clases(manual, CLASES)


def test_voc_a_yolo():
  assert voc_a_yolo(0, 0, 100, 50, 200, 200) == (0.25, 0.125, 0.5, 0.25)
