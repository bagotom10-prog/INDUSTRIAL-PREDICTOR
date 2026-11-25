"""Estación 2: búsqueda en el manual de calidad (la parte de recuperación del RAG).

Divide el manual (PDF o Markdown) en secciones numeradas y las ordena por
relevancia con BM25. BM25 puntúa cada sección según cuántas palabras de la
consulta contiene, dando más peso a las palabras raras en el manual y
penalizando las secciones largas. Para un manual de decenas de secciones
funciona tan bien como los embeddings, no necesita descargar ningún modelo y
es determinista.
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MANUAL_PDF_POR_DEFECTO = "manual/manual_calidad.pdf"

# "4. Grietas finas en red (crazing)" al principio de una línea.
_TITULO = re.compile(r"^(\d+)\.\s+(\S.*)$", re.MULTILINE)
_PALABRAS_VACIAS = set(
    "a al como con de del el en es la las lo los o para por que se si sin su"
    " un una y".split()
)


@dataclass
class Seccion:
  numero: int
  titulo: str
  texto: str

  @property
  def referencia(self) -> str:
    return f"Sección {self.numero}. {self.titulo}"

  def campo(self, nombre: str) -> str | None:
    """Devuelve el valor de una línea 'Nombre: valor' de la sección."""
    m = re.search(rf"^{nombre}:\s*(.+?)(?=\n\S+(?: \S+)?:|\Z)", self.texto,
                  re.MULTILINE | re.DOTALL)
    return " ".join(m.group(1).split()) if m else None


def _tokens(texto: str) -> list[str]:
  sin_tildes = unicodedata.normalize("NFKD", texto.lower())
  sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
  return [t for t in re.findall(r"[a-z0-9]+", sin_tildes)
          if t not in _PALABRAS_VACIAS]


def leer_texto(ruta: str | Path) -> str:
  ruta = Path(ruta)
  if ruta.suffix.lower() == ".pdf":
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(ruta).pages)
  # Markdown o texto: quita las almohadillas de los títulos.
  return re.sub(r"^#+\s*", "", ruta.read_text(encoding="utf-8"), flags=re.M)


def dividir_secciones(texto: str) -> list[Seccion]:
  titulos = list(_TITULO.finditer(texto))
  secciones = []
  for i, m in enumerate(titulos):
    fin = titulos[i + 1].start() if i + 1 < len(titulos) else len(texto)
    cuerpo = texto[m.end():fin].strip()
    secciones.append(Seccion(int(m.group(1)), m.group(2).strip(), cuerpo))
  return secciones


class BuscadorManual:
  def __init__(self, secciones: list[Seccion], k1: float = 1.5, b: float = 0.75):
    if not secciones:
      raise ValueError("El manual no tiene secciones numeradas ('1. Título').")
    self.secciones = secciones
    self.k1, self.b = k1, b
    # El título cuenta doble: es lo más descriptivo de cada sección.
    self._docs = [Counter(_tokens(f"{s.titulo} {s.titulo} {s.texto}"))
                  for s in secciones]
    self._largo = [sum(d.values()) for d in self._docs]
    self._largo_medio = sum(self._largo) / len(self._largo)
    n = len(self._docs)
    df = Counter(t for d in self._docs for t in d)
    self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

  @classmethod
  def desde_archivo(cls, ruta: str | Path) -> "BuscadorManual":
    return cls(dividir_secciones(leer_texto(ruta)))

  def puntuar(self, consulta: str) -> list[float]:
    terminos = _tokens(consulta)
    puntos = []
    for doc, largo in zip(self._docs, self._largo):
      norm = self.k1 * (1 - self.b + self.b * largo / self._largo_medio)
      puntos.append(sum(
          self._idf.get(t, 0) * doc[t] * (self.k1 + 1) / (doc[t] + norm)
          for t in terminos if t in doc
      ))
    return puntos

  def buscar(self, consulta: str, k: int = 3) -> list[tuple[Seccion, float]]:
    puntos = self.puntuar(consulta)
    orden = sorted(range(len(puntos)), key=lambda i: puntos[i], reverse=True)
    return [(self.secciones[i], round(puntos[i], 3))
            for i in orden[:k] if puntos[i] > 0]
