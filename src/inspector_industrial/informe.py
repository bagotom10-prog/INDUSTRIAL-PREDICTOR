"""Genera el informe técnico HTML a partir del reporte de inspección.

El HTML se guarda junto a las imágenes marcadas y las enlaza por nombre, así
que la carpeta de salida se puede copiar o comprimir entera.
"""

import json
from collections import Counter
from html import escape
from pathlib import Path

COLORES = {"APROBADA": "#1a7f37", "REVISION": "#9a6700", "RECHAZADA": "#cf222e"}
ORDEN = {"RECHAZADA": 0, "REVISION": 1, "APROBADA": 2}

CSS = """
:root { --borde:#d0d7de; --suave:#f6f8fa; --texto:#1f2328; --gris:#59636e; }
* { box-sizing:border-box; }
body { font:14px/1.5 system-ui,sans-serif; color:var(--texto); margin:0;
       padding:24px clamp(16px,4vw,48px); background:#fff; }
h1 { margin:0 0 4px; font-size:24px; }
h2 { margin:32px 0 12px; font-size:18px; border-bottom:1px solid var(--borde);
     padding-bottom:6px; }
.meta { color:var(--gris); margin:0; }
.tarjetas { display:flex; flex-wrap:wrap; gap:12px; margin-top:20px; }
.tarjeta { border:1px solid var(--borde); border-radius:8px; padding:12px 16px;
           min-width:140px; flex:1; }
.tarjeta b { display:block; font-size:28px; font-variant-numeric:tabular-nums; }
.tabla { overflow-x:auto; }
table { border-collapse:collapse; width:100%; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--borde);
         vertical-align:top; }
th { background:var(--suave); font-weight:600; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.estado { color:#fff; border-radius:12px; padding:1px 10px; font-size:12px;
          font-weight:600; white-space:nowrap; }
img.mini { width:96px; height:96px; object-fit:cover; border-radius:4px;
           border:1px solid var(--borde); }
ul { margin:0; padding-left:18px; }
.filtros button { border:1px solid var(--borde); background:var(--suave);
                  border-radius:6px; padding:4px 12px; margin:0 6px 12px 0;
                  cursor:pointer; font:inherit; }
.filtros button[aria-pressed=true] { background:var(--texto); color:#fff; }
.aviso { background:#fff8c5; border:1px solid #d4a72c; border-radius:6px;
         padding:8px 12px; margin-top:16px; }
"""

JS = """
document.querySelectorAll('.filtros button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.filtros button')
    .forEach(o => o.setAttribute('aria-pressed', o === b));
  document.querySelectorAll('tr[data-estado]').forEach(tr =>
    tr.hidden = b.dataset.f !== 'TODAS' && tr.dataset.estado !== b.dataset.f);
});
"""


def _estado(estado: str) -> str:
  return (f'<span class="estado" style="background:{COLORES[estado]}">'
          f'{escape(estado)}</span>')


def generar_html(reporte: dict, ruta: str | Path) -> Path:
  ruta = Path(ruta)
  r = reporte["resumen"]
  manual = reporte["manual"]
  total = max(r["total_piezas"], 1)

  tarjetas = "".join(
      f'<div class="tarjeta">{etq}<b style="color:{color}">{n}</b>'
      f'{n / total:.0%} del lote</div>'
      for etq, n, color in [
          ("Aprobadas", r["aprobadas"], COLORES["APROBADA"]),
          ("Revisión manual", r["revision"], COLORES["REVISION"]),
          ("Rechazadas", r["rechazadas"], COLORES["RECHAZADA"]),
      ]
  )

  # Defectos por clase y su dictamen.
  por_clase = Counter()
  por_dictamen = Counter()
  for p in reporte["piezas"]:
    for d in p["defectos"]:
      por_clase[d["clase"]] += 1
      por_dictamen[(d["clase"], d["dictamen"])] += 1
  filas_clases = "".join(
      f"<tr><td>{escape(c)}</td>"
      f"<td>{escape(regla['gravedad'])}</td>"
      f"<td class=num>{regla['conf_deteccion']:.2f}</td>"
      f"<td class=num>{por_clase[c]}</td>"
      f"<td class=num>{por_dictamen[(c, 'rechazo')]}</td>"
      f"<td class=num>{por_dictamen[(c, 'revision')]}</td>"
      f"<td class=num>{por_dictamen[(c, 'tolerado')]}</td>"
      f"<td>{escape(regla.get('descripcion', ''))}</td></tr>"
      for c, regla in manual["clases"].items()
  )

  piezas = sorted(reporte["piezas"],
                  key=lambda p: (ORDEN[p["estado"]], p["imagen"]))
  filas_piezas = []
  for p in piezas:
    img = escape(p["imagen"])
    defectos = "".join(
        f"<li>{escape(d['clase'])} · conf. {d['confianza']:.2f} · "
        f"{d['area_relativa']:.0%} · {escape(d['dictamen'])}</li>"
        for d in p["defectos"]
    ) or "<li>—</li>"
    motivos = "".join(f"<li>{escape(m)}</li>" for m in p["motivos"])
    filas_piezas.append(
        f'<tr data-estado="{escape(p["estado"])}">'
        f'<td><a href="{img}"><img class="mini" src="{img}" alt="{img}" '
        f'loading="lazy"></a></td>'
        f"<td>{img}</td><td>{_estado(p['estado'])}</td>"
        f"<td><ul>{defectos}</ul></td><td><ul>{motivos}</ul></td></tr>"
    )

  filtros = "".join(
      f'<button data-f="{f}" aria-pressed="{str(f == "TODAS").lower()}">'
      f"{f.capitalize()}</button>"
      for f in ("TODAS", "RECHAZADA", "REVISION", "APROBADA")
  )

  html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe de inspección {escape(reporte['fecha'])}</title>
<style>{CSS}</style></head><body>
<h1>Informe técnico de inspección</h1>
<p class="meta">{escape(reporte['fecha'])} · Modelo:
<code>{escape(reporte['modelo'])}</code> · Imágenes:
<code>{escape(reporte.get('fuente', ''))}</code> · Manual
{escape(str(manual.get('version', '')))}</p>
<div class="tarjetas"><div class="tarjeta">Piezas inspeccionadas
<b>{r['total_piezas']}</b>&nbsp;</div>{tarjetas}</div>
<p class="aviso">Los criterios de aceptación del manual
v{escape(str(manual.get('version', '')))} son valores de ejemplo. Las piezas en
<b>revisión</b> necesitan la decisión de un inspector.</p>

<h2>Criterios aplicados y defectos por clase</h2>
<p class="meta">Rechazo automático solo con confianza ≥
{manual['min_confidence']:.2f}; por debajo, la pieza pasa a revisión.
Los defectos NO CRITICO se toleran si ocupan como máximo el
{manual['max_box_size']:.0%} de la pieza.</p>
<div class="tabla"><table>
<tr><th>Clase</th><th>Gravedad</th><th>Umbral detección</th><th>Detectados</th>
<th>Rechazo</th><th>Revisión</th><th>Tolerados</th><th>Descripción</th></tr>
{filas_clases}</table></div>

<h2>Detalle por pieza</h2>
<div class="filtros">{filtros}</div>
<div class="tabla"><table>
<tr><th>Imagen</th><th>Archivo</th><th>Estado</th>
<th>Defectos (clase · confianza · área · dictamen)</th><th>Motivo</th></tr>
{''.join(filas_piezas)}</table></div>
<script>{JS}</script>
</body></html>
"""
  ruta.write_text(html, encoding="utf-8")
  return ruta


def generar_desde_json(ruta_json: str | Path) -> Path:
  ruta_json = Path(ruta_json)
  reporte = json.loads(ruta_json.read_text(encoding="utf-8"))
  return generar_html(reporte, ruta_json.with_name("informe_inspeccion.html"))
