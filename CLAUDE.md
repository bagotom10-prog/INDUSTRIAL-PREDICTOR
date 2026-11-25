# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Proyecto **inspector-industrial**: detección de defectos superficiales en piezas de acero con YOLO (Ultralytics), decisión aprobar/revisar/rechazar según un manual de criterios e informe técnico HTML.

## Entorno

- **Sistema:** WSL2 sobre Windows (Ubuntu). WSL tiene **8 GB de RAM** (`.wslconfig`); si un proceso la agota, Windows reinicia WSL entero.
- **Python 3.14** con **uv**; entorno en `.venv/`. Dependencias con `uv add` (dev: `uv add --dev`), nunca `pip install`.
- **GPU:** NVIDIA RTX 3060 Laptop, 6 GB de VRAM, CUDA disponible desde WSL2.

## Comandos

```bash
uv sync                                        # instalar dependencias
uv run pytest                                  # tests
uv run pytest tests/test_criterios.py::test_critico_rechaza   # un test
uv run inspector-industrial preparar           # descargar y convertir NEU-DET
uv run inspector-industrial entrenar           # entrenar (yolo11n, 640 px, 50 épocas)
uv run inspector-industrial inspeccionar       # inspección -> JSON + HTML
uv run inspector-industrial informe            # regenerar HTML desde el JSON
```

`preparar_datos.py`, `entrenar.py` e `inspeccionar.py` en la raíz son accesos directos a esos subcomandos (aceptan las mismas opciones).

## Arquitectura (`src/inspector_industrial/`)

Flujo: `datos` → `entrenamiento` → `inspeccion` (usa `criterios` e `informe`).

- `__init__.py` — CLI `argparse` con subcomandos; importa los módulos pesados (ultralytics) de forma perezosa.
- `datos.py` — descarga NEU-DET con kagglehub, convierte XML Pascal VOC a `.txt` YOLO, escribe `datasets/NEU-DET/data.yaml`.
- `entrenamiento.py` — `YOLO.train` con `workers=2, batch=8` para no agotar la RAM de WSL a 640 px.
- `inspeccion.py` — predice con el umbral más bajo del manual, filtra por el `conf_deteccion` de cada clase, guarda las imágenes marcadas, aplica `criterios` y escribe `reporte_inspeccion.json` + `informe_inspeccion.html` en `runs/detect/inspeccion_real/`. `PESOS_POR_DEFECTO` es el modelo en producción.
- `criterios.py` — lógica pura (sin YOLO, cubierta por `tests/`): carga/valida `manual_criterios.yaml` y dictamina cada defecto (rechazo/revision/tolerado) y cada pieza (RECHAZADA > REVISION > APROBADA).
- `informe.py` — HTML autocontenido; enlaza las imágenes por nombre relativo, así que vive en la misma carpeta que ellas.

`manual_criterios.yaml` (raíz): gravedades `CRITICO`/`REVISAR`/`NO CRITICO`, `max_box_size`, `min_confidence` y por clase `gravedad` + `conf_deteccion`. **Los valores son de ejemplo**, no de una norma real.

## Trampas conocidas

- **Python 3.14 usa `forkserver`**: los procesos de carga de datos reimportan el script principal. Todo `train`/`val` debe ir bajo `if __name__ == "__main__":` y desde un archivo (no `python -` ni `-c` con workers > 0).
- En `predict`, una ruta `project` relativa hace que Ultralytics anide la salida en `runs/detect/runs/detect/...`; usar rutas absolutas.
- El modelo de producción se elige a mano (`PESOS_POR_DEFECTO` o `--pesos`), nunca por orden alfabético de carpetas.

## Datos y modelos

- NEU-DET: imágenes 200×200 en `datasets/NEU-DET/images/{train,val}`, 6 clases. **Todas las imágenes tienen defecto**: no hay piezas sanas; la clase real está en el nombre del archivo (`crazing_241.jpg`).
- `datasets/`, `runs/` y `*.pt` no se versionan.

| Carpeta `runs/detect/` | Modelo | Épocas | imgsz | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| `train` | yolo11n | 20 | 200 | 0.62 | 0.69 | 0.726 | 0.383 |
| `train-2` | yolo11s | 50 | 224 | 0.65 | 0.66 | 0.701 | 0.374 |
| **`n_640` (producción)** | yolo11n | 50 | 640 | 0.73 | 0.67 | 0.754 | 0.400 |

- mAP50 por clase de `n_640`: patches 0.94, scratches 0.85, inclusion 0.84, pitted_surface 0.83, rolled-in_scale 0.52, crazing 0.48. Un modelo mayor no ayudó: el límite son las etiquetas difusas de crazing/rolled-in_scale.
- `crazing` y `rolled-in_scale` no generan falsas alarmas en otras clases ni con conf 0.10; por eso su `conf_deteccion` es bajo (0.15/0.20).
- Al experimentar, cambiar un solo parámetro por entrenamiento y darle `--nombre`.

## Rol de Claude Code

Claude Code **solo asiste en el desarrollo**: código, documentación y configuración. **No forma parte de la ejecución en tiempo real**: no procesa imágenes en producción ni decide sobre piezas; toda la inferencia la hace YOLO localmente.

**Los entrenamientos los lanza el usuario.** Claude edita el código y lee resultados (`results.csv`, `args.yaml`, reportes JSON); solo ejecuta validaciones o inspecciones si se le pide expresamente, y nunca entrenamientos sin permiso (riesgo de agotar la RAM de WSL).
