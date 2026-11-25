# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Proyecto **inspector-industrial**: inspección de piezas de acero en cuatro estaciones:
1. Visión (YOLO).
2. Búsqueda en el manual PDF (RAG con BM25).
3. Diagnóstico redactado por Claude.
4. API FastAPI, que se empaquetará en Docker.

La decisión APROBADA / REVISION / RECHAZADA la toma un manual de criterios determinista, no el LLM.

## Entorno

- **Sistema:** WSL2 sobre Windows (Ubuntu). WSL tiene **8 GB de RAM** (`.wslconfig`); si un proceso la agota, Windows reinicia WSL entero.
- **Python 3.14** con **uv**; entorno en `.venv/`. Dependencias con `uv add` (dev: `uv add --dev`), nunca `pip install`.
- **GPU:** NVIDIA RTX 3060 Laptop, 6 GB de VRAM, CUDA disponible desde WSL2.
- **Docker:** previsto como siguiente paso. La configuración ya va por variables de entorno (`.env.example`).

## Comandos

```bash
uv sync                                        # instalar dependencias
uv run pytest                                  # tests (45; los de API necesitan pesos y dataset)
uv run pytest tests/test_criterios.py::test_critico_rechaza   # un test
uv run inspector-industrial preparar           # descargar y convertir NEU-DET
uv run inspector-industrial entrenar           # entrenar (yolo11n, 640 px, 50 épocas)
uv run --env-file .env inspector-industrial inspeccionar   # lote -> JSON + HTML (--diagnostico auto|llm|reglas|ninguno)
uv run inspector-industrial servir             # API en :8000 (/docs)
uv run inspector-industrial buscar "consulta"  # probar la búsqueda en el manual
uv run python scripts/generar_manual_pdf.py    # regenerar manual/manual_calidad.pdf desde el .md
```

`preparar_datos.py`, `entrenar.py` e `inspeccionar.py` en la raíz son accesos directos a esos subcomandos.

## Arquitectura (`src/inspector_industrial/`)

Flujo de una pieza:

```
YOLO -> criterios -> manual (BM25) -> diagnostico -> JSON / HTML / API
```

- `__init__.py`: CLI `argparse` con subcomandos. Importa ultralytics de forma perezosa.
- `datos.py`: descarga NEU-DET y convierte de Pascal VOC a YOLO.
- `entrenamiento.py`: `YOLO.train` con `workers=2, batch=8` para no agotar la RAM.
- `inspeccion.py`: clase `Inspector`, que carga una vez el modelo, el manual YAML, el buscador y el diagnosticador.
  - `predecir()` usa el umbral más bajo del manual; `analizar()` filtra por el `conf_deteccion` de cada clase y llama a criterios y a diagnóstico.
  - `inspeccionar()` es el modo por lotes.
  - `PESOS_POR_DEFECTO` es el modelo en producción (sobrescribible con `INSPECTOR_PESOS`).
- `criterios.py`: lógica pura. Carga y valida `manual_criterios.yaml` y dictamina cada defecto (rechazo/revision/tolerado) y cada pieza (RECHAZADA > REVISION > APROBADA).
- `manual.py`: divide el manual (PDF con pypdf, o .md) en secciones `N. Título` y las ordena con BM25 en Python puro. `Seccion.campo("Acción")` extrae líneas `Campo: valor`.
- `diagnostico.py`: `Diagnosticador` con modos `llm` (Claude, salida `json_schema` validada con Pydantic) y `reglas` (copia Acción y Plazo del manual).
  - `auto` usa `llm` si existe `ANTHROPIC_API_KEY`.
  - Cualquier fallo del LLM cae a `reglas`.
  - Guarda el diagnóstico por (estado, clases y dictámenes) para no llamar al LLM por cada pieza.
  - El estado de la pieza se pasa como ya decidido: el LLM no lo cambia.
- `informe.py`: HTML autocontenido. Enlaza las imágenes por nombre relativo, así que vive en su misma carpeta.
- `api.py`: `crear_app(inspector=None)`. El modelo se carga en el `lifespan`; un `threading.Lock` serializa la inferencia (Ultralytics no es seguro entre hilos). Los tests inyectan su propio `Inspector`.

**Coherencia entre manuales:** `manual_criterios.yaml` decide; `manual/manual_calidad.md` (y su PDF) aporta el texto del diagnóstico. La gravedad de cada clase debe coincidir en ambos; `tests/test_manual.py` lo comprueba. **Los dos son de ejemplo**, no de una norma real.

## Trampas conocidas

- **Python 3.14 usa `forkserver`:** los procesos de carga de datos reimportan el script principal. Todo `train`/`val` debe ir bajo `if __name__ == "__main__":` y desde un archivo (no `python -` ni `-c` con workers > 0).
- En `predict`, una ruta `project` relativa hace que Ultralytics anide la salida en `runs/detect/runs/detect/...`; usar rutas absolutas.
- El modelo de producción se elige a mano, nunca por orden alfabético de carpetas.
- La clave de Anthropic va en `.env` (ignorado por git), **nunca en `.env.example`**, que sí se versiona. El programa no lee `.env` solo: usar `uv run --env-file .env ...` (y `docker run --env-file .env` en Docker).
- Modo `llm` probado con la API real el 2026-11-25: lote de 360 piezas = 13 llamadas, todas válidas. El diagnóstico se reutiliza entre piezas equivalentes, por eso el prompt prohíbe cifras concretas de una pieza.

## Datos y modelos

- NEU-DET: imágenes 200×200 en `datasets/NEU-DET/images/{train,val}`, 6 clases. **Todas las imágenes tienen defecto**: no hay piezas sanas. La clase real está en el nombre del archivo (`crazing_241.jpg`).
- `datasets/`, `runs/`, `*.pt` y `.env` no se versionan.

| Carpeta `runs/detect/` | Modelo | Épocas | imgsz | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| `train` | yolo11n | 20 | 200 | 0.62 | 0.69 | 0.726 | 0.383 |
| `train-2` | yolo11s | 50 | 224 | 0.65 | 0.66 | 0.701 | 0.374 |
| **`n_640` (producción)** | yolo11n | 50 | 640 | 0.73 | 0.67 | 0.754 | 0.400 |

- mAP50 por clase de `n_640`: patches 0.94, scratches 0.85, inclusion 0.84, pitted_surface 0.83, rolled-in_scale 0.52, crazing 0.48. Un modelo mayor no ayudó: el límite son las etiquetas difusas de crazing y rolled-in_scale.
- `crazing` y `rolled-in_scale` no generan falsas alarmas en otras clases ni con conf 0.10; por eso su `conf_deteccion` es bajo (0.15 y 0.20).
- Resultado actual del lote de validación: 122 aprobadas, 170 en revisión y 68 rechazadas; solo 4 piezas sin ninguna detección.
- Al experimentar, cambiar un solo parámetro por entrenamiento y darle `--nombre`.

## Rol de Claude Code

Claude Code **solo asiste en el desarrollo** (código, documentación, configuración). El producto sí usa la API de Claude en tiempo de ejecución, pero solo para *redactar* diagnósticos. Eso es independiente de Claude Code y nunca decide sobre las piezas.

**Los entrenamientos los lanza el usuario.** Claude edita el código y lee resultados (`results.csv`, `args.yaml`, reportes JSON). Solo ejecuta validaciones o inspecciones si se le pide, y nunca entrenamientos sin permiso (riesgo de agotar la RAM de WSL). Tampoco hace push a GitHub: lo hace el usuario.
