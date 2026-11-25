# Inspector industrial

Sistema de inspección visual de piezas de acero que funciona como una cadena de cuatro estaciones:

1. **Visión:** YOLO11 detecta defectos superficiales en la foto de la pieza.
2. **Manual (RAG):** busca en el manual de calidad en PDF las secciones que tratan cada defecto.
3. **Diagnóstico:** Claude lee los defectos y esas secciones y redacta un diagnóstico estructurado: defecto, gravedad, acción, plazo, justificación y referencias al manual.
4. **API:** todo se ofrece como un servicio web con FastAPI, preparado para Docker.

Antes del diagnóstico, un **manual de criterios** decide de forma determinista si cada pieza queda **APROBADA**, **RECHAZADA** o en **REVISION** manual. El LLM redacta el diagnóstico pero no puede cambiar esa decisión, así que cada aprobación o rechazo es reproducible y auditable.

Se entrena y evalúa con el dataset público [NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database): 1.800 imágenes de acero laminado en caliente con 6 tipos de defecto (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`).

## Requisitos

- Python 3.14 y [uv](https://docs.astral.sh/uv/)
- GPU NVIDIA con CUDA (recomendado). Desarrollado en WSL2 con una RTX 3060 de 6 GB.
- Cuenta de Kaggle configurada para `kagglehub` (solo para descargar el dataset).
- Opcional: una clave de la API de Anthropic para el diagnóstico con LLM. Sin ella, el diagnóstico se construye directamente con el texto del manual.

```bash
uv sync
cp .env.example .env   # y añade ANTHROPIC_API_KEY si la tienes
```

## Uso

```bash
uv run inspector-industrial preparar       # descarga NEU-DET y lo convierte a formato YOLO
uv run inspector-industrial entrenar       # entrena yolo11n a 640 px durante 50 épocas
uv run inspector-industrial inspeccionar   # inspecciona un lote -> reporte JSON + informe HTML
uv run inspector-industrial servir         # API web en http://localhost:8000/docs
uv run inspector-industrial buscar "grietas en la superficie"   # consulta el manual
uv run inspector-industrial informe        # regenera el HTML a partir de un JSON existente
```

Cada comando admite `--help`. Por ejemplo, para inspeccionar otra carpeta con otro modelo:

```bash
uv run inspector-industrial inspeccionar --fuente fotos/ --pesos runs/detect/otro/weights/best.pt
```

### Inspección por lotes

Los resultados quedan en `runs/detect/inspeccion_real/`:

- Imágenes con los defectos marcados.
- `reporte_inspeccion.json`: para cada pieza, sus defectos (confianza, área, dictamen), su estado, los motivos y el diagnóstico.
- `informe_inspeccion.html`: informe técnico con resumen, criterios aplicados y detalle por pieza filtrable por estado.

Para usar Claude, pasa el archivo `.env` con la clave:

```bash
uv run --env-file .env inspector-industrial inspeccionar --diagnostico llm
```

Se llama a Claude una sola vez por cada combinación distinta de defectos y estado, no por cada pieza: las 360 piezas de validación necesitan 13 llamadas.

### API web

| Método | Ruta | Qué hace |
|---|---|---|
| `POST` | `/inspecciones` | Sube una foto y devuelve defectos, estado, motivos y diagnóstico. Con `?devolver_imagen=true` incluye la imagen marcada en PNG (base64). |
| `GET` | `/manual/buscar?q=...` | Busca en el manual de calidad. |
| `GET` | `/salud` | Estado del servicio y configuración cargada. |
| `GET` | `/docs` | Documentación interactiva (Swagger). |

```bash
curl -F "imagen=@pieza.jpg;type=image/jpeg" http://localhost:8000/inspecciones
```

Se configura con variables de entorno (ver [`.env.example`](.env.example)): `INSPECTOR_PESOS`, `INSPECTOR_CRITERIOS`, `INSPECTOR_MANUAL_PDF`, `INSPECTOR_DIAGNOSTICO`, `INSPECTOR_MODELO_LLM` y `ANTHROPIC_API_KEY`.

## Cómo se decide cada pieza

Las reglas están en [`manual_criterios.yaml`](manual_criterios.yaml):

| Gravedad de la clase | Dictamen |
|---|---|
| `CRITICO` (crazing, inclusion) | Rechazo |
| `REVISAR` (pitted_surface, rolled-in_scale) | Revisión manual |
| `NO CRITICO` (patches, scratches) | Tolerado si ocupa ≤ 80 % de la pieza; si no, rechazo |

Un rechazo con confianza inferior a 0,5 no es fiable, así que la pieza pasa a revisión. Cada clase tiene su propio umbral de detección, calibrado sobre el conjunto de validación.

## Cómo se redacta el diagnóstico

1. Para cada defecto de la pieza se busca su sección en [`manual/manual_calidad.pdf`](manual/manual_calidad.pdf) con BM25, un algoritmo de búsqueda por palabras clave. Si la pieza va a revisión, también se recupera la sección del procedimiento de revisión.
2. **Modo `llm`:** Claude recibe la pieza (estado ya decidido, defectos y motivos) y esas secciones, y devuelve un JSON validado contra un esquema fijo.
3. **Modo `reglas`:** sin clave de API, o si el LLM falla, la acción y el plazo se copian de las secciones del manual.

El manual PDF se genera desde [`manual/manual_calidad.md`](manual/manual_calidad.md) con `uv run python scripts/generar_manual_pdf.py`.

> El manual de calidad y los criterios son **de ejemplo**: no proceden de ninguna norma real. Hay que sustituirlos por los de la planta antes de usar el sistema en producción.

## Resultados del modelo

`yolo11n`, 50 épocas, 640 px, sobre las 360 imágenes de validación:

| Clase | mAP50 | mAP50-95 |
|---|---|---|
| patches | 0,935 | 0,606 |
| scratches | 0,847 | 0,442 |
| inclusion | 0,835 | 0,461 |
| pitted_surface | 0,834 | 0,459 |
| rolled-in_scale | 0,521 | 0,249 |
| crazing | 0,477 | 0,192 |
| **Total** | **0,754** | **0,400** |

Con el manual aplicado, cada clase llega al estado que le corresponde. Solo 4 de las 360 piezas defectuosas quedan sin ninguna detección.

## Limitaciones

- NEU-DET no contiene piezas sin defectos, así que no se puede medir cuántas piezas sanas se rechazarían por error.
- `crazing` y `rolled-in_scale` son defectos de textura difusa con etiquetas ambiguas; el modelo los detecta con poca confianza y la mayoría pasan a revisión.
- Los umbrales por clase se calibraron sobre el mismo conjunto de validación que se usa para evaluar, así que los resultados son algo optimistas.

## Tests

```bash
uv run pytest
```

Los tests del diagnóstico simulan el cliente de Anthropic, así que no hacen llamadas reales ni gastan dinero. Los de la API usan el modelo entrenado y se saltan si no está.
