# Inspector industrial

Sistema de inspección visual que detecta defectos superficiales en piezas de acero con YOLO11, decide si cada pieza se **aprueba**, se **rechaza** o pasa a **revisión manual** según un manual de criterios, y genera un informe técnico.

Se entrena y evalúa con el dataset público [NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database): 1.800 imágenes de acero laminado en caliente con 6 tipos de defecto (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`).

## Requisitos

- Python 3.14 y [uv](https://docs.astral.sh/uv/)
- GPU NVIDIA con CUDA (recomendado). Desarrollado en WSL2 con una RTX 3060 de 6 GB.
- Cuenta de Kaggle configurada para `kagglehub` (solo para descargar el dataset).

```bash
uv sync
```

## Uso

```bash
uv run inspector-industrial preparar       # descarga NEU-DET y lo convierte a formato YOLO
uv run inspector-industrial entrenar       # entrena yolo11n a 640 px durante 50 épocas
uv run inspector-industrial inspeccionar   # inspecciona y genera reporte JSON + informe HTML
uv run inspector-industrial informe        # regenera el HTML a partir de un JSON existente
```

Cada comando admite `--help`. Por ejemplo, para inspeccionar otra carpeta con otro modelo:

```bash
uv run inspector-industrial inspeccionar --fuente fotos/ --pesos runs/detect/otro/weights/best.pt
```

Los resultados quedan en `runs/detect/inspeccion_real/`:

- Imágenes con los defectos marcados.
- `reporte_inspeccion.json`: defectos, confianza, área, dictamen y estado de cada pieza.
- `informe_inspeccion.html`: informe técnico con resumen, criterios aplicados y detalle filtrable por estado.

## Cómo se decide cada pieza

Las reglas están en [`manual_criterios.yaml`](manual_criterios.yaml):

| Gravedad de la clase | Dictamen |
|---|---|
| `CRITICO` (crazing, inclusion) | Rechazo |
| `REVISAR` (pitted_surface, rolled-in_scale) | Revisión manual |
| `NO CRITICO` (patches, scratches) | Tolerado si ocupa ≤ 80 % de la pieza; si no, rechazo |

Además, un rechazo con confianza inferior a 0,5 no es fiable y pasa a revisión. Cada clase tiene su propio umbral de detección, calibrado sobre el conjunto de validación.

> Los valores del manual son **de ejemplo**: hay que sustituirlos por los del manual de calidad real antes de usar el sistema en planta.

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

## Limitaciones

- NEU-DET no contiene piezas sin defectos, así que no se puede medir cuántas piezas sanas se rechazarían por error.
- `crazing` y `rolled-in_scale` son defectos de textura difusa con etiquetas ambiguas; el modelo los detecta con poca confianza.
- Los umbrales por clase se calibraron sobre el mismo conjunto de validación que se usa para evaluar, así que los resultados son algo optimistas.

## Tests

```bash
uv run pytest
```
