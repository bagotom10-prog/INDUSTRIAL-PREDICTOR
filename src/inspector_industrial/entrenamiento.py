"""Entrena YOLO sobre NEU-DET."""

from .datos import DIRECTORIO_DATASET


def entrenar(modelo="yolo11n.pt", epocas=50, imgsz=640, nombre="n_640",
             workers=2, batch=8):
  # WSL2 tiene 8 GB de RAM. Con 20 núcleos, Ultralytics lanza 8 procesos de
  # carga de imágenes y a 640 px agotan la memoria: WSL se reinicia.
  # workers=2 y batch=8 mantienen el consumo bajo.
  #
  # Importante: quien llame a esta función debe hacerlo bajo
  # `if __name__ == "__main__":`. Python 3.14 crea los procesos de carga con
  # forkserver, que reimporta el script principal; sin esa guarda, cada
  # proceso lanzaría otro entrenamiento.
  from ultralytics import YOLO

  YOLO(modelo).train(
      data=str(DIRECTORIO_DATASET / "data.yaml"),
      epochs=epocas,
      imgsz=imgsz,
      name=nombre,
      workers=workers,
      batch=batch,
      cache=False,
  )
