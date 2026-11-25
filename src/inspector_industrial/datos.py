"""Descarga NEU-DET de Kaggle y lo convierte al formato que espera YOLO."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
DIRECTORIO_DATASET = Path("datasets/NEU-DET")


def voc_a_yolo(xmin, ymin, xmax, ymax, ancho, alto):
  """Caja en píxeles (esquinas) -> centro y tamaño normalizados entre 0 y 1."""
  return (
      ((xmin + xmax) / 2) / ancho,
      ((ymin + ymax) / 2) / alto,
      (xmax - xmin) / ancho,
      (ymax - ymin) / alto,
  )


def preparar(destino: Path = DIRECTORIO_DATASET) -> None:
  import kagglehub

  # 1. Descarga el dataset público NEU-DET (Kaggle lo guarda en caché).
  origen = Path(
      kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
  )

  # 2. Crea images/{train,val} y labels/{train,val}.
  for split_origen, split_yolo in [("train", "train"), ("validation", "val")]:
    img_dest = destino / "images" / split_yolo
    lbl_dest = destino / "labels" / split_yolo
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)

    # 3. Convierte cada anotación XML (Pascal VOC) a un .txt de YOLO.
    for xml_file in origen.rglob(f"*{split_origen}*/annotations/*.xml"):
      raiz = ET.parse(xml_file).getroot()
      ancho = int(raiz.find("size/width").text)
      alto = int(raiz.find("size/height").text)

      imagenes = list(
          xml_file.parent.parent.glob(f"images/**/{xml_file.stem}.jpg")
      )
      if not imagenes:
        continue
      shutil.copy(imagenes[0], img_dest / f"{xml_file.stem}.jpg")

      lineas = []
      for obj in raiz.findall("object"):
        nombre = obj.find("name").text.strip()
        if nombre not in CLASES:
          continue
        caja = obj.find("bndbox")
        coords = [float(caja.find(k).text)
                  for k in ("xmin", "ymin", "xmax", "ymax")]
        xc, yc, w, h = voc_a_yolo(*coords, ancho, alto)
        lineas.append(
            f"{CLASES.index(nombre)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        )

      (lbl_dest / f"{xml_file.stem}.txt").write_text("\n".join(lineas) + "\n")

  # 4. Escribe data.yaml, que le dice a YOLO dónde están los datos.
  (destino / "data.yaml").write_text(
      f"path: {destino.resolve()}\ntrain: images/train\nval: images/val\n"
      f"nc: {len(CLASES)}\nnames: {CLASES}\n"
  )
  n_train = len(list((destino / "images/train").glob("*.jpg")))
  print(f"Dataset NEU-DET listo en {destino} | Imágenes train: {n_train}")
