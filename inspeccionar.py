"""Acceso directo a: uv run inspector-industrial inspeccionar [opciones]"""

import sys

from inspector_industrial import main

if __name__ == "__main__":
  main(["inspeccionar", *sys.argv[1:]])
