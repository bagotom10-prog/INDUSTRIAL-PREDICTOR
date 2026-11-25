"""Acceso directo a: uv run inspector-industrial preparar [opciones]"""

import sys

from inspector_industrial import main

if __name__ == "__main__":
  main(["preparar", *sys.argv[1:]])
