"""Acceso directo a: uv run inspector-industrial entrenar [opciones]"""

import sys

from inspector_industrial import main

if __name__ == "__main__":
  main(["entrenar", *sys.argv[1:]])
