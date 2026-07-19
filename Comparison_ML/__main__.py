#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import runpy

_ROOT_MAIN = Path(__file__).resolve().parents[1] / "main.py"
runpy.run_path(str(_ROOT_MAIN), run_name="__main__")
