import subprocess
import json
import re
import os
import sys

HARNESS_EXE = r"C:\Users\Choosen\Documents\Estudo-Executor\RocketSim\build\differential_harness.exe"
THRESHOLDS_JSON = r"C:\Users\Choosen\Documents\Estudo-Executor\RocketSim\docs\parity_thresholds.json"
REPO_DIR = r"C:\Users\Choosen\Documents\Estudo-Executor\RocketSim"

with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
    thresh_data = json.load(f)

scenarios = sorted(thresh_data.get("scenarios", {}).keys())
print(f"Loaded {len(scenarios)} scenarios from {THRESHOLDS_JSON}")
