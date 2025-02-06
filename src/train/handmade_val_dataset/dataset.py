import json

from dinora import PROJECT_ROOT

with open(PROJECT_ROOT / "data/handmade_val/handmade.json", encoding="utf8") as f:
    POSITIONS = json.load(f)
