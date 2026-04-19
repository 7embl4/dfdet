import json
from pathlib import Path


ROOT_PATH = Path(__file__).parent.parent.parent

def read_json(file: str | Path):
    file = str(file)
    with open(file, "r") as f:
        data = json.load(f)
    return data

def write_json(data, filename: str | Path):
    filename = str(filename)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
