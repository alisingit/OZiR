from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir or PROJECT_ROOT).joinpath(candidate).resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def normalize_russian_text(text: str) -> str:
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = text.replace("„", "\"").replace("“", "\"").replace("”", "\"")
    text = text.replace("«", "\"").replace("»", "\"")
    text = text.replace("―", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\\([.!?\-])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slugify(value: str) -> str:
    value = normalize_russian_text(value).lower()
    value = re.sub(r"[^a-zа-я0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "sample"
