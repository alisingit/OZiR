from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from common import normalize_russian_text, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize validation metadata.csv into wav files for MCD evaluation."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--config", required=True, help="Path to generated Coqui config.json.")
    parser.add_argument(
        "--metadata",
        required=True,
        help="LJSpeech-format metadata.csv (id|text|normalized_text).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write synthesized wav files (same IDs as metadata).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of utterances to synthesize (useful for fast smoke tests).",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Disable CUDA for inference.")
    return parser.parse_args()


def read_metadata(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="|")
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            text = row[-1] if len(row) > 2 else row[1]
            rows.append((row[0].strip(), normalize_russian_text(text)))
    return rows


def main() -> None:
    args = parse_args()
    from TTS.api import TTS

    checkpoint = resolve_path(args.checkpoint)
    config = resolve_path(args.config)
    metadata_path = resolve_path(args.metadata)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = (not args.force_cpu) and torch.cuda.is_available()
    print(f"Inference device: {'cuda' if use_gpu else 'cpu'}")
    tts = TTS(model_path=str(checkpoint), config_path=str(config), progress_bar=False, gpu=use_gpu)

    rows = read_metadata(metadata_path)
    if args.max_items:
        rows = rows[: args.max_items]

    timings: list[dict[str, float | str]] = []
    for item_id, text in tqdm(rows, desc="Synthesizing validation"):
        wav_path = output_dir / f"{item_id}.wav"
        start = time.perf_counter()
        try:
            tts.tts_to_file(text=text, file_path=str(wav_path))
        except (RuntimeError, ValueError) as exc:
            print(f"Skip {item_id}: {exc}")
            continue
        timings.append(
            {
                "id": item_id,
                "text": text,
                "synthesis_sec": round(time.perf_counter() - start, 4),
            }
        )

    summary_path = output_dir.parent / f"{output_dir.name}_synthesis_summary.json"
    summary_path.write_text(
        json.dumps(timings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Synthesized {len(timings)} files. Summary: {summary_path}")


if __name__ == "__main__":
    main()
