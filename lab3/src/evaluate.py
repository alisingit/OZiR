from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from jiwer import cer, wer
from scipy.spatial.distance import cdist

from common import normalize_russian_text, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthesized speech with objective metrics.")
    parser.add_argument("--metadata", required=True, help="Validation metadata.csv in LJSpeech format.")
    parser.add_argument("--reference-wavs", default=None, help="Directory with reference wavs. Defaults to metadata/wavs.")
    parser.add_argument("--synth-dir", required=True, help="Directory with synthesized wav files.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics JSON/CSV.")
    parser.add_argument(
        "--asr-command",
        default=None,
        help=(
            "Optional ASR command template. Use {audio} placeholder. "
            "The command must print recognized text to stdout."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=22050, help="Sample rate for MCD computation.")
    return parser.parse_args()


def read_metadata(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0], normalize_russian_text(row[-1])))
    return rows


def mfcc(path: Path, sample_rate: int) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sample_rate, mono=True)
    features = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=13)
    return features.T


def dtw_mcd(reference_path: Path, synthesized_path: Path, sample_rate: int) -> float:
    reference = mfcc(reference_path, sample_rate)
    synthesized = mfcc(synthesized_path, sample_rate)
    if len(reference) == 0 or len(synthesized) == 0:
        return math.nan
    distances = cdist(reference[:, 1:], synthesized[:, 1:], metric="euclidean")
    accumulated = librosa.sequence.dtw(C=distances, backtrack=False)
    normalizer = accumulated.shape[0] + accumulated.shape[1]
    coefficient = 10.0 / math.log(10.0) * math.sqrt(2.0)
    return float(coefficient * accumulated[-1, -1] / normalizer)


def run_asr(command_template: str, audio_path: Path) -> str:
    command = command_template.format(audio=str(audio_path))
    completed = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    return normalize_russian_text(completed.stdout)


def audio_duration(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def evaluate(args: argparse.Namespace) -> None:
    metadata_path = resolve_path(args.metadata)
    synth_dir = resolve_path(args.synth_dir)
    output_dir = resolve_path(args.output_dir)
    reference_wavs = resolve_path(args.reference_wavs) if args.reference_wavs else metadata_path.parent / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mcd_values = []
    wer_values = []
    cer_values = []

    for item_id, text in read_metadata(metadata_path):
        reference_path = reference_wavs / f"{item_id}.wav"
        synth_path = synth_dir / f"{item_id}.wav"
        if not reference_path.exists() or not synth_path.exists():
            continue

        mcd = dtw_mcd(reference_path, synth_path, args.sample_rate)
        mcd_values.append(mcd)
        row = {
            "id": item_id,
            "reference_text": text,
            "mcd": round(mcd, 4),
            "reference_duration_sec": round(audio_duration(reference_path), 4),
            "synth_duration_sec": round(audio_duration(synth_path), 4),
        }

        if args.asr_command:
            hypothesis = run_asr(args.asr_command, synth_path)
            row["asr_hypothesis"] = hypothesis
            row["wer"] = round(wer(text, hypothesis), 4)
            row["cer"] = round(cer(text, hypothesis), 4)
            wer_values.append(row["wer"])
            cer_values.append(row["cer"])

        rows.append(row)

    summary = {
        "items": len(rows),
        "mean_mcd": round(float(np.nanmean(mcd_values)), 4) if mcd_values else None,
        "mean_wer": round(float(np.mean(wer_values)), 4) if wer_values else None,
        "mean_cer": round(float(np.mean(cer_values)), 4) if cer_values else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "per_sample_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def demo_asr_command() -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav") as _:
        return "python my_asr.py --audio {audio}"


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
