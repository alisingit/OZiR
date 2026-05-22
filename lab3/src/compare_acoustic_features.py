from __future__ import annotations

# flake8: noqa: E501

import argparse
import csv
import json
import math
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from common import normalize_russian_text, resolve_path


DB_TO_LN_SCALE = 10.0 / math.log(10.0)
MCD_COEFFICIENT = math.sqrt(2.0) * DB_TO_LN_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare informative acoustic features for reference and generated speech."
    )
    parser.add_argument("--metadata", required=True, help="Validation metadata.csv in LJSpeech format.")
    parser.add_argument(
        "--reference-wavs",
        default=None,
        help="Directory with reference wavs. Defaults to metadata parent / wavs.",
    )
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        help="Generated system in name=path form, for example pretrained=outputs/audio/xtts_pretrained_val.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for CSV, JSON and plots.")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Analysis sample rate.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for quick checks.")
    return parser.parse_args()


def read_metadata(path: Path, max_items: int | None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="|")
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                rows.append((row[0].strip(), normalize_russian_text(row[-1])))
            if max_items is not None and len(rows) >= max_items:
                break
    return rows


def parse_systems(values: list[str]) -> dict[str, Path]:
    systems: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"System must have name=path format: {value}")
        name, path = value.split("=", 1)
        systems[name.strip()] = resolve_path(path.strip())
    return systems


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sample_rate, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    return wav


def mfcc_features(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    features = librosa.feature.mfcc(
        y=wav,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
    )
    return features.T / DB_TO_LN_SCALE


def dtw_mcd(reference_wav: np.ndarray, candidate_wav: np.ndarray, sample_rate: int) -> float:
    reference = mfcc_features(reference_wav, sample_rate)
    candidate = mfcc_features(candidate_wav, sample_rate)
    if len(reference) == 0 or len(candidate) == 0:
        return math.nan
    distances = cdist(reference[:, 1:], candidate[:, 1:], metric="euclidean")
    accumulated, path = librosa.sequence.dtw(C=distances, backtrack=True)
    return float(MCD_COEFFICIENT * accumulated[-1, -1] / max(len(path), 1))


def pitch_stats(wav: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    if len(wav) < sample_rate // 4:
        return math.nan, math.nan, 0.0
    f0, _, _ = librosa.pyin(
        wav,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
    )
    voiced = f0[~np.isnan(f0)]
    if len(voiced) == 0:
        return math.nan, math.nan, 0.0
    return float(np.mean(voiced)), float(np.std(voiced)), float(len(voiced) / len(f0))


def extract_features(path: Path, sample_rate: int) -> dict[str, float]:
    wav = load_audio(path, sample_rate)
    duration = len(wav) / sample_rate
    rms = librosa.feature.rms(y=wav)[0]
    zcr = librosa.feature.zero_crossing_rate(y=wav)[0]
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sample_rate)[0]
    mel = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_mels=80)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mfcc = mfcc_features(wav, sample_rate)
    f0_mean, f0_std, voiced_ratio = pitch_stats(wav, sample_rate)

    features = {
        "duration_sec": duration,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "log_mel_mean": float(np.mean(log_mel)),
        "log_mel_std": float(np.std(log_mel)),
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "voiced_ratio": voiced_ratio,
    }
    for index in range(min(13, mfcc.shape[1])):
        features[f"mfcc_{index}_mean"] = float(np.mean(mfcc[:, index]))
        features[f"mfcc_{index}_std"] = float(np.std(mfcc[:, index]))
    return features


def summarize(rows: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    dataframe = pd.DataFrame(rows)
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    summary: dict[str, dict[str, float]] = {}
    for system_name, group in dataframe.groupby("system"):
        summary[system_name] = {
            f"{column}_mean": round(float(group[column].mean()), 6)
            for column in numeric_columns
            if not group[column].isna().all()
        }
    return summary


def write_plots(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    dataframe = pd.DataFrame(rows)
    metrics = ["duration_sec", "rms_mean", "spectral_centroid_mean", "f0_mean", "voiced_ratio", "mcd_to_reference"]
    for metric in metrics:
        if metric not in dataframe or dataframe[metric].isna().all():
            continue
        plt.figure(figsize=(8, 4))
        dataframe.boxplot(column=metric, by="system", grid=False, rot=20)
        plt.title(metric)
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=160)
        plt.close()


def compare(args: argparse.Namespace) -> None:
    metadata_path = resolve_path(args.metadata)
    reference_wavs = resolve_path(args.reference_wavs) if args.reference_wavs else metadata_path.parent / "wavs"
    systems = parse_systems(args.system)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for item_id, text in read_metadata(metadata_path, args.max_items):
        reference_path = reference_wavs / f"{item_id}.wav"
        if not reference_path.exists():
            continue
        reference_wav = load_audio(reference_path, args.sample_rate)
        reference_features = extract_features(reference_path, args.sample_rate)
        rows.append(
            {
                "id": item_id,
                "system": "reference",
                "text": text,
                "mcd_to_reference": 0.0,
                **reference_features,
            }
        )
        for system_name, system_dir in systems.items():
            candidate_path = system_dir / f"{item_id}.wav"
            if not candidate_path.exists():
                continue
            candidate_wav = load_audio(candidate_path, args.sample_rate)
            features = extract_features(candidate_path, args.sample_rate)
            rows.append(
                {
                    "id": item_id,
                    "system": system_name,
                    "text": text,
                    "mcd_to_reference": dtw_mcd(reference_wav, candidate_wav, args.sample_rate),
                    **features,
                }
            )

    if not rows:
        raise RuntimeError("No comparable audio files found.")

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_dir / "per_file_features.csv", index=False)
    summary = summarize(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame.from_dict(summary, orient="index").to_csv(output_dir / "summary.csv")
    write_plots(rows, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    compare(parse_args())


if __name__ == "__main__":
    main()
