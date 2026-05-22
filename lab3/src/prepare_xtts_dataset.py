from __future__ import annotations

# flake8: noqa: E501

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from common import normalize_russian_text, resolve_path


@dataclass(frozen=True)
class MetadataRow:
    item_id: str
    text: str


@dataclass(frozen=True)
class PreparedRow:
    audio_file: str
    text: str
    speaker_name: str
    duration_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a prepared LJSpeech-style dataset to the Coqui XTTS fine-tuning format."
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed/ru_ss_ljspeech",
        help="Directory with metadata.csv, metadata_train.csv, metadata_val.csv and wavs/.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/ru_ss_xtts",
        help="Directory for XTTS metadata and normalized wav files.",
    )
    parser.add_argument("--speaker-name", default="ru_single_speaker", help="Speaker name written to XTTS CSV.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="XTTS GPT training sample rate.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split if train/val files are absent.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on utterances.")
    parser.add_argument("--min-duration", type=float, default=1.0, help="Drop shorter clips.")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=11.6,
        help="Drop longer clips; XTTS demo defaults to roughly 11.6 seconds.",
    )
    return parser.parse_args()


def read_ljspeech_metadata(path: Path) -> list[MetadataRow]:
    rows: list[MetadataRow] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="|")
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            text = row[2] if len(row) > 2 and row[2].strip() else row[1]
            normalized = normalize_russian_text(text)
            if normalized:
                rows.append(MetadataRow(item_id=row[0].strip(), text=normalized))
    return rows


def load_split(input_dir: Path, val_ratio: float, seed: int, limit: int | None) -> tuple[list[MetadataRow], list[MetadataRow]]:
    train_path = input_dir / "metadata_train.csv"
    val_path = input_dir / "metadata_val.csv"
    if train_path.exists() and val_path.exists():
        train_rows = read_ljspeech_metadata(train_path)
        val_rows = read_ljspeech_metadata(val_path)
    else:
        rows = read_ljspeech_metadata(input_dir / "metadata.csv")
        random.Random(seed).shuffle(rows)
        val_count = max(1, round(len(rows) * val_ratio))
        val_rows = rows[:val_count]
        train_rows = rows[val_count:]

    if limit is not None:
        train_limit = max(1, int(limit * (1.0 - val_ratio)))
        val_limit = max(1, limit - train_limit)
        train_rows = train_rows[:train_limit]
        val_rows = val_rows[:val_limit]
    return train_rows, val_rows


def convert_audio(source: Path, target: Path, sample_rate: int) -> float:
    waveform, source_sample_rate = torchaudio.load(source)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sample_rate, sample_rate)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, waveform.squeeze(0).numpy(), sample_rate)
    return waveform.shape[1] / sample_rate


def prepare_rows(
    rows: list[MetadataRow],
    input_wavs_dir: Path,
    output_wavs_dir: Path,
    sample_rate: int,
    speaker_name: str,
    min_duration: float,
    max_duration: float,
    split_name: str,
) -> tuple[list[PreparedRow], list[str]]:
    prepared: list[PreparedRow] = []
    skipped: list[str] = []
    for row in tqdm(rows, desc=f"Preparing {split_name}"):
        source = input_wavs_dir / f"{row.item_id}.wav"
        target_rel = f"wavs/{row.item_id}.wav"
        target = output_wavs_dir / f"{row.item_id}.wav"
        if not source.exists():
            skipped.append(f"{row.item_id}: missing source wav")
            continue
        try:
            duration = convert_audio(source, target, sample_rate)
        except (RuntimeError, ValueError, OSError) as exc:
            skipped.append(f"{row.item_id}: {exc}")
            continue
        if duration < min_duration or duration > max_duration:
            target.unlink(missing_ok=True)
            skipped.append(f"{row.item_id}: duration {duration:.2f}s outside range")
            continue
        prepared.append(
            PreparedRow(
                audio_file=target_rel,
                text=row.text,
                speaker_name=speaker_name,
                duration_sec=duration,
            )
        )
    return prepared, skipped


def write_xtts_metadata(rows: list[PreparedRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_file": row.audio_file,
                    "text": row.text,
                    "speaker_name": row.speaker_name,
                }
            )


def choose_speaker_reference(rows: list[PreparedRow], output_dir: Path) -> str | None:
    candidates = [row for row in rows if row.duration_sec >= 3.0]
    if not candidates:
        return None
    selected = max(candidates, key=lambda row: row.duration_sec)
    return str(output_dir / selected.audio_file)


def prepare_xtts_dataset(args: argparse.Namespace) -> None:
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    input_wavs_dir = input_dir / "wavs"
    output_wavs_dir = output_dir / "wavs"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_dir}")
    if not input_wavs_dir.exists():
        raise FileNotFoundError(f"Input wavs directory does not exist: {input_wavs_dir}")

    train_rows, val_rows = load_split(input_dir, args.val_ratio, args.seed, args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wavs_dir.mkdir(parents=True, exist_ok=True)

    train_prepared, train_skipped = prepare_rows(
        train_rows,
        input_wavs_dir,
        output_wavs_dir,
        args.sample_rate,
        args.speaker_name,
        args.min_duration,
        args.max_duration,
        "train",
    )
    val_prepared, val_skipped = prepare_rows(
        val_rows,
        input_wavs_dir,
        output_wavs_dir,
        args.sample_rate,
        args.speaker_name,
        args.min_duration,
        args.max_duration,
        "validation",
    )

    write_xtts_metadata(train_prepared, output_dir / "metadata_train.csv")
    write_xtts_metadata(val_prepared, output_dir / "metadata_eval.csv")

    speaker_reference = choose_speaker_reference(val_prepared + train_prepared, output_dir)
    if speaker_reference:
        (output_dir / "speaker_ref.txt").write_text(f"{speaker_reference}\n", encoding="utf-8")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "sample_rate": args.sample_rate,
        "speaker_name": args.speaker_name,
        "train": len(train_prepared),
        "validation": len(val_prepared),
        "skipped": len(train_skipped) + len(val_skipped),
        "speaker_reference": speaker_reference,
        "skipped_examples": (train_skipped + val_skipped)[:20],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    prepare_xtts_dataset(parse_args())


if __name__ == "__main__":
    main()
