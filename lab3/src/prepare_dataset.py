from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from common import normalize_russian_text, resolve_path


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass(frozen=True)
class Utterance:
    audio_path: Path
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a Russian TTS dataset in LJSpeech/Coqui format."
    )
    parser.add_argument("--raw-dir", required=True, help="Directory with raw audio and transcripts.")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed dataset.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--min-duration", type=float, default=0.4, help="Drop shorter audio files.")
    parser.add_argument("--max-duration", type=float, default=20.0, help="Drop longer audio files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of utterances.")
    return parser.parse_args()


def load_metadata(raw_dir: Path) -> list[Utterance]:
    jsonl_path = raw_dir / "metadata.jsonl"
    csv_path = raw_dir / "metadata.csv"

    if jsonl_path.exists():
        return load_jsonl_metadata(jsonl_path, raw_dir)
    if csv_path.exists():
        return load_csv_metadata(csv_path, raw_dir)
    return load_audio_txt_pairs(raw_dir)


def load_jsonl_metadata(path: Path, raw_dir: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            audio_value = item.get("audio_filepath") or item.get("audio") or item.get("path")
            text_value = item.get("text") or item.get("sentence") or item.get("transcript")
            if not audio_value or not text_value:
                continue
            utterances.append(Utterance(resolve_audio_path(raw_dir, audio_value), text_value))
    return utterances


def load_csv_metadata(path: Path, raw_dir: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        delimiter = "|" if "|" in sample else ","
        reader = csv.reader(file, delimiter=delimiter)
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            audio_value = row[0].strip()
            text_value = row[2].strip() if len(row) > 2 and row[2].strip() else row[1].strip()
            if audio_value.lower() in {"wav", "audio", "path", "audio_filepath"}:
                continue
            utterances.append(Utterance(resolve_audio_path(raw_dir, audio_value), text_value))
    return utterances


def load_audio_txt_pairs(raw_dir: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    for audio_path in sorted(raw_dir.rglob("*")):
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        text_path = audio_path.with_suffix(".txt")
        if text_path.exists():
            utterances.append(Utterance(audio_path, text_path.read_text(encoding="utf-8")))
    return utterances


def resolve_audio_path(raw_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    direct = raw_dir / candidate
    if direct.exists():
        return direct
    wavs = raw_dir / "wavs" / candidate
    if wavs.exists():
        return wavs
    if candidate.suffix:
        return direct
    return raw_dir / "wavs" / f"{candidate}.wav"


def convert_audio(source: Path, target: Path, sample_rate: int) -> float:
    waveform, source_sample_rate = torchaudio.load(source)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sample_rate, sample_rate)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, waveform.squeeze(0).numpy(), sample_rate)
    return waveform.shape[1] / sample_rate


def write_metadata(rows: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="|", lineterminator="\n")
        for item_id, text in rows:
            writer.writerow([item_id, text, text])


def prepare_dataset(args: argparse.Namespace) -> None:
    raw_dir = resolve_path(args.raw_dir)
    output_dir = resolve_path(args.output_dir)
    wav_dir = output_dir / "wavs"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")

    utterances = load_metadata(raw_dir)
    if args.limit:
        utterances = utterances[: args.limit]
    if not utterances:
        raise RuntimeError(f"No utterances found in {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    for index, utterance in enumerate(tqdm(utterances, desc="Preparing audio"), start=1):
        text = normalize_russian_text(utterance.text)
        if not utterance.audio_path.exists() or not text:
            skipped.append(str(utterance.audio_path))
            continue
        item_id = f"ru_{index:06d}"
        target = wav_dir / f"{item_id}.wav"
        try:
            duration = convert_audio(utterance.audio_path, target, args.sample_rate)
        except (RuntimeError, ValueError, OSError):
            skipped.append(str(utterance.audio_path))
            continue
        if duration < args.min_duration or duration > args.max_duration:
            target.unlink(missing_ok=True)
            skipped.append(str(utterance.audio_path))
            continue
        rows.append((item_id, text))

    random.Random(args.seed).shuffle(rows)
    val_count = max(1, round(len(rows) * args.val_ratio))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]

    write_metadata(rows, output_dir / "metadata.csv")
    write_metadata(train_rows, output_dir / "metadata_train.csv")
    write_metadata(val_rows, output_dir / "metadata_val.csv")

    summary = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "sample_rate": args.sample_rate,
        "total": len(rows),
        "train": len(train_rows),
        "validation": len(val_rows),
        "skipped": len(skipped),
        "skipped_examples": skipped[:20],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(output_dir / "metadata_train.csv", output_dir / "metadata_train_ljspeech.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    prepare_dataset(parse_args())


if __name__ == "__main__":
    main()
