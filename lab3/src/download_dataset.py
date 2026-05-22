from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from common import resolve_path


HF_REPO_ID = "niobures/russian-single-speaker-speech-dataset"
TRANSCRIPT_FILENAME = "transcript.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a single-speaker Russian TTS subset from HuggingFace "
            f"({HF_REPO_ID}) into data/raw."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/ru_ss",
        help="Target directory for raw audio and metadata.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1500,
        help="Cap on number of utterances to download.",
    )
    parser.add_argument(
        "--folder",
        default="early_short_stories",
        help="Subfolder of the HuggingFace dataset to take samples from.",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=15,
        help="Skip utterances with normalized text shorter than this many characters.",
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=200,
        help="Skip utterances with normalized text longer than this many characters.",
    )
    return parser.parse_args()


def parse_transcript(transcript_path: Path) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with transcript_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            audio_rel_path, _raw_text, normalized_text = parts[0], parts[1], parts[2]
            folder = audio_rel_path.split("/", 1)[0]
            grouped[folder].append((audio_rel_path, normalized_text.strip()))
    return grouped


def download_subset(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    wavs_dir = output_dir / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = Path(
        hf_hub_download(HF_REPO_ID, TRANSCRIPT_FILENAME, repo_type="dataset")
    )

    grouped = parse_transcript(transcript_path)
    if args.folder not in grouped:
        available = ", ".join(sorted(grouped.keys()))
        raise ValueError(f"Folder '{args.folder}' not in dataset. Available: {available}")

    candidates = grouped[args.folder]
    print(f"Total utterances in '{args.folder}': {len(candidates)}")

    selected: list[tuple[str, str]] = []
    for audio_rel, text in candidates:
        if not text:
            continue
        if not (args.min_text_len <= len(text) <= args.max_text_len):
            continue
        selected.append((audio_rel, text))
        if len(selected) >= args.max_samples:
            break

    print(f"Will download {len(selected)} samples to {wavs_dir}")

    metadata_path = output_dir / "metadata.csv"
    written: list[tuple[str, str]] = []
    for audio_rel, text in tqdm(selected, desc="Downloading audio"):
        item_id = Path(audio_rel).stem
        local_audio = wavs_dir / f"{item_id}.wav"
        if local_audio.exists() and local_audio.stat().st_size > 0:
            written.append((item_id, text))
            continue
        try:
            downloaded = hf_hub_download(HF_REPO_ID, audio_rel, repo_type="dataset")
        except (OSError, RuntimeError) as exc:
            print(f"Skip {audio_rel}: {exc}")
            continue
        shutil.copyfile(downloaded, local_audio)
        written.append((item_id, text))

    with metadata_path.open("w", encoding="utf-8") as file:
        for item_id, text in written:
            file.write(f"{item_id}|{text}|{text}\n")
    print(f"Wrote metadata to {metadata_path} ({len(written)} rows).")


def main() -> None:
    download_subset(parse_args())


if __name__ == "__main__":
    main()
