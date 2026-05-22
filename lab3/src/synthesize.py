from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from common import normalize_russian_text, resolve_path, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize control Russian texts with a trained Coqui TTS model.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--config", required=True, help="Path to generated Coqui config.json.")
    parser.add_argument("--texts", default="texts/control_texts.json", help="JSON list with id/text fields.")
    parser.add_argument("--output-dir", required=True, help="Directory for synthesized wav files.")
    parser.add_argument("--spectrogram-dir", default=None, help="Directory for mel-spectrogram png files.")
    parser.add_argument("--speaker", default=None, help="Optional speaker id/name for multi-speaker models.")
    return parser.parse_args()


def load_texts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = [{"id": key, "text": value} for key, value in data.items()]
    return data


def save_mel_spectrogram(wav_path: Path, output_path: Path, sample_rate: int) -> None:
    wav, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    mel = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sample_rate, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def synthesize(args: argparse.Namespace) -> None:
    from TTS.api import TTS

    checkpoint = resolve_path(args.checkpoint)
    config = resolve_path(args.config)
    texts_path = resolve_path(args.texts)
    output_dir = resolve_path(args.output_dir)
    spectrogram_dir = resolve_path(args.spectrogram_dir) if args.spectrogram_dir else output_dir.parent / "spectrograms"
    output_dir.mkdir(parents=True, exist_ok=True)
    spectrogram_dir.mkdir(parents=True, exist_ok=True)

    tts = TTS(model_path=str(checkpoint), config_path=str(config), progress_bar=True, gpu=False)
    metrics: list[dict[str, float | str]] = []

    for item in load_texts(texts_path):
        text = normalize_russian_text(item["text"])
        item_id = item.get("id") or slugify(text[:50])
        wav_path = output_dir / f"{item_id}.wav"
        start = time.perf_counter()
        kwargs = {"text": text, "file_path": str(wav_path)}
        if args.speaker:
            kwargs["speaker"] = args.speaker
        tts.tts_to_file(**kwargs)
        elapsed = time.perf_counter() - start
        audio, sample_rate = sf.read(wav_path)
        duration = len(audio) / sample_rate
        save_mel_spectrogram(wav_path, spectrogram_dir / f"{item_id}.png", sample_rate)
        metrics.append(
            {
                "id": item_id,
                "text": text,
                "duration_sec": round(duration, 4),
                "synthesis_sec": round(elapsed, 4),
                "real_time_factor": round(elapsed / duration, 4) if duration else 0.0,
            }
        )

    metrics_path = output_dir.parent / "synthesis_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved audio to {output_dir}")
    print(f"Saved spectrograms to {spectrogram_dir}")
    print(f"Saved synthesis metrics to {metrics_path}")


def main() -> None:
    synthesize(parse_args())


if __name__ == "__main__":
    main()
