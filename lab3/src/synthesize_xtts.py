from __future__ import annotations

# flake8: noqa: E501

import argparse
import csv
import json
import re
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from common import normalize_russian_text, resolve_path, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize Russian control texts with XTTS-v2.")
    parser.add_argument("--checkpoint", required=True, help="XTTS checkpoint: original model.pth or fine-tuned best_model.pth.")
    parser.add_argument("--config", required=True, help="XTTS config.json.")
    parser.add_argument("--vocab", required=True, help="XTTS vocab.json.")
    parser.add_argument("--speaker-wav", required=True, help="Reference wav or a .txt file containing the path.")
    parser.add_argument("--texts", default="texts/control_texts.json", help="JSON list with id/text fields.")
    parser.add_argument("--metadata", default=None, help="Optional LJSpeech metadata.csv to synthesize validation ids.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated wav files.")
    parser.add_argument("--spectrogram-dir", default=None, help="Directory for mel-spectrogram PNG files.")
    parser.add_argument("--language", default="ru", help="XTTS language code.")
    parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top-p", type=float, default=0.85, help="Top-p sampling.")
    parser.add_argument("--force-cpu", action="store_true", help="Disable CUDA for inference.")
    return parser.parse_args()


def load_texts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        return [{"id": key, "text": value} for key, value in data.items()]
    return data


def load_metadata_texts(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="|")
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            text = row[2] if len(row) > 2 and row[2].strip() else row[1]
            rows.append({"id": row[0].strip(), "text": text})
    return rows


def resolve_speaker_wav(path: Path) -> Path:
    if path.suffix == ".txt":
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Speaker reference file is empty: {path}")
        return resolve_path(content)
    return path


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


def split_text(text: str, max_chars: int = 170) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                sentence[index : index + max_chars].strip()
                for index in range(0, len(sentence), max_chars)
            )
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks or [text]


def load_xtts_model(checkpoint: Path, config_path: Path, vocab_path: Path, use_gpu: bool):
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(checkpoint), vocab_path=str(vocab_path), use_deepspeed=False)
    if use_gpu:
        model.cuda()
    return model, config


def synthesize(args: argparse.Namespace) -> None:
    checkpoint = resolve_path(args.checkpoint)
    config_path = resolve_path(args.config)
    vocab_path = resolve_path(args.vocab)
    texts_path = resolve_path(args.texts)
    metadata_path = resolve_path(args.metadata) if args.metadata else None
    speaker_wav = resolve_speaker_wav(resolve_path(args.speaker_wav))
    output_dir = resolve_path(args.output_dir)
    spectrogram_dir = resolve_path(args.spectrogram_dir) if args.spectrogram_dir else output_dir.parent / "spectrograms" / output_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    spectrogram_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = (not args.force_cpu) and torch.cuda.is_available()
    print(f"Inference device: {'cuda' if use_gpu else 'cpu'}")
    model, config = load_xtts_model(checkpoint, config_path, vocab_path, use_gpu)

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=str(speaker_wav),
        gpt_cond_len=config.gpt_cond_len,
        max_ref_length=config.max_ref_len,
        sound_norm_refs=config.sound_norm_refs,
    )

    metrics: list[dict[str, float | str]] = []
    sample_rate = config.audio.output_sample_rate
    items = load_metadata_texts(metadata_path) if metadata_path else load_texts(texts_path)
    for item in items:
        text = normalize_russian_text(item["text"])
        item_id = item.get("id") or slugify(text[:50])
        wav_path = output_dir / f"{item_id}.wav"
        start = time.perf_counter()
        wav_chunks = []
        silence = np.zeros(int(sample_rate * 0.18), dtype=np.float32)
        for chunk in split_text(text):
            out = model.inference(
                text=chunk,
                language=args.language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            wav_chunks.append(np.asarray(out["wav"], dtype=np.float32))
            wav_chunks.append(silence)
        wav_array = np.concatenate(wav_chunks[:-1] if len(wav_chunks) > 1 else wav_chunks)
        elapsed = time.perf_counter() - start
        wav = torch.tensor(wav_array).unsqueeze(0)
        torchaudio.save(str(wav_path), wav, sample_rate)
        duration = wav.shape[-1] / sample_rate
        save_mel_spectrogram(wav_path, spectrogram_dir / f"{item_id}.png", sample_rate)
        metrics.append(
            {
                "id": item_id,
                "text": text,
                "speaker_wav": str(speaker_wav),
                "duration_sec": round(duration, 4),
                "synthesis_sec": round(elapsed, 4),
                "real_time_factor": round(elapsed / duration, 4) if duration else 0.0,
            }
        )

    metrics_path = output_dir.parent / f"{output_dir.name}_synthesis_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved audio to {output_dir}")
    print(f"Saved spectrograms to {spectrogram_dir}")
    print(f"Saved synthesis metrics to {metrics_path}")


def main() -> None:
    synthesize(parse_args())


if __name__ == "__main__":
    main()
