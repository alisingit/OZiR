import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_VOICE = "Milena"
SYSTEM_ENGINES = {
    "Silero TTS": "silero",
    "RHVoice": "rhvoice",
    "Piper": "piper",
    "Coqui XTTS": "coqui_xtts",
    "eSpeak NG": "espeak_ng",
}


def configure_local_model_cache():
    lab_dir = Path(__file__).resolve().parents[1]
    cache_dir = lab_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("TTS_HOME", str(cache_dir / "tts"))
    os.environ.setdefault("COQUI_TOS_AGREED", "1")


def read_sentences(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"category", "item", "text", "targets", "target_count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            joined = ", ".join(sorted(missing_columns))
            raise ValueError(f"В файле {path} отсутствуют колонки: {joined}")
        return list(reader)


def ensure_macos_say_available():
    say_path = shutil.which("say")
    if say_path is None:
        raise RuntimeError("Команда macOS say не найдена, аудио не может быть сгенерировано.")
    return say_path


def ensure_command_available(command):
    command_path = shutil.which(command)
    if command_path is not None:
        return command_path

    candidate_dirs = [Path(sys.executable).parent]
    lab_venv_bin = Path(__file__).resolve().parents[1] / ".venv" / "bin"
    candidate_dirs.append(lab_venv_bin)

    for candidate_dir in candidate_dirs:
        venv_command_path = candidate_dir / command
        if venv_command_path.exists():
            return str(venv_command_path)

    raise RuntimeError(f"Команда {command} не найдена.")


def synthesize_with_say(say_path, text, output_path, voice):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [say_path, "-v", voice, "-o", str(output_path), text]
    subprocess.run(command, check=True)


def synthesize_with_espeak(espeak_path, text, output_path, voice):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [espeak_path, "-v", voice, "-w", str(output_path), text]
    subprocess.run(command, check=True)


def synthesize_with_rhvoice(rhvoice_path, text, output_path, voice):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [rhvoice_path, "-o", str(output_path)]
    if voice:
        command.extend(["-p", voice])
    subprocess.run(command, input=text, text=True, check=True)


def synthesize_with_piper(piper_path, text, output_path, model_path):
    if model_path is None:
        raise ValueError("Для Piper нужно передать путь к модели через --piper-model.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [piper_path, "--model", str(model_path), "--output_file", str(output_path)]
    subprocess.run(command, input=text, text=True, check=True)


def load_silero_model():
    import torch

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
    )
    model.to(torch.device("cpu"))
    return model


def synthesize_with_silero(model, text, output_path, voice):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = split_text(text, max_length=160)
    if len(chunks) == 1:
        model.save_wav(text=text, speaker=voice, sample_rate=48000, audio_path=str(output_path))
        return

    ffmpeg_path = ensure_command_available("ffmpeg")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        chunk_paths = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_path = temp_path / f"chunk_{index:02d}.wav"
            model.save_wav(text=chunk, speaker=voice, sample_rate=48000, audio_path=str(chunk_path))
            chunk_paths.append(chunk_path)

        concat_list_path = temp_path / "concat.txt"
        concat_list_path.write_text(
            "\n".join(f"file '{chunk_path}'" for chunk_path in chunk_paths),
            encoding="utf-8",
        )
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def split_text(text, max_length):
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = []
    current_length = 0
    sentences = text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_length:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_length = 0
            word_chunk = []
            word_chunk_length = 0
            for word in sentence.split():
                if word_chunk and word_chunk_length + len(word) + 1 > max_length:
                    chunks.append(" ".join(word_chunk))
                    word_chunk = []
                    word_chunk_length = 0
                word_chunk.append(word)
                word_chunk_length += len(word) + 1
            if word_chunk:
                chunks.append(" ".join(word_chunk))
            continue
        if current and current_length + len(sentence) + 1 > max_length:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        current.append(sentence)
        current_length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


class LocalCoquiXtts:
    def __init__(self, model_dir):
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        if not config_path.exists() or not (model_dir / "model.pth").exists():
            raise FileNotFoundError(
                f"В {model_dir} должны быть файлы model.pth и config.json для Coqui XTTS."
            )

        self.config = XttsConfig()
        self.config.load_json(str(config_path))
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=str(model_dir), eval=True)

    def tts_to_file(self, text, speaker_wav, language, file_path):
        import soundfile as sf

        output = self.model.synthesize(
            text,
            self.config,
            speaker_wav=str(speaker_wav),
            language=language,
        )
        sample_rate = self.config.audio.get("output_sample_rate", 24000)
        sf.write(str(file_path), output["wav"], sample_rate)


def load_coqui_model(model_dir):
    if model_dir is not None:
        return LocalCoquiXtts(model_dir)

    from TTS.api import TTS

    return TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)


def synthesize_with_coqui(model, text, output_path, speaker_wav):
    if speaker_wav is None:
        raise ValueError("Для Coqui XTTS нужно передать speaker WAV через --coqui-speaker-wav.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = split_text(text, max_length=140)
    if len(chunks) == 1:
        model.tts_to_file(text=text, speaker_wav=str(speaker_wav), language="ru", file_path=str(output_path))
        return

    ffmpeg_path = ensure_command_available("ffmpeg")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        chunk_paths = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_path = temp_path / f"chunk_{index:02d}.wav"
            model.tts_to_file(text=chunk, speaker_wav=str(speaker_wav), language="ru", file_path=str(chunk_path))
            chunk_paths.append(chunk_path)

        concat_list_path = temp_path / "concat.txt"
        concat_list_path.write_text(
            "\n".join(f"file '{chunk_path}'" for chunk_path in chunk_paths),
            encoding="utf-8",
        )
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["engine", "voice", "category", "item", "targets", "target_count", "text", "audio_path"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Генерация аудиопрогонов для лабораторной работы №4.")
    parser.add_argument("--sentences", type=Path, default=project_dir / "data" / "test_sentences.csv")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "audio")
    parser.add_argument(
        "--engine",
        choices=["macos_say", "espeak_ng", "rhvoice", "piper", "silero", "coqui_xtts"],
        required=True,
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--piper-model", type=Path)
    parser.add_argument("--coqui-model-dir", type=Path)
    parser.add_argument("--coqui-speaker-wav", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    configure_local_model_cache()
    args = parse_args()
    sentences = read_sentences(args.sentences)

    executable_path = None
    model = None
    if args.engine == "macos_say":
        executable_path = ensure_macos_say_available()
        file_extension = ".aiff"
    elif args.engine == "espeak_ng":
        executable_path = ensure_command_available("espeak-ng")
        file_extension = ".wav"
    elif args.engine == "rhvoice":
        executable_path = ensure_command_available("RHVoice-test")
        file_extension = ".wav"
    elif args.engine == "piper":
        executable_path = ensure_command_available("piper")
        file_extension = ".wav"
    elif args.engine == "silero":
        model = load_silero_model()
        file_extension = ".wav"
    elif args.engine == "coqui_xtts":
        model = load_coqui_model(args.coqui_model_dir)
        file_extension = ".wav"

    manifest_rows = []
    engine_dir = args.output_dir / args.engine
    for row in sentences:
        category = row["category"]
        item = int(row["item"])
        output_path = engine_dir / category / f"{item:02d}{file_extension}"
        if args.overwrite or not output_path.exists():
            if args.engine == "macos_say":
                synthesize_with_say(executable_path, row["text"], output_path, args.voice)
            elif args.engine == "espeak_ng":
                synthesize_with_espeak(executable_path, row["text"], output_path, args.voice)
            elif args.engine == "rhvoice":
                synthesize_with_rhvoice(executable_path, row["text"], output_path, args.voice)
            elif args.engine == "piper":
                synthesize_with_piper(executable_path, row["text"], output_path, args.piper_model)
            elif args.engine == "silero":
                synthesize_with_silero(model, row["text"], output_path, args.voice)
            elif args.engine == "coqui_xtts":
                synthesize_with_coqui(model, row["text"], output_path, args.coqui_speaker_wav)
        manifest_rows.append(
            {
                "engine": args.engine,
                "voice": args.voice,
                "category": category,
                "item": item,
                "targets": row["targets"],
                "target_count": row["target_count"],
                "text": row["text"],
                "audio_path": output_path.relative_to(args.output_dir.parent),
            }
        )

    manifest_path = args.output_dir / "manifests" / f"{args.engine}.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"Сгенерировано или проверено файлов: {len(manifest_rows)}")
    print(f"Манифест: {manifest_path}")


if __name__ == "__main__":
    main()
