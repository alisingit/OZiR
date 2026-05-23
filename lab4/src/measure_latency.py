import argparse
import csv
import tempfile
import time
from pathlib import Path

from generate_audio_runs import (
    configure_local_model_cache,
    ensure_command_available,
    load_coqui_model,
    load_silero_model,
    synthesize_with_coqui,
    synthesize_with_espeak,
    synthesize_with_piper,
    synthesize_with_rhvoice,
    synthesize_with_silero,
)


TEST_TEXT = "Сегодня проводится экспериментальная оценка синтеза русской речи."
SYSTEM_NAMES = {
    "silero": "Silero TTS",
    "rhvoice": "RHVoice",
    "piper": "Piper",
    "coqui_xtts": "Coqui XTTS",
    "espeak_ng": "eSpeak NG",
}
COMMENTS = {
    "silero": "автоматический замер времени синтеза после прогрева модели",
    "rhvoice": "автоматический замер времени выполнения локального синтеза",
    "piper": "автоматический замер времени ONNX-синтеза после прогрева",
    "coqui_xtts": "автоматический замер времени синтеза после прогрева крупной модели",
    "espeak_ng": "автоматический замер времени формантного синтеза",
}
FIELDNAMES = ["system", "repeat", "text", "latency_seconds", "comment"]


def build_synthesizer(args):
    if args.engine == "silero":
        model = load_silero_model()
        return lambda text, output_path: synthesize_with_silero(model, text, output_path, args.voice)

    if args.engine == "rhvoice":
        rhvoice_path = ensure_command_available("RHVoice-test")
        return lambda text, output_path: synthesize_with_rhvoice(rhvoice_path, text, output_path, args.voice)

    if args.engine == "piper":
        piper_path = ensure_command_available("piper")
        return lambda text, output_path: synthesize_with_piper(piper_path, text, output_path, args.piper_model)

    if args.engine == "coqui_xtts":
        model = load_coqui_model(args.coqui_model_dir)
        return lambda text, output_path: synthesize_with_coqui(
            model,
            text,
            output_path,
            args.coqui_speaker_wav,
        )

    if args.engine == "espeak_ng":
        espeak_path = ensure_command_available("espeak-ng")
        return lambda text, output_path: synthesize_with_espeak(espeak_path, text, output_path, args.voice)

    raise ValueError(f"Неизвестный синтезатор: {args.engine}")


def measure_repeats(args):
    synthesize = build_synthesizer(args)
    rows = []

    with tempfile.TemporaryDirectory(prefix=f"lab4-latency-{args.engine}-") as temp_dir:
        temp_path = Path(temp_dir)

        for warmup_index in range(args.warmup):
            synthesize(args.text, temp_path / f"warmup_{warmup_index + 1}.wav")

        for repeat in range(1, args.repeats + 1):
            output_path = temp_path / f"repeat_{repeat}.wav"
            started_at = time.perf_counter()
            synthesize(args.text, output_path)
            elapsed = time.perf_counter() - started_at
            rows.append(
                {
                    "system": SYSTEM_NAMES[args.engine],
                    "repeat": repeat,
                    "text": args.text,
                    "latency_seconds": f"{elapsed:.2f}",
                    "comment": COMMENTS[args.engine],
                }
            )

    return rows


def write_rows(path, rows, append):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not append or not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args():
    lab_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Автоматический замер задержки синтеза для лабораторной работы №4."
    )
    parser.add_argument(
        "--engine",
        choices=["silero", "rhvoice", "piper", "coqui_xtts", "espeak_ng"],
        required=True,
    )
    parser.add_argument("--output", type=Path, default=lab_dir / "data" / "latency.csv")
    parser.add_argument("--text", default=TEST_TEXT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--voice", default="")
    parser.add_argument("--piper-model", type=Path)
    parser.add_argument("--coqui-model-dir", type=Path)
    parser.add_argument("--coqui-speaker-wav", type=Path)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main():
    configure_local_model_cache()
    args = parse_args()
    rows = measure_repeats(args)
    write_rows(args.output, rows, args.append)
    print(f"Записано замеров: {len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
