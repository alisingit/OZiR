from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import read_json, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Coqui TTS config and optionally start training.")
    parser.add_argument("--config", required=True, help="Experiment config from configs/*.json.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate paths and write generated Coqui config. Does not start training.",
    )
    return parser.parse_args()


def build_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    return {
        "formatter": dataset.get("formatter", "ljspeech"),
        "dataset_name": dataset.get("name", "russian_tts"),
        "path": str(resolve_path(dataset["path"])),
        "meta_file_train": dataset.get("meta_file_train", "metadata_train.csv"),
        "meta_file_val": dataset.get("meta_file_val", "metadata_val.csv"),
        "language": dataset.get("language", "ru"),
    }


def build_common_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    audio = config["audio"]
    hyperparameters = config["hyperparameters"]
    output_path = resolve_path(training["output_path"])
    log_path = resolve_path(training["log_path"])

    return {
        "run_name": config["experiment_name"],
        "output_path": str(output_path),
        "logger_uri": str(log_path),
        "datasets": [build_dataset_config(config)],
        "audio": {
            "sample_rate": audio["sample_rate"],
            "num_mels": audio["num_mels"],
            "fft_size": audio["fft_size"],
            "hop_length": audio["hop_length"],
            "win_length": audio["win_length"],
            "mel_fmin": audio["mel_fmin"],
            "mel_fmax": audio["mel_fmax"],
        },
        "batch_size": training["batch_size"],
        "eval_batch_size": training["eval_batch_size"],
        "num_loader_workers": training["num_loader_workers"],
        "num_eval_loader_workers": training["num_eval_loader_workers"],
        "epochs": training["epochs"],
        "save_step": training["save_step"],
        "print_step": training["print_step"],
        "mixed_precision": training["mixed_precision"],
        "test_delay_epochs": -1,
        "run_eval": True,
        "save_checkpoints": True,
        "save_best_after": 1000,
        "text_cleaner": hyperparameters["text_cleaner"],
        "use_phonemes": hyperparameters["use_phonemes"],
        "phoneme_language": "ru",
        "training_seed": training["seed"],
    }


def build_vits_config(config: dict[str, Any]) -> dict[str, Any]:
    coqui_config = build_common_config(config)
    hyperparameters = config["hyperparameters"]
    coqui_config.update(
        {
            "model": "vits",
            "lr_gen": hyperparameters["lr_gen"],
            "lr_disc": hyperparameters["lr_disc"],
            "mel_loss_alpha": hyperparameters["mel_loss_alpha"],
            "kl_loss_alpha": hyperparameters["kl_loss_alpha"],
            "dur_loss_alpha": hyperparameters["dur_loss_alpha"],
            "compute_input_seq_cache": True,
            "return_wav": True,
        }
    )
    return coqui_config


def build_tacotron2_config(config: dict[str, Any]) -> dict[str, Any]:
    coqui_config = build_common_config(config)
    hyperparameters = config["hyperparameters"]
    coqui_config.update(
        {
            "model": "tacotron2",
            "lr": hyperparameters["lr"],
            "r": hyperparameters["r"],
            "grad_clip": hyperparameters["grad_clip"],
            "decoder_loss_alpha": 0.25,
            "postnet_loss_alpha": 0.25,
            "ga_alpha": 5.0,
            "use_guided_attention_loss": True,
        }
    )
    return coqui_config


def build_coqui_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"].lower()
    if model == "vits":
        return build_vits_config(config)
    if model == "tacotron2":
        return build_tacotron2_config(config)
    raise ValueError(f"Unsupported model: {model}")


def validate_dataset(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    dataset_path = resolve_path(dataset["path"])
    train_metadata = dataset_path / dataset.get("meta_file_train", "metadata_train.csv")
    val_metadata = dataset_path / dataset.get("meta_file_val", "metadata_val.csv")
    wavs_path = dataset_path / "wavs"
    missing = [path for path in (dataset_path, train_metadata, val_metadata, wavs_path) if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Dataset is not prepared yet. Missing paths:\n{joined}")


def run_training(generated_config_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "TTS.bin.train_tts",
        "--config_path",
        str(generated_config_path),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = read_json(config_path)
    coqui_config = build_coqui_config(config)

    generated_config_path = resolve_path(config["training"]["output_path"]) / "generated_coqui_config.json"
    write_json(coqui_config, generated_config_path)
    print(f"Generated Coqui config: {generated_config_path}")

    validate_dataset(config)
    if args.dry_run:
        print("Dry run finished: dataset paths are present, training was not started.")
        return
    run_training(generated_config_path)


if __name__ == "__main__":
    main()
