from __future__ import annotations

# flake8: noqa: E501

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from common import read_json, resolve_path, write_json


XTTS_REPO_ID = "coqui/XTTS-v2"
XTTS_FILES = {
    "checkpoint": "model.pth",
    "config": "config.json",
    "vocab": "vocab.json",
    "dvae": "dvae.pth",
    "mel_norm": "mel_stats.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained XTTS-v2 on a Coqui-format dataset.")
    parser.add_argument("--config", required=True, help="Path to configs/xtts_*.json.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset without training.")
    parser.add_argument(
        "--download-pretrained",
        action="store_true",
        help="Download XTTS-v2 files during dry-run. Training always downloads missing files.",
    )
    parser.add_argument(
        "--restore-path",
        default=None,
        help="Optional checkpoint path to restore model weights from.",
    )
    parser.add_argument(
        "--continue-path",
        default=None,
        help="Existing trainer run directory to continue without deleting checkpoints.",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Disable CUDA even if it is available.")
    return parser.parse_args()


def configure_device(force_cpu: bool) -> str:
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if torch.cuda.is_available():
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("CUDA is not available, falling back to CPU.")
    return "cpu"


def validate_dataset(raw_config: dict[str, Any]) -> None:
    dataset = raw_config["dataset"]
    dataset_path = resolve_path(dataset["path"])
    train_csv = dataset_path / dataset.get("meta_file_train", "metadata_train.csv")
    eval_csv = dataset_path / dataset.get("meta_file_val", "metadata_eval.csv")
    missing = [path for path in (dataset_path, train_csv, eval_csv, dataset_path / "wavs") if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"XTTS dataset is not prepared yet. Missing paths:\n{joined}")
    for metadata_path in (train_csv, eval_csv):
        with metadata_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter="|")
            if not reader.fieldnames or not {"audio_file", "text"}.issubset(reader.fieldnames):
                raise ValueError(f"{metadata_path} must contain audio_file|text columns.")


def download_pretrained_files(model_dir: Path) -> dict[str, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for key, filename in XTTS_FILES.items():
        target = model_dir / filename
        if not target.exists():
            cached = Path(hf_hub_download(XTTS_REPO_ID, filename, repo_type="model"))
            shutil.copyfile(cached, target)
        resolved[key] = target
    return resolved


def resolve_pretrained_files(model_dir: Path) -> dict[str, Path | None]:
    return {key: model_dir / filename if (model_dir / filename).exists() else None for key, filename in XTTS_FILES.items()}


def build_dataset_config(raw_config: dict[str, Any]):
    from TTS.config.shared_configs import BaseDatasetConfig

    dataset = raw_config["dataset"]
    return BaseDatasetConfig(
        formatter=dataset.get("formatter", "coqui"),
        dataset_name=dataset.get("name", "ru_single_speaker_xtts"),
        path=str(resolve_path(dataset["path"])),
        meta_file_train=dataset.get("meta_file_train", "metadata_train.csv"),
        meta_file_val=dataset.get("meta_file_val", "metadata_eval.csv"),
        language=raw_config.get("language", "ru"),
    )


def build_trainer_config(raw_config: dict[str, Any], pretrained_files: dict[str, Path]):
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig

    training = raw_config["training"]
    hyperparameters = raw_config["hyperparameters"]
    output_path = resolve_path(training["output_path"])
    log_path = resolve_path(training["log_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    max_audio_length = int(training.get("max_audio_length_sec", 11.6) * 22050)
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=max_audio_length,
        max_text_length=200,
        mel_norm_file=str(pretrained_files["mel_norm"]),
        dvae_checkpoint=str(pretrained_files["dvae"]),
        xtts_checkpoint=str(pretrained_files["checkpoint"]),
        tokenizer_file=str(pretrained_files["vocab"]),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    return GPTTrainerConfig(
        epochs=training["epochs"],
        output_path=str(output_path),
        model_args=model_args,
        run_name=raw_config["experiment_name"],
        project_name="lab3_xtts_finetune",
        run_description="XTTS-v2 fine-tuning for laboratory work 3.",
        dashboard_logger="tensorboard",
        logger_uri=str(log_path),
        audio=audio_config,
        batch_size=training["batch_size"],
        batch_group_size=48,
        eval_batch_size=training["eval_batch_size"],
        num_loader_workers=training["num_loader_workers"],
        num_eval_loader_workers=training.get("num_eval_loader_workers", 1),
        eval_split_max_size=256,
        print_step=training["print_step"],
        plot_step=training["plot_step"],
        log_model_step=training.get("plot_step", 100),
        save_step=training["save_step"],
        save_n_checkpoints=training["save_n_checkpoints"],
        save_checkpoints=True,
        print_eval=False,
        mixed_precision=training.get("mixed_precision", False),
        training_seed=training["seed"],
        optimizer=hyperparameters.get("optimizer", "AdamW"),
        optimizer_wd_only_on_weights=True,
        optimizer_params=hyperparameters["optimizer_params"],
        lr=hyperparameters["lr"],
        lr_scheduler=hyperparameters.get("lr_scheduler", "MultiStepLR"),
        lr_scheduler_params=hyperparameters["lr_scheduler_params"],
        test_sentences=[],
    )


def load_samples(raw_config: dict[str, Any], trainer_config):
    from TTS.tts.datasets import load_tts_samples

    dataset_config = build_dataset_config(raw_config)
    return load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=trainer_config.eval_split_max_size,
        eval_split_size=trainer_config.eval_split_size,
    )


def write_model_sidecars(pretrained_files: dict[str, Path], run_path: Path) -> None:
    shutil.copyfile(pretrained_files["config"], run_path / "config.json")
    shutil.copyfile(pretrained_files["vocab"], run_path / "vocab.json")


def run_training(
    raw_config: dict[str, Any],
    trainer_config,
    pretrained_files: dict[str, Path],
    device: str,
    restore_path: str | None,
    continue_path: str | None,
) -> None:
    from trainer import Trainer, TrainerArgs

    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer

    random.seed(raw_config["training"]["seed"])
    torch.manual_seed(raw_config["training"]["seed"])

    train_samples, eval_samples = load_samples(raw_config, trainer_config)
    print(f"Loaded {len(train_samples)} train and {len(eval_samples)} eval samples.")

    model = GPTTrainer.init_from_config(trainer_config)
    trainer = Trainer(
        TrainerArgs(
            restore_path=restore_path,
            continue_path=continue_path,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=raw_config["training"]["grad_accum_steps"],
        ),
        trainer_config,
        trainer_config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    run_path = Path(trainer.output_path)
    write_model_sidecars(pretrained_files, run_path)
    summary = {
        "experiment": raw_config["experiment_name"],
        "language": raw_config.get("language", "ru"),
        "device": device,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "trainer_output_path": str(run_path),
        "best_model_path": str(run_path / "best_model.pth"),
        "config_path": str(run_path / "config.json"),
        "vocab_path": str(run_path / "vocab.json"),
        "learning_rate": raw_config["hyperparameters"]["lr"],
        "restore_path": restore_path,
        "continue_path": continue_path,
    }
    write_json(summary, resolve_path(raw_config["training"]["output_path"]) / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    raw_config = read_json(resolve_path(args.config))
    validate_dataset(raw_config)

    model_dir = resolve_path(raw_config["pretrained_model"]["model_dir"])
    should_download = (not args.dry_run) or args.download_pretrained
    if should_download:
        pretrained_files = download_pretrained_files(model_dir)
    else:
        maybe_files = resolve_pretrained_files(model_dir)
        missing = [name for name, path in maybe_files.items() if path is None]
        if missing:
            print(f"Pretrained files not downloaded for dry-run: {', '.join(missing)}")
            print("Run again with --download-pretrained or start training to download them.")
            return
        pretrained_files = {key: path for key, path in maybe_files.items() if path is not None}

    trainer_config = build_trainer_config(raw_config, pretrained_files)
    generated_config_path = resolve_path(raw_config["training"]["output_path"]) / "generated_xtts_gpt_config.json"
    trainer_config.save_json(str(generated_config_path))
    print(f"Generated XTTS GPT config: {generated_config_path}")

    train_samples, eval_samples = load_samples(raw_config, trainer_config)
    print(f"Dry validation loaded {len(train_samples)} train and {len(eval_samples)} eval samples.")
    if args.dry_run:
        print("Dry run finished: training was not started.")
        return

    device = configure_device(args.force_cpu)
    run_training(
        raw_config,
        trainer_config,
        pretrained_files,
        device,
        args.restore_path,
        args.continue_path,
    )


if __name__ == "__main__":
    main()
