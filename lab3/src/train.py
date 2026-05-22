from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger
from trainer import Trainer, TrainerArgs

from common import read_json, resolve_path, write_json


RUSSIAN_CHARACTERS = " абвгдежзийклмнопрстуфхцчшщъыьэюя"
RUSSIAN_PUNCTUATIONS = '!\'(),-.:;? "'
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
BOS_TOKEN = "<BOS>"
BLANK_TOKEN = "<BLNK>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VITS/Tacotron2 on a Russian LJSpeech-style dataset using Coqui TTS Python API.",
    )
    parser.add_argument("--config", required=True, help="Experiment config from configs/*.json.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset and dump generated Coqui config without launching the Trainer.",
    )
    parser.add_argument(
        "--restore-path",
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--continue-path",
        default=None,
        help="Path to an existing run directory to continue from.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable CUDA even if available. By default GPU is used when present.",
    )
    return parser.parse_args()


def build_characters_config() -> CharactersConfig:
    return CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad=PAD_TOKEN,
        eos=EOS_TOKEN,
        bos=BOS_TOKEN,
        blank=BLANK_TOKEN,
        characters=RUSSIAN_CHARACTERS,
        punctuations=RUSSIAN_PUNCTUATIONS,
        phonemes="",
        is_unique=True,
        is_sorted=True,
    )


def build_audio_for_vits(audio: dict[str, Any]) -> VitsAudioConfig:
    return VitsAudioConfig(
        sample_rate=audio["sample_rate"],
        win_length=audio["win_length"],
        hop_length=audio["hop_length"],
        num_mels=audio["num_mels"],
        mel_fmin=audio["mel_fmin"],
        mel_fmax=audio["mel_fmax"],
        fft_size=audio["fft_size"],
    )


def build_audio_for_tacotron(audio: dict[str, Any]) -> BaseAudioConfig:
    return BaseAudioConfig(
        sample_rate=audio["sample_rate"],
        win_length=audio["win_length"],
        hop_length=audio["hop_length"],
        num_mels=audio["num_mels"],
        mel_fmin=audio["mel_fmin"],
        mel_fmax=audio["mel_fmax"],
        fft_size=audio["fft_size"],
        preemphasis=0.0,
        ref_level_db=20,
        log_func="np.log",
        do_trim_silence=True,
        trim_db=45,
        mel_spec_gain=20.0,
        do_amp_to_db_linear=True,
        do_amp_to_db_mel=True,
        signal_norm=True,
        symmetric_norm=True,
        max_norm=4.0,
        clip_norm=True,
        stats_path=None,
    )


def build_dataset_config(dataset: dict[str, Any]) -> BaseDatasetConfig:
    dataset_path = resolve_path(dataset["path"])
    return BaseDatasetConfig(
        formatter=dataset.get("formatter", "ljspeech"),
        dataset_name=dataset.get("name", "russian_tts"),
        meta_file_train=dataset.get("meta_file_train", "metadata_train.csv"),
        meta_file_val=dataset.get("meta_file_val", "metadata_val.csv"),
        path=str(dataset_path),
        language=dataset.get("language", "ru"),
    )


def build_vits_config(config: dict[str, Any]) -> VitsConfig:
    training = config["training"]
    hyperparameters = config["hyperparameters"]
    output_path = resolve_path(training["output_path"])
    log_path = resolve_path(training["log_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    vits_args = VitsArgs(
        use_sdp=hyperparameters.get("use_sdp", True),
        num_speakers=hyperparameters.get("num_speakers", 0),
        use_speaker_embedding=False,
        use_language_embedding=False,
    )

    return VitsConfig(
        model="vits",
        run_name=config["experiment_name"],
        output_path=str(output_path),
        logger_uri=str(log_path),
        dashboard_logger="tensorboard",
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        eval_batch_size=training["eval_batch_size"],
        num_loader_workers=training["num_loader_workers"],
        num_eval_loader_workers=training["num_eval_loader_workers"],
        print_step=training["print_step"],
        save_step=training["save_step"],
        save_n_checkpoints=2,
        save_checkpoints=True,
        save_best_after=training.get("save_best_after", 1000),
        run_eval=True,
        test_delay_epochs=-1,
        mixed_precision=training.get("mixed_precision", False),
        cudnn_benchmark=True,
        training_seed=training["seed"],
        audio=build_audio_for_vits(config["audio"]),
        model_args=vits_args,
        text_cleaner=hyperparameters.get("text_cleaner", "multilingual_cleaners"),
        use_phonemes=hyperparameters.get("use_phonemes", False),
        phoneme_language="ru",
        phoneme_cache_path=str(output_path / "phoneme_cache"),
        compute_input_seq_cache=True,
        add_blank=True,
        characters=build_characters_config(),
        datasets=[build_dataset_config(config["dataset"])],
        lr_gen=hyperparameters["lr_gen"],
        lr_disc=hyperparameters["lr_disc"],
        mel_loss_alpha=hyperparameters["mel_loss_alpha"],
        kl_loss_alpha=hyperparameters["kl_loss_alpha"],
        dur_loss_alpha=hyperparameters["dur_loss_alpha"],
        min_audio_len=training.get("min_audio_len", 22050 * 0.5),
        max_audio_len=training.get("max_audio_len", 22050 * 12),
        min_text_len=training.get("min_text_len", 5),
        max_text_len=training.get("max_text_len", 200),
        test_sentences=hyperparameters.get(
            "test_sentences",
            [
                "Сегодня ясный солнечный день.",
                "Привет, как у тебя дела?",
                "Москва - столица России.",
            ],
        ),
    )


def build_tacotron2_config(config: dict[str, Any]) -> Tacotron2Config:
    training = config["training"]
    hyperparameters = config["hyperparameters"]
    output_path = resolve_path(training["output_path"])
    log_path = resolve_path(training["log_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    return Tacotron2Config(
        model="tacotron2",
        run_name=config["experiment_name"],
        output_path=str(output_path),
        logger_uri=str(log_path),
        dashboard_logger="tensorboard",
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        eval_batch_size=training["eval_batch_size"],
        num_loader_workers=training["num_loader_workers"],
        num_eval_loader_workers=training["num_eval_loader_workers"],
        print_step=training["print_step"],
        save_step=training["save_step"],
        save_n_checkpoints=2,
        save_checkpoints=True,
        save_best_after=training.get("save_best_after", 1000),
        run_eval=True,
        test_delay_epochs=-1,
        mixed_precision=training.get("mixed_precision", False),
        cudnn_benchmark=True,
        training_seed=training["seed"],
        audio=build_audio_for_tacotron(config["audio"]),
        text_cleaner=hyperparameters.get("text_cleaner", "multilingual_cleaners"),
        use_phonemes=hyperparameters.get("use_phonemes", False),
        phoneme_language="ru",
        phoneme_cache_path=str(output_path / "phoneme_cache"),
        compute_input_seq_cache=True,
        add_blank=True,
        characters=build_characters_config(),
        datasets=[build_dataset_config(config["dataset"])],
        lr=hyperparameters["lr"],
        r=hyperparameters["r"],
        grad_clip=hyperparameters.get("grad_clip", 1.0),
        decoder_loss_alpha=hyperparameters.get("decoder_loss_alpha", 0.25),
        postnet_loss_alpha=hyperparameters.get("postnet_loss_alpha", 0.25),
        ga_alpha=hyperparameters.get("ga_alpha", 5.0),
        use_guided_attention_loss=True,
        min_audio_len=training.get("min_audio_len", 22050 * 0.5),
        max_audio_len=training.get("max_audio_len", 22050 * 12),
        min_text_len=training.get("min_text_len", 5),
        max_text_len=training.get("max_text_len", 200),
        test_sentences=hyperparameters.get(
            "test_sentences",
            [
                "Сегодня ясный солнечный день.",
                "Привет, как у тебя дела?",
                "Москва - столица России.",
            ],
        ),
    )


def build_coqui_config(config: dict[str, Any]):
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


def init_model(coqui_config, train_samples, eval_samples):
    model_name = coqui_config.model.lower()
    if model_name == "vits":
        audio_processor = AudioProcessor.init_from_config(coqui_config)
        tokenizer, coqui_config = TTSTokenizer.init_from_config(coqui_config)
        model = Vits(coqui_config, audio_processor, tokenizer, speaker_manager=None)
        return model, coqui_config
    if model_name == "tacotron2":
        audio_processor = AudioProcessor.init_from_config(coqui_config)
        tokenizer, coqui_config = TTSTokenizer.init_from_config(coqui_config)
        model = Tacotron2(coqui_config, audio_processor, tokenizer, speaker_manager=None)
        return model, coqui_config
    raise ValueError(f"Unsupported model: {model_name}")


def configure_device(force_cpu: bool) -> str:
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA detected: {device_name}, training will use GPU.")
        return "cuda"
    print("CUDA is not available, falling back to CPU.")
    return "cpu"


def main() -> None:
    args = parse_args()
    setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())

    config_path = resolve_path(args.config)
    raw_config = read_json(config_path)
    validate_dataset(raw_config)

    coqui_config = build_coqui_config(raw_config)
    output_path = Path(coqui_config.output_path)
    generated_config_path = output_path / "generated_coqui_config.json"
    coqui_config.save_json(str(generated_config_path))
    print(f"Generated Coqui config: {generated_config_path}")

    if args.dry_run:
        print("Dry run finished: dataset paths are present, training was not started.")
        return

    device = configure_device(args.force_cpu)

    train_samples, eval_samples = load_tts_samples(
        coqui_config.datasets,
        eval_split=True,
        eval_split_max_size=coqui_config.eval_split_max_size,
        eval_split_size=coqui_config.eval_split_size,
    )
    print(f"Loaded {len(train_samples)} train and {len(eval_samples)} eval samples.")

    model, coqui_config = init_model(coqui_config, train_samples, eval_samples)

    trainer_args = TrainerArgs(
        restore_path=args.restore_path,
        continue_path=args.continue_path,
    )

    trainer = Trainer(
        trainer_args,
        coqui_config,
        coqui_config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    print(f"Trainer device: {trainer.use_cuda=}, expected={device}")
    trainer.fit()

    summary = {
        "experiment": coqui_config.run_name,
        "output_path": coqui_config.output_path,
        "log_path": coqui_config.logger_uri,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "device": device,
        "best_model_path": str(Path(coqui_config.output_path) / "best_model.pth"),
    }
    write_json(summary, output_path / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
