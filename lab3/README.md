# Лабораторная работа 3. Построение систем синтеза речи

Проект содержит воспроизводимый пайплайн fine-tuning готовой multilingual XTTS-v2 модели на русском single-speaker датасете. Основное сравнение: `XTTS-v2 pretrained` против `XTTS-v2 fine-tuned`; Tacotron2/WaveNet разобраны в отчете как классические модели из условия.

## Структура

- `configs/xtts_finetune*.json` - два запуска XTTS с разным learning rate.
- `src/download_dataset.py` - скачивание русского датасета с HuggingFace.
- `src/prepare_dataset.py` - приведение raw audio к LJSpeech-style формату.
- `src/prepare_xtts_dataset.py` - конвертация LJSpeech-style данных в Coqui XTTS CSV.
- `src/train_xtts.py` - fine-tuning XTTS-v2 через `GPTTrainer`.
- `src/synthesize_xtts.py` - синтез контрольных текстов до и после дообучения.
- `src/compare_acoustic_features.py` - MCD и акустические признаки для reference/pretrained/fine-tuned.
- `texts/control_texts.json` - контрольный русский текст с вопросами, восклицаниями, двоеточием и тире.

Локальные артефакты (`data/`, `runs/`, `checkpoints/`, `outputs/`, `.venv/`) не предназначены для git.

## Установка

```bash
cd lab3
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchaudio==2.5.1
pip install -r requirements.txt
```

## Подготовка данных

Скачайте subset публичного русского single-speaker датасета и подготовьте две версии метаданных:

```bash
python src/download_dataset.py --max-samples 1500 --folder early_short_stories

python src/prepare_dataset.py \
  --raw-dir data/raw/ru_ss \
  --output-dir data/processed/ru_ss_ljspeech \
  --sample-rate 22050 \
  --val-ratio 0.05

python src/prepare_xtts_dataset.py \
  --input-dir data/processed/ru_ss_ljspeech \
  --output-dir data/processed/ru_ss_xtts \
  --speaker-name ru_single_speaker
```

`prepare_xtts_dataset.py` создает `metadata_train.csv`, `metadata_eval.csv` с колонками `audio_file|text|speaker_name` и файл `speaker_ref.txt` для voice cloning conditioning.

## Fine-Tuning

Dry-run без обучения:

```bash
python src/train_xtts.py --config configs/xtts_finetune.json --dry-run
```

Dry-run со скачиванием исходной XTTS-v2:

```bash
python src/train_xtts.py --config configs/xtts_finetune.json --dry-run --download-pretrained
```

Запуски для сравнения learning rate. Конфиги используют fp32: на RTX 3090 mixed precision для этого XTTS trainer дал `nan` loss на первых шагах.

```bash
python src/train_xtts.py --config configs/xtts_finetune.json
python src/train_xtts.py --config configs/xtts_finetune_lr_low.json
```

Логи TensorBoard и learning rate пишутся в `runs/`, чекпойнты - в `checkpoints/`. Графики:

```bash
tensorboard --logdir runs
```

## Синтез До И После

Исходная предобученная модель:

```bash
python src/synthesize_xtts.py \
  --checkpoint checkpoints/xtts_original/model.pth \
  --config checkpoints/xtts_original/config.json \
  --vocab checkpoints/xtts_original/vocab.json \
  --speaker-wav data/processed/ru_ss_xtts/speaker_ref.txt \
  --texts texts/control_texts.json \
  --output-dir outputs/audio/xtts_pretrained
```

Модель после fine-tuning:

```bash
RUN_DIR=$(ls -1d checkpoints/xtts_finetune/xtts_finetune-* | head -1)
python src/synthesize_xtts.py \
  --checkpoint "$RUN_DIR/best_model.pth" \
  --config "$RUN_DIR/config.json" \
  --vocab "$RUN_DIR/vocab.json" \
  --speaker-wav data/processed/ru_ss_xtts/speaker_ref.txt \
  --texts texts/control_texts.json \
  --output-dir outputs/audio/xtts_finetuned
```

## Оценка

Для validation-сравнения с исходными записями сгенерируйте wav с теми же `id`:

```bash
python src/synthesize_xtts.py \
  --checkpoint checkpoints/xtts_original/model.pth \
  --config checkpoints/xtts_original/config.json \
  --vocab checkpoints/xtts_original/vocab.json \
  --speaker-wav data/processed/ru_ss_xtts/speaker_ref.txt \
  --metadata data/processed/ru_ss_ljspeech/metadata_val.csv \
  --output-dir outputs/audio/xtts_pretrained_val

python src/synthesize_xtts.py \
  --checkpoint "$RUN_DIR/best_model.pth" \
  --config "$RUN_DIR/config.json" \
  --vocab "$RUN_DIR/vocab.json" \
  --speaker-wav data/processed/ru_ss_xtts/speaker_ref.txt \
  --metadata data/processed/ru_ss_ljspeech/metadata_val.csv \
  --output-dir outputs/audio/xtts_finetuned_val
```

После этого запустите анализ:

```bash
python src/compare_acoustic_features.py \
  --metadata data/processed/ru_ss_ljspeech/metadata_val.csv \
  --reference-wavs data/processed/ru_ss_ljspeech/wavs \
  --system pretrained=outputs/audio/xtts_pretrained_val \
  --system finetuned=outputs/audio/xtts_finetuned_val \
  --output-dir outputs/metrics/xtts_comparison
```

Скрипт сохраняет `per_file_features.csv`, `summary.csv`, `summary.json` и PNG-графики для MCD, F0, RMS, spectral centroid, voiced ratio и длительности.
