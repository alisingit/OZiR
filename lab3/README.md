# Лабораторная работа 3. Построение систем синтеза речи

Работа готовит воспроизводимый эксперимент по обучению русских open source TTS-моделей с нуля. Обучение намеренно не запускается автоматически: скрипты и конфиги подготовлены так, чтобы запустить их позже на локальной машине или GPU-среде.

## Структура

- `configs/` - конфиги экспериментов `VITS` и `Tacotron2`.
- `src/` - подготовка датасета, запуск обучения, синтез и оценка.
- `texts/control_texts.json` - контрольные русские тексты из задания.
- `notebooks/lab3_tts.ipynb` - ноутбук для выполнения и оформления результатов.
- `report.md` - шаблон отчёта с методикой, метриками и ответами на вопросы.
- `data/`, `runs/`, `checkpoints/`, `outputs/` - локальные артефакты, не предназначенные для git.

## Быстрый старт

```bash
cd lab3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Установка PyTorch с CUDA:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchaudio==2.5.1
pip install -r requirements.txt
```

Скачайте подготовленный single-speaker русский датасет и приведите к LJSpeech-формату:

```bash
python src/download_dataset.py --max-samples 1500 --folder early_short_stories
python src/prepare_dataset.py \
  --raw-dir data/raw/ru_ss \
  --output-dir data/processed/ru_ss_ljspeech \
  --sample-rate 22050 \
  --val-ratio 0.05
```

Также поддерживаются собственные датасеты в `data/raw/<name>`:

- `metadata.csv` в стиле LJSpeech: `wav|text` или `wav|text|normalized_text`;
- `metadata.jsonl` с полями `audio_filepath`/`audio`/`path` и `text`/`sentence`/`transcript`;
- пары файлов `sample.wav` и `sample.txt` в одной директории.

## Обучение

Скрипт `src/train.py` собирает полноценный `VitsConfig`/`Tacotron2Config` из JSON в `configs/` и запускает Coqui-TTS Trainer на GPU при наличии CUDA. Проверка без запуска:

```bash
python src/train.py --config configs/vits_baseline.json --dry-run
```

Полный запуск (RTX 3090 ~35 минут на 80 эпох, mixed precision):

```bash
python src/train.py --config configs/vits_baseline.json
python src/train.py --config configs/vits_lr_changed.json
python src/train.py --config configs/tacotron2_baseline.json
```

Флаг `--force-cpu` отключает CUDA, `--restore-path` / `--continue-path` возобновляют обучение. Логи TensorBoard сохраняются в `runs/`, чекпойнты - в `checkpoints/`.

## Синтез и оценка

После обучения (путь к чекпойнту берётся из последней папки в `checkpoints/vits_baseline/`):

```bash
RUN_DIR=$(ls -1d checkpoints/vits_baseline/vits_baseline-* | head -1)

python src/synthesize.py \
  --checkpoint "$RUN_DIR/best_model.pth" \
  --config "$RUN_DIR/config.json" \
  --texts texts/control_texts.json \
  --output-dir outputs/audio/vits_baseline

python src/synthesize_validation.py \
  --checkpoint "$RUN_DIR/best_model.pth" \
  --config "$RUN_DIR/config.json" \
  --metadata data/processed/ru_ss_ljspeech/metadata_val.csv \
  --output-dir outputs/audio/vits_baseline_val

python src/evaluate.py \
  --metadata data/processed/ru_ss_ljspeech/metadata_val.csv \
  --synth-dir outputs/audio/vits_baseline_val \
  --output-dir outputs/metrics/vits_baseline
```

Для просмотра графиков обучения:

```bash
tensorboard --logdir runs
```
