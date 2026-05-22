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

Положите датасет `audio + transcript` в `data/raw`. Поддерживаются:

- `metadata.csv` в стиле LJSpeech: `wav|text` или `wav|text|normalized_text`;
- `metadata.jsonl` с полями `audio_filepath`/`audio`/`path` и `text`/`sentence`/`transcript`;
- пары файлов `sample.wav` и `sample.txt` в одной директории.

Пример подготовки:

```bash
python src/prepare_dataset.py \
  --raw-dir data/raw/denis \
  --output-dir data/processed/denis_ljspeech \
  --sample-rate 22050 \
  --val-ratio 0.05
```

## Запуск обучения позже

Перед полноценным обучением можно проверить конфиг и датасет:

```bash
python src/train.py --config configs/vits_baseline.json --dry-run
```

Когда будете готовы обучать:

```bash
python src/train.py --config configs/vits_baseline.json
python src/train.py --config configs/vits_lr_changed.json
python src/train.py --config configs/tacotron2_baseline.json
```

Логи TensorBoard сохраняются в `runs/`, checkpoints - в `checkpoints/`.

## Синтез и оценка

После обучения:

```bash
python src/synthesize.py \
  --checkpoint checkpoints/vits_baseline/best_model.pth \
  --config checkpoints/vits_baseline/config.json \
  --texts texts/control_texts.json \
  --output-dir outputs/audio/vits_baseline
```

```bash
python src/evaluate.py \
  --metadata data/processed/denis_ljspeech/metadata_val.csv \
  --synth-dir outputs/audio/vits_baseline \
  --output-dir outputs/metrics/vits_baseline
```

Для просмотра графиков обучения:

```bash
tensorboard --logdir runs
```
