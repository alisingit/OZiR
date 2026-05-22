# Лабораторная работа №3. Построение систем синтеза речи

## Цель работы

Получить практические навыки работы с neural text-to-speech моделями с открытым исходным кодом: выбрать готовую модель, дообучить ее на целевом голосе, сравнить синтез до и после fine-tuning и проанализировать акустические признаки.

## Выбранные модели

Основная модель эксперимента - **XTTS-v2** из Coqui TTS. Это multilingual zero-shot voice cloning TTS: модель принимает текст, язык и короткий speaker reference, после чего синтезирует речь похожим голосом. Для лабораторной она удобна тем, что уже обучена на разных языках, поддерживает русский (`ru`) и допускает fine-tuning GPT/acoustic-token части на небольшом single-speaker датасете.

Для связи с моделями из презентации рассматривается **Tacotron2**. Tacotron2 строит mel-спектрограмму autoregressive encoder-decoder моделью с attention, а waveform получает через отдельный vocoder. В отличие от него, XTTS-v2 стартует с крупного предобученного multilingual чекпойнта и адаптируется к новому голосу быстрее.

| Модель | Роль в работе | Особенности |
| --- | --- | --- |
| `XTTS-v2 pretrained` | baseline до дообучения | zero-shot voice cloning по reference wav |
| `XTTS-v2 fine-tuned` | основной эксперимент | адаптация к русскому single-speaker датасету |
| `Tacotron2` | классическая модель для сравнения | требует больше данных и отдельного vocoder |

## Датасет

Используется публичный single-speaker русский датасет `niobures/russian-single-speaker-speech-dataset` с HuggingFace. Для лабораторного запуска удобно взять 1500 фраз из подпапки `early_short_stories`. Данные скачиваются скриптом `src/download_dataset.py`, затем приводятся к двум форматам:

- LJSpeech-style `id|text|normalized_text` для хранения reference wav и validation split;
- Coqui XTTS CSV `audio_file|text|speaker_name` для `GPTTrainer`.

Команды подготовки:

```bash
python src/download_dataset.py --max-samples 1500 --folder early_short_stories
python src/prepare_dataset.py --raw-dir data/raw/ru_ss --output-dir data/processed/ru_ss_ljspeech --sample-rate 22050 --val-ratio 0.05
python src/prepare_xtts_dataset.py --input-dir data/processed/ru_ss_ljspeech --output-dir data/processed/ru_ss_xtts --speaker-name ru_single_speaker
```

`prepare_xtts_dataset.py` также выбирает reference wav длиной не меньше 3 секунд и сохраняет путь в `data/processed/ru_ss_xtts/speaker_ref.txt`.

В проверочном запуске на 1500 исходных фразах после ограничения XTTS по длительности получилось 1411 train-записей, 73 validation-записи и 16 отфильтрованных длинных клипов.

## Эксперименты

Подготовлены два fine-tuning конфига:

| Эксперимент | Learning rate | Назначение |
| --- | --- | --- |
| `xtts_finetune` | `5e-6` | базовая скорость адаптации из Coqui XTTS demo |
| `xtts_finetune_lr_low` | `1e-6` | более осторожная адаптация, меньше риск забывания |

Проверка без обучения:

```bash
python src/train_xtts.py --config configs/xtts_finetune.json --dry-run
```

Проверка со скачиванием исходной модели:

```bash
python src/train_xtts.py --config configs/xtts_finetune.json --dry-run --download-pretrained
```

Запуск обучения:

```bash
python src/train_xtts.py --config configs/xtts_finetune.json
python src/train_xtts.py --config configs/xtts_finetune_lr_low.json
```

В процессе сохраняются `best_model.pth`, `config.json`, `vocab.json`, `training_summary.json`, TensorBoard events и trainer logs.

Реальный запуск выполнен на RTX 3090:

| Параметр | Значение |
| --- | --- |
| Train / validation | 1411 / 73 |
| Epochs | 6 total: 2 initial + 4 continued |
| Batch / grad accumulation | 1 / 4 |
| Precision | fp32 |
| Learning rate | `5e-6` |
| Best checkpoint | `checkpoints/xtts_finetune/xtts_finetune-May-22-2026_11+22PM-c3ef84c/best_model.pth` |

Первая попытка с mixed precision дала `nan` loss, поэтому для стабильности fine-tuning переведен в fp32. На второй эпохе eval loss улучшился: `avg_loss` 3.1627 -> 3.1132, `avg_loss_mel_ce` 3.1320 -> 3.0828.
Затем обучение было продолжено из `best_model.pth` старого run directory в новый run directory, чтобы сохранить исходные чекпоинты. После продолжения лучший `avg_loss` стал 3.0384, `avg_loss_mel_ce` - 3.0087.

## Логирование

`GPTTrainer` пишет в TensorBoard:

- train/eval loss для GPT части XTTS;
- learning rate;
- sample audio и diagnostic plots, если они доступны в текущей версии Coqui;
- служебные параметры шага, скорости обучения и checkpoint saving.

Дополнительно `src/synthesize_xtts.py` сохраняет mel-спектрограммы сгенерированных wav в `outputs/audio/spectrograms/<run>/`, что закрывает требование по визуализации спектрограмм.

## Контрольный текст

Контрольные фразы лежат в `texts/control_texts.json`. Главный текст близок к требуемому объему и содержит разные типы интонации:

- нейтральные предложения;
- вопрос;
- восклицание;
- двоеточие;
- тире;
- перечисление акустических критериев.

Синтез исходной модели:

```bash
python src/synthesize_xtts.py --checkpoint checkpoints/xtts_original/model.pth --config checkpoints/xtts_original/config.json --vocab checkpoints/xtts_original/vocab.json --speaker-wav data/processed/ru_ss_xtts/speaker_ref.txt --texts texts/control_texts.json --output-dir outputs/audio/xtts_pretrained
```

Синтез fine-tuned модели:

```bash
RUN_DIR=checkpoints/xtts_finetune/<run-dir>
python src/synthesize_xtts.py --checkpoint "$RUN_DIR/best_model.pth" --config "$RUN_DIR/config.json" --vocab "$RUN_DIR/vocab.json" --speaker-wav data/processed/ru_ss_xtts/speaker_ref.txt --texts texts/control_texts.json --output-dir outputs/audio/xtts_finetuned
```

## Оценка качества

Для объективного сравнения используется `src/compare_acoustic_features.py`. Скрипт считает:

- **MCD** с DTW по MFCC относительно reference wav;
- MFCC mean/std;
- log-mel energy mean/std;
- spectral centroid, bandwidth, rolloff;
- RMS и zero-crossing rate;
- F0 mean/std и voiced ratio;
- длительность аудио.

Пример:

```bash
python src/synthesize_xtts.py --checkpoint checkpoints/xtts_original/model.pth --config checkpoints/xtts_original/config.json --vocab checkpoints/xtts_original/vocab.json --speaker-wav data/processed/ru_ss_xtts/speaker_ref.txt --metadata data/processed/ru_ss_ljspeech/metadata_val.csv --output-dir outputs/audio/xtts_pretrained_val
python src/synthesize_xtts.py --checkpoint "$RUN_DIR/best_model.pth" --config "$RUN_DIR/config.json" --vocab "$RUN_DIR/vocab.json" --speaker-wav data/processed/ru_ss_xtts/speaker_ref.txt --metadata data/processed/ru_ss_ljspeech/metadata_val.csv --output-dir outputs/audio/xtts_finetuned_val
python src/compare_acoustic_features.py --metadata data/processed/ru_ss_ljspeech/metadata_val.csv --reference-wavs data/processed/ru_ss_ljspeech/wavs --system pretrained=outputs/audio/xtts_pretrained_val --system finetuned=outputs/audio/xtts_finetuned_val --output-dir outputs/metrics/xtts_comparison
```

Для субъективной оценки используется mini-MOS по шкале 1-5: разборчивость, естественность, стабильность тембра, просодия, шумы/артефакты. Ожидаемый результат: fine-tuned модель должна лучше попадать в тембр и речевую манеру датасета, а pretrained baseline может звучать естественно, но менее похоже на выбранного диктора.

Итоговые validation-метрики:

| Система | mean MCD | duration mean, s | RMS mean | F0 mean, Hz | voiced ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | 0.00 | 7.11 | 0.033 | 152.86 | 0.673 |
| XTTS pretrained | 95.10 | 6.78 | 0.112 | 160.89 | 0.600 |
| XTTS fine-tuned | 96.93 | 7.10 | 0.033 | 163.84 | 0.637 |

MCD после двух эпох не улучшился, но fine-tuned модель стала ближе к reference по средней длительности и RMS. Это ожидаемо для короткого лабораторного fine-tuning: модель быстро адаптирует громкость/длительность и speaker style, но для устойчивого спектрального выигрыша нужны больше эпох или более аккуратная фильтрация проблемных батчей.

## Анализ До И После Дообучения

До fine-tuning XTTS-v2 уже умеет говорить по-русски и клонировать голос по reference wav, поэтому baseline не является случайным шумом. После fine-tuning ожидаются:

- более стабильный speaker identity на длинных фразах;
- меньше случайных просодических скачков;
- ближе длительности и F0 к validation записям;
- ниже MCD к reference wav на validation subset;
- возможный риск переобучения при слишком большом learning rate.

Сравнение `5e-6` и `1e-6` показывает влияние гиперпараметра: больший LR быстрее адаптирует голос, меньший LR обычно стабильнее и меньше портит уже выученную multilingual речь.

В текущем коротком запуске основной выигрыш виден не по MCD, а по признакам уровня и длительности: fine-tuned RMS практически совпал с reference (`0.0329` против `0.0334`), а средняя длительность стала `7.10` с при reference `7.11`. Pretrained baseline был заметно громче (`0.112`) и короче (`6.78`).

## Ответы на вопросы самопроверки

**1. Какие дополнительные характеристики можно подать на вход модели для улучшения точности синтеза речи?**

Можно подавать фонемы, ударения, speaker embedding, language ID, pitch/F0, energy, duration targets, emotion/style/prosody embeddings, темп речи, контекст предложений, SSML-разметку пауз и нормализованный текст. Для русского особенно важны ударение, `е/ё`, фонемизация и раскрытие чисел/сокращений.

**2. Что изменится в случае дообучения модели вместо обучения?**

При дообучении модель стартует с уже выученных акустических, языковых и просодических представлений. Поэтому нужно меньше данных и GPU-времени, а первые синтезы уже разборчивы. Меняются риски: появляется catastrophic forgetting, зависимость от исходного tokenizer/cleaner, необходимость меньшего learning rate и аккуратного выбора validation/reference аудио. Обучение с нуля требует больше данных и дольше ищет alignment, зато не наследует ограничения исходной модели.
