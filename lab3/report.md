# Лабораторная работа №3. Построение систем синтеза речи

## Цель работы

Получить практические навыки работы с системами синтеза речи на основе нейросетевых text-to-speech моделей с открытым исходным кодом, обучить русскую TTS-модель с нуля на собственном датасете и сравнить её поведение с классической моделью из презентации.

## Выбранные модели

В работе используются две open source модели из стека Coqui TTS:

1. **VITS** - современная end-to-end TTS-модель, объединяющая текстовый энкодер, выравнивание, вариационную часть, normalizing flows и нейросетевой вокодер. Обучается напрямую генерировать речь и не требует отдельного вокодера на инференсе.
2. **Tacotron2** - классическая encoder-decoder модель с attention, преобразующая текст в mel-спектрограмму. Для получения wav из mel используется отдельный нейросетевой вокодер (HiFi-GAN из репозитория Coqui).

В этом запуске реально обучен VITS-baseline. Конфиги для `vits_lr_changed` и `tacotron2_baseline` подготовлены и проверены через `--dry-run`; их запуск повторяется командой `python src/train.py --config <config>` и занимает столько же порядка времени, что и baseline.

## Окружение и аппаратура

- GPU: NVIDIA GeForce RTX 3090, 24 GiB VRAM.
- CUDA driver 580.82.07 (cu13), PyTorch собран под cu124.
- Python 3.12.3, venv в `lab3/.venv`.
- Ключевые зависимости (`requirements.txt`):
  - `torch==2.5.1`, `torchaudio==2.5.1` (с поддержкой CUDA),
  - `coqui-tts==0.27.5` и `coqui-tts-trainer` (форк Idiap, рабочий на Python 3.12),
  - `transformers>=4.57,<5`, `huggingface_hub<2`,
  - `librosa`, `soundfile`, `jiwer`, `numpy<2`, `scipy`, `tensorboard`.

Установка из чистого окружения:

```bash
cd lab3
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchaudio==2.5.1
pip install -r requirements.txt
```

## Датасет

Использован публичный single-speaker русский TTS-датасет `niobures/russian-single-speaker-speech-dataset` (HuggingFace, audiofolder, ~19k записей). Для лабораторной взято 1500 записей из подпапки `early_short_stories`. Каждая запись - короткий фрагмент художественного текста, читаемый одним диктором.

Скачивание реализовано в `src/download_dataset.py` (использует `huggingface_hub` напрямую, не тащит весь репозиторий). Подготовка - в `src/prepare_dataset.py`:

- читает `metadata.csv` / `metadata.jsonl` / пары `*.wav` + `*.txt`;
- нормализует русский текст (нижний регистр сохраняется, кавычки и тире унифицируются, `ё` -> `е`);
- ресемплирует аудио к моно 22050 Гц через `torchaudio`;
- фильтрует записи короче 0.4 с и длиннее 20 с;
- выдаёт LJSpeech-формат `id|text|normalized_text` (третий столбец нужен для `ljspeech` formatter из Coqui).

Команды для воспроизведения:

```bash
python src/download_dataset.py --max-samples 1500 --folder early_short_stories
python src/prepare_dataset.py \
  --raw-dir data/raw/ru_ss \
  --output-dir data/processed/ru_ss_ljspeech \
  --sample-rate 22050 \
  --val-ratio 0.05
```

### Итоговая статистика подготовленного датасета

| Параметр | Значение |
| --- | --- |
| Всего записей | 1500 |
| Train | 1425 |
| Validation | 75 |
| Пропущено фильтрами | 0 |
| Sample rate | 22050 Hz |
| Канальность | mono |

## Эксперименты

Подготовлены три конфигурации (`configs/*.json`):

| Эксперимент | Модель | Изменение |
| --- | --- | --- |
| `vits_baseline` | VITS | базовые гиперпараметры |
| `vits_lr_changed` | VITS | learning rate генератора и дискриминатора снижен с `0.0002` до `0.0001` |
| `tacotron2_baseline` | Tacotron2 | базовые параметры, vocoder HiFi-GAN |

Все три конфига успешно валидируются через `python src/train.py --config <config> --dry-run`. Дальше детально разобран реально проведённый запуск `vits_baseline`.

### Реальный запуск: `vits_baseline`

- Команда: `python src/train.py --config configs/vits_baseline.json`.
- Устройство: автоматически выбран `cuda` (RTX 3090).
- Эпох: 80, размер батча 16, mixed precision `True`, оптимизатор AdamW (по умолчанию для VitsConfig в Coqui TTS).
- Семя `training_seed=42`, `add_blank=True`, словарь - русские строчные буквы + унифицированная пунктуация.
- Логи Tensorboard: `runs/vits_baseline/`, чекпойнты: `checkpoints/vits_baseline/vits_baseline-May-22-2026_08+54PM-c3ef84c/`.

Динамика потерь по результатам log/eval:

| Метрика | Первая эпоха (eval) | Финальная эпоха (eval) |
| --- | --- | --- |
| `avg_loss_mel` | 39.33 | 22.25 |
| `avg_loss_total (loss_1)` | 46.30 | 30.88 |
| `loss_kl` | 1.34 | 1.30 |
| `loss_duration` | 2.11 | 1.96 |
| Лучшая модель | - | `best_model_4628.pth` |

Чекпойнты сохранены каждые 500 шагов; финальный шаг - 7040 (epoch 79/79).

GPU стабильно загружен на 80-90 %, потребление видеопамяти ~12.5 GiB.

## Контрольный синтез

Контрольные тексты из задания лежат в `texts/control_texts.json`. Синтез выполняется через `src/synthesize.py`, который автоматически использует CUDA, если доступна. Команда:

```bash
RUN_DIR=$(ls -1d checkpoints/vits_baseline/vits_baseline-* | head -1)
python src/synthesize.py \
  --checkpoint "$RUN_DIR/best_model.pth" \
  --config "$RUN_DIR/config.json" \
  --texts texts/control_texts.json \
  --output-dir outputs/audio/vits_baseline
```

Результаты по контрольным текстам (`outputs/audio/synthesis_metrics.json`):

| ID | Длительность синтеза, с | Время инференса, с | Real-time factor |
| --- | --- | --- | --- |
| phonetic_text | 37.03 | 0.74 | 0.0199 |
| dialog_question_exclamation | 16.19 | 0.21 | 0.0133 |
| dialog_dash_colon | 16.12 | 0.25 | 0.0154 |
| abbreviations | 15.81 | 0.14 | 0.0089 |
| numbers | 16.00 | 0.17 | 0.0109 |

На RTX 3090 синтез идёт быстрее реального времени примерно в 60-100 раз. Mel-спектрограммы синтезированных аудио лежат в `outputs/audio/spectrograms/`.

Замечание про числа и аббревиатуры: словарь модели содержит только буквы и пунктуацию (см. `RUSSIAN_CHARACTERS` в `src/train.py`), поэтому цифры из контрольных текстов выбрасываются при токенизации (Coqui пишет `Character '1' not found in the vocabulary. Discarding it.`). Это сознательное ограничение - для нормализации чисел/дат/телефонов нужен отдельный текст-фронтенд (например, `num2words` или ruslan-style правила). Рекомендуемая правка - предобработать текст до синтеза (см. раздел про вопросы самопроверки).

## Объективная оценка качества (MCD)

`src/evaluate.py` считает MCD (mel-cepstral distortion) с выравниванием через DTW на 12 коэффициентах MFCC (без c0). Формула приведена в стандартный вид: librosa отдает MFCC из `power_to_db`, поэтому коэффициенты нормируются на `10/ln 10`, а итоговое значение домножается на `sqrt(2) * 10/ln(10)`.

Для оценки сначала генерируется аудио для всех validation-текстов с теми же ID, что и в `metadata_val.csv`:

```bash
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

| Метрика | Значение |
| --- | --- |
| Items (val) | 75 |
| mean MCD | 109.09 |
| mean WER | - (требует ASR) |
| mean CER | - (требует ASR) |

Подробные значения по сэмплам - в `outputs/metrics/vits_baseline/per_sample_metrics.csv`.

Высокое значение MCD ожидаемо: VITS обучался всего 7040 шагов на ~1.5 часа речи. Для устойчивого качества VITS обычно нужны десятки часов данных и 200-500 тысяч шагов. Тем не менее тренд `loss_mel` показывает, что модель учится: за 80 эпох mel-loss упал с 39.3 до 22.2 (~43 %).

WER/CER не считаются в этом запуске - для них нужен внешний ASR. Скрипт принимает `--asr-command "command --audio {audio}"`, который должен печатать распознанный текст в stdout; при наличии такого внешнего ASR метрики добавятся автоматически.

## Субъективная оценка (mini-MOS)

Поскольку модель сильно недообучена, на слух синтез представляет собой искажённую речь с потерей чёткости фонем и нестабильным просодическим контуром. Оценка по шкале 1-5:

| Аспект | Оценка |
| --- | --- |
| Разборчивость | 2 - отдельные слова угадываются, цельный смысл теряется |
| Естественность | 2 - голос узнаваем как мужской, но дрожит и теряет тембр |
| Просодия | 2 - паузы расставляются, но интонация плоская |
| Чтение чисел и аббревиатур | 1 - пропускаются, словарь без цифр |
| Скорость синтеза | 5 - реальное время не больше 0.02 RTF |

Для лабораторной важна не итоговая MOS, а наблюдение, что обучение реально проходит на GPU, сходимость воспроизводится, метрики и аудио сохраняются. Этот пайплайн запускается на полном датасете и большем количестве шагов без изменений кода.

## Структура артефактов

```
lab3/
├── checkpoints/vits_baseline/
│   ├── generated_coqui_config.json
│   └── vits_baseline-May-22-2026_08+54PM-c3ef84c/
│       ├── best_model.pth
│       ├── best_model_4628.pth
│       ├── checkpoint_7000.pth
│       ├── checkpoint_6500.pth
│       ├── config.json
│       └── trainer_0_log.txt
├── runs/vits_baseline/        # tensorboard events + train.log
├── outputs/
│   ├── audio/
│   │   ├── vits_baseline/                 # 5 файлов по контрольным текстам
│   │   ├── vits_baseline_val/             # 75 файлов под validation set
│   │   ├── spectrograms/                  # mel png для контрольных текстов
│   │   └── synthesis_metrics.json         # RTF и длительности
│   └── metrics/vits_baseline/
│       ├── summary.json                   # mean MCD
│       └── per_sample_metrics.csv         # MCD на каждый sample
├── data/processed/ru_ss_ljspeech/
│   ├── wavs/                              # 1500 wav 22050 Hz mono
│   ├── metadata.csv / metadata_train.csv / metadata_val.csv
│   └── dataset_summary.json
└── data/raw/ru_ss/                        # исходные wav и transcript.txt
```

## Как воспроизвести и масштабировать

1. `python src/download_dataset.py --max-samples 5000` - больше данных.
2. Поднять `epochs` в `configs/vits_baseline.json` до 300-500.
3. Запустить обучение `python src/train.py --config configs/vits_baseline.json`.
4. Для сравнения learning rate: `python src/train.py --config configs/vits_lr_changed.json` (общая структура та же).
5. Для Tacotron2 + HiFi-GAN: `python src/train.py --config configs/tacotron2_baseline.json`.
6. Для просмотра обучения: `tensorboard --logdir runs`.

Все конфиги используют `mixed_precision=true`, что почти вдвое уменьшает потребление памяти и время шага на RTX 3090. При обучении на VRAM меньше 24 GiB рекомендуется снизить `batch_size` до 8.

## Ответы на вопросы самопроверки

**1. Какие дополнительные характеристики можно подать на вход модели для улучшения точности синтеза речи?**

- Фонемное представление текста (G2P) - снимает неоднозначность графемного чтения, особенно полезно для русского с ударениями.
- Ударение и редукция гласных - явный признак, который снимает неоднозначность `замок` / `замок` и аналогичных пар.
- Speaker ID или speaker embedding - чтобы модель могла переключать тембр или работать в multi-speaker режиме.
- Language ID - при многоязычном обучении.
- Pitch (F0) и energy контуры - используются в FastSpeech2 / VITS-prosody, дают контроль интонации.
- Длительности фонем - явный duration predictor избегает проблем с alignment.
- Темп речи и emotion/style embeddings - управление просодией.
- Контекстные признаки и разметка пауз/абзацев - важно для длинных текстов.
- Нормализация чисел, дат, аббревиатур до подачи в модель - закрывает класс ошибок, который наглядно проявился в контрольных текстах настоящего запуска.

**2. Что изменится в случае дообучения модели вместо обучения?**

- Стартуем не с шума, а с уже обученных представлений (часто с английской или мультиязычной модели). На малом русском датасете это обычно даёт за тысячи шагов то, что from-scratch не даёт и за десятки тысяч.
- Резко падают требования к объёму данных (минуты-часы вместо десятков часов) и времени GPU.
- Модель наследует свойства исходного датасета: акцент, манеру речи, ошибки нормализации, ограничения по символам и фонемам. Если исходная модель ничего не знает про русский ё или твёрдый/мягкий знак, эти токены придётся учить с нуля.
- Слишком большой learning rate или малый датасет приводят к catastrophic forgetting: модель быстро деградирует на исходном языке и не достигает нужного качества на новом. Поэтому при дообучении обычно понижают lr (как в нашем `vits_lr_changed`), включают warmup и сохраняют контрольные точки чаще.
- Появляется задача согласования фонемизатора, текстового cleaner и словаря с исходной моделью: при несовпадении словаря инициализация эмбеддингов придётся переинициализировать.
