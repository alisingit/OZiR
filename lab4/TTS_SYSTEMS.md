# Локальные синтезаторы речи

В работе выбраны пять бесплатных систем, которые можно запускать локально или без платных облачных API. Воспроизводимая генерация настроена в папке `lab4`: Python-зависимости ставятся в `.venv` и `.venv-coqui`, модели скачиваются в `models`, кэши — в `.cache`, итоговые WAV-файлы сохраняются в `audio`.

Полный запуск:

```bash
./lab4/setup_env.sh
./lab4/generate_all_audio.sh
```

## Silero TTS

Silero TTS — нейросетевой синтезатор с русскими голосами. Для эксперимента подходит голос `aidar`, `baya`, `kseniya`, `xenia` или другой русский голос из пакета Silero.

Пример запуска через Python:

```bash
python3 - <<'PY'
import torch

language = "ru"
speaker = "baya"
model_id = "v4_ru"
device = torch.device("cpu")

model, example_text = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language=language,
    speaker=model_id,
)
model.to(device)
model.save_wav(text="Тестовая фраза для синтеза речи.", speaker=speaker, sample_rate=48000, audio_path="outputs/silero.wav")
PY
```

## RHVoice

RHVoice — локальный синтезатор речи с русскими голосами. Он хорошо подходит для офлайн-сравнения, так как не требует удалённого сервиса.

Пример запуска:

```bash
echo "Тестовая фраза для синтеза речи." | RHVoice-test -p anna -o outputs/rhvoice.wav
```

## Piper

Piper — быстрый локальный TTS на ONNX. Для русской речи нужна русская модель Piper, например файл `ru_RU-*.onnx` и соответствующий `*.json`.

Пример запуска:

```bash
echo "Тестовая фраза для синтеза речи." | piper --model models/ru_RU-model.onnx --output_file outputs/piper.wav
```

## Coqui XTTS

Coqui XTTS — многоязычный нейросетевой синтезатор. Для русской речи используется параметр языка `ru`. Если используется клонирование голоса, нужен короткий референсный WAV.

В проекте модель XTTS-v2 скачивается в `models/coqui_xtts/XTTS-v2` и запускается из окружения `.venv-coqui`, чтобы не зависеть от системного кэша Coqui.

Пример запуска:

```bash
COQUI_TOS_AGREED=1 ./lab4/.venv-coqui/bin/python lab4/src/generate_audio_runs.py \
  --engine coqui_xtts \
  --coqui-model-dir lab4/models/coqui_xtts/XTTS-v2 \
  --coqui-speaker-wav lab4/audio/rhvoice/stress/01.wav
```

## eSpeak NG

eSpeak NG — формантный синтезатор. Он звучит менее естественно, но полезен как базовая контрольная система.

Пример запуска:

```bash
espeak-ng -v ru "Тестовая фраза для синтеза речи." -w outputs/espeak_ng.wav
```

## Общий порядок измерений

1. Для каждого предложения генерируется аудио одним и тем же голосом выбранной системы.
2. Для скорости запуска речи используется короткая фраза: `Сегодня проводится экспериментальная оценка синтеза русской речи.`
3. Задержка измеряется секундомером от запуска воспроизведения или команды синтеза до первого слышимого звука.
4. Каждая система измеряется пять раз.
5. Ошибки чтения фиксируются вручную при прослушивании аудио и заносятся в `data/results.csv`.
