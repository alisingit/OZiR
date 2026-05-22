import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


CATEGORY_TITLES = {
    "graph_abbrev": "Графические сокращения",
    "initial_abbrev": "Инициальные аббревиатуры и сложносокращённые слова",
    "numbers": "Цифровые обозначения",
}

SYSTEM_ORDER = ["Silero TTS", "RHVoice", "Piper", "Coqui XTTS", "eSpeak NG"]


def read_csv(path, required_columns):
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = set(required_columns) - set(reader.fieldnames or [])
        if missing_columns:
            joined = ", ".join(sorted(missing_columns))
            raise ValueError(f"В файле {path} отсутствуют колонки: {joined}")
        return list(reader)


def parse_positive_int(value, field_name, path):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Поле {field_name} в {path} должно быть целым числом: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"Поле {field_name} в {path} не может быть отрицательным: {value!r}")
    return parsed


def parse_positive_float(value, field_name, path):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Поле {field_name} в {path} должно быть числом: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"Поле {field_name} в {path} не может быть отрицательным: {value!r}")
    return parsed


def format_float(value, digits=2):
    return f"{value:.{digits}f}"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def analyze_latency(rows, path):
    values_by_system = defaultdict(list)
    for row in rows:
        system = row["system"].strip()
        latency = parse_positive_float(row["latency_seconds"], "latency_seconds", path)
        values_by_system[system].append(latency)

    table_rows = []
    for system in SYSTEM_ORDER:
        values = values_by_system.get(system, [])
        if not values:
            raise ValueError(f"Для системы {system} нет замеров задержки")
        table_rows.append(
            [
                system,
                len(values),
                format_float(mean(values)),
                format_float(min(values)),
                format_float(max(values)),
            ]
        )
    return table_rows


def analyze_quality(rows, path):
    grouped = defaultdict(lambda: {"target_count": 0, "error_count": 0, "items": 0})
    for row in rows:
        system = row["system"].strip()
        category = row["category"].strip()
        target_count = parse_positive_int(row["target_count"], "target_count", path)
        error_count = parse_positive_int(row["error_count"], "error_count", path)
        if error_count > target_count:
            raise ValueError(
                f"В {path} ошибка разметки: error_count больше target_count "
                f"для {system}, {category}, item={row['item']}"
            )
        grouped[(system, category)]["target_count"] += target_count
        grouped[(system, category)]["error_count"] += error_count
        grouped[(system, category)]["items"] += 1

    table_rows_by_category = {}
    for category in CATEGORY_TITLES:
        category_rows = []
        for system in SYSTEM_ORDER:
            stats = grouped.get((system, category))
            if not stats:
                raise ValueError(f"Для системы {system} нет результатов категории {category}")
            error_percent = 100 * stats["error_count"] / stats["target_count"]
            category_rows.append(
                [
                    system,
                    stats["items"],
                    stats["target_count"],
                    stats["error_count"],
                    format_float(error_percent),
                ]
            )
        table_rows_by_category[category] = category_rows
    return table_rows_by_category


def analyze_stress(rows):
    table_rows = []
    by_system = {row["system"].strip(): row for row in rows}
    for system in SYSTEM_ORDER:
        row = by_system.get(system)
        if row is None:
            raise ValueError(f"Для системы {system} нет наблюдений по ударениям")
        table_rows.append([system, row["problem_words"], row["summary"]])
    return table_rows


def write_outputs(output_dir, latency_rows, quality_rows_by_category, stress_rows):
    output_dir.mkdir(parents=True, exist_ok=True)

    latency_markdown = markdown_table(
        ["Синтезатор", "Повторов", "Средняя задержка, с", "Минимум, с", "Максимум, с"],
        latency_rows,
    )
    (output_dir / "latency_summary.md").write_text(latency_markdown, encoding="utf-8")

    for category, rows in quality_rows_by_category.items():
        filename = f"{category}_summary.md"
        title = CATEGORY_TITLES[category]
        table = markdown_table(
            ["Синтезатор", "Предложений", "Проверяемых элементов", "Ошибок", "Ошибок, %"],
            rows,
        )
        (output_dir / filename).write_text(f"### {title}\n\n{table}", encoding="utf-8")

    stress_markdown = markdown_table(["Синтезатор", "Проблемные слова", "Вывод"], stress_rows)
    (output_dir / "stress_summary.md").write_text(stress_markdown, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Подсчёт результатов лабораторной работы №4.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    latency_path = args.data_dir / "latency.csv"
    results_path = args.data_dir / "results.csv"
    stress_path = args.data_dir / "stress_observations.csv"

    latency_rows = read_csv(latency_path, ["system", "repeat", "text", "latency_seconds", "comment"])
    results_rows = read_csv(
        results_path,
        ["system", "category", "item", "target_count", "error_count", "status", "comment"],
    )
    stress_rows = read_csv(stress_path, ["system", "problem_words", "summary"])

    latency_summary = analyze_latency(latency_rows, latency_path)
    quality_summary = analyze_quality(results_rows, results_path)
    stress_summary = analyze_stress(stress_rows)

    write_outputs(args.output_dir, latency_summary, quality_summary, stress_summary)
    print(f"Сводные таблицы сохранены в {args.output_dir}")


if __name__ == "__main__":
    main()
