import csv
from pathlib import Path


def main():
    audio_dir = Path(__file__).resolve().parents[1] / "audio"
    manifest_dir = audio_dir / "manifests"
    output_path = audio_dir / "manifest.csv"
    manifest_paths = sorted(manifest_dir.glob("*.csv"))
    if not manifest_paths:
        raise RuntimeError(f"В {manifest_dir} нет манифестов для объединения.")

    fieldnames = None
    rows = []
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"Колонки в {manifest_path} отличаются от остальных манифестов.")
            rows.extend(reader)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Объединено записей: {len(rows)}")
    print(f"Манифест: {output_path}")


if __name__ == "__main__":
    main()
