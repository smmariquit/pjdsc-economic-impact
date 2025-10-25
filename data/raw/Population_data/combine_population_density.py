import csv
import re
from pathlib import Path
import sys


FILENAME_PATTERN = re.compile(r"^(?P<year>\d{4})_population_density\.csv$")


def find_input_files(directory: Path) -> list[Path]:
    csv_files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    matched = []
    for path in csv_files:
        match = FILENAME_PATTERN.match(path.name)
        if match:
            matched.append((int(match.group("year")), path))
    matched.sort(key=lambda t: t[0])
    return [p for _, p in matched]


def build_header(files: list[Path]) -> list[str]:
    header_order: list[str] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            for field in reader.fieldnames:
                if field not in header_order:
                    header_order.append(field)
    final_header = ["Year"] + header_order if "Year" not in header_order else header_order
    return final_header


def combine_files(files: list[Path], output_path: Path) -> None:
    if not files:
        print("No matching input files found. Skipping.")
        return

    header = build_header(files)

    with output_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=header)
        writer.writeheader()

        for file_path in files:
            year_match = FILENAME_PATTERN.match(file_path.name)
            year_value = year_match.group("year") if year_match else ""

            with file_path.open("r", encoding="utf-8-sig", newline="") as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    row_out = {k: row.get(k, "") for k in header}
                    if "Year" in header:
                        row_out["Year"] = year_value
                    writer.writerow(row_out)


def main() -> None:
    script_dir = Path(__file__).parent
    input_dir = script_dir
    default_output = script_dir / "population_density_all_years.csv"

    # Optional: allow custom output path via CLI arg
    output_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default_output

    files = find_input_files(input_dir)
    if not files:
        print(f"No files matched pattern in: {input_dir}")
        return

    print(f"Combining {len(files)} files → {output_path}")
    combine_files(files, output_path)
    print("Done.")


if __name__ == "__main__":
    main()


