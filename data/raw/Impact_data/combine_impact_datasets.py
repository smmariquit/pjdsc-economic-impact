import csv
import re
from pathlib import Path
import sys
from typing import List, Dict, Optional, Tuple

# Patterns for file discovery
PEOPLE_PATTERN = re.compile(r"^(?P<year>\d{4})_(?P<storm>[A-Za-z]+)_people_affected\.csv$")
HOUSES_PATTERN = re.compile(r"^(?P<year>\d{4})_(?P<storm>[A-Za-z]+)_houses\.csv$")

# Output headers
PEOPLE_HEADER = ["Year", "Storm", "Province", "Affected"]
HOUSES_HEADER = [
    "Year",
    "Storm",
    "Province",
    "Houses destroyed",
    "Houses damaged",
    "Total Houses",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, header: List[str], rows: List[Dict[str, str]]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _float0(v: str) -> float:
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return 0.0


def _parse_from_filename(path: Path, pattern: re.Pattern) -> Tuple[str, str]:
    m = pattern.match(path.name)
    if not m:
        return "", ""
    return m.group("year"), m.group("storm")


# =====================
# People Affected
# =====================

def find_people_files(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.rglob("*_people_affected.csv") if p.is_file()])


def combine_people(base_dir: Path, out_csv: Path) -> int:
    files = find_people_files(base_dir)
    if not files:
        print(f"No *_people_affected.csv found under: {base_dir}")
        return 0

    out_rows: List[Dict[str, str]] = []
    for file_path in files:
        year, storm = _parse_from_filename(file_path, PEOPLE_PATTERN)
        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            # find Affected col case-insensitively
            affected_key = None
            for k in (reader.fieldnames or []):
                if k and k.lower() == "affected":
                    affected_key = k
                    break
            for r in reader:
                val = r.get(affected_key, "") if affected_key else ""
                if _float0(val) == 0.0:
                    continue
                out_rows.append({
                    "Year": year or str(r.get("Year", "")).strip(),
                    "Storm": storm or str(r.get("Storm", "")).strip(),
                    "Province": str(r.get("Province", "")).strip(),
                    "Affected": str(r.get(affected_key, r.get("Affected", ""))).strip(),
                })

    _write_csv(out_csv, PEOPLE_HEADER, out_rows)
    print(f"People: wrote {len(out_rows)} rows → {out_csv}")
    return len(out_rows)


def append_legacy_people(legacy_csv: Path, people_csv: Path):
    if not legacy_csv.exists() or not people_csv.exists():
        return
    current = _read_csv(people_csv)
    legacy_raw = _read_csv(legacy_csv)

    # Standardize legacy
    legacy = []
    for r in legacy_raw:
        ph = str(r.get("PH_Name", "")).strip()
        intl = str(r.get("Intl_Name", "")).strip()
        storm = ph if ph else intl
        row = {
            "Year": str(r.get("Year", "")).strip(),
            "Storm": storm,
            "Province": str(r.get("Province", "")).strip(),
            "Affected": str(r.get("Affected", "")).strip(),
        }
        if _float0(row["Affected"]) == 0.0:
            continue
        legacy.append(row)

    # Fill missing storm names in current using legacy mapping by (Year, Province, Affected)
    def key3(r: Dict[str, str]) -> Tuple[str, str, str]:
        return (r.get("Year", ""), r.get("Province", ""), r.get("Affected", ""))

    map_legacy: Dict[Tuple[str, str, str], str] = {}
    for r in legacy:
        k = key3(r)
        if r["Storm"] and k not in map_legacy:
            map_legacy[k] = r["Storm"]

    for r in current:
        if not r.get("Storm", "").strip():
            k = key3(r)
            if k in map_legacy:
                r["Storm"] = map_legacy[k]

    # Merge and dedupe by (Year, Province, Affected) preferring rows with Storm
    merged: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for r in current + legacy:
        k = key3(r)
        ex = merged.get(k)
        if ex is None or (not ex.get("Storm", "").strip() and r.get("Storm", "").strip()):
            merged[k] = r

    rows = list(merged.values())
    rows.sort(key=lambda r: (int(r.get("Year", "0") or 0), r.get("Storm", ""), r.get("Province", "")))
    _write_csv(people_csv, PEOPLE_HEADER, rows)
    print(f"People: merged with legacy, total {len(rows)} rows → {people_csv}")


# =====================
# Houses
# =====================

def find_houses_files(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.rglob("*_houses.csv") if p.is_file()])


def _normalize_house_row(r: Dict[str, str]) -> Dict[str, str]:
    # Map possible case variants to standard output header
    def find_key(options: List[str]) -> Optional[str]:
        for k in r.keys():
            kl = k.lower()
            for opt in options:
                if kl == opt:
                    return k
        return None

    k_destroyed = find_key(["houses destroyed", "destroyed", "houses_destroyed"]) or "Houses destroyed"
    k_damaged = find_key(["houses damaged", "damaged", "houses_damaged"]) or "Houses damaged"
    k_total = find_key(["total houses", "total", "total_houses"]) or "Total Houses"

    out = {
        "Year": str(r.get("Year", "")).strip(),
        "Storm": str(r.get("Storm", "")).strip(),
        "Province": str(r.get("Province", "")).strip(),
        "Houses destroyed": str(r.get(k_destroyed, r.get("Houses destroyed", ""))).strip(),
        "Houses damaged": str(r.get(k_damaged, r.get("Houses damaged", ""))).strip(),
        "Total Houses": str(r.get(k_total, r.get("Total Houses", ""))).strip(),
    }
    return out


def combine_houses(base_dir: Path, out_csv: Path) -> int:
    files = find_houses_files(base_dir)
    if not files:
        print(f"No *_houses.csv found under: {base_dir}")
        return 0

    out_rows: List[Dict[str, str]] = []
    for file_path in files:
        year, storm = _parse_from_filename(file_path, HOUSES_PATTERN)
        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                nr = _normalize_house_row(r)
                # set Year/Storm from filename if missing
                if not nr["Year"]:
                    nr["Year"] = year
                if not nr["Storm"]:
                    nr["Storm"] = storm
                # filter rows where all counts are zero (use Total first; else sum destroyed+damaged)
                total = _float0(nr.get("Total Houses", ""))
                if total == 0.0:
                    d = _float0(nr.get("Houses destroyed", ""))
                    g = _float0(nr.get("Houses damaged", ""))
                    if d + g == 0.0:
                        continue
                out_rows.append(nr)

    _write_csv(out_csv, HOUSES_HEADER, out_rows)
    print(f"Houses: wrote {len(out_rows)} rows → {out_csv}")
    return len(out_rows)


def append_legacy_houses(legacy_csv: Path, houses_csv: Path):
    if not legacy_csv.exists() or not houses_csv.exists():
        return

    current = _read_csv(houses_csv)
    legacy_raw = _read_csv(legacy_csv)

    # Standardize legacy
    legacy: List[Dict[str, str]] = []
    for r in legacy_raw:
        ph = str(r.get("PH_Name", "")).strip()
        intl = str(r.get("Intl_Name", "")).strip()
        storm = ph if ph else intl
        row = {
            "Year": str(r.get("Year", "")).strip(),
            "Storm": storm,
            "Province": str(r.get("Province", "")).strip(),
            "Houses destroyed": str(r.get("Houses destroyed", r.get("Houses Destroyed", r.get("Destroyed", "")))).strip(),
            "Houses damaged": str(r.get("Houses damaged", r.get("Houses Damaged", r.get("Damaged", "")))).strip(),
            "Total Houses": str(r.get("Total Houses", r.get("Total", ""))).strip(),
        }
        total = _float0(row.get("Total Houses", ""))
        if total == 0.0:
            d = _float0(row.get("Houses destroyed", ""))
            g = _float0(row.get("Houses damaged", ""))
            if d + g == 0.0:
                continue
        legacy.append(row)

    # Fill missing storm names in current using legacy by (Year, Province, Total Houses)
    def key3h(r: Dict[str, str]) -> Tuple[str, str, str]:
        return (r.get("Year", ""), r.get("Province", ""), r.get("Total Houses", ""))

    map_legacy: Dict[Tuple[str, str, str], str] = {}
    for r in legacy:
        k = key3h(r)
        if r["Storm"] and k not in map_legacy:
            map_legacy[k] = r["Storm"]

    for r in current:
        if not r.get("Storm", "").strip():
            k = key3h(r)
            if k in map_legacy:
                r["Storm"] = map_legacy[k]

    # Merge and dedupe by (Year, Province, Total Houses, Destroyed, Damaged)
    def key5(r: Dict[str, str]) -> Tuple[str, str, str, str, str]:
        return (
            r.get("Year", ""),
            r.get("Province", ""),
            r.get("Total Houses", ""),
            r.get("Houses destroyed", ""),
            r.get("Houses damaged", ""),
        )

    merged: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}
    for r in current + legacy:
        k = key5(r)
        ex = merged.get(k)
        if ex is None or (not ex.get("Storm", "").strip() and r.get("Storm", "").strip()):
            merged[k] = r

    rows = list(merged.values())
    rows.sort(key=lambda r: (int(r.get("Year", "0") or 0), r.get("Storm", ""), r.get("Province", "")))
    _write_csv(houses_csv, HOUSES_HEADER, rows)
    print(f"Houses: merged with legacy, total {len(rows)} rows → {houses_csv}")


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir / "Impact_Data 2021-2025"
    legacy_csv = script_dir / "Impact_Data 2010-2020.csv"

    # Outputs
    people_out = script_dir / "people_affected_all_years.csv"
    houses_out = script_dir / "houses_all_years.csv"

    # Allow overrides via CLI
    # python combine_impact_datasets.py [base_dir] [legacy_csv] [people_out] [houses_out]
    if len(sys.argv) >= 2:
        bd = Path(sys.argv[1]).resolve()
        if bd.exists():
            base_dir = bd
    if len(sys.argv) >= 3:
        lc = Path(sys.argv[2]).resolve()
        if lc.exists():
            legacy_csv = lc
    if len(sys.argv) >= 4:
        people_out = Path(sys.argv[3]).resolve()
    if len(sys.argv) >= 5:
        houses_out = Path(sys.argv[4]).resolve()

    print(f"Base dir: {base_dir}")
    print(f"Legacy file: {legacy_csv}")

    # People
    combine_people(base_dir, people_out)
    if legacy_csv.exists():
        append_legacy_people(legacy_csv, people_out)

    # Houses
    combine_houses(base_dir, houses_out)
    if legacy_csv.exists():
        append_legacy_houses(legacy_csv, houses_out)

    print("Done.")


if __name__ == "__main__":
    main()
