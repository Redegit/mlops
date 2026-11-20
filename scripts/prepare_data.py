import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


COLUMNS = {
    "url": "sNO6x href",
    "rooms": "s6KKu",
    "area": "s6KKu 2",
    "floor": "s6KKu 3",
    "price": "s5oAg",
}
FIELDS = ["id", "rooms", "area", "floor", "total_floors", "price",
          "property_type", "address", "source_url", "source_file"]
REJECT_FIELDS = ["source_file", "source_row", "source_url", "reason",
                 "raw_rooms", "raw_area", "raw_floor", "raw_price"]


def normalize(value):
    text = unicodedata.normalize("NFKC", value or "")
    return " ".join(text.split()).replace("–", "-").replace("\u2014", "-")


def parse_listing(raw):
    values = {name: normalize(raw.get(column)) for name, column in COLUMNS.items()}
    aggregate_label = r"\d+ предложени[яйе] (?:такой|этой) планировки"
    if (values["price"].lower().startswith("от ")
            or re.search(r"\d\s*-\s*\d", values["floor"])
            or re.search(r"\d\s*-\s*\d", values["area"])
            or any(re.fullmatch(aggregate_label, normalize(v).lower())
                   for v in raw.values() if isinstance(v, str))):
        raise ValueError("aggregated_offer")

    identifier = re.search(r"/card/[a-z_]+__(\d+)(?:$|[/?#])", values["url"])
    if not identifier:
        raise ValueError("invalid_id")

    title = values["rooms"].lower()
    rooms = re.fullmatch(r"(\d+)-комн\. (квартира|апартаменты)", title)
    if title in {"студия", "апартаменты-студия"}:
        room_count = 0
    elif rooms and int(rooms[1]) > 0:
        room_count = int(rooms[1])
    else:
        raise ValueError("invalid_rooms")

    area = re.fullmatch(r"(\d+(?:[.,]\d+)?) м2", values["area"])
    if not area or float(area[1].replace(",", ".")) <= 0:
        raise ValueError("invalid_area")

    floors = re.fullmatch(r"(\d+)\s*/\s*(\d+) эт\.", values["floor"])
    if not floors or not 1 <= int(floors[1]) <= int(floors[2]):
        raise ValueError("invalid_floor")

    price = re.fullmatch(r"([\d ]+) ₽", values["price"])
    if not price or int(price[1].replace(" ", "")) <= 0:
        raise ValueError("missing_or_invalid_price")

    return {
        "id": identifier[1],
        "rooms": room_count,
        "area": float(area[1].replace(",", ".")),
        "floor": int(floors[1]),
        "total_floors": int(floors[2]),
        "price": int(price[1].replace(" ", "")),
        "property_type": "apartments" if "апартамент" in title else "flat",
        "address": normalize(raw.get("sS4xs")),
        "source_url": values["url"],
    }


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(input_dir, output_dir):
    files = sorted(input_dir.glob("*.csv"), key=lambda path: path.stem)
    if not files:
        raise ValueError(f"No CSV files in {input_dir}")

    accepted = {}
    origins = {}
    rejected = []
    input_rows = 0
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(COLUMNS.values()) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
            for number, raw in enumerate(reader, start=2):
                input_rows += 1
                origin = {
                    "source_file": path.name,
                    "source_row": number,
                    "source_url": raw.get(COLUMNS["url"], ""),
                    "raw_rooms": raw.get(COLUMNS["rooms"], ""),
                    "raw_area": raw.get(COLUMNS["area"], ""),
                    "raw_floor": raw.get(COLUMNS["floor"], ""),
                    "raw_price": raw.get(COLUMNS["price"], ""),
                }
                try:
                    row = parse_listing(raw)
                except ValueError as error:
                    rejected.append(origin | {"reason": str(error)})
                    continue
                identifier = row["id"]
                if identifier in accepted:
                    rejected.append(origins[identifier] | {"reason": "duplicate_id"})
                # ISO dates in filenames make the latest valid snapshot win.
                accepted[identifier] = row | {"source_file": path.name}
                origins[identifier] = origin

    rows = sorted(accepted.values(), key=lambda row: int(row["id"]))
    report = {
        "input_files": [path.name for path in files],
        "input_rows": input_rows,
        "accepted_rows": len(rows),
        "excluded_rows": len(rejected),
        "excluded_by_reason": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        "property_types": dict(sorted(Counter(row["property_type"] for row in rows).items())),
        "price_rub": {"min": min((r["price"] for r in rows), default=None),
                      "max": max((r["price"] for r in rows), default=None)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "listings.csv", FIELDS, rows)
    write_csv(output_dir / "excluded.csv", REJECT_FIELDS, rejected)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.input_dir, args.output_dir),
                     ensure_ascii=False, indent=2))
