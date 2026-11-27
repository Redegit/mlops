import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_data import parse_listing, prepare_dataset


def listing(**changes):
    row = {
        "sNO6x href": "https://domclick.ru/card/sale__flat__123",
        "s6KKu": "2-комн. квартира",
        "s6KKu 2": "67,1 м²",
        "s6KKu 3": "5/19 эт.",
        "s5oAg": "24 169 420 ₽",
        "sS4xs": "Москва",
        "link-link-777-13-1-9": "В ипотеку от 194 308 ₽/мес",
    }
    return row | changes


class PrepareDataTests(unittest.TestCase):
    def test_full_price_is_used_instead_of_mortgage(self):
        row = parse_listing(listing())
        self.assertEqual(row["price"], 24_169_420)
        self.assertEqual(row["area"], 67.1)

    def test_studio_has_zero_rooms(self):
        row = parse_listing(listing(s6KKu="Апартаменты-студия"))
        self.assertEqual(row["rooms"], 0)
        self.assertEqual(row["property_type"], "apartments")

    def test_new_build_with_exact_floor_is_kept(self):
        row = parse_listing(listing(**{
            "sNO6x href": "https://domclick.ru/card/sale__new_flat__456",
        }))
        self.assertEqual(row["id"], "456")

    def test_mortgage_and_unit_prices_are_not_targets(self):
        for price in ("", "В ипотеку от 194 308 ₽/мес", "372 500 ₽/м²"):
            with self.subTest(price=price), self.assertRaises(ValueError):
                parse_listing(listing(s5oAg=price))

    def test_aggregated_offers_are_rejected(self):
        for changes in ({"s5oAg": "от 10 000 000 ₽"},
                        {"s6KKu 3": "16–19/28 эт."},
                        {"s6KKu 2": "68-72 м²"}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "aggregated_offer"):
                    parse_listing(listing(**changes))

    def test_exact_multiple_offer_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "aggregated_offer"):
            parse_listing(listing(sCVgA="2 предложения такой планировки"))

    def test_description_words_do_not_make_an_aggregate(self):
        row = listing(s7NDB="Уникальное предложение. Есть варианты планировки.")
        self.assertEqual(parse_listing(row)["price"], 24_169_420)

    def test_shifted_fields_are_not_guessed(self):
        with self.assertRaisesRegex(ValueError, "invalid_rooms"):
            parse_listing(listing(s6KKu="133,2 м²"))

    def test_floor_must_belong_to_building(self):
        with self.assertRaisesRegex(ValueError, "invalid_floor"):
            parse_listing(listing(**{"s6KKu 3": "20/19 эт."}))

    def test_object_duplicates_keep_lowest_id_regardless_of_price_or_input_order(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "raw"
            raw.mkdir()
            records = [
                listing(**{"sNO6x href": "https://domclick.ru/card/sale__flat__200",
                           "s5oAg": "11 000 000 ₽"}),
                listing(**{"sNO6x href": "https://domclick.ru/card/sale__flat__99"}),
                listing(**{"sNO6x href": "https://domclick.ru/card/sale__flat__300",
                           "s6KKu 3": "6/19 эт."}),
                listing(**{"sNO6x href": "https://domclick.ru/card/sale__flat__400",
                           "sS4xs": ""}),
                listing(**{"sNO6x href": "https://domclick.ru/card/sale__flat__500",
                           "sS4xs": ""}),
            ]
            with (raw / "snapshot.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            output = Path(folder) / "processed"
            report = prepare_dataset(raw, output)
            with (output / "listings.csv").open() as f:
                self.assertEqual([r["id"] for r in csv.DictReader(f)],
                                 ["99", "300", "400", "500"])
            with (output / "excluded.csv").open() as f:
                excluded = list(csv.DictReader(f))
            self.assertEqual(report["excluded_by_reason"], {"duplicate_object": 1})
            self.assertEqual(excluded[0]["source_url"],
                             "https://domclick.ru/card/sale__flat__200")
            self.assertEqual(excluded[0]["reason"], "duplicate_object")
            self.assertEqual(report["input_rows"],
                             report["accepted_rows"] + report["excluded_rows"])

    def test_later_valid_snapshot_replaces_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "raw"
            raw.mkdir()
            for filename, price in (("domclick_2026-11-20.csv", "10 000 000 ₽"),
                                    ("domclick_2026-11-20-large.csv", "11 000 000 ₽")):
                row = listing(s5oAg=price)
                with (raw / filename).open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    writer.writeheader()
                    writer.writerow(row)
            report = prepare_dataset(raw, Path(folder) / "processed")
            with (Path(folder) / "processed/listings.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["price"], "11000000")
            self.assertEqual(report["excluded_by_reason"], {"duplicate_id": 1})


if __name__ == "__main__":
    unittest.main()
