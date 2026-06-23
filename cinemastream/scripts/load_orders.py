# cinemastream/scripts/load_orders.py
"""
Load and clean subscription order data from a raw CSV export.
Writes validated rows to data/clean_orders.csv and logs rejected rows.

Usage:
    python scripts/load_orders.py
"""

import csv
import json
from pathlib import Path


# ---- Configuration ----
DATA_DIR      = Path(__file__).parent.parent / "data"
INPUT_FILE    = DATA_DIR / "raw_orders.csv"
OUTPUT_FILE   = DATA_DIR / "clean_orders.csv"
REJECTED_FILE = DATA_DIR / "rejected_orders.json"

VALID_PLANS      = {"Free", "Basic", "Premium"}
VALID_CURRENCIES = {"SGD", "MYR", "IDR", "PHP", "THB", "VND", "INR"}


# ---- Validation ----
class OrderValidationError(ValueError):
    """Raised when an order row fails business-rule validation."""
    pass


def validate_order(row: dict) -> dict:
    """
    Validate and coerce a raw order row.
    Returns a cleaned dict. Raises OrderValidationError on failure.
    """
    try:
        order_id = int(row["order_id"])
        user_id  = int(row["user_id"])
    except (ValueError, KeyError) as e:
        raise OrderValidationError(f"Bad ID fields: {e}")

    plan = row.get("plan", "").strip()
    if plan not in VALID_PLANS:
        raise OrderValidationError(f"Unknown plan '{plan}'")

    currency = row.get("currency", "").strip().upper()
    if currency not in VALID_CURRENCIES:
        raise OrderValidationError(f"Unknown currency '{currency}'")

    try:
        amount = float(row["amount_local"])
        if amount < 0:
            raise OrderValidationError(f"Negative amount: {amount}")
    except (ValueError, KeyError) as e:
        raise OrderValidationError(f"Bad amount: {e}")

    return {
        "order_id":     order_id,
        "user_id":      user_id,
        "plan":         plan,
        "currency":     currency,
        "amount_local": amount,
    }


# ---- Main ----
def load_orders(input_path: Path, output_path: Path, rejected_path: Path) -> None:
    """Load raw orders, validate, write clean and rejected files."""
    clean_rows = []
    rejected   = []

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                clean_rows.append(validate_order(row))
            except OrderValidationError as e:
                rejected.append({"row": dict(row), "reason": str(e)})

    # Write clean CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if clean_rows:
        fieldnames = list(clean_rows[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)

    # Write rejected JSON log
    if rejected:
        with open(rejected_path, "w", encoding="utf-8") as f:
            json.dump(rejected, f, indent=2)

    print(f"Loaded {len(clean_rows)} valid orders, {len(rejected)} rejected")
    print(f"Clean:    {output_path}")
    if rejected:
        print(f"Rejected: {rejected_path}")


if __name__ == "__main__":
    load_orders(INPUT_FILE, OUTPUT_FILE, REJECTED_FILE)
