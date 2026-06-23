# cinemastream/scripts/currency_api.py
"""
Fetch live currency exchange rates and convert subscription amounts to SGD.

Usage:
    python scripts/currency_api.py --amount 100 --from MYR --to SGD
    python scripts/currency_api.py --list-rates

Environment variables:
    OPENEXCHANGE_APP_ID   API key (optional — uses fallback rates if absent)
"""

import argparse
import os
import sys
import requests
from requests.exceptions import RequestException

FALLBACK_RATES_TO_SGD: dict = {
    "SGD": 1.000,
    "MYR": 0.296,
    "IDR": 0.000087,
    "PHP": 0.024,
    "THB": 0.037,
    "VND": 0.000054,
    "INR": 0.016,
}

SUPPORTED_CURRENCIES = set(FALLBACK_RATES_TO_SGD.keys())


def fetch_rates_from_api(app_id: str):
    url = f"https://openexchangerates.org/api/latest.json?app_id={app_id}&base=USD"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("rates", {})
    except RequestException as e:
        print(f"API fetch failed: {e}. Using fallback rates.", file=sys.stderr)
        return None


def get_rates_to_sgd(app_id=None):
    if not app_id:
        return FALLBACK_RATES_TO_SGD
    usd_rates = fetch_rates_from_api(app_id)
    if usd_rates is None:
        return FALLBACK_RATES_TO_SGD
    usd_to_sgd = usd_rates.get("SGD", 1.35)
    return {
        currency: usd_to_sgd / rate
        for currency, rate in usd_rates.items()
        if currency in SUPPORTED_CURRENCIES and rate != 0
    }


def convert_to_sgd(amount: float, from_currency: str, rates: dict) -> float:
    if from_currency == "SGD":
        return amount
    rate = rates.get(from_currency)
    if rate is None:
        raise ValueError(f"Unsupported currency: {from_currency}")
    return amount * rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CinemaStream currency conversion")
    parser.add_argument("--from", dest="from_currency", metavar="CURRENCY")
    parser.add_argument("--to", dest="to_currency", default="SGD", metavar="CURRENCY")
    parser.add_argument("--amount", type=float)
    parser.add_argument("--list-rates", action="store_true")
    args = parser.parse_args()

    rates = get_rates_to_sgd(os.environ.get("OPENEXCHANGE_APP_ID"))

    if args.list_rates:
        print("Exchange rates -> SGD:")
        for currency, rate in sorted(rates.items()):
            print(f"  1 {currency} = {rate:.6f} SGD")
        sys.exit(0)

    if not args.from_currency or args.amount is None:
        print("ERROR: --from and --amount are required", file=sys.stderr)
        sys.exit(1)

    try:
        sgd_amount = convert_to_sgd(args.amount, args.from_currency, rates)
        print(f"{args.amount:.2f} {args.from_currency} = {sgd_amount:.4f} SGD")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
