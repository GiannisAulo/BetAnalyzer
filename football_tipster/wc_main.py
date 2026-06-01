#!/usr/bin/env python3
"""
WC2026 main entry point.

Run from the football_tipster directory:
    python wc_main.py
"""

import sys
from pathlib import Path

# Windows terminals default to cp1253 here — force UTF-8 so the box-drawing
# and € characters in the prediction output don't raise UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))


def _menu() -> str:
    print()
    print("═" * 40)
    print("  WC2026 Prediction System")
    print("═" * 40)
    print("  1 — Predict specific game")
    print("  2 — Run all group predictions  (coming soon)")
    print("  0 — Exit")
    print()
    return input("  Select: ").strip()


def main() -> None:
    while True:
        choice = _menu()

        if choice == "0":
            print("  Bye.")
            break

        elif choice == "1":
            print()
            print("  Loading specific_game.json ...")
            from wc_analyzer import predict_specific_game
            predict_specific_game()
            input("  Press Enter to continue ...")

        elif choice == "2":
            print()
            print("  Group predictions not yet implemented.")
            input("  Press Enter to continue ...")

        else:
            print("  Invalid option — try again.")


if __name__ == "__main__":
    main()
