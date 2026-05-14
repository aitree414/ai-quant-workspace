#!/usr/bin/env python3
"""Quick-start entry point for the AI Investment Committee.

Usage
-----
    python run_committee.py                         # default: 2330.TW
    python run_committee.py --ticker AAPL --fallback
    python run_committee.py --ticker TSLA --start 2024-01-01 --json
"""

import sys

from strategies.committee import main

if __name__ == "__main__":
    sys.exit(main())
