from __future__ import annotations

import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engines.overview_metrics import _format_datetime_from_utc_us  # noqa: E402


def test_utc_timestamp_format_is_timezone_explicit_and_warning_free():
    utc_us = 1_704_067_200_000_000  # 2024-01-01 00:00:00 UTC
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert _format_datetime_from_utc_us(utc_us) == "01-01-2024 00:00"
        assert _format_datetime_from_utc_us(utc_us, offset_min=540) == "01-01-2024 09:00"
