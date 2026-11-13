"""Project-wide settings. Change values here rather than inside modules."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("P2ML_RAW_DIR", ROOT / "data" / "raw"))
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"


@dataclass
class SplitConfig:
    """GKX (2020) style split: expanding train, fixed-length rolling validation,
    one-year test block, refit once a year."""
    first_train_year: int = 1957
    first_valid_year: int = 1975      # GKX: train 1957-1974 (18y)
    valid_years: int = 12             # GKX: validation 1975-1986 (12y)
    first_test_year: int = 1987       # GKX: test 1987-2016
    last_test_year: int = 2016
    test_years: int = 1               # refit frequency in years
    embargo_months: int = 0           # set to horizon-1 for multi-month targets


@dataclass
class LagConfig:
    """Publication lags used by GKX to avoid look-ahead bias (months)."""
    annual: int = 6
    quarterly: int = 4
    monthly: int = 1


@dataclass
class UniverseConfig:
    share_codes: tuple = (10, 11)     # common stocks
    exchange_codes: tuple = (1, 2, 3) # NYSE, AMEX, NASDAQ
    min_price: float = 0.0            # e.g. 5.0 to drop penny stocks
    drop_microcaps: bool = False      # below NYSE 20th pct of market cap


@dataclass
class Config:
    split: SplitConfig = field(default_factory=SplitConfig)
    lags: LagConfig = field(default_factory=LagConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    horizon: int = 1                  # forecast horizon in months (1, 3, 6, 12)
    seed: int = 42
