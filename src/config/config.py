from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NODE_DIR = RAW_DIR / "Nodes"
EDGE_DIR = RAW_DIR / "Edges"
TEMPORAL_DIR = RAW_DIR / "Temporal Data"
PROC_DIR = DATA_DIR / "processed"

TRAIN_RATIO: float = 0.6
VAL_RATIO: float = 0.2
TEST_RATIO: float = 0.2


TemporalType = Literal["unit", "weight"]


@dataclass(frozen=True)
class ExperimentConfig:
    temporal_type: TemporalType
    horizons: Sequence[int]
    lag_windows: Sequence[int]
    gru_seq_lengths: Sequence[int]
    # GRU output config: chỉ 1 trong 2 nên True tại 1 thời điểm
    is_softplus: bool = False
    is_log1p: bool = False


DEFAULT_EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        temporal_type="unit",
        horizons=[7],
        lag_windows=[7, 14],
        gru_seq_lengths=[7, 14],
        is_softplus=False,
        is_log1p=False,
    ),
]
