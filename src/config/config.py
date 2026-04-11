from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NODE_DIR = RAW_DIR / "Nodes"
EDGE_DIR = RAW_DIR / "Edges"
TEMPORAL_DIR = RAW_DIR / "Temporal Data"
PROC_DIR = DATA_DIR / "processed"

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

LAG_WINDOWS = [7, 14, 30] 