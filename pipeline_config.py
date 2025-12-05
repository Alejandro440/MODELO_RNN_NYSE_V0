from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_STOCKS_DIR = DATA_DIR / "raw" / "stocks"
RAW_INDICES_DIR = DATA_DIR / "raw" / "indices"
PROCESSED_STOCKS_DIR = DATA_DIR / "processed" / "stocks"
TARGET_STOCKS_DIR = DATA_DIR / "processed" / "with_target"
INDEX_PROCESSED_DIR = DATA_DIR / "processed" / "indices"
MERGED_WITH_INDEX_DIR = DATA_DIR / "processed" / "with_index"
MODEL_OUTPUT_DIR = DATA_DIR / "models"


def ensure_data_directories() -> None:
    for path in [
        RAW_STOCKS_DIR,
        RAW_INDICES_DIR,
        PROCESSED_STOCKS_DIR,
        TARGET_STOCKS_DIR,
        INDEX_PROCESSED_DIR,
        MERGED_WITH_INDEX_DIR,
        MODEL_OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)