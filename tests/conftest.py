import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
BOT_DIR = PROJECT_DIR / "bot"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.append(str(BOT_DIR))
