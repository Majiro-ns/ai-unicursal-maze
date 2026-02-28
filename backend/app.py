import logging
from pathlib import Path

from fastapi import FastAPI

from .api.routes import router as api_router

# logs/ ディレクトリを自動作成して logs/app.log にも出力する。
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_fmt = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_fmt,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8"),
    ],
)

app = FastAPI(title="AI Unicursal Maze Generator API")

app.include_router(api_router, prefix="/api")
