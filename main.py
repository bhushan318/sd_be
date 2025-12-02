"""
System Dynamics Simulation Backend
Entry point for the FastAPI application
"""

import uvicorn
from app.api import app
from app.utils.logging_config import setup_logging, get_logger
from app.config import get_settings

# Get configuration
settings = get_settings()

# Setup logging before starting the app
setup_logging(
    level=settings.log_level,
    json_format=settings.log_format_json,
    log_file=settings.log_file,
)

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting System Dynamics Simulation Backend")
    logger.info(
        f"Environment: {settings.env}, "
        f"Log level: {settings.log_level}, "
        f"Format: {'JSON' if settings.log_format_json else 'Human-readable'}"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
