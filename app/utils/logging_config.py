"""
Logging configuration for System Dynamics Backend
Supports JSON logs in production and human-readable logs in development
Includes request ID middleware support
"""

import logging
import sys
import json
import uuid
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from datetime import datetime, timezone
import os

# Context variable for request ID (thread-safe)
request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID from context or record attribute
        request_id = request_id_context.get()
        if request_id:
            log_data["request_id"] = request_id
        elif hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "request_id",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter with request ID support"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable string"""
        # Get request ID from context or record attribute
        request_id = request_id_context.get()
        if not request_id and hasattr(record, "request_id"):
            request_id = record.request_id

        # Build base format
        base_format = "%(asctime)s - %(name)s - %(levelname)s"
        if request_id:
            base_format += f" - [request_id={request_id}]"
        base_format += " - %(message)s"

        # Create temporary formatter with request ID
        formatter = logging.Formatter(base_format, datefmt="%Y-%m-%d %H:%M:%S")

        return formatter.format(record)


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Setup logging configuration

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (for production)
        log_file: Optional log file path
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    """
    # Determine format
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = HumanReadableFormatter()

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set level for third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID in context

    Args:
        request_id: Optional request ID. If None, generates a new UUID.

    Returns:
        The request ID (generated or provided)
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_context.set(request_id)
    return request_id


def get_request_id() -> Optional[str]:
    """
    Get current request ID from context

    Returns:
        Current request ID or None
    """
    return request_id_context.get()


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
