"""
Configuration module for System Dynamics Backend
Centralizes all environment variable access and configuration settings
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    
    All settings can be overridden via environment variables.
    Default values are provided for development.
    """

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "human"
    log_file: Optional[str] = None
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_backup_count: int = 5

    # Environment
    env: str = "development"  # "development" or "production"
    debug: bool = False

    # CORS configuration
    allowed_origins: str = "*"  # Comma-separated list of origins

    # Request limits
    max_request_size: int = 1 * 1024 * 1024  # 1 MB
    max_ast_depth: int = 20
    max_ast_nodes: int = 1000  # Maximum total AST nodes per equation (DoS protection)

    # Simulation configuration
    simulation_timeout: int = 300  # 5 minutes in seconds
    progress_update_interval: float = 0.1  # Progress update frequency in seconds (WebSocket)
    progress_report_interval: int = 100  # Progress reporting interval (every N steps)

    # Result store configuration
    result_store_max_size: int = 1000
    result_store_ttl_hours: int = 24  # Default TTL for results

    # Cache configuration
    cache_max_size: int = 50
    
    # Lookup table limits
    max_lookup_table_points: int = 1000  # Maximum points per lookup table
    
    # Optimization defaults
    default_optimization_tolerance: float = 0.01  # Default tolerance for optimization algorithms

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def log_format_json(self) -> bool:
        """Check if logging should use JSON format"""
        return self.log_format.lower() == "json"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.env.lower() == "production"

    @property
    def allowed_origins_list(self) -> List[str]:
        """Get allowed origins as a list"""
        origins = [origin.strip() for origin in self.allowed_origins.split(",")]
        if "*" in origins:
            return ["*"]
        return origins

    def __init__(self, **kwargs):
        """Initialize settings with environment variable overrides"""
        super().__init__(**kwargs)
        # Force JSON logging in production if not explicitly set
        if self.is_production and not self.log_format_json:
            self.log_format = "json"


# Global settings instance (singleton)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance (singleton pattern)
    
    Returns:
        Settings instance with current configuration
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

