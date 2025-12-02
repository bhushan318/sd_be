"""
Type definitions for System Dynamics Backend
Provides TypedDict and other type hints for structured data
"""

from typing import TypedDict, List, Dict, Any, Optional


class SimulationResultDict(TypedDict):
    """
    Typed dictionary for simulation results
    
    This provides type safety for simulation result dictionaries
    returned by the simulation engine.
    """
    time: List[float]
    results: Dict[str, List[float]]


class ValidationSummaryDict(TypedDict, total=False):
    """
    Typed dictionary for validation summary
    
    All fields are optional to match the actual validation response structure.
    """
    valid: bool
    error_count: int
    warning_count: int
    errors_by_code: Dict[str, int]
    errors_by_element: Dict[str, int]


class CacheStatsDict(TypedDict):
    """
    Typed dictionary for cache statistics
    """
    size: int
    max_size: int
    hits: int
    misses: int
    evictions: int
    hit_rate: float


class ResultStoreStatsDict(TypedDict):
    """
    Typed dictionary for result store statistics
    """
    size: int
    max_size: int
    results: List[Dict[str, Any]]

