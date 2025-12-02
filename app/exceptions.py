"""
Structured exception classes for System Dynamics Backend
Provides unified error handling with structured error responses
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class ValidationError(BaseModel):
    """
    Structured validation error with detailed information

    Attributes:
        code: Error code for programmatic handling
        message: Human-readable error message
        element_id: ID of the element causing the error (if applicable)
        field: Field name within the element (if applicable)
        suggestion: Optional suggestion for fixing the error
        context: Optional additional context information
    """

    code: str
    message: str
    element_id: Optional[str] = None
    field: Optional[str] = None
    suggestion: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """String representation for logging"""
        parts = [f"[{self.code}] {self.message}"]
        if self.element_id:
            parts.append(f"Element: {self.element_id}")
        if self.field:
            parts.append(f"Field: {self.field}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class SimulationError(Exception):
    """
    Exception raised during simulation execution

    Attributes:
        code: Error code for programmatic handling
        message: Human-readable error message
        details: Additional error details
    """

    def __init__(
        self, code: str, message: str, details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response"""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        """String representation for logging"""
        return f"[{self.code}] {self.message}"


class EvaluationError(Exception):
    """
    Exception raised during equation evaluation

    Attributes:
        code: Error code for programmatic handling
        message: Human-readable error message
        element_id: ID of the element being evaluated
        equation: The equation that failed
        details: Additional error details
    """

    def __init__(
        self,
        code: str,
        message: str,
        element_id: Optional[str] = None,
        equation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message
        self.element_id = element_id
        self.equation = equation
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response with unified format"""
        result = {
            "code": self.code,
            "message": self.message,
            "details": self.details.copy() if self.details else {},
        }
        if self.element_id:
            result["details"]["element_id"] = self.element_id
        if self.equation:
            result["details"]["equation"] = self.equation
        return result

    def __str__(self) -> str:
        """String representation for logging"""
        parts = [f"[{self.code}] {self.message}"]
        if self.element_id:
            parts.append(f"Element: {self.element_id}")
        if self.equation:
            parts.append(f"Equation: {self.equation}")
        return " | ".join(parts)


class CircularDependencyError(SimulationError):
    """
    Specific exception for circular dependency detection

    Attributes:
        cycle: List of element IDs forming the cycle
    """

    def __init__(
        self,
        message: str,
        cycle: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.cycle = cycle or []
        super().__init__(
            code="circular_dependency",
            message=message,
            details={**(details or {}), "cycle": self.cycle},
        )


class ModelStructureError(SimulationError):
    """
    Exception for structural issues in the model

    Used for issues like disconnected elements, missing required components, etc.
    """

    def __init__(
        self, message: str, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            code="model_structure_error",
            message=message,
            details=details,
        )