"""
Pydantic models for System Dynamics Backend
Defines data structures for elements, links, and simulation configuration
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from app.exceptions import ValidationError


class LookupTable(BaseModel):
    """
    Represents a lookup table for interpolation

    Attributes:
        points: List of [x, y] pairs representing the lookup table
        interpolation: Interpolation method ('linear' or 'step')
    """

    points: List[List[float]] = Field(
        ..., description="List of [x, y] coordinate pairs"
    )
    interpolation: str = Field(
        "linear", description="Interpolation method: 'linear' or 'step'"
    )

    @field_validator("points")
    @classmethod
    def validate_points(cls, v: List[List[float]]) -> List[List[float]]:
        """
        Validate that points are properly formatted
        
        Args:
            v: List of points to validate
            
        Returns:
            Validated points list
            
        Raises:
            ValueError: If points are invalid
        """
        if len(v) < 2:
            raise ValueError("Lookup table must have at least 2 points")
        
        # Security: Limit lookup table size to prevent DoS
        from app.config import get_settings
        settings = get_settings()
        if len(v) > settings.max_lookup_table_points:
            raise ValueError(
                f"Lookup table has {len(v)} points, exceeding maximum of {settings.max_lookup_table_points}"
            )
        
        for i, point in enumerate(v):
            if len(point) != 2:
                raise ValueError(f"Point {i} must have exactly 2 coordinates [x, y]")
            if not all(isinstance(coord, (int, float)) for coord in point):
                raise ValueError(f"Point {i} must contain numeric values only")
        
        # Check for duplicate x-values (warn but allow)
        x_values = [p[0] for p in v]
        if len(x_values) != len(set(x_values)):
            # This is a warning, not an error, but we should note it
            pass
        
        return v

    @field_validator("interpolation")
    @classmethod
    def validate_interpolation(cls, v: str) -> str:
        """Validate interpolation method"""
        if v not in ["linear", "step"]:
            raise ValueError("Interpolation must be 'linear' or 'step'")
        return v


class Element(BaseModel):
    """
    Represents a model element (stock, flow, parameter, or variable)

    Attributes:
        id: Unique identifier for the element
        type: Element type ('stock', 'flow', 'parameter', 'variable')
        name: Human-readable name
        initial: Initial value (required for stocks, optional for parameters)
        equation: Mathematical equation string (required for flows/variables)
        value: Static value (for parameters)
        lookup_table: Optional lookup table for LOOKUP function
    """

    id: str
    type: str = Field(
        ..., description="Element type: 'stock', 'flow', 'parameter', or 'variable'"
    )
    name: str
    initial: Optional[float] = 0.0
    equation: Optional[str] = ""
    value: Optional[float] = None
    lookup_table: Optional[LookupTable] = None

    def get_value(self) -> Optional[float]:
        """Get the element's value (for parameters)"""
        if self.value is not None:
            return self.value
        return self.initial

    def has_equation(self) -> bool:
        """Check if element has a non-empty equation"""
        return bool(self.equation and self.equation.strip())


class Link(BaseModel):
    """
    Represents a connection between two elements

    Attributes:
        id: Unique identifier for the link
        source: ID of the source element
        target: ID of the target element
    """

    id: str
    source: str
    target: str


class SimulationConfig(BaseModel):
    """
    Configuration for simulation execution

    Attributes:
        start_time: Simulation start time
        end_time: Simulation end time
        time_step: Time step size (must be > 0)
        method: Integration method ('euler' or 'rk4')
        verbose: Enable detailed logging
    """

    start_time: float = 0.0
    end_time: float = 100.0
    time_step: float = Field(1.0, gt=0, description="Time step must be greater than 0")
    method: str = Field("euler", description="Integration method: 'euler' or 'rk4'")
    verbose: bool = False  # Changed default to False for cleaner output

    def get_num_steps(self) -> int:
        """Calculate the number of simulation steps"""
        if self.time_step <= 0:
            return 0
        return int((self.end_time - self.start_time) / self.time_step) + 1


class SimulationRequest(BaseModel):
    """
    Complete simulation request payload

    Attributes:
        elements: List of model elements
        links: List of connections between elements
        config: Simulation configuration
    """

    elements: List[Element]
    links: List[Link]
    config: SimulationConfig


class SimulationResponse(BaseModel):
    """
    Simulation execution result

    Attributes:
        success: Whether simulation completed successfully
        time: List of time points
        results: Dictionary mapping element IDs to time series data
        error: Error message if simulation failed
        result_id: Optional unique identifier for retrieving/exporting results
    """

    success: bool
    time: List[float] = []
    results: Dict[str, List[float]] = {}
    error: Optional[str] = None
    result_id: Optional[str] = None


class ValidationResponse(BaseModel):
    """
    Model validation result

    Attributes:
        valid: Whether the model is valid
        errors: List of validation errors
        warnings: List of validation warnings
        summary: Summary statistics (if valid)
    """

    valid: bool
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []
    summary: Optional[Dict[str, Any]] = None


class CompareRequest(BaseModel):
    """
    Request for comparing two simulation results

    Attributes:
        result_id_1: First simulation result ID
        result_id_2: Second simulation result ID
        elements: Optional list of element IDs to compare (if None, compares all)
    """

    result_id_1: str
    result_id_2: str
    elements: Optional[List[str]] = None


class CompareResponse(BaseModel):
    """
    Comparison result between two simulations

    Attributes:
        differences: Dictionary mapping element IDs to difference metrics
        summary: Summary statistics of differences
    """

    differences: Dict[str, Dict[str, float]]
    summary: Dict[str, Any]


class SensitivityParameter(BaseModel):
    """
    Parameter configuration for sensitivity analysis

    Attributes:
        element_id: ID of the parameter element to vary
        values: List of values to test
    """

    element_id: str
    values: List[float]


class SensitivityRequest(BaseModel):
    """
    Request for sensitivity analysis

    Attributes:
        elements: List of model elements
        links: List of connections between elements
        config: Base simulation configuration
        parameters: List of parameters to vary
        metrics: Optional list of element IDs to compute metrics for
    """

    elements: List[Element]
    links: List[Link]
    config: SimulationConfig
    parameters: List[SensitivityParameter]
    metrics: Optional[List[str]] = None


class SensitivityResponse(BaseModel):
    """
    Sensitivity analysis result

    Attributes:
        runs: List of simulation results for each parameter combination
        metrics: Dictionary mapping element IDs to sensitivity metrics
        parameter_values: List of parameter value combinations tested
    """

    runs: List[Dict[str, Any]]
    metrics: Dict[str, Dict[str, float]]
    parameter_values: List[Dict[str, float]]


class OptimizationParameter(BaseModel):
    """
    Parameter configuration for optimization

    Attributes:
        element_id: ID of the parameter element to optimize
        min_value: Lower bound for parameter value
        max_value: Upper bound for parameter value
        initial_value: Optional starting value for optimization
    """

    element_id: str
    min_value: float
    max_value: float
    initial_value: Optional[float] = None

    def get_initial_or_midpoint(self) -> float:
        """Get initial value or midpoint of bounds"""
        if self.initial_value is not None:
            return self.initial_value
        return (self.min_value + self.max_value) / 2.0


class OptimizationRequest(BaseModel):
    """
    Request for parameter optimization

    Attributes:
        elements: List of model elements
        links: List of connections between elements
        config: Base simulation configuration
        parameters: List of parameters to optimize
        objective: Element ID to optimize
        objective_type: "maximize" or "minimize"
        metric: Metric to extract from objective element
        algorithm: Optimization algorithm to use
        max_iterations: Maximum number of function evaluations
        tolerance: Convergence tolerance
        seed: Random seed for reproducibility
    """

    elements: List[Element]
    links: List[Link]
    config: SimulationConfig
    parameters: List[OptimizationParameter]
    objective: str
    objective_type: str = Field(
        "minimize", description="Optimization type: 'maximize' or 'minimize'"
    )
    metric: str = Field(
        "final", description="Metric: 'final', 'mean', 'max', 'min', 'integral'"
    )
    algorithm: str = Field(
        "differential_evolution", description="Optimization algorithm"
    )
    max_iterations: Optional[int] = None
    tolerance: Optional[float] = None
    seed: Optional[int] = None


class OptimizationResponse(BaseModel):
    """
    Optimization result

    Attributes:
        success: Whether optimization completed successfully
        optimal_parameters: Dictionary mapping parameter IDs to optimal values
        optimal_value: Objective function value at optimum
        iterations: Number of iterations/evaluations performed
        convergence_info: Algorithm-specific convergence information
        result_id: Optional ID of best simulation result
        error: Error message if optimization failed
    """

    success: bool
    optimal_parameters: Dict[str, float] = {}
    optimal_value: Optional[float] = None
    iterations: int = 0
    convergence_info: Dict[str, Any] = {}
    result_id: Optional[str] = None
    error: Optional[str] = None


class ObservedDataPoint(BaseModel):
    """
    Single observed data point for calibration

    Attributes:
        time: Time point of observation
        values: Dictionary mapping element IDs to observed values
    """

    time: float
    values: Dict[str, float]


class CalibrationRequest(BaseModel):
    """
    Request for model calibration

    Attributes:
        elements: List of model elements
        links: List of connections between elements
        config: Base simulation configuration
        parameters: List of parameters to calibrate
        observed_data: List of observed data points
        elements_to_fit: List of element IDs to fit
        error_metric: Error metric to minimize
        algorithm: Optimization algorithm to use
        max_iterations: Maximum number of function evaluations
        tolerance: Convergence tolerance
        seed: Random seed for reproducibility
        interpolation: Interpolation method for matching model to observed times
    """

    elements: List[Element]
    links: List[Link]
    config: SimulationConfig
    parameters: List[OptimizationParameter]
    observed_data: List[ObservedDataPoint]
    elements_to_fit: List[str]
    error_metric: str = Field(
        "rmse", description="Error metric: 'rmse', 'mae', 'mse', 'mape', 'r2'"
    )
    algorithm: str = Field(
        "differential_evolution", description="Optimization algorithm"
    )
    max_iterations: Optional[int] = None
    tolerance: Optional[float] = None
    seed: Optional[int] = None
    interpolation: str = Field(
        "linear", description="Interpolation method: 'linear' or 'step'"
    )


class CalibrationResponse(BaseModel):
    """
    Calibration result

    Attributes:
        success: Whether calibration completed successfully
        calibrated_parameters: Dictionary mapping parameter IDs to calibrated values
        error_value: Final error metric value
        iterations: Number of iterations/evaluations performed
        convergence_info: Algorithm-specific convergence information
        fitted_results: Model outputs at observed time points
        error_by_element: Error metric for each element
        result_id: Optional ID of best simulation result
        error: Error message if calibration failed
    """

    success: bool
    calibrated_parameters: Dict[str, float] = {}
    error_value: float = 0.0
    iterations: int = 0
    convergence_info: Dict[str, Any] = {}
    fitted_results: Dict[str, List[float]] = {}
    error_by_element: Dict[str, float] = {}
    result_id: Optional[str] = None
    error: Optional[str] = None