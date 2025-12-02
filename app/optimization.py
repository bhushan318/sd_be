"""
Optimization and calibration engine for System Dynamics models
Provides functions for parameter optimization and model calibration
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from scipy.optimize import (
    differential_evolution,
    basinhopping,
    minimize,
    dual_annealing,
)
from scipy.interpolate import interp1d

from app.models import (
    Element,
    Link,
    SimulationConfig,
    OptimizationParameter,
    ObservedDataPoint,
)
from app.simulation import SystemDynamicsModel
from app.exceptions import SimulationError, EvaluationError
from app.utils.model_utils import apply_parameter_values

logger = logging.getLogger(__name__)


# ============================================================================
# Metric Extraction
# ============================================================================


def extract_metric(
    values: List[float], time: List[float], metric: str
) -> float:
    """
    Extract a scalar metric from a time series

    Args:
        values: Time series values
        time: Time points
        metric: Metric type ("final", "mean", "max", "min", "integral")

    Returns:
        Extracted metric value

    Raises:
        ValueError: If metric type is unknown
    """
    if not values:
        return float("inf")

    values_array = np.array(values)
    time_array = np.array(time)

    if metric == "final":
        return float(values_array[-1])
    elif metric == "mean":
        return float(np.mean(values_array))
    elif metric == "max":
        return float(np.max(values_array))
    elif metric == "min":
        return float(np.min(values_array))
    elif metric == "integral":
        # Trapezoidal integration
        return float(np.trapz(values_array, time_array))
    else:
        raise ValueError(f"Unknown metric: {metric}. Valid options: final, mean, max, min, integral")


# ============================================================================
# Error Metrics
# ============================================================================


def compute_error_metric(
    model_values: np.ndarray,
    observed_values: np.ndarray,
    error_type: str,
) -> float:
    """
    Compute error metric between model and observed values

    Args:
        model_values: Model output values (1D array)
        observed_values: Observed values (1D array)
        error_type: Error metric type ("rmse", "mae", "mse", "mape", "r2")

    Returns:
        Error metric value (lower is better, except r2 which is negated)

    Raises:
        ValueError: If arrays have different lengths or error type is unknown
    """
    model_values = np.asarray(model_values)
    observed_values = np.asarray(observed_values)

    if len(model_values) != len(observed_values):
        raise ValueError(
            f"Model and observed values must have same length: "
            f"{len(model_values)} vs {len(observed_values)}"
        )

    if len(model_values) == 0:
        return float("inf")

    if error_type == "rmse":
        # Root Mean Square Error
        return float(np.sqrt(np.mean((model_values - observed_values) ** 2)))

    elif error_type == "mae":
        # Mean Absolute Error
        return float(np.mean(np.abs(model_values - observed_values)))

    elif error_type == "mse":
        # Mean Square Error
        return float(np.mean((model_values - observed_values) ** 2))

    elif error_type == "mape":
        # Mean Absolute Percentage Error
        mask = observed_values != 0
        if not np.any(mask):
            return float("inf")
        return float(
            np.mean(np.abs((model_values[mask] - observed_values[mask]) / observed_values[mask])) * 100
        )

    elif error_type == "r2":
        # R-squared (coefficient of determination)
        # Return negative R2 for minimization (we want to maximize R2)
        ss_res = np.sum((observed_values - model_values) ** 2)
        ss_tot = np.sum((observed_values - np.mean(observed_values)) ** 2)
        if ss_tot == 0:
            return 0.0  # Perfect fit to constant data
        r2 = 1.0 - (ss_res / ss_tot)
        return float(-r2)  # Negate for minimization

    else:
        raise ValueError(
            f"Unknown error metric: {error_type}. "
            f"Valid options: rmse, mae, mse, mape, r2"
        )


# ============================================================================
# Objective Functions
# ============================================================================


class ObjectiveFunction:
    """
    Objective function wrapper for optimization

    Wraps the simulation model and extracts a scalar metric
    for optimization algorithms.
    """

    def __init__(
        self,
        elements: List[Element],
        links: List[Link],
        config: SimulationConfig,
        parameters: List[OptimizationParameter],
        objective: str,
        objective_type: str,
        metric: str,
    ):
        """
        Initialize objective function

        Args:
            elements: Model elements
            links: Model links
            config: Simulation configuration
            parameters: Parameters to optimize (with bounds)
            objective: Element ID to extract metric from
            objective_type: "maximize" or "minimize"
            metric: Metric to extract ("final", "mean", "max", "min", "integral")
        """
        self.elements = elements
        self.links = links
        self.config = config
        self.parameters = parameters
        self.objective = objective
        self.objective_type = objective_type.lower()
        self.metric = metric
        self.param_ids = [p.element_id for p in parameters]
        self.evaluation_count = 0
        self.best_value = float("inf")
        self.best_params: Optional[np.ndarray] = None

    def __call__(self, param_values: np.ndarray) -> float:
        """
        Evaluate objective function

        Args:
            param_values: Parameter values (array matching parameters order)

        Returns:
            Objective function value (always formatted for minimization)
        """
        self.evaluation_count += 1

        # Create parameter dictionary
        param_dict = dict(zip(self.param_ids, param_values))

        # Create modified elements with new parameter values
        modified_elements = self._create_modified_elements(param_dict)

        # Run simulation
        try:
            model = SystemDynamicsModel(modified_elements, self.links, verbose=False)
            result = model.simulate(self.config)

            # Extract metric from objective element
            if self.objective not in result["results"]:
                logger.warning(f"Objective element '{self.objective}' not in results")
                return float("inf")

            values = result["results"][self.objective]
            objective_value = extract_metric(values, result["time"], self.metric)

            # Handle NaN or infinite values
            if not np.isfinite(objective_value):
                return float("inf")

            # Negate for maximization (optimizer always minimizes)
            if self.objective_type == "maximize":
                objective_value = -objective_value

            # Track best result
            if objective_value < self.best_value:
                self.best_value = objective_value
                self.best_params = param_values.copy()

            return float(objective_value)

        except (SimulationError, EvaluationError) as e:
            logger.debug(f"Simulation failed: {e.message}")
            return float("inf")
        except Exception as e:
            logger.debug(f"Unexpected error in objective function: {str(e)}")
            return float("inf")

    def _create_modified_elements(self, param_dict: Dict[str, float]) -> List[Element]:
        """Create elements with modified parameter values"""
        return apply_parameter_values(self.elements, param_dict)


class CalibrationFunction:
    """
    Calibration error function wrapper

    Computes the error between model output and observed data
    for calibration/parameter estimation.
    """

    def __init__(
        self,
        elements: List[Element],
        links: List[Link],
        config: SimulationConfig,
        parameters: List[OptimizationParameter],
        observed_data: List[ObservedDataPoint],
        elements_to_fit: List[str],
        error_metric: str,
        interpolation: str,
    ):
        """
        Initialize calibration function

        Args:
            elements: Model elements
            links: Model links
            config: Simulation configuration
            parameters: Parameters to calibrate (with bounds)
            observed_data: List of ObservedDataPoint objects
            elements_to_fit: Element IDs to fit (must match observed_data keys)
            error_metric: Error metric type ("rmse", "mae", "mse", "mape", "r2")
            interpolation: Interpolation method ("linear" or "step")
        """
        self.elements = elements
        self.links = links
        self.config = config
        self.parameters = parameters
        self.elements_to_fit = elements_to_fit
        self.error_metric = error_metric
        self.interpolation = interpolation
        self.param_ids = [p.element_id for p in parameters]
        self.evaluation_count = 0
        self.best_error = float("inf")
        self.best_params: Optional[np.ndarray] = None

        # Extract observed data into arrays
        self.observed_times = np.array([d.time for d in observed_data])
        self.observed_values: Dict[str, np.ndarray] = {}

        for elem_id in elements_to_fit:
            values = []
            for data_point in observed_data:
                values.append(data_point.values.get(elem_id, np.nan))
            self.observed_values[elem_id] = np.array(values)

    def __call__(self, param_values: np.ndarray) -> float:
        """
        Evaluate calibration error

        Args:
            param_values: Parameter values (array matching parameters order)

        Returns:
            Total error metric value (to be minimized)
        """
        self.evaluation_count += 1

        # Create parameter dictionary
        param_dict = dict(zip(self.param_ids, param_values))

        # Create modified elements
        modified_elements = self._create_modified_elements(param_dict)

        # Run simulation
        try:
            model = SystemDynamicsModel(modified_elements, self.links, verbose=False)
            result = model.simulate(self.config)

            model_time = np.array(result["time"])

            # Compute total error across all elements to fit
            total_error = 0.0
            valid_elements = 0

            for elem_id in self.elements_to_fit:
                if elem_id not in result["results"]:
                    logger.warning(f"Element '{elem_id}' not in simulation results")
                    continue

                model_values_series = np.array(result["results"][elem_id])
                observed_values = self.observed_values[elem_id]

                # Interpolate model to observed time points
                model_values = self._interpolate_to_observed_times(
                    model_time, model_values_series
                )

                # Remove NaN values
                valid_mask = ~np.isnan(observed_values) & ~np.isnan(model_values)
                if not np.any(valid_mask):
                    continue

                # Compute error for this element
                element_error = compute_error_metric(
                    model_values[valid_mask],
                    observed_values[valid_mask],
                    self.error_metric,
                )

                if np.isfinite(element_error):
                    total_error += element_error
                    valid_elements += 1

            if valid_elements == 0:
                return float("inf")

            # Average error across elements (optional: could use sum instead)
            # Using sum to match original behavior
            final_error = total_error

            # Track best result
            if final_error < self.best_error:
                self.best_error = final_error
                self.best_params = param_values.copy()

            return float(final_error)

        except (SimulationError, EvaluationError) as e:
            logger.debug(f"Simulation failed during calibration: {e.message}")
            return float("inf")
        except Exception as e:
            logger.debug(f"Unexpected error in calibration: {str(e)}")
            return float("inf")

    def _create_modified_elements(self, param_dict: Dict[str, float]) -> List[Element]:
        """Create elements with modified parameter values"""
        return apply_parameter_values(self.elements, param_dict)

    def _interpolate_to_observed_times(
        self, model_time: np.ndarray, model_values: np.ndarray
    ) -> np.ndarray:
        """Interpolate model values to observed time points"""
        if len(model_time) <= 1:
            return np.full_like(self.observed_times, model_values[0] if len(model_values) > 0 else np.nan)

        kind = "linear" if self.interpolation == "linear" else "nearest"

        try:
            interp_func = interp1d(
                model_time,
                model_values,
                kind=kind,
                bounds_error=False,
                fill_value="extrapolate",
            )
            return interp_func(self.observed_times)
        except Exception as e:
            logger.debug(f"Interpolation failed: {e}")
            return np.full_like(self.observed_times, np.nan)


# ============================================================================
# Optimization Engine
# ============================================================================


class OptimizationEngine:
    """
    Engine for running various optimization algorithms

    Supports:
    - differential_evolution: Global optimization, robust for nonlinear problems
    - basinhopping: Global optimization with local refinement
    - L-BFGS-B: Local optimization, fast for smooth problems
    - dual_annealing: Global optimization, good for many local minima
    """

    SUPPORTED_ALGORITHMS = {
        "differential_evolution",
        "basinhopping",
        "L-BFGS-B",
        "dual_annealing",
    }

    @staticmethod
    def run_optimization(
        objective_func: ObjectiveFunction,
        parameters: List[OptimizationParameter],
        algorithm: str = "differential_evolution",
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, float], float, int, Dict[str, Any]]:
        """
        Run optimization using specified algorithm

        Args:
            objective_func: Objective function to minimize
            parameters: Parameter configurations with bounds
            algorithm: Algorithm name (see SUPPORTED_ALGORITHMS)
            max_iterations: Maximum iterations/evaluations
            tolerance: Convergence tolerance
            seed: Random seed for reproducibility

        Returns:
            Tuple of (optimal_parameters, optimal_value, iterations, convergence_info)

        Raises:
            ValueError: If algorithm is not supported
        """
        if algorithm not in OptimizationEngine.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Supported: {OptimizationEngine.SUPPORTED_ALGORITHMS}"
            )

        # Set up bounds
        bounds = [(p.min_value, p.max_value) for p in parameters]
        param_ids = [p.element_id for p in parameters]

        # Set up initial guess
        x0 = [p.get_initial_or_midpoint() for p in parameters]

        # Default max_iterations
        if max_iterations is None:
            max_iterations = 1000

        logger.info(f"Starting optimization with {algorithm}")
        logger.info(f"Parameters: {param_ids}")
        logger.info(f"Bounds: {bounds}")

        convergence_info: Dict[str, Any] = {}

        try:
            if algorithm == "differential_evolution":
                # Use configured default tolerance if not provided
                from app.config import get_settings
                default_tolerance = get_settings().default_optimization_tolerance
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=seed,
                    maxiter=max_iterations,
                    tol=tolerance if tolerance else default_tolerance,
                    polish=True,
                    disp=False,
                )

            elif algorithm == "basinhopping":
                result = basinhopping(
                    objective_func,
                    x0,
                    niter=max_iterations,
                    minimizer_kwargs={
                        "method": "L-BFGS-B",
                        "bounds": bounds,
                    },
                    seed=seed,
                )

            elif algorithm == "L-BFGS-B":
                options = {"maxiter": max_iterations}
                if tolerance:
                    options["ftol"] = tolerance

                result = minimize(
                    objective_func,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options,
                )

            elif algorithm == "dual_annealing":
                result = dual_annealing(
                    objective_func,
                    bounds,
                    maxiter=max_iterations,
                    seed=seed,
                )

            # Extract results
            optimal_params = dict(zip(param_ids, result.x))
            optimal_value = float(result.fun)
            iterations = objective_func.evaluation_count

            # Build convergence info
            convergence_info = {
                "success": getattr(result, "success", True),
                "message": getattr(result, "message", "Optimization completed"),
                "nfev": getattr(result, "nfev", iterations),
                "nit": getattr(result, "nit", 0),
            }

            logger.info(f"Optimization completed: {convergence_info['message']}")
            logger.info(f"Optimal value: {optimal_value}")
            logger.info(f"Evaluations: {iterations}")

            return optimal_params, optimal_value, iterations, convergence_info

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise

    @staticmethod
    def run_calibration(
        calibration_func: CalibrationFunction,
        parameters: List[OptimizationParameter],
        algorithm: str = "differential_evolution",
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, float], float, int, Dict[str, Any]]:
        """
        Run calibration (parameter estimation)

        This is functionally identical to optimization but uses
        a CalibrationFunction that computes error vs observed data.

        Args:
            calibration_func: Calibration error function
            parameters: Parameter configurations with bounds
            algorithm: Algorithm name
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            seed: Random seed

        Returns:
            Tuple of (calibrated_parameters, error_value, iterations, convergence_info)
        """
        logger.info("Starting model calibration")

        # Use the same optimization machinery
        # CalibrationFunction has the same interface as ObjectiveFunction
        return OptimizationEngine.run_optimization(
            calibration_func,  # type: ignore (compatible interface)
            parameters,
            algorithm,
            max_iterations,
            tolerance,
            seed,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def optimize_model(
    elements: List[Element],
    links: List[Link],
    config: SimulationConfig,
    parameters: List[OptimizationParameter],
    objective: str,
    objective_type: str = "minimize",
    metric: str = "final",
    algorithm: str = "differential_evolution",
    max_iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function for model optimization

    Args:
        elements: Model elements
        links: Model links
        config: Simulation configuration
        parameters: Parameters to optimize
        objective: Element ID to optimize
        objective_type: "maximize" or "minimize"
        metric: Metric to extract
        algorithm: Optimization algorithm
        max_iterations: Maximum iterations
        seed: Random seed

    Returns:
        Dictionary with optimization results
    """
    objective_func = ObjectiveFunction(
        elements=elements,
        links=links,
        config=config,
        parameters=parameters,
        objective=objective,
        objective_type=objective_type,
        metric=metric,
    )

    optimal_params, optimal_value, iterations, convergence_info = (
        OptimizationEngine.run_optimization(
            objective_func,
            parameters,
            algorithm,
            max_iterations,
            seed=seed,
        )
    )

    return {
        "success": convergence_info.get("success", True),
        "optimal_parameters": optimal_params,
        "optimal_value": optimal_value,
        "iterations": iterations,
        "convergence_info": convergence_info,
    }


def calibrate_model(
    elements: List[Element],
    links: List[Link],
    config: SimulationConfig,
    parameters: List[OptimizationParameter],
    observed_data: List[ObservedDataPoint],
    elements_to_fit: List[str],
    error_metric: str = "rmse",
    algorithm: str = "differential_evolution",
    max_iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function for model calibration

    Args:
        elements: Model elements
        links: Model links
        config: Simulation configuration
        parameters: Parameters to calibrate
        observed_data: Observed data points
        elements_to_fit: Element IDs to fit
        error_metric: Error metric type
        algorithm: Optimization algorithm
        max_iterations: Maximum iterations
        seed: Random seed

    Returns:
        Dictionary with calibration results
    """
    calibration_func = CalibrationFunction(
        elements=elements,
        links=links,
        config=config,
        parameters=parameters,
        observed_data=observed_data,
        elements_to_fit=elements_to_fit,
        error_metric=error_metric,
        interpolation="linear",
    )

    calibrated_params, error_value, iterations, convergence_info = (
        OptimizationEngine.run_calibration(
            calibration_func,
            parameters,
            algorithm,
            max_iterations,
            seed=seed,
        )
    )

    return {
        "success": convergence_info.get("success", True),
        "calibrated_parameters": calibrated_params,
        "error_value": error_value,
        "iterations": iterations,
        "convergence_info": convergence_info,
    }