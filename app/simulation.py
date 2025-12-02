"""
Simulation engine for System Dynamics models
Implements Euler and RK4 integration methods
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
import logging

from app.models import Element, Link, SimulationConfig
from app.evaluator import (
    SafeEquationEvaluator,
    build_dependency_graph,
    topological_sort,
)
from app.exceptions import SimulationError, EvaluationError, CircularDependencyError
from app.validation import validate_model
from app.constants import BUILT_IN_VARIABLES

logger = logging.getLogger(__name__)


class DelayStateManager:
    """
    Manages state for DELAY1, DELAY3, and SMOOTH functions

    Tracks historical values needed for delay and smoothing calculations.
    This class maintains the internal state required for time-delay functions
    in system dynamics models.
    """

    def __init__(self, time_step: float):
        """
        Initialize state manager

        Args:
            time_step: Simulation time step (dt)
        """
        self.time_step = time_step
        self.delay_states: Dict[str, Any] = {}
        self.smooth_states: Dict[str, float] = {}

    def reset(self) -> None:
        """Reset all delay and smooth states"""
        self.delay_states.clear()
        self.smooth_states.clear()

    def get_delay1_key(self, call_signature: str, delay: float) -> str:
        """Generate unique key for DELAY1 function instance"""
        return f"DELAY1_{call_signature}_{delay}"

    def get_delay3_key(self, call_signature: str, delay: float) -> str:
        """Generate unique key for DELAY3 function instance"""
        return f"DELAY3_{call_signature}_{delay}"

    def get_smooth_key(self, call_signature: str, smooth_time: float) -> str:
        """Generate unique key for SMOOTH function instance"""
        return f"SMOOTH_{call_signature}_{smooth_time}"

    def delay1(self, input_value: float, delay: float, key: str) -> float:
        """
        First-order exponential delay

        Implements: d(output)/dt = (input - output) / delay
        Using Euler integration: output += (input - output) * dt / delay

        Args:
            input_value: Current input value
            delay: Delay time constant (must be positive)
            key: Unique key for this delay instance

        Returns:
            Delayed output value
        """
        if delay <= 0:
            return input_value

        if key not in self.delay_states:
            self.delay_states[key] = input_value

        output = self.delay_states[key]
        output += (input_value - output) * self.time_step / delay
        self.delay_states[key] = output

        return output

    def delay3(self, input_value: float, delay: float, key: str) -> float:
        """
        Third-order delay (three cascaded first-order delays)

        Provides a more realistic delay with an S-shaped response curve.

        Args:
            input_value: Current input value
            delay: Total delay time (split into 3 equal stages)
            key: Unique key for this delay instance

        Returns:
            Delayed output value
        """
        if delay <= 0:
            return input_value

        stage_delay = delay / 3.0

        if key not in self.delay_states:
            self.delay_states[key] = [input_value, input_value, input_value]

        stages = self.delay_states[key]

        # Cascade three first-order delays
        stages[0] += (input_value - stages[0]) * self.time_step / stage_delay
        stages[1] += (stages[0] - stages[1]) * self.time_step / stage_delay
        stages[2] += (stages[1] - stages[2]) * self.time_step / stage_delay

        self.delay_states[key] = stages
        return stages[2]

    def smooth(self, input_value: float, smooth_time: float, key: str) -> float:
        """
        Exponential smoothing (first-order exponential filter)

        Implements: d(output)/dt = (input - output) / smooth_time

        Args:
            input_value: Current input value
            smooth_time: Smoothing time constant (must be positive)
            key: Unique key for this smooth instance

        Returns:
            Smoothed output value
        """
        if smooth_time <= 0:
            return input_value

        if key not in self.smooth_states:
            self.smooth_states[key] = input_value

        output = self.smooth_states[key]
        output += (input_value - output) * self.time_step / smooth_time
        self.smooth_states[key] = output

        return output


class SystemDynamicsModel:
    """
    System dynamics model executor

    Handles initialization, state computation, and simulation execution
    using Euler or RK4 integration methods.

    The simulation follows the standard system dynamics paradigm:
    1. Stocks hold state (integrated values)
    2. Flows determine rates of change for stocks
    3. Variables are intermediate calculations
    4. Parameters are constants (or time-varying inputs)
    """

    def __init__(
        self,
        elements: List[Element],
        links: List[Link],
        verbose: bool = False,
    ):
        """
        Initialize model with elements and links

        Args:
            elements: List of model elements (stocks, flows, parameters, variables)
            links: List of connections between elements
            verbose: Enable detailed logging

        Raises:
            SimulationError: If model structure is invalid
        """
        self.elements = {e.id: e for e in elements}
        self.elements_list = elements
        self.links = links
        self.evaluator = SafeEquationEvaluator()
        self.verbose = verbose
        self.delay_state_manager: Optional[DelayStateManager] = None

        # Categorize elements by type
        self.stocks = {e.id: e for e in elements if e.type == "stock"}
        self.flows = {e.id: e for e in elements if e.type == "flow"}
        self.parameters = {e.id: e for e in elements if e.type == "parameter"}
        self.variables = {e.id: e for e in elements if e.type == "variable"}

        # Log model structure
        if self.verbose:
            self._log_model_structure()

        # Build dependency graph and evaluation order
        self._build_dependency_graph()

    def _log_model_structure(self) -> None:
        """Log model structure for debugging"""
        logger.info("=" * 60)
        logger.info("MODEL INITIALIZATION")
        logger.info("=" * 60)

        logger.info(f"Stocks: {len(self.stocks)}")
        for stock_id, stock in self.stocks.items():
            logger.info(f"  - {stock.name} (ID: {stock_id})")
            logger.info(f"    Initial: {stock.initial}")
            logger.info(f"    Equation: '{stock.equation}'")

        logger.info(f"Flows: {len(self.flows)}")
        for flow_id, flow in self.flows.items():
            logger.info(f"  - {flow.name} (ID: {flow_id})")
            logger.info(f"    Equation: '{flow.equation}'")

        logger.info(f"Parameters: {len(self.parameters)}")
        for param_id, param in self.parameters.items():
            value = param.value if param.value is not None else param.initial
            logger.info(f"  - {param.name} (ID: {param_id})")
            logger.info(f"    Value: {value}")

        logger.info(f"Variables: {len(self.variables)}")
        for var_id, var in self.variables.items():
            logger.info(f"  - {var.name} (ID: {var_id})")
            logger.info(f"    Equation: '{var.equation}'")

        logger.info(f"Links: {len(self.links)}")
        for link in self.links:
            source_name = self.elements.get(link.source, link).name if link.source in self.elements else link.source
            target_name = self.elements.get(link.target, link).name if link.target in self.elements else link.target
            logger.info(f"  - {source_name} → {target_name}")

        logger.info("=" * 60)

    def _build_dependency_graph(self) -> None:
        """
        Build dependency graph and compute evaluation order

        Uses topological sort to determine the order in which
        variables and flows should be evaluated.

        Raises:
            SimulationError: If circular dependency is detected among non-stock elements
        """
        try:
            self.dependencies = build_dependency_graph(
                self.elements_list, self.links, use_cache=True
            )

            self.evaluation_order = topological_sort(
                self.dependencies,
                elements=self.elements_list,
                links=self.links,
            )

            if self.verbose:
                logger.info(f"Evaluation order: {self.evaluation_order}")

        except CircularDependencyError as e:
            # Re-raise with more context
            raise SimulationError(
                code="circular_dependency",
                message=f"Circular dependency detected: {e.message}",
                details={"cycle": e.cycle},
            ) from e
        except ValueError as e:
            # Handle legacy ValueError from topological_sort
            error_msg = str(e)
            raise SimulationError(
                code="circular_dependency",
                message=f"Circular dependency detected: {error_msg}",
                details={"suggestion": "Break the cycle by modifying equations"},
            ) from e

    def _setup_evaluator(self, time_step: float) -> None:
        """
        Setup evaluator with delay state manager and lookup tables

        Args:
            time_step: Simulation time step
        """
        # Initialize or reset delay state manager
        self.delay_state_manager = DelayStateManager(time_step)
        self.evaluator.set_delay_state_manager(self.delay_state_manager)

        # Collect lookup tables from elements (by both ID and name)
        lookup_tables: Dict[str, Any] = {}
        for elem_id, elem in self.elements.items():
            if elem.lookup_table:
                lookup_tables[elem_id] = elem.lookup_table
                lookup_tables[elem.name] = elem.lookup_table

        self.evaluator.set_lookup_tables(lookup_tables)
        self.evaluator.clear_cache()

    def _compute_state_variables(
        self, stock_values: Dict[str, float], t: float
    ) -> Dict[str, float]:
        """
        Compute all state variables given current stock values and time

        Evaluation order:
        1. Time variables (t, time)
        2. Stock values (known from integration)
        3. Parameters (constants or time-varying)
        4. Variables and flows in topological order

        Args:
            stock_values: Current stock values
            t: Current simulation time

        Returns:
            Dictionary of all computed state values

        Raises:
            EvaluationError: If any equation evaluation fails
        """
        state: Dict[str, float] = {}

        # 1. Add time variables
        state["t"] = t
        state["time"] = t

        # 2. Add stock values (by both ID and name)
        for stock_id, value in stock_values.items():
            state[stock_id] = value
            if stock_id in self.elements:
                state[self.elements[stock_id].name] = value

        # 3. Add parameters
        for param_id, param in self.parameters.items():
            if param.equation and param.equation.strip():
                # Time-varying parameter - evaluate equation
                self.evaluator.set_variables(state)
                try:
                    value = self.evaluator.evaluate(param.equation, param_id)
                except EvaluationError as e:
                    raise EvaluationError(
                        code=e.code,
                        message=f"Error computing parameter '{param.name}': {e.message}",
                        element_id=param_id,
                        equation=param.equation,
                    ) from e
            else:
                # Constant parameter
                value = param.value if param.value is not None else (param.initial or 0.0)

            state[param_id] = value
            state[param.name] = value

        self.evaluator.set_variables(state)

        # 4. Compute variables and flows in topological order
        for elem_id in self.evaluation_order:
            element = self.elements.get(elem_id)
            if element is None:
                continue

            # Skip stocks (already have their values) and parameters (already computed)
            if element.type in ("stock", "parameter"):
                continue

            # Compute variables and flows
            if element.type in ("variable", "flow"):
                if element.equation and element.equation.strip():
                    try:
                        value = self.evaluator.evaluate(element.equation, elem_id)
                        state[elem_id] = value
                        state[element.name] = value
                        self.evaluator.set_variables(state)
                    except EvaluationError as e:
                        raise EvaluationError(
                            code=e.code,
                            message=f"Error computing {element.type} '{element.name}': {e.message}",
                            element_id=elem_id,
                            equation=element.equation,
                        ) from e

        return state

    def compute_derivatives(
        self, stock_values: Dict[str, float], t: float
    ) -> Dict[str, float]:
        """
        Compute derivatives (rates of change) for all stocks

        Each stock's equation defines its rate of change (typically
        the sum of inflows minus outflows).

        Args:
            stock_values: Current stock values
            t: Current simulation time

        Returns:
            Dictionary mapping stock IDs to their derivatives

        Raises:
            EvaluationError: If derivative computation fails
        """
        state = self._compute_state_variables(stock_values, t)
        self.evaluator.set_variables(state)

        derivatives: Dict[str, float] = {}

        for stock_id, stock in self.stocks.items():
            if not stock.equation or not stock.equation.strip():
                # Stock with no equation has zero rate of change
                derivatives[stock_id] = 0.0
                continue

            try:
                deriv = self.evaluator.evaluate(stock.equation, stock_id)
                derivatives[stock_id] = deriv

                if self.verbose and t == 0:
                    logger.debug(f"  Stock '{stock.name}' at t=0:")
                    logger.debug(f"    Value: {stock_values[stock_id]:.4f}")
                    logger.debug(f"    Derivative: {deriv:.4f}")

            except EvaluationError as e:
                raise EvaluationError(
                    code=e.code,
                    message=f"Error computing derivative for stock '{stock.name}': {e.message}",
                    element_id=stock_id,
                    equation=stock.equation,
                ) from e

        return derivatives

    def _validate_before_simulation(self, config: SimulationConfig) -> None:
        """
        Run validation before simulation

        Args:
            config: Simulation configuration

        Raises:
            SimulationError: If validation fails
        """
        validation_result = validate_model(self.elements_list, self.links, config)

        if not validation_result.valid:
            error_messages = [e.message for e in validation_result.errors]
            raise SimulationError(
                code="validation_failed",
                message="Model validation failed",
                details={
                    "errors": error_messages,
                    "error_count": len(validation_result.errors),
                },
            )

        # Log warnings if any
        if self.verbose and validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Validation warning: {warning.message}")

    def _initialize_results(self, n_steps: int) -> Dict[str, np.ndarray]:
        """Initialize result arrays for all elements"""
        results: Dict[str, np.ndarray] = {}

        for stock_id in self.stocks:
            results[stock_id] = np.zeros(n_steps)

        for flow_id in self.flows:
            results[flow_id] = np.zeros(n_steps)

        for var_id in self.variables:
            results[var_id] = np.zeros(n_steps)

        for param_id in self.parameters:
            results[param_id] = np.zeros(n_steps)

        return results

    def _store_results(
        self,
        results: Dict[str, np.ndarray],
        state: Dict[str, float],
        stock_values: Dict[str, float],
        index: int,
    ) -> None:
        """Store current state in results arrays"""
        for stock_id in self.stocks:
            results[stock_id][index] = stock_values[stock_id]

        for flow_id in self.flows:
            results[flow_id][index] = state.get(flow_id, 0.0)

        for var_id in self.variables:
            results[var_id][index] = state.get(var_id, 0.0)

        for param_id in self.parameters:
            results[param_id][index] = state.get(param_id, 0.0)

    def simulate_euler(
        self,
        config: SimulationConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
        skip_validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Run simulation using Euler method

        The Euler method is a first-order numerical integration method:
        x(t + dt) = x(t) + dx/dt * dt

        Mathematical Formula:
        For a differential equation dx/dt = f(t, x), the Euler method computes:
        x_{n+1} = x_n + h * f(t_n, x_n)
        
        where:
        - x_n is the value at time t_n
        - h is the time step (dt)
        - f(t_n, x_n) is the derivative at time t_n

        Args:
            config: Simulation configuration
            progress_callback: Optional callback(progress: 0.0-1.0) for progress updates
            skip_validation: Skip model validation (use with caution)

        Returns:
            Dictionary with 'time' and 'results' keys

        Raises:
            SimulationError: If simulation fails
        """
        # Validate model
        if not skip_validation:
            self._validate_before_simulation(config)

        # Setup evaluator
        self._setup_evaluator(config.time_step)

        # Generate time points
        t = np.arange(
            config.start_time,
            config.end_time + config.time_step * 0.5,  # Avoid floating point issues
            config.time_step,
        )
        n_steps = len(t)

        if self.verbose:
            logger.info("=" * 60)
            logger.info("SIMULATION START (Euler)")
            logger.info(f"Time: {config.start_time} to {config.end_time}, dt={config.time_step}")
            logger.info(f"Steps: {n_steps}")
            logger.info("=" * 60)

        # Initialize results
        results = self._initialize_results(n_steps)

        # Initialize stock values
        stock_values = {
            stock_id: float(stock.initial or 0.0)
            for stock_id, stock in self.stocks.items()
        }

        if self.verbose:
            logger.info("Initial conditions:")
            for stock_id, stock in self.stocks.items():
                logger.info(f"  {stock.name}: {stock_values[stock_id]:.4f}")

        # Progress reporting frequency (use configurable interval)
        from app.config import get_settings
        settings = get_settings()
        report_interval = max(1, n_steps // settings.progress_report_interval)

        # Simulation loop
        for i, time in enumerate(t):
            # Compute current state
            try:
                state = self._compute_state_variables(stock_values, time)
            except EvaluationError as e:
                raise SimulationError(
                    code="evaluation_error",
                    message=f"Error at t={time:.4f}: {e.message}",
                    details={"time": time, "element_id": e.element_id},
                ) from e

            # Store results
            self._store_results(results, state, stock_values, i)

            # Report progress
            if progress_callback and (i == 0 or i == n_steps - 1 or (i + 1) % report_interval == 0):
                progress_callback((i + 1) / n_steps)

            # Compute derivatives and update stocks (except for last step)
            if i < n_steps - 1:
                try:
                    derivatives = self.compute_derivatives(stock_values, time)
                except EvaluationError as e:
                    raise SimulationError(
                        code="derivative_error",
                        message=f"Error computing derivatives at t={time:.4f}: {e.message}",
                        details={"time": time, "element_id": e.element_id},
                    ) from e

                # Euler update: x(t+dt) = x(t) + dx/dt * dt
                for stock_id in self.stocks:
                    stock_values[stock_id] += derivatives[stock_id] * config.time_step

        if self.verbose:
            logger.info("=" * 60)
            logger.info("SIMULATION COMPLETE")
            logger.info("Final values:")
            for stock_id, stock in self.stocks.items():
                logger.info(f"  {stock.name}: {stock_values[stock_id]:.4f}")
            logger.info("=" * 60)

        return {
            "time": t.tolist(),
            "results": {elem_id: vals.tolist() for elem_id, vals in results.items()},
        }

    def simulate_rk4(
        self,
        config: SimulationConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
        skip_validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Run simulation using RK4 (Runge-Kutta 4th order) method

        RK4 provides higher accuracy than Euler by evaluating derivatives
        at multiple points within each time step. It is a 4th-order method,
        meaning the local truncation error is O(h^5) and the global error is O(h^4).

        Mathematical Formula:
        For a differential equation dy/dt = f(t, y), the RK4 method computes:
        
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h*k1/2)
        k3 = f(t_n + h/2, y_n + h*k2/2)
        k4 = f(t_n + h, y_n + h*k3)
        y_{n+1} = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        where:
        - y_n is the value at time t_n
        - h is the time step (dt)
        - f(t, y) is the derivative function
        - k1, k2, k3, k4 are intermediate derivative estimates

        Args:
            config: Simulation configuration
            progress_callback: Optional callback(progress: 0.0-1.0) for progress updates
            skip_validation: Skip model validation (use with caution)

        Returns:
            Dictionary with 'time' and 'results' keys

        Raises:
            SimulationError: If simulation fails
        """
        # Validate model
        if not skip_validation:
            self._validate_before_simulation(config)

        # Setup evaluator
        self._setup_evaluator(config.time_step)

        # Generate time points
        t = np.arange(
            config.start_time,
            config.end_time + config.time_step * 0.5,
            config.time_step,
        )
        n_steps = len(t)

        if self.verbose:
            logger.info("=" * 60)
            logger.info("SIMULATION START (RK4)")
            logger.info(f"Time: {config.start_time} to {config.end_time}, dt={config.time_step}")
            logger.info(f"Steps: {n_steps}")
            logger.info("=" * 60)

        # Initialize results
        results = self._initialize_results(n_steps)

        # Initialize stock values
        stock_values = {
            stock_id: float(stock.initial or 0.0)
            for stock_id, stock in self.stocks.items()
        }

        if self.verbose:
            logger.info("Initial conditions:")
            for stock_id, stock in self.stocks.items():
                logger.info(f"  {stock.name}: {stock_values[stock_id]:.4f}")

        # Progress reporting frequency (use configurable interval)
        from app.config import get_settings
        settings = get_settings()
        report_interval = max(1, n_steps // settings.progress_report_interval)
        dt = config.time_step

        # Simulation loop
        for i, time in enumerate(t):
            # Compute current state
            try:
                state = self._compute_state_variables(stock_values, time)
            except EvaluationError as e:
                raise SimulationError(
                    code="evaluation_error",
                    message=f"Error at t={time:.4f}: {e.message}",
                    details={"time": time, "element_id": e.element_id},
                ) from e

            # Store results
            self._store_results(results, state, stock_values, i)

            # Report progress
            if progress_callback and (i == 0 or i == n_steps - 1 or (i + 1) % report_interval == 0):
                progress_callback((i + 1) / n_steps)

            # RK4 integration (except for last step)
            if i < n_steps - 1:
                try:
                    # k1: derivative at current point
                    k1 = self.compute_derivatives(stock_values, time)

                    # k2: derivative at midpoint using k1
                    stock_k2 = {
                        sid: stock_values[sid] + 0.5 * dt * k1[sid]
                        for sid in self.stocks
                    }
                    k2 = self.compute_derivatives(stock_k2, time + 0.5 * dt)

                    # k3: derivative at midpoint using k2
                    stock_k3 = {
                        sid: stock_values[sid] + 0.5 * dt * k2[sid]
                        for sid in self.stocks
                    }
                    k3 = self.compute_derivatives(stock_k3, time + 0.5 * dt)

                    # k4: derivative at endpoint using k3
                    stock_k4 = {
                        sid: stock_values[sid] + dt * k3[sid]
                        for sid in self.stocks
                    }
                    k4 = self.compute_derivatives(stock_k4, time + dt)

                    # Weighted average update
                    for stock_id in self.stocks:
                        stock_values[stock_id] += (dt / 6.0) * (
                            k1[stock_id]
                            + 2.0 * k2[stock_id]
                            + 2.0 * k3[stock_id]
                            + k4[stock_id]
                        )

                except EvaluationError as e:
                    raise SimulationError(
                        code="derivative_error",
                        message=f"RK4 error at t={time:.4f}: {e.message}",
                        details={"time": time, "element_id": e.element_id},
                    ) from e

        if self.verbose:
            logger.info("=" * 60)
            logger.info("SIMULATION COMPLETE (RK4)")
            logger.info("Final values:")
            for stock_id, stock in self.stocks.items():
                logger.info(f"  {stock.name}: {stock_values[stock_id]:.4f}")
            logger.info("=" * 60)

        return {
            "time": t.tolist(),
            "results": {elem_id: vals.tolist() for elem_id, vals in results.items()},
        }

    def simulate(
        self,
        config: SimulationConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run simulation using the method specified in config

        This is the main entry point for running simulations.
        Automatically selects Euler or RK4 based on config.method.

        Args:
            config: Simulation configuration
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with 'time' and 'results' keys

        Raises:
            SimulationError: If simulation fails
        """
        if config.method == "rk4":
            return self.simulate_rk4(config, progress_callback)
        else:
            return self.simulate_euler(config, progress_callback)


def run_simulation(
    elements: List[Element],
    links: List[Link],
    config: SimulationConfig,
    verbose: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run a simulation

    Creates a SystemDynamicsModel and runs the simulation in one call.

    Args:
        elements: List of model elements
        links: List of links between elements
        config: Simulation configuration
        verbose: Enable detailed logging
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with 'time' and 'results' keys

    Raises:
        SimulationError: If simulation fails
    """
    model = SystemDynamicsModel(elements, links, verbose=verbose)
    return model.simulate(config, progress_callback)