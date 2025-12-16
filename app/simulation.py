"""
Simulation engine for System Dynamics models
Implements Euler and RK4 integration methods
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
import logging
import random
import math

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


class EventManager:
    """
    Manages event scheduling and execution during simulation
    
    Handles three types of events:
    - Timeout events: Execute at fixed intervals
    - Rate events: Execute at exponential intervals (Poisson process)
    - Condition events: Execute when a condition becomes true
    """
    
    def __init__(
        self,
        events: List[Element],
        elements: Dict[str, Element],
        start_time: float,
        end_time: float,
        verbose: bool = False,
    ):
        """
        Initialize event manager
        
        Args:
            events: List of event elements
            elements: Dictionary of all elements (for element access in actions)
            start_time: Simulation start time
            end_time: Simulation end time
            verbose: Enable detailed logging
        """
        self.events = events
        self.elements = elements
        self.start_time = start_time
        self.end_time = end_time
        self.verbose = verbose
        
        # Event scheduling state
        self.timeout_events: Dict[str, float] = {}  # event_id -> next_trigger_time
        self.rate_events: Dict[str, float] = {}  # event_id -> next_trigger_time
        self.condition_events: Dict[str, bool] = {}  # event_id -> last_condition_state
        
        # Element access objects (will be created during simulation)
        self.element_objects: Dict[str, Any] = {}
        
        # Initialize event schedules
        self._initialize_events()
    
    def _initialize_events(self) -> None:
        """Initialize event schedules based on trigger types"""
        for event in self.events:
            event_id = event.id
            element_id = event.get_element_id()
            
            if event.trigger_type == "timeout":
                # Schedule first occurrence at start_time + trigger
                trigger_value = float(event.trigger)
                self.timeout_events[event_id] = self.start_time + trigger_value
                
            elif event.trigger_type == "rate":
                # Schedule first occurrence using exponential distribution
                rate = float(event.trigger)
                if rate > 0:
                    # Exponential: next = current_time + (-log(random()) / rate)
                    interval = -math.log(random.random()) / rate
                    self.rate_events[event_id] = self.start_time + interval
                else:
                    # Invalid rate, disable event
                    logger.warning(f"Event '{event.name}' has invalid rate: {rate}")
                    
            elif event.trigger_type == "condition":
                # Initialize condition state (will be checked during simulation)
                self.condition_events[event_id] = False
    
    def _create_element_access_objects(
        self,
        stock_values: Dict[str, float],
        state: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Create objects that allow event actions to access element values
        
        Args:
            stock_values: Current stock values
            state: Current state (all computed values)
            
        Returns:
            Dictionary mapping elementId to access objects
        """
        element_objects = {}
        
        # Create access objects for all elements by elementId
        for elem_id, element in self.elements.items():
            element_id = element.get_element_id()  # This is the identifier used in event actions
            
            # Create a simple object that allows reading/writing values
            class ElementAccessor:
                def __init__(self, elem: Element, elem_id_internal: str, stock_vals: Dict[str, float], state_vals: Dict[str, float]):
                    self._element = elem
                    self._stock_values = stock_vals
                    self._state = state_vals
                    self._element_id_internal = elem_id_internal  # Use element's actual id for lookups
                
                @property
                def value(self) -> float:
                    """Get current element value"""
                    if self._element.type == "stock":
                        return self._stock_values.get(self._element_id_internal, 0.0)
                    else:
                        return self._state.get(self._element_id_internal, 0.0)
                
                @value.setter
                def value(self, val: float) -> None:
                    """Set element value (for stocks, parameters, variables)"""
                    if self._element.type == "stock":
                        self._stock_values[self._element_id_internal] = float(val)
                    else:
                        self._state[self._element_id_internal] = float(val)
                
                @property
                def equation(self) -> Optional[str]:
                    """Get element equation (if available)"""
                    return self._element.equation if self._element.equation else None
            
            element_objects[element_id] = ElementAccessor(element, elem_id, stock_values, state)
        
        return element_objects
    
    def _execute_action(
        self,
        event: Element,
        stock_values: Dict[str, float],
        state: Dict[str, float],
        current_time: float,
    ) -> None:
        """
        Execute event action code
        
        Args:
            event: Event element to execute
            stock_values: Current stock values (can be modified)
            state: Current state values (can be modified)
            current_time: Current simulation time
        """
        if not event.action or not event.action.strip():
            return
        
        # Create element access objects
        element_objects = self._create_element_access_objects(stock_values, state)
        
        # Build execution namespace
        namespace = {
            "time": current_time,
            "t": current_time,
            "__builtins__": {
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "int": int,
                "float": float,
                "bool": bool,
                "str": str,
                "len": len,
                "range": range,
                "sum": sum,
                "math": __import__("math"),
                "np": __import__("numpy"),
            },
        }
        
        # Add all elements by elementId
        for element_id, accessor in element_objects.items():
            namespace[element_id] = accessor
        
        # Also add elements by their name for convenience (if name differs from elementId)
        for elem_id, element in self.elements.items():
            element_name = element.name
            element_id_for_access = element.get_element_id()
            if element_name != element_id_for_access and element_id_for_access in element_objects:
                namespace[element_name] = element_objects[element_id_for_access]
        
        # Execute action code
        try:
            # Compile and execute the action code
            code = compile(event.action, f"<event_{event.id}>", "exec")
            exec(code, namespace)
            
            if self.verbose:
                logger.debug(f"Executed event '{event.name}' at t={current_time:.4f}")
                
        except Exception as e:
            raise SimulationError(
                code="event_action_error",
                message=f"Error executing action for event '{event.name}': {str(e)}",
                details={
                    "event_id": event.id,
                    "time": current_time,
                    "action": event.action,
                    "error_type": type(e).__name__,
                },
            ) from e
    
    def _check_condition(self, event: Element, state: Dict[str, float], stock_values: Dict[str, float], current_time: float) -> bool:
        """
        Check if condition event should trigger
        
        Args:
            event: Event element with condition trigger
            state: Current state values
            stock_values: Current stock values
            current_time: Current simulation time
            
        Returns:
            True if condition is satisfied
        """
        if event.trigger_type != "condition" or not isinstance(event.trigger, str):
            return False
        
        # Create element access objects for condition evaluation
        element_objects = self._create_element_access_objects(stock_values, state)
        
        # Build evaluation namespace
        namespace = {
            "time": current_time,
            "t": current_time,
            "__builtins__": {
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "int": int,
                "float": float,
                "bool": bool,
            },
        }
        
        # Add all elements by elementId
        for element_id, accessor in element_objects.items():
            namespace[element_id] = accessor
        
        # Also add elements by their name for convenience (if name differs from elementId)
        for elem_id, element in self.elements.items():
            element_name = element.name
            element_id_for_access = element.get_element_id()
            if element_name != element_id_for_access and element_id_for_access in element_objects:
                namespace[element_name] = element_objects[element_id_for_access]
        
        try:
            # Evaluate condition
            condition_code = compile(event.trigger, f"<condition_{event.id}>", "eval")
            result = eval(condition_code, namespace)
            return bool(result)
        except Exception as e:
            logger.warning(f"Error evaluating condition for event '{event.name}': {str(e)}")
            return False
    
    def check_and_execute_events(
        self,
        current_time: float,
        stock_values: Dict[str, float],
        state: Dict[str, float],
    ) -> None:
        """
        Check all events and execute those that should fire at current_time
        
        Args:
            current_time: Current simulation time
            stock_values: Current stock values (can be modified by events)
            state: Current state values (can be modified by events)
        """
        events_to_execute = []
        
        # Check timeout events
        for event_id, next_time in list(self.timeout_events.items()):
            if current_time >= next_time:
                event = next(e for e in self.events if e.id == event_id)
                events_to_execute.append(("timeout", event))
                # Reschedule next occurrence
                trigger_value = float(event.trigger)
                self.timeout_events[event_id] = current_time + trigger_value
        
        # Check rate events
        for event_id, next_time in list(self.rate_events.items()):
            if current_time >= next_time:
                event = next(e for e in self.events if e.id == event_id)
                events_to_execute.append(("rate", event))
                # Reschedule next occurrence using exponential distribution
                rate = float(event.trigger)
                if rate > 0:
                    interval = -math.log(random.random()) / rate
                    self.rate_events[event_id] = current_time + interval
        
        # Check condition events
        for event_id in list(self.condition_events.keys()):
            event = next(e for e in self.events if e.id == event_id)
            current_condition = self._check_condition(event, state, stock_values, current_time)
            last_condition = self.condition_events[event_id]
            
            # Trigger when condition transitions from False to True
            if current_condition and not last_condition:
                events_to_execute.append(("condition", event))
            
            # Update condition state
            self.condition_events[event_id] = current_condition
        
        # Execute events in order: timeout, rate, condition
        for event_type, event in events_to_execute:
            if current_time <= self.end_time:
                self._execute_action(event, stock_values, state, current_time)


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
        self.events = [e for e in elements if e.type == "event"]
        
        # Event manager (will be initialized when simulation starts)
        self.event_manager: Optional[EventManager] = None

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

        logger.info(f"Events: {len(self.events)}")
        for event in self.events:
            logger.info(f"  - {event.name} (ID: {event.id})")
            logger.info(f"    Type: {event.trigger_type}, Trigger: {event.trigger}")

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
            # Include full error objects (not just messages) so frontend can display them
            error_objects = [
                {
                    "code": e.code,
                    "message": e.message,
                    "element_id": e.element_id,
                    "field": e.field,
                    "suggestion": e.suggestion,
                    "context": e.context,
                }
                for e in validation_result.errors
            ]
            error_messages = [e.message for e in validation_result.errors]
            
            # Log all validation errors for debugging
            if self.verbose or logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Validation failed with {len(validation_result.errors)} error(s):")
                for i, error in enumerate(validation_result.errors, 1):
                    logger.warning(f"  {i}. [{error.code}] {error.message}" + 
                                 (f" (Element: {error.element_id})" if error.element_id else ""))
            
            raise SimulationError(
                code="validation_failed",
                message=f"Model validation failed: {len(validation_result.errors)} error(s) found. Please fix errors before running.",
                details={
                    "errors": error_objects,  # Full structured error objects
                    "error_messages": error_messages,  # Also include messages for backward compatibility
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

        # Initialize event manager
        if self.events:
            self.event_manager = EventManager(
                self.events,
                self.elements,
                config.start_time,
                config.end_time,
                self.verbose,
            )

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

            # Check and execute events (before storing results and computing derivatives)
            if self.event_manager:
                try:
                    self.event_manager.check_and_execute_events(time, stock_values, state)
                except SimulationError:
                    raise
                except Exception as e:
                    raise SimulationError(
                        code="event_execution_error",
                        message=f"Error executing events at t={time:.4f}: {str(e)}",
                        details={"time": time},
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

        # Initialize event manager
        if self.events:
            self.event_manager = EventManager(
                self.events,
                self.elements,
                config.start_time,
                config.end_time,
                self.verbose,
            )

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

            # Check and execute events (before storing results and computing derivatives)
            if self.event_manager:
                try:
                    self.event_manager.check_and_execute_events(time, stock_values, state)
                except SimulationError:
                    raise
                except Exception as e:
                    raise SimulationError(
                        code="event_execution_error",
                        message=f"Error executing events at t={time:.4f}: {str(e)}",
                        details={"time": time},
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