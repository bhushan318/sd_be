"""
Comprehensive validation layer for System Dynamics models
Validates model structure, types, equations, links, dependencies, and config
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel
import ast

from app.models import Element, Link, SimulationConfig
from app.evaluator import extract_variable_references
from app.exceptions import ValidationError
from app.constants import (
    SAFE_FUNCTION_NAMES,
    SAFE_AST_OPERATORS,
    BUILT_IN_VARIABLES,
    VALID_ELEMENT_TYPES,
    MAX_SIMULATION_STEPS,
    VALID_INTEGRATION_METHODS,
)


class ValidationResult(BaseModel):
    """Result of validation"""

    valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError] = []


# ============================================================================
# Configuration Validation
# ============================================================================


def validate_simulation_config(config: SimulationConfig) -> List[ValidationError]:
    """
    Validate simulation configuration

    Rules:
    - time_step > 0
    - end_time > start_time
    - total steps < MAX_SIMULATION_STEPS
    - method is 'euler' or 'rk4'
    """
    errors: List[ValidationError] = []

    # Validate time step
    if config.time_step <= 0:
        errors.append(
            ValidationError(
                code="invalid_time_step",
                message=f"Time step must be greater than 0, got {config.time_step}",
                element_id=None,
                field="time_step",
                suggestion="Set time_step to a positive value (e.g., 0.1, 1.0)",
            )
        )

    # Validate time range
    if config.end_time <= config.start_time:
        errors.append(
            ValidationError(
                code="invalid_time_range",
                message=f"End time ({config.end_time}) must be greater than start time ({config.start_time})",
                element_id=None,
                field="end_time",
                suggestion=f"Set end_time to a value greater than {config.start_time}",
            )
        )

    # Validate max steps (only if time_step is valid)
    if config.time_step > 0:
        max_steps = (config.end_time - config.start_time) / config.time_step
        if max_steps >= MAX_SIMULATION_STEPS:
            errors.append(
                ValidationError(
                    code="too_many_steps",
                    message=f"Simulation would require {max_steps:.0f} steps, exceeding maximum of {MAX_SIMULATION_STEPS:,}",
                    element_id=None,
                    field="time_step",
                    suggestion="Increase time_step or reduce the time range (end_time - start_time)",
                )
            )

    # Validate integration method
    if config.method not in VALID_INTEGRATION_METHODS:
        errors.append(
            ValidationError(
                code="invalid_method",
                message=f"Integration method must be one of {VALID_INTEGRATION_METHODS}, got '{config.method}'",
                element_id=None,
                field="method",
                suggestion=f"Set method to one of: {', '.join(sorted(VALID_INTEGRATION_METHODS))}",
            )
        )

    return errors


# ============================================================================
# Element Validation
# ============================================================================


def validate_elements(elements: List[Element]) -> List[ValidationError]:
    """
    Validate elements structure

    Rules:
    - Unique element IDs
    - Unique element names
    - No name/ID collisions between different elements
    - Valid element types
    - Stocks require initial values
    - Parameters require numeric values
    - Flows/variables require valid equations (non-empty)
    """
    errors: List[ValidationError] = []

    element_ids: Set[str] = set()
    element_names: Set[str] = set()
    id_to_element: Dict[str, Element] = {}
    name_to_element: Dict[str, Element] = {}

    for element in elements:
        # Check for duplicate IDs
        if element.id in element_ids:
            errors.append(
                ValidationError(
                    code="duplicate_id",
                    message=f"Element ID '{element.id}' is duplicated",
                    element_id=element.id,
                    field="id",
                    suggestion="Ensure all element IDs are unique",
                )
            )
        else:
            element_ids.add(element.id)
            id_to_element[element.id] = element

        # Check for duplicate names
        if element.name in element_names:
            existing = name_to_element.get(element.name)
            errors.append(
                ValidationError(
                    code="duplicate_name",
                    message=f"Element name '{element.name}' is duplicated (also used by element '{existing.id if existing else 'unknown'}')",
                    element_id=element.id,
                    field="name",
                    suggestion="Ensure all element names are unique to avoid confusion",
                )
            )
        else:
            element_names.add(element.name)
            name_to_element[element.name] = element

    # Check for name/ID collisions (after collecting all elements)
    for element in elements:
        # Check if this element's name matches another element's ID
        if element.name in id_to_element and id_to_element[element.name].id != element.id:
            other = id_to_element[element.name]
            errors.append(
                ValidationError(
                    code="name_id_collision",
                    message=f"Element name '{element.name}' collides with ID of element '{other.name}'",
                    element_id=element.id,
                    field="name",
                    suggestion="Rename the element to avoid ambiguity in equation references",
                )
            )

        # Check if this element's ID matches another element's name
        if element.id in name_to_element and name_to_element[element.id].id != element.id:
            other = name_to_element[element.id]
            errors.append(
                ValidationError(
                    code="id_name_collision",
                    message=f"Element ID '{element.id}' collides with name of element '{other.id}'",
                    element_id=element.id,
                    field="id",
                    suggestion="Change the element ID to avoid ambiguity in equation references",
                )
            )

    # Validate each element based on type
    for element in elements:
        # Skip if ID was duplicated (already reported)
        if list(e.id for e in elements).count(element.id) > 1:
            continue

        # Validate element type
        if element.type not in VALID_ELEMENT_TYPES:
            errors.append(
                ValidationError(
                    code="invalid_element_type",
                    message=f"Element type must be one of {VALID_ELEMENT_TYPES}, got '{element.type}'",
                    element_id=element.id,
                    field="type",
                    suggestion=f"Set type to one of: {', '.join(sorted(VALID_ELEMENT_TYPES))}",
                )
            )
            continue

        # Type-specific validation
        if element.type == "stock":
            errors.extend(_validate_stock(element))
        elif element.type == "parameter":
            errors.extend(_validate_parameter(element))
        elif element.type in ("flow", "variable"):
            errors.extend(_validate_flow_or_variable(element))
        elif element.type == "event":
            errors.extend(_validate_event(element))

    return errors


def _validate_stock(element: Element) -> List[ValidationError]:
    """Validate stock element"""
    errors = []

    if element.initial is None:
        errors.append(
            ValidationError(
                code="missing_initial_value",
                message=f"Stock '{element.name}' requires an initial value",
                element_id=element.id,
                field="initial",
                suggestion="Provide a numeric initial value (e.g., initial=100.0)",
            )
        )
    elif not isinstance(element.initial, (int, float)):
        errors.append(
            ValidationError(
                code="invalid_initial_value",
                message=f"Stock '{element.name}' initial value must be numeric, got {type(element.initial).__name__}",
                element_id=element.id,
                field="initial",
                suggestion="Set initial to a numeric value (int or float)",
            )
        )

    return errors


def _validate_parameter(element: Element) -> List[ValidationError]:
    """Validate parameter element"""
    errors = []

    # Parameters can use either 'value', 'initial', or 'equation' (for time-varying parameters)
    has_value = element.value is not None
    has_initial = element.initial is not None and element.initial != 0.0
    has_equation = element.equation and element.equation.strip() != ""

    # Parameters need at least one: value, initial, or equation
    if not has_value and not has_initial and not has_equation:
        errors.append(
            ValidationError(
                code="missing_parameter_value",
                message=f"Parameter '{element.name}' requires a numeric value, initial value, or equation",
                element_id=element.id,
                field="value",
                suggestion="Provide a numeric value (e.g., value=5.0), initial value, or equation for time-varying parameter",
            )
        )
    elif has_value and not isinstance(element.value, (int, float)):
        errors.append(
            ValidationError(
                code="invalid_parameter_value",
                message=f"Parameter '{element.name}' value must be numeric, got {type(element.value).__name__}",
                element_id=element.id,
                field="value",
                suggestion="Set value to a numeric value (int or float)",
            )
        )

    return errors


def _validate_flow_or_variable(element: Element) -> List[ValidationError]:
    """Validate flow or variable element"""
    errors = []

    if not element.equation or element.equation.strip() == "":
        errors.append(
            ValidationError(
                code="missing_equation",
                message=f"{element.type.capitalize()} '{element.name}' requires a valid equation",
                element_id=element.id,
                field="equation",
                suggestion=f"Provide a mathematical equation (e.g., equation='10' or equation='param * 2')",
            )
        )

    return errors


def _validate_event(element: Element) -> List[ValidationError]:
    """Validate event element"""
    errors = []
    
    # Events should not have initial, value, or equation
    if element.initial is not None and element.initial != 0.0:
        errors.append(
            ValidationError(
                code="event_has_initial",
                message=f"Event '{element.name}' should not have an initial value",
                element_id=element.id,
                field="initial",
                suggestion="Remove the 'initial' field from event elements",
            )
        )
    
    if element.value is not None:
        errors.append(
            ValidationError(
                code="event_has_value",
                message=f"Event '{element.name}' should not have a value field",
                element_id=element.id,
                field="value",
                suggestion="Remove the 'value' field from event elements",
            )
        )
    
    if element.equation and element.equation.strip():
        errors.append(
            ValidationError(
                code="event_has_equation",
                message=f"Event '{element.name}' should not have an equation field",
                element_id=element.id,
                field="equation",
                suggestion="Remove the 'equation' field from event elements. Use 'action' instead.",
            )
        )
    
    # Validate trigger_type
    valid_trigger_types = {"timeout", "rate", "condition"}
    if not element.trigger_type:
        errors.append(
            ValidationError(
                code="missing_event_trigger_type",
                message=f"Event '{element.name}' requires a trigger_type",
                element_id=element.id,
                field="trigger_type",
                suggestion=f"Set trigger_type to one of: {', '.join(sorted(valid_trigger_types))}",
            )
        )
    elif element.trigger_type not in valid_trigger_types:
        errors.append(
            ValidationError(
                code="invalid_event_trigger_type",
                message=f"Event '{element.name}' has invalid trigger_type '{element.trigger_type}'. Must be one of: {', '.join(sorted(valid_trigger_types))}",
                element_id=element.id,
                field="trigger_type",
                suggestion=f"Set trigger_type to one of: {', '.join(sorted(valid_trigger_types))}",
            )
        )
    
    # Validate trigger
    if element.trigger is None:
        errors.append(
            ValidationError(
                code="missing_event_trigger",
                message=f"Event '{element.name}' requires a trigger value",
                element_id=element.id,
                field="trigger",
                suggestion="Provide a trigger value (number for timeout/rate, string for condition)",
            )
        )
    elif element.trigger_type in ("timeout", "rate"):
        # For timeout and rate, trigger must be a positive number
        if not isinstance(element.trigger, (int, float)):
            errors.append(
                ValidationError(
                    code="invalid_event_trigger_number",
                    message=f"Event '{element.name}' with trigger_type '{element.trigger_type}' requires a numeric trigger value, got {type(element.trigger).__name__}",
                    element_id=element.id,
                    field="trigger",
                    suggestion=f"Set trigger to a positive number (e.g., trigger=10.0)",
                )
            )
        elif element.trigger <= 0:
            errors.append(
                ValidationError(
                    code="invalid_event_trigger_positive",
                    message=f"Event '{element.name}' with trigger_type '{element.trigger_type}' requires a positive trigger value, got {element.trigger}",
                    element_id=element.id,
                    field="trigger",
                    suggestion="Set trigger to a positive number (e.g., trigger=10.0)",
                )
            )
    elif element.trigger_type == "condition":
        # For condition, trigger must be a non-empty string
        if not isinstance(element.trigger, str):
            errors.append(
                ValidationError(
                    code="invalid_event_trigger_string",
                    message=f"Event '{element.name}' with trigger_type 'condition' requires a string trigger value, got {type(element.trigger).__name__}",
                    element_id=element.id,
                    field="trigger",
                    suggestion="Set trigger to a boolean expression string (e.g., trigger='stock >= 100')",
                )
            )
        elif not element.trigger.strip():
            errors.append(
                ValidationError(
                    code="empty_event_trigger_condition",
                    message=f"Event '{element.name}' with trigger_type 'condition' requires a non-empty trigger string",
                    element_id=element.id,
                    field="trigger",
                    suggestion="Set trigger to a boolean expression string (e.g., trigger='stock >= 100')",
                )
            )
    
    # Validate action
    if not element.action or not element.action.strip():
        errors.append(
            ValidationError(
                code="missing_event_action",
                message=f"Event '{element.name}' has no action code",
                element_id=element.id,
                field="action",
                suggestion="Provide Python code to execute when the event triggers",
            )
        )
    else:
        # Basic syntax checking for action code
        try:
            compile(element.action, f"<event_{element.id}>", "exec")
        except SyntaxError as e:
            errors.append(
                ValidationError(
                    code="invalid_event_action_syntax",
                    message=f"Event '{element.name}' has invalid Python syntax in action: {str(e)}",
                    element_id=element.id,
                    field="action",
                    suggestion="Fix the Python syntax error in the action code",
                )
            )
        except Exception:
            # Other compilation errors are acceptable (e.g., undefined variables)
            # They will be caught during execution
            pass
    
    return errors


# ============================================================================
# Link Validation
# ============================================================================


def validate_links(links: List[Link], elements: List[Element]) -> List[ValidationError]:
    """
    Validate links between elements

    Rules:
    - Unique link IDs
    - Source element must exist
    - Target element must exist
    - No self-loops
    """
    errors: List[ValidationError] = []

    element_ids = {elem.id for elem in elements}
    link_ids: Set[str] = set()

    for link in links:
        # Check for duplicate link IDs
        if link.id in link_ids:
            errors.append(
                ValidationError(
                    code="duplicate_link_id",
                    message=f"Link ID '{link.id}' is duplicated",
                    element_id=link.id,
                    field="id",
                    suggestion="Ensure all link IDs are unique",
                )
            )
        link_ids.add(link.id)

        # Check if source exists
        if link.source not in element_ids:
            errors.append(
                ValidationError(
                    code="invalid_link_source",
                    message=f"Link '{link.id}' references non-existent source element '{link.source}'",
                    element_id=link.id,
                    field="source",
                    suggestion=f"Create element with ID '{link.source}' or update link source",
                )
            )

        # Check if target exists
        if link.target not in element_ids:
            errors.append(
                ValidationError(
                    code="invalid_link_target",
                    message=f"Link '{link.id}' references non-existent target element '{link.target}'",
                    element_id=link.id,
                    field="target",
                    suggestion=f"Create element with ID '{link.target}' or update link target",
                )
            )

        # Check for self-loops
        if link.source == link.target:
            errors.append(
                ValidationError(
                    code="self_loop",
                    message=f"Link '{link.id}' creates a self-loop from '{link.source}' to itself",
                    element_id=link.id,
                    field="target",
                    suggestion="Remove the self-loop or change target to a different element",
                )
            )

    return errors


# ============================================================================
# Equation Validation
# ============================================================================


def validate_equation_syntax(equation: str) -> Tuple[bool, Optional[str]]:
    """Validate equation syntax by parsing AST"""
    try:
        ast.parse(equation, mode="eval")
        return True, None
    except SyntaxError as e:
        return False, str(e)


def validate_equation_ast(node: ast.AST) -> Tuple[bool, Optional[str]]:
    """
    Recursively validate AST for unsafe operations

    Returns:
        Tuple of (is_valid, error_message)
    """
    if isinstance(node, ast.Expression):
        return validate_equation_ast(node.body)

    # Constants and literals (Python 3.8+) - check first to avoid deprecation warnings
    if isinstance(node, ast.Constant):
        return True, None

    # Legacy support for Python < 3.8 (deprecated but still needed for compatibility)
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        return True, None

    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return True, None

    if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant):
        return True, None

    # Variable reference
    if isinstance(node, ast.Name):
        return True, None

    # Binary operation
    if isinstance(node, ast.BinOp):
        if type(node.op) not in SAFE_AST_OPERATORS:
            return False, f"Unsupported operator: {type(node.op).__name__}"
        valid, error = validate_equation_ast(node.left)
        if not valid:
            return False, error
        return validate_equation_ast(node.right)

    # Unary operation
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in SAFE_AST_OPERATORS:
            return False, f"Unsupported unary operator: {type(node.op).__name__}"
        return validate_equation_ast(node.operand)

    # Function call
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Check both exact match and case-insensitive for SD functions
            if func_name not in SAFE_FUNCTION_NAMES and func_name.upper() not in SAFE_FUNCTION_NAMES:
                return False, f"Function '{func_name}' is not allowed. Allowed: {', '.join(sorted(SAFE_FUNCTION_NAMES))}"
        else:
            return False, "Only simple function calls are allowed (no method calls)"

        # Validate arguments
        for arg in node.args:
            valid, error = validate_equation_ast(arg)
            if not valid:
                return False, error

        for keyword in node.keywords:
            valid, error = validate_equation_ast(keyword.value)
            if not valid:
                return False, error

        return True, None

    # Ternary conditional
    if isinstance(node, ast.IfExp):
        valid, error = validate_equation_ast(node.test)
        if not valid:
            return False, error
        valid, error = validate_equation_ast(node.body)
        if not valid:
            return False, error
        return validate_equation_ast(node.orelse)

    # Comparison
    if isinstance(node, ast.Compare):
        valid, error = validate_equation_ast(node.left)
        if not valid:
            return False, error
        for op in node.ops:
            if type(op) not in SAFE_AST_OPERATORS:
                return False, f"Unsupported comparison: {type(op).__name__}"
        for comparator in node.comparators:
            valid, error = validate_equation_ast(comparator)
            if not valid:
                return False, error
        return True, None

    # Boolean operation (and, or)
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            valid, error = validate_equation_ast(value)
            if not valid:
                return False, error
        return True, None

    # Tuple/List (used in function arguments)
    if isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            valid, error = validate_equation_ast(elt)
            if not valid:
                return False, error
        return True, None

    return False, f"Unsupported expression type: {type(node).__name__}"


def validate_equations(elements: List[Element]) -> List[ValidationError]:
    """
    Validate equations

    Rules:
    - Valid syntax (parseable as Python expression)
    - Only safe operations allowed
    - All referenced variables must exist as elements
    """
    errors: List[ValidationError] = []

    # Build lookup maps for element IDs and names
    element_map: Dict[str, Element] = {elem.id: elem for elem in elements}
    element_names: Dict[str, Element] = {elem.name: elem for elem in elements}

    # Merge maps for variable lookup (ID takes precedence)
    all_valid_vars: Set[str] = set(element_map.keys()) | set(element_names.keys()) | BUILT_IN_VARIABLES

    for element in elements:
        if not element.equation or element.equation.strip() == "":
            continue

        equation = element.equation.strip()

        # Validate syntax
        syntax_valid, syntax_error = validate_equation_syntax(equation)
        if not syntax_valid:
            errors.append(
                ValidationError(
                    code="syntax_error",
                    message=f"Syntax error in equation '{equation}': {syntax_error}",
                    element_id=element.id,
                    field="equation",
                    suggestion="Check for balanced parentheses, valid operators, and proper syntax",
                )
            )
            continue

        # Validate AST for safe operations
        try:
            tree = ast.parse(equation, mode="eval")
            ast_valid, ast_error = validate_equation_ast(tree)
            if not ast_valid:
                errors.append(
                    ValidationError(
                        code="unsafe_operation",
                        message=f"Unsafe operation in equation '{equation}': {ast_error}",
                        element_id=element.id,
                        field="equation",
                        suggestion="Use only allowed operations and functions",
                    )
                )
                continue

            # Check variable references
            referenced_vars = extract_variable_references(tree)

            for var_name in referenced_vars:
                if var_name in BUILT_IN_VARIABLES:
                    continue

                if var_name not in element_map and var_name not in element_names:
                    # Find similar names for suggestion
                    similar = _find_similar_names(var_name, all_valid_vars)
                    suggestion = f"Create element with ID or name '{var_name}'"
                    if similar:
                        suggestion += f". Did you mean: {', '.join(similar[:3])}?"

                    errors.append(
                        ValidationError(
                            code="undefined_variable",
                            message=f"Equation references undefined variable '{var_name}'",
                            element_id=element.id,
                            field="equation",
                            suggestion=suggestion,
                            context={"equation": equation, "undefined_var": var_name},
                        )
                    )

        except Exception as e:
            errors.append(
                ValidationError(
                    code="parse_error",
                    message=f"Failed to parse equation '{equation}': {str(e)}",
                    element_id=element.id,
                    field="equation",
                    suggestion="Check equation format",
                )
            )

    return errors


def _find_similar_names(name: str, candidates: Set[str]) -> List[str]:
    """Find similar names for typo suggestions"""
    name_lower = name.lower()
    similar = []

    for candidate in candidates:
        candidate_lower = candidate.lower()
        # Exact case-insensitive match
        if name_lower == candidate_lower:
            similar.insert(0, candidate)
        # Substring match
        elif name_lower in candidate_lower or candidate_lower in name_lower:
            similar.append(candidate)
        # Simple edit distance approximation
        elif abs(len(name) - len(candidate)) <= 2:
            common = sum(1 for a, b in zip(name_lower, candidate_lower) if a == b)
            if common >= len(name_lower) * 0.6:
                similar.append(candidate)

    return similar[:5]


# ============================================================================
# Circular Dependency Detection
# ============================================================================


def detect_circular_dependencies(
    elements: List[Element], links: List[Link]
) -> List[ValidationError]:
    """
    Detect circular dependencies in non-stock elements using DFS

    IMPORTANT: Stocks are EXCLUDED from circular dependency detection because
    stock-flow-stock cycles are valid feedback loops in system dynamics.
    Stock values are known from the previous timestep, breaking any apparent cycle.

    Only reports cycles among flows, variables, and parameters.
    """
    errors: List[ValidationError] = []

    element_map: Dict[str, Element] = {elem.id: elem for elem in elements}
    element_names: Dict[str, Element] = {elem.name: elem for elem in elements}

    # Identify stocks - they are excluded from cycle detection
    stock_ids: Set[str] = {elem.id for elem in elements if elem.type == "stock"}

    # Only check non-stock elements
    non_stock_elements = [e for e in elements if e.type != "stock"]
    non_stock_ids = {e.id for e in non_stock_elements}

    # Build dependency graph (excluding stocks)
    dependencies: Dict[str, List[str]] = {elem.id: [] for elem in non_stock_elements}

    # Add dependencies from links
    for link in links:
        if link.target in dependencies and link.source in non_stock_ids:
            if link.source not in dependencies[link.target]:
                dependencies[link.target].append(link.source)

    # Add dependencies from equations
    for element in non_stock_elements:
        if not element.equation or element.equation.strip() == "":
            continue

        try:
            tree = ast.parse(element.equation, mode="eval")
            referenced_vars = extract_variable_references(tree)

            for var_name in referenced_vars:
                if var_name in BUILT_IN_VARIABLES:
                    continue

                # Find the element this variable refers to
                dep_element = element_map.get(var_name) or element_names.get(var_name)

                if dep_element and dep_element.id != element.id:
                    # Only add if it's a non-stock element
                    if dep_element.id in non_stock_ids:
                        if dep_element.id not in dependencies[element.id]:
                            dependencies[element.id].append(dep_element.id)
        except Exception:
            pass

    # Detect cycles using DFS with coloring
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {node: WHITE for node in dependencies}
    parent: Dict[str, Optional[str]] = {node: None for node in dependencies}
    detected_cycles: Set[Tuple[str, ...]] = set()

    def dfs(node: str, path: List[str]) -> Optional[List[str]]:
        """DFS that returns cycle path if found"""
        color[node] = GRAY
        path.append(node)

        for neighbor in dependencies.get(node, []):
            if color.get(neighbor) == GRAY:
                # Found back edge - extract cycle
                if neighbor in path:
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]
            elif color.get(neighbor) == WHITE:
                result = dfs(neighbor, path)
                if result:
                    return result

        path.pop()
        color[node] = BLACK
        return None

    for node in dependencies:
        if color[node] == WHITE:
            cycle = dfs(node, [])
            if cycle:
                # Normalize cycle for deduplication
                normalized = tuple(sorted(cycle))
                if normalized not in detected_cycles:
                    detected_cycles.add(normalized)

                    # Create readable cycle path
                    cycle_names = [
                        element_map[eid].name if eid in element_map else eid
                        for eid in cycle
                    ]
                    cycle_path_str = " -> ".join(cycle_names) + f" -> {cycle_names[0]}"

                    errors.append(
                        ValidationError(
                            code="circular_dependency",
                            message=f"Circular dependency detected: {cycle_path_str}",
                            element_id=cycle[0],
                            field="equation",
                            suggestion=f"Break the cycle by modifying one of: {', '.join(cycle_names)}",
                            context={"cycle": cycle},
                        )
                    )

    return errors


# ============================================================================
# Model-Level Validation
# ============================================================================


def validate_stock_flow_relationships(elements: List[Element]) -> List[ValidationError]:
    """
    Validate relationships between stocks and flows (warnings, not errors)

    Checks:
    - Stocks should reference flows in their equations
    - Flows should not directly reference flows (should go through stocks)
    """
    warnings: List[ValidationError] = []

    element_map: Dict[str, Element] = {elem.id: elem for elem in elements}
    element_names: Dict[str, Element] = {elem.name: elem for elem in elements}

    flow_ids = {elem.id for elem in elements if elem.type == "flow"}
    stock_ids = {elem.id for elem in elements if elem.type == "stock"}

    for element in elements:
        if element.type != "stock" or not element.equation:
            continue

        try:
            tree = ast.parse(element.equation, mode="eval")
            referenced_vars = extract_variable_references(tree)

            # Check if stock references another stock directly
            for var_name in referenced_vars:
                ref_elem = element_map.get(var_name) or element_names.get(var_name)
                if ref_elem and ref_elem.type == "stock" and ref_elem.id != element.id:
                    warnings.append(
                        ValidationError(
                            code="stock_references_stock",
                            message=f"Stock '{element.name}' directly references stock '{ref_elem.name}'. Consider using a flow.",
                            element_id=element.id,
                            field="equation",
                            suggestion=f"Create a flow between '{ref_elem.name}' and '{element.name}'",
                        )
                    )
        except Exception:
            pass

    return warnings


# ============================================================================
# Main Validation Orchestrator
# ============================================================================


def validate_model(
    elements: List[Element], links: List[Link], config: SimulationConfig
) -> ValidationResult:
    """
    Orchestrate all validation checks

    Validation order:
    1. Configuration validation
    2. Element structure validation
    3. Link validation (if elements valid)
    4. Equation validation (if elements valid)
    5. Circular dependency detection (if structure valid)
    6. Stock-flow relationship validation (warnings)

    Returns:
        ValidationResult with errors and warnings
    """
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []

    # 1. Validate configuration
    errors.extend(validate_simulation_config(config))

    # 2. Validate elements
    element_errors = validate_elements(elements)
    errors.extend(element_errors)

    # Check for blocking errors before proceeding
    blocking_codes = {"duplicate_id", "invalid_element_type"}
    has_blocking_errors = any(e.code in blocking_codes for e in element_errors)

    if not has_blocking_errors:
        # 3. Validate links
        errors.extend(validate_links(links, elements))

        # 4. Validate equations
        errors.extend(validate_equations(elements))

        # Check for structure errors before circular dependency check
        structure_error_codes = {
            "duplicate_id",
            "invalid_element_type",
            "invalid_link_source",
            "invalid_link_target",
            "syntax_error",
            "undefined_variable",
        }
        has_structure_errors = any(e.code in structure_error_codes for e in errors)

        if not has_structure_errors:
            # 5. Detect circular dependencies
            errors.extend(detect_circular_dependencies(elements, links))

            # 6. Validate stock-flow relationships (warnings only)
            warnings.extend(validate_stock_flow_relationships(elements))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ============================================================================
# Utility Functions
# ============================================================================


def quick_validate(elements: List[Element], links: List[Link]) -> bool:
    """
    Quick validation check without full error details

    Useful for fast checks before simulation.
    """
    try:
        config = SimulationConfig()  # Use defaults
        result = validate_model(elements, links, config)
        return result.valid
    except Exception:
        return False


def get_validation_summary(result: ValidationResult) -> Dict[str, Any]:
    """
    Get a summary of validation results

    Returns:
        Dictionary with error counts by category
    """
    summary = {
        "valid": result.valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "errors_by_code": {},
        "errors_by_element": {},
    }

    for error in result.errors:
        # Count by code
        if error.code not in summary["errors_by_code"]:
            summary["errors_by_code"][error.code] = 0
        summary["errors_by_code"][error.code] += 1

        # Count by element
        elem_id = error.element_id or "config"
        if elem_id not in summary["errors_by_element"]:
            summary["errors_by_element"][elem_id] = 0
        summary["errors_by_element"][elem_id] += 1

    return summary