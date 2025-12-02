"""
Equation evaluator for System Dynamics Backend
Safely evaluates mathematical equations with controlled namespace and AST parsing
"""

from typing import Dict, Set, Optional, List, Any, Tuple
import ast
import operator
import math
import logging

from app.exceptions import EvaluationError, CircularDependencyError
from app.constants import (
    SAFE_FUNCTION_NAMES,
    SAFE_AST_OPERATORS,
    BUILT_IN_VARIABLES,
)

# Conditional import for cache utilities
try:
    from app.utils.cache import get_cache, hash_model
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    
    def get_cache():
        return None
    
    def hash_model(elements, links):
        return None

# Conditional import for models
try:
    from app.models import LookupTable
except ImportError:
    LookupTable = None

logger = logging.getLogger(__name__)


class SafeEquationEvaluator:
    """
    Safely evaluate mathematical equations with controlled namespace

    Supports:
    - Basic arithmetic operations (+, -, *, /, **, %, //)
    - Comparison operations (<, <=, >, >=, ==, !=)
    - Boolean operations (and, or, not)
    - Mathematical functions (sin, cos, tan, exp, log, sqrt, etc.)
    - Ternary conditional expressions (x if condition else y)
    - Variable references
    - System dynamics functions (DELAY1, DELAY3, SMOOTH, LOOKUP)
    """

    # Operator implementations
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Function implementations
    SAFE_FUNCTIONS = {
        # Basic math
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "round": round,
        # Exponential and logarithmic
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "sqrt": math.sqrt,
        # Trigonometric
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        # Hyperbolic
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        # Rounding
        "ceil": math.ceil,
        "floor": math.floor,
    }

    # System dynamics functions that require special handling
    SD_FUNCTIONS = {"DELAY1", "DELAY3", "SMOOTH", "LOOKUP", "delay1", "delay3", "smooth", "lookup"}

    def __init__(self):
        """Initialize evaluator with empty variable namespace"""
        self.variables: Dict[str, float] = {}
        self.delay_state_manager: Optional[Any] = None
        self.lookup_tables: Dict[str, Any] = {}
        self._ast_cache: Dict[str, ast.Expression] = {}

    def set_variables(self, variables: Dict[str, float]) -> None:
        """
        Set available variables for evaluation

        Args:
            variables: Dictionary mapping variable names to values
        """
        self.variables = variables.copy()

    def set_delay_state_manager(self, delay_state_manager: Any) -> None:
        """
        Set delay state manager for DELAY1, DELAY3, SMOOTH functions

        Args:
            delay_state_manager: DelayStateManager instance
        """
        self.delay_state_manager = delay_state_manager

    def set_lookup_tables(self, lookup_tables: Dict[str, Any]) -> None:
        """
        Set lookup tables for LOOKUP function

        Args:
            lookup_tables: Dictionary mapping element IDs/names to LookupTable objects
        """
        self.lookup_tables = lookup_tables.copy()

    def clear_cache(self) -> None:
        """Clear the internal AST cache"""
        self._ast_cache.clear()

    def parse_equation(
        self, equation: str, element_id: Optional[str] = None, use_cache: bool = True
    ) -> ast.Expression:
        """
        Parse equation string into AST with optional caching

        Args:
            equation: Mathematical equation string
            element_id: Optional element ID for caching
            use_cache: Whether to use internal cache

        Returns:
            Parsed AST expression

        Raises:
            EvaluationError: If equation syntax is invalid
        """
        # Strip whitespace before parsing to handle edge cases (leading/trailing whitespace)
        equation = equation.strip()
        cache_key = equation
        
        if use_cache and cache_key in self._ast_cache:
            return self._ast_cache[cache_key]

        try:
            tree = ast.parse(equation, mode="eval")
            if use_cache:
                self._ast_cache[cache_key] = tree
            return tree
        except SyntaxError as e:
            raise EvaluationError(
                code="syntax_error",
                message=f"Syntax error in equation: {equation}. {str(e)}",
                element_id=element_id,
                equation=equation,
            ) from e

    def eval_node(self, node: ast.AST, element_id: Optional[str] = None) -> Any:
        """
        Recursively evaluate AST node

        Args:
            node: AST node to evaluate
            element_id: Optional element ID for error reporting

        Returns:
            Evaluated result (usually float, but can be bool or list)

        Raises:
            EvaluationError: If evaluation fails
        """
        # Expression wrapper
        if isinstance(node, ast.Expression):
            return self.eval_node(node.body, element_id)

        # Constants (Python 3.8+) - check first to avoid deprecation warnings
        if isinstance(node, ast.Constant):
            return node.value

        # Legacy support for Python < 3.8 (deprecated but still needed for compatibility)
        # Numbers (Python 3.7 compatibility)
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return node.n

        # Strings (Python 3.7 compatibility)
        if hasattr(ast, "Str") and isinstance(node, ast.Str):
            return node.s

        # NameConstant for True/False/None (Python 3.7 compatibility)
        if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant):
            return node.value

        # Variable reference
        if isinstance(node, ast.Name):
            var_name = node.id
            if var_name in self.variables:
                return self.variables[var_name]
            else:
                raise EvaluationError(
                    code="undefined_variable",
                    message=f"Undefined variable: {var_name}",
                    element_id=element_id,
                    equation=var_name,
                )

        # Binary operations (+, -, *, /, etc.)
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node, element_id)

        # Unary operations (-, +)
        if isinstance(node, ast.UnaryOp):
            return self._eval_unaryop(node, element_id)

        # Function calls
        if isinstance(node, ast.Call):
            return self._eval_call(node, element_id)

        # Ternary conditional (x if condition else y)
        if isinstance(node, ast.IfExp):
            return self._eval_ifexp(node, element_id)

        # Comparison operations (<, <=, >, >=, ==, !=)
        if isinstance(node, ast.Compare):
            return self._eval_compare(node, element_id)

        # Boolean operations (and, or)
        if isinstance(node, ast.BoolOp):
            return self._eval_boolop(node, element_id)

        # Unary not
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            operand = self.eval_node(node.operand, element_id)
            return not operand

        # Tuple/List (used in function arguments)
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self.eval_node(elt, element_id) for elt in node.elts]

        # Unsupported node type
        raise EvaluationError(
            code="unsupported_node_type",
            message=f"Unsupported expression type: {type(node).__name__}",
            element_id=element_id,
        )

    def _eval_binop(self, node: ast.BinOp, element_id: Optional[str]) -> float:
        """Evaluate binary operation"""
        op_type = type(node.op)
        if op_type not in self.SAFE_OPERATORS:
            raise EvaluationError(
                code="unsupported_operator",
                message=f"Unsupported operator: {op_type.__name__}",
                element_id=element_id,
            )

        left = self.eval_node(node.left, element_id)
        right = self.eval_node(node.right, element_id)

        # Handle division by zero
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise EvaluationError(
                code="division_by_zero",
                message=f"Division by zero in expression",
                element_id=element_id,
            )

        try:
            return self.SAFE_OPERATORS[op_type](left, right)
        except Exception as e:
            raise EvaluationError(
                code="arithmetic_error",
                message=f"Arithmetic error: {str(e)}",
                element_id=element_id,
            ) from e

    def _eval_unaryop(self, node: ast.UnaryOp, element_id: Optional[str]) -> Any:
        """Evaluate unary operation"""
        op_type = type(node.op)
        operand = self.eval_node(node.operand, element_id)

        if op_type == ast.Not:
            return not operand
        elif op_type in self.SAFE_OPERATORS:
            return self.SAFE_OPERATORS[op_type](operand)
        else:
            raise EvaluationError(
                code="unsupported_unary_operator",
                message=f"Unsupported unary operator: {op_type.__name__}",
                element_id=element_id,
            )

    def _eval_call(self, node: ast.Call, element_id: Optional[str]) -> float:
        """Evaluate function call"""
        if not isinstance(node.func, ast.Name):
            raise EvaluationError(
                code="invalid_function_call",
                message="Function call must use a named function",
                element_id=element_id,
            )

        func_name = node.func.id
        func_name_upper = func_name.upper()

        # Handle System Dynamics special functions
        if func_name_upper == "LOOKUP":
            return self._eval_lookup(node.args, node, element_id)

        # Evaluate arguments for other functions
        args = [self.eval_node(arg, element_id) for arg in node.args]

        if func_name_upper == "DELAY1":
            return self._eval_delay1(args, node, element_id)
        elif func_name_upper == "DELAY3":
            return self._eval_delay3(args, node, element_id)
        elif func_name_upper == "SMOOTH":
            return self._eval_smooth(args, node, element_id)
        elif func_name in self.SAFE_FUNCTIONS:
            try:
                return self.SAFE_FUNCTIONS[func_name](*args)
            except ValueError as e:
                raise EvaluationError(
                    code="math_domain_error",
                    message=f"Math domain error in {func_name}: {str(e)}",
                    element_id=element_id,
                ) from e
            except Exception as e:
                raise EvaluationError(
                    code="function_evaluation_error",
                    message=f"Error evaluating function {func_name}: {str(e)}",
                    element_id=element_id,
                ) from e
        else:
            raise EvaluationError(
                code="function_not_allowed",
                message=f"Function not allowed: {func_name}. Allowed functions: {', '.join(sorted(self.SAFE_FUNCTIONS.keys()))}",
                element_id=element_id,
            )

    def _eval_ifexp(self, node: ast.IfExp, element_id: Optional[str]) -> float:
        """Evaluate ternary conditional expression"""
        condition = self.eval_node(node.test, element_id)
        if condition:
            return self.eval_node(node.body, element_id)
        else:
            return self.eval_node(node.orelse, element_id)

    def _eval_compare(self, node: ast.Compare, element_id: Optional[str]) -> bool:
        """Evaluate comparison expression"""
        left = self.eval_node(node.left, element_id)

        for op, comparator in zip(node.ops, node.comparators):
            right = self.eval_node(comparator, element_id)

            if isinstance(op, ast.Lt):
                result = left < right
            elif isinstance(op, ast.LtE):
                result = left <= right
            elif isinstance(op, ast.Gt):
                result = left > right
            elif isinstance(op, ast.GtE):
                result = left >= right
            elif isinstance(op, ast.Eq):
                result = left == right
            elif isinstance(op, ast.NotEq):
                result = left != right
            else:
                raise EvaluationError(
                    code="unsupported_comparison",
                    message=f"Unsupported comparison operator: {type(op).__name__}",
                    element_id=element_id,
                )

            if not result:
                return False
            left = right

        return True

    def _eval_boolop(self, node: ast.BoolOp, element_id: Optional[str]) -> bool:
        """Evaluate boolean operation (and, or)"""
        if isinstance(node.op, ast.And):
            for value in node.values:
                if not self.eval_node(value, element_id):
                    return False
            return True
        elif isinstance(node.op, ast.Or):
            for value in node.values:
                if self.eval_node(value, element_id):
                    return True
            return False
        else:
            raise EvaluationError(
                code="unsupported_bool_op",
                message=f"Unsupported boolean operator: {type(node.op).__name__}",
                element_id=element_id,
            )

    def _eval_delay1(
        self, args: List[float], node: ast.Call, element_id: Optional[str]
    ) -> float:
        """Evaluate DELAY1(input, delay_time) function"""
        if len(args) != 2:
            raise EvaluationError(
                code="invalid_function_args",
                message="DELAY1 requires 2 arguments: DELAY1(input, delay_time)",
                element_id=element_id,
            )

        if self.delay_state_manager is None:
            raise EvaluationError(
                code="delay_state_not_available",
                message="DELAY1 function requires delay state manager (only available during simulation)",
                element_id=element_id,
            )

        input_value, delay_time = args[0], args[1]

        if delay_time <= 0:
            raise EvaluationError(
                code="invalid_delay_time",
                message=f"DELAY1 delay time must be positive, got {delay_time}",
                element_id=element_id,
            )

        call_signature = f"DELAY1_{hash(ast.dump(node))}_{delay_time}"
        key = self.delay_state_manager.get_delay1_key(call_signature, delay_time)
        return self.delay_state_manager.delay1(input_value, delay_time, key)

    def _eval_delay3(
        self, args: List[float], node: ast.Call, element_id: Optional[str]
    ) -> float:
        """Evaluate DELAY3(input, delay_time) function"""
        if len(args) != 2:
            raise EvaluationError(
                code="invalid_function_args",
                message="DELAY3 requires 2 arguments: DELAY3(input, delay_time)",
                element_id=element_id,
            )

        if self.delay_state_manager is None:
            raise EvaluationError(
                code="delay_state_not_available",
                message="DELAY3 function requires delay state manager (only available during simulation)",
                element_id=element_id,
            )

        input_value, delay_time = args[0], args[1]

        if delay_time <= 0:
            raise EvaluationError(
                code="invalid_delay_time",
                message=f"DELAY3 delay time must be positive, got {delay_time}",
                element_id=element_id,
            )

        call_signature = f"DELAY3_{hash(ast.dump(node))}_{delay_time}"
        key = self.delay_state_manager.get_delay3_key(call_signature, delay_time)
        return self.delay_state_manager.delay3(input_value, delay_time, key)

    def _eval_smooth(
        self, args: List[float], node: ast.Call, element_id: Optional[str]
    ) -> float:
        """Evaluate SMOOTH(input, smooth_time) function"""
        if len(args) != 2:
            raise EvaluationError(
                code="invalid_function_args",
                message="SMOOTH requires 2 arguments: SMOOTH(input, smooth_time)",
                element_id=element_id,
            )

        if self.delay_state_manager is None:
            raise EvaluationError(
                code="delay_state_not_available",
                message="SMOOTH function requires delay state manager (only available during simulation)",
                element_id=element_id,
            )

        input_value, smooth_time = args[0], args[1]

        if smooth_time <= 0:
            raise EvaluationError(
                code="invalid_smooth_time",
                message=f"SMOOTH time must be positive, got {smooth_time}",
                element_id=element_id,
            )

        call_signature = f"SMOOTH_{hash(ast.dump(node))}_{smooth_time}"
        key = self.delay_state_manager.get_smooth_key(call_signature, smooth_time)
        return self.delay_state_manager.smooth(input_value, smooth_time, key)

    def _eval_lookup(
        self, arg_nodes: List[ast.AST], node: ast.Call, element_id: Optional[str]
    ) -> float:
        """Evaluate LOOKUP(x, table_name) function"""
        if len(arg_nodes) != 2:
            raise EvaluationError(
                code="invalid_function_args",
                message="LOOKUP requires 2 arguments: LOOKUP(x, table_name)",
                element_id=element_id,
            )

        # Evaluate first argument (x value)
        x = self.eval_node(arg_nodes[0], element_id)

        # Get table name from second argument
        table_name = self._extract_table_name(arg_nodes[1], element_id)

        # Find the lookup table
        lookup_table = self._find_lookup_table(table_name, element_id)

        # Perform interpolation
        return self._interpolate_lookup(x, lookup_table, element_id)

    def _extract_table_name(
        self, node: ast.AST, element_id: Optional[str]
    ) -> str:
        """Extract table name from AST node"""
        # String constant (Python 3.8+)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value

        # String (Python 3.7)
        if hasattr(ast, "Str") and isinstance(node, ast.Str):
            return node.s

        # Variable reference
        if isinstance(node, ast.Name):
            return node.id

        raise EvaluationError(
            code="invalid_lookup_table_reference",
            message="LOOKUP table must be specified as a string or variable name",
            element_id=element_id,
        )

    def _find_lookup_table(self, table_name: str, element_id: Optional[str]) -> Any:
        """Find lookup table by name"""
        if table_name in self.lookup_tables:
            return self.lookup_tables[table_name]

        # Try case-insensitive match
        for key in self.lookup_tables:
            if key.lower() == table_name.lower():
                return self.lookup_tables[key]

        available = list(self.lookup_tables.keys()) if self.lookup_tables else []
        raise EvaluationError(
            code="lookup_table_not_found",
            message=f"Lookup table '{table_name}' not found. Available tables: {available}",
            element_id=element_id,
        )

    def _interpolate_lookup(
        self, x: float, lookup_table: Any, element_id: Optional[str]
    ) -> float:
        """Perform lookup table interpolation"""
        points = lookup_table.points
        if len(points) < 2:
            raise EvaluationError(
                code="invalid_lookup_table",
                message="Lookup table must have at least 2 points",
                element_id=element_id,
            )

        # Sort points by x value
        sorted_points = sorted(points, key=lambda p: p[0])

        # Handle out-of-range values (extrapolation: use boundary values)
        if x <= sorted_points[0][0]:
            return sorted_points[0][1]
        if x >= sorted_points[-1][0]:
            return sorted_points[-1][1]

        # Find interpolation segment
        for i in range(len(sorted_points) - 1):
            x1, y1 = sorted_points[i]
            x2, y2 = sorted_points[i + 1]

            if x1 <= x <= x2:
                if lookup_table.interpolation == "step":
                    return y1
                else:
                    # Linear interpolation
                    if x2 == x1:
                        # Edge case: duplicate x-values - return first y-value
                        # This could indicate a data quality issue, but we handle it gracefully
                        logger.debug(
                            f"Lookup table has duplicate x-values at {x1}, "
                            f"returning y={y1}"
                        )
                        return y1
                    t = (x - x1) / (x2 - x1)
                    return y1 + t * (y2 - y1)

        # Fallback (should not reach here)
        return sorted_points[-1][1]

    def evaluate(self, equation: str, element_id: Optional[str] = None) -> float:
        """
        Evaluate equation string with current variables

        Args:
            equation: Mathematical equation string
            element_id: Optional element ID for error reporting

        Returns:
            Evaluated numeric result (boolean comparisons converted to 1.0/0.0)

        Raises:
            EvaluationError: If evaluation fails
        """
        if not equation or equation.strip() == "":
            return 0.0

        try:
            tree = self.parse_equation(equation, element_id)
            result = self.eval_node(tree, element_id)
            # Convert boolean to float (True -> 1.0, False -> 0.0) for system dynamics compatibility
            if isinstance(result, bool):
                return 1.0 if result else 0.0
            return float(result)
        except EvaluationError:
            raise
        except Exception as e:
            raise EvaluationError(
                code="evaluation_error",
                message=f"Error evaluating '{equation}': {str(e)}",
                element_id=element_id,
                equation=equation,
            ) from e


# ============================================================================
# Variable Reference Extraction
# ============================================================================


def extract_variable_references(node: ast.AST) -> Set[str]:
    """
    Extract all variable names referenced in an AST node

    This function extracts only actual variable references, excluding:
    - Function names (handled separately)
    - String literals (used in LOOKUP)
    - Numeric constants

    Args:
        node: AST node to analyze

    Returns:
        Set of variable names referenced
    """
    variables: Set[str] = set()
    _extract_variables_recursive(node, variables)
    return variables


def _extract_variables_recursive(node: ast.AST, variables: Set[str]) -> None:
    """Recursive helper for variable extraction"""
    if node is None:
        return

    if isinstance(node, ast.Expression):
        _extract_variables_recursive(node.body, variables)

    elif isinstance(node, ast.Name):
        # Only add if not a known function name
        if node.id not in SAFE_FUNCTION_NAMES:
            variables.add(node.id)

    elif isinstance(node, ast.Constant):
        # Constants don't reference variables
        pass
    elif hasattr(ast, "Num") and isinstance(node, ast.Num):
        # Legacy: Numbers don't reference variables
        pass

    elif hasattr(ast, "Str") and isinstance(node, ast.Str):
        # Legacy: String literals don't reference variables
        pass

    elif hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant):
        # Legacy: True/False/None don't reference variables
        pass

    elif isinstance(node, ast.BinOp):
        _extract_variables_recursive(node.left, variables)
        _extract_variables_recursive(node.right, variables)

    elif isinstance(node, ast.UnaryOp):
        _extract_variables_recursive(node.operand, variables)

    elif isinstance(node, ast.Call):
        # Don't add function name as variable
        # But do extract from arguments
        for arg in node.args:
            _extract_variables_recursive(arg, variables)
        for keyword in node.keywords:
            _extract_variables_recursive(keyword.value, variables)

    elif isinstance(node, ast.IfExp):
        _extract_variables_recursive(node.test, variables)
        _extract_variables_recursive(node.body, variables)
        _extract_variables_recursive(node.orelse, variables)

    elif isinstance(node, ast.Compare):
        _extract_variables_recursive(node.left, variables)
        for comparator in node.comparators:
            _extract_variables_recursive(comparator, variables)

    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            _extract_variables_recursive(value, variables)

    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _extract_variables_recursive(elt, variables)

    elif isinstance(node, ast.Subscript):
        _extract_variables_recursive(node.value, variables)
        _extract_variables_recursive(node.slice, variables)

    elif isinstance(node, ast.Index):  # Python 3.7/3.8 compatibility
        _extract_variables_recursive(node.value, variables)


# ============================================================================
# Dependency Graph Building
# ============================================================================


def build_dependency_graph(
    elements: List, links: List, use_cache: bool = True
) -> Dict[str, List[str]]:
    """
    Build dependency graph from elements and links

    IMPORTANT: Stock elements are excluded as dependencies because their values
    are known from the previous timestep (or initial conditions). This allows
    the classic stock-flow-stock feedback loops in system dynamics models.

    Args:
        elements: List of model elements
        links: List of links between elements
        use_cache: Whether to use cache (default: True)

    Returns:
        Dictionary mapping element IDs to list of dependencies (element IDs they depend on)
    """
    cache = get_cache() if CACHE_AVAILABLE else None
    model_hash = None

    # Try to get from cache
    if use_cache and cache is not None:
        try:
            model_hash = hash_model(elements, links)
            cached_data = cache.get(model_hash)
            if cached_data and "dependency_graph" in cached_data:
                logger.debug("Using cached dependency graph")
                return cached_data["dependency_graph"]
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

    # Build element lookup maps
    element_map: Dict[str, Any] = {elem.id: elem for elem in elements}
    element_names: Dict[str, Any] = {elem.name: elem for elem in elements}

    # Identify stocks - their values are pre-computed from integration
    stock_ids: Set[str] = {elem.id for elem in elements if elem.type == "stock"}

    # Initialize dependency graph for all elements
    dependencies: Dict[str, List[str]] = {elem.id: [] for elem in elements}

    # Add dependencies from explicit links
    for link in links:
        if link.target in dependencies and link.source in element_map:
            # Exclude stocks as dependencies
            if link.source not in stock_ids:
                if link.source not in dependencies[link.target]:
                    dependencies[link.target].append(link.source)

    # Add dependencies from equation variable references
    for element in elements:
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
                    # Exclude stocks as dependencies
                    if dep_element.id not in stock_ids:
                        if dep_element.id not in dependencies[element.id]:
                            dependencies[element.id].append(dep_element.id)
        except SyntaxError:
            # Skip if equation can't be parsed (handled by validation)
            logger.debug(f"Skipping unparseable equation for element {element.id}")

    # Cache the results
    if use_cache and cache is not None and model_hash is not None:
        try:
            evaluation_order = topological_sort(dependencies)
            cache.put(model_hash, {}, dependencies, evaluation_order)
        except Exception as e:
            logger.debug(f"Failed to cache dependency graph: {e}")

    return dependencies


# ============================================================================
# Topological Sort
# ============================================================================


def topological_sort(
    dependencies: Dict[str, List[str]],
    elements: Optional[List] = None,
    links: Optional[List] = None,
) -> List[str]:
    """
    Perform topological sort on dependency graph using Kahn's algorithm

    Args:
        dependencies: Dictionary mapping element IDs to their dependencies
        elements: Optional list of elements (for cache key generation)
        links: Optional list of links (for cache key generation)

    Returns:
        List of element IDs in valid evaluation order (dependencies first)

    Raises:
        CircularDependencyError: If circular dependency is detected
    """
    # Build in-degree map (count of dependencies for each node)
    in_degree: Dict[str, int] = {node: len(deps) for node, deps in dependencies.items()}

    # Build reverse graph (who depends on each node)
    dependents: Dict[str, List[str]] = {node: [] for node in dependencies}
    for node, deps in dependencies.items():
        for dep in deps:
            if dep in dependents:
                dependents[dep].append(node)

    # Start with nodes that have no dependencies
    queue: List[str] = [node for node, degree in in_degree.items() if degree == 0]
    result: List[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)

        # Update in-degrees for dependent nodes
        for dependent in dependents.get(node, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # Check for cycles
    if len(result) != len(dependencies):
        remaining = set(dependencies.keys()) - set(result)
        cycle = _find_cycle(dependencies, remaining)
        
        if cycle:
            cycle_str = " -> ".join(cycle)
            # Raise as ValueError for backward compatibility with tests
            # CircularDependencyError is a subclass of SimulationError, but tests expect ValueError
            raise ValueError(
                f"Circular dependency detected. Cycle: {cycle_str}"
            )
        else:
            raise ValueError(
                f"Circular dependency detected involving: {', '.join(sorted(remaining))}"
            )

    return result


def _find_cycle(
    dependencies: Dict[str, List[str]], candidates: Set[str]
) -> Optional[List[str]]:
    """Find a cycle in the dependency graph among candidate nodes"""
    visited: Set[str] = set()
    rec_stack: Set[str] = set()
    path: List[str] = []

    def dfs(node: str) -> Optional[List[str]]:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in dependencies.get(node, []):
            if neighbor not in candidates:
                continue

            if neighbor not in visited:
                cycle = dfs(neighbor)
                if cycle:
                    return cycle
            elif neighbor in rec_stack:
                # Found cycle
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]

        path.pop()
        rec_stack.remove(node)
        return None

    for node in candidates:
        if node not in visited:
            cycle = dfs(node)
            if cycle:
                return cycle

    return None


# ============================================================================
# Utility Functions
# ============================================================================


def get_evaluation_order(elements: List, links: List) -> List[str]:
    """
    Get the order in which elements should be evaluated

    Convenience function that builds dependency graph and performs topological sort.

    Args:
        elements: List of model elements
        links: List of links between elements

    Returns:
        List of element IDs in valid evaluation order
    """
    dependencies = build_dependency_graph(elements, links)
    return topological_sort(dependencies, elements, links)


def validate_equation_references(
    equation: str, available_elements: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Check if all variable references in an equation are valid

    Args:
        equation: Equation string to check
        available_elements: Dictionary of available element IDs/names

    Returns:
        Tuple of (is_valid, list_of_undefined_variables)
    """
    try:
        tree = ast.parse(equation, mode="eval")
        referenced = extract_variable_references(tree)

        undefined = []
        for var in referenced:
            if var not in BUILT_IN_VARIABLES and var not in available_elements:
                undefined.append(var)

        return len(undefined) == 0, undefined
    except SyntaxError:
        return False, []