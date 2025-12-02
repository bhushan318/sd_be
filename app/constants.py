"""
Shared constants for System Dynamics Backend
Centralizes configuration to prevent inconsistencies between validator and evaluator
"""

import ast

# ============================================================================
# Built-in Variables
# ============================================================================

# Variables automatically available in all equations
BUILT_IN_VARIABLES = {"t", "time"}

# ============================================================================
# Safe Functions
# ============================================================================

# Function names allowed in equations
# This set is used by the validator to check equation safety
SAFE_FUNCTION_NAMES = {
    # Basic math
    "abs",
    "min",
    "max",
    "pow",
    "round",
    # Exponential and logarithmic
    "exp",
    "log",
    "log10",
    "sqrt",
    # Trigonometric
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    # Hyperbolic
    "sinh",
    "cosh",
    "tanh",
    # Rounding
    "ceil",
    "floor",
    # System Dynamics special functions
    "DELAY1",
    "DELAY3",
    "SMOOTH",
    "LOOKUP",
    # Alternative lowercase versions for convenience
    "delay1",
    "delay3",
    "smooth",
    "lookup",
}

# ============================================================================
# Safe AST Operators
# ============================================================================

# Binary and unary operators allowed in equations
SAFE_AST_OPERATORS = {
    # Arithmetic
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    # Unary
    ast.USub,
    ast.UAdd,
    # Comparison
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    # Boolean
    ast.And,
    ast.Or,
    ast.Not,
}

# ============================================================================
# Element Types
# ============================================================================

VALID_ELEMENT_TYPES = {"stock", "flow", "parameter", "variable"}

# ============================================================================
# Simulation Limits
# ============================================================================

MAX_SIMULATION_STEPS = 1_000_000
VALID_INTEGRATION_METHODS = {"euler", "rk4"}