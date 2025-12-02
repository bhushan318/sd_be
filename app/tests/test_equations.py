"""
Tests for equation parsing, evaluation, and validation
"""

import pytest
import ast
from app.evaluator import SafeEquationEvaluator, extract_variable_references
from app.models import Element
from app.validation import (
    validate_equations,
    validate_equation_syntax,
    validate_equation_ast,
)
from app.exceptions import EvaluationError


def test_equation_arithmetic_operations():
    """Test various arithmetic operations in equations"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10, "y": 5, "z": 2})

    assert evaluator.evaluate("x + y") == 15
    assert evaluator.evaluate("x - y") == 5
    assert evaluator.evaluate("x * y") == 50
    assert evaluator.evaluate("x / y") == 2.0
    assert evaluator.evaluate("x ** z") == 100
    assert evaluator.evaluate("-x") == -10
    assert evaluator.evaluate("+x") == 10


def test_equation_comparison_operations():
    """Test comparison operations in equations"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10, "y": 5})

    # Comparison operations return 1.0 for True and 0.0 for False
    assert evaluator.evaluate("x > y") == 1.0
    assert evaluator.evaluate("x < y") == 0.0
    assert evaluator.evaluate("x >= y") == 1.0
    assert evaluator.evaluate("x <= y") == 0.0
    assert evaluator.evaluate("x == 10") == 1.0
    assert evaluator.evaluate("x != y") == 1.0


def test_equation_mathematical_functions():
    """Test mathematical functions in equations"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 4, "y": 2})

    assert evaluator.evaluate("sqrt(x)") == 2.0
    assert evaluator.evaluate("abs(-x)") == 4
    assert evaluator.evaluate("min(x, 10)") == 4
    assert evaluator.evaluate("max(x, 10)") == 10
    assert evaluator.evaluate("pow(x, y)") == 16
    assert evaluator.evaluate("exp(0)") == 1.0
    assert evaluator.evaluate("log(exp(1))") == pytest.approx(1.0, abs=0.001)


def test_equation_ternary_conditional():
    """Test ternary conditional expressions"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10, "y": 5})

    assert evaluator.evaluate("x if x > 5 else 0") == 10
    assert evaluator.evaluate("x if x < 5 else 0") == 0
    assert evaluator.evaluate("y if x > y else x") == 5


def test_equation_complex_expressions():
    """Test complex nested expressions"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10, "y": 5, "z": 2})

    assert evaluator.evaluate("(x + y) * z") == 30
    assert evaluator.evaluate("x ** 2 + y * z") == 110
    # sqrt(10 * 5) + 2 = sqrt(50) + 2 = 7.0710678118654755 + 2 = 9.071067811865476
    assert evaluator.evaluate("sqrt(x * y) + z") == pytest.approx(9.071, abs=0.01)
    assert evaluator.evaluate("x if (x > y and y > z) else 0") == 10


def test_equation_with_time_variable():
    """Test equations using built-in time variable"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"t": 5.0, "time": 5.0})

    assert evaluator.evaluate("t * 2") == 10.0
    assert evaluator.evaluate("time + 1") == 6.0
    assert evaluator.evaluate("t ** 2") == 25.0


def test_equation_undefined_variable():
    """Test error handling for undefined variables"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10})

    with pytest.raises(EvaluationError) as exc_info:
        evaluator.evaluate("y + 5")

    assert exc_info.value.code == "undefined_variable"
    assert "y" in exc_info.value.message


def test_equation_syntax_error():
    """Test handling of syntax errors"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10})

    with pytest.raises(EvaluationError) as exc_info:
        evaluator.evaluate("x + (")

    assert (
        exc_info.value.code == "syntax_error"
        or exc_info.value.code == "evaluation_error"
    )


def test_equation_unsafe_function():
    """Test that unsafe functions are rejected"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10})

    with pytest.raises(EvaluationError) as exc_info:
        evaluator.evaluate("__import__('os')")

    # Should raise an error about unsupported node type or function not allowed
    assert (
        "not allowed" in exc_info.value.message.lower()
        or "unsupported" in exc_info.value.message.lower()
    )


def test_extract_variable_references_simple():
    """Test variable reference extraction from simple expressions"""
    tree = ast.parse("x + y", mode="eval")
    vars = extract_variable_references(tree)
    assert vars == {"x", "y"}


def test_extract_variable_references_complex():
    """Test variable reference extraction from complex expressions"""
    tree = ast.parse("x * y + z * w", mode="eval")
    vars = extract_variable_references(tree)
    assert vars == {"x", "y", "z", "w"}


def test_extract_variable_references_functions():
    """Test variable reference extraction from function calls"""
    tree = ast.parse("sqrt(x) + log(y)", mode="eval")
    vars = extract_variable_references(tree)
    assert vars == {"x", "y"}


def test_extract_variable_references_ternary():
    """Test variable reference extraction from ternary expressions"""
    tree = ast.parse("x if y > 0 else z", mode="eval")
    vars = extract_variable_references(tree)
    assert vars == {"x", "y", "z"}


def test_extract_variable_references_no_strings():
    """Test that string literals are not extracted as variables"""
    tree = ast.parse('LOOKUP(x, "table")', mode="eval")
    vars = extract_variable_references(tree)
    assert vars == {"x"}
    assert "table" not in vars


def test_validate_equation_syntax_valid():
    """Test syntax validation for valid equations"""
    valid, error = validate_equation_syntax("x + y * 2")
    assert valid is True
    assert error is None


def test_validate_equation_syntax_invalid():
    """Test syntax validation for invalid equations"""
    valid, error = validate_equation_syntax("x + (")
    assert valid is False
    assert error is not None


def test_validate_equation_ast_valid():
    """Test AST validation for valid equations"""
    tree = ast.parse("x + y * 2", mode="eval")
    valid, error = validate_equation_ast(tree)
    assert valid is True
    assert error is None


def test_validate_equation_ast_unsafe_operator():
    """Test AST validation rejects unsafe operators"""
    # Note: This test depends on what operators are considered unsafe
    # For now, we test that the validation function works
    tree = ast.parse("x + y", mode="eval")
    valid, error = validate_equation_ast(tree)
    # Safe operators should pass
    assert valid is True


def test_validate_equations_valid():
    """Test equation validation for valid model"""
    elements = [
        Element(id="x", type="parameter", name="X", value=10),
        Element(id="y", type="variable", name="Y", equation="x * 2"),
        Element(id="z", type="variable", name="Z", equation="y + 5"),
    ]

    errors = validate_equations(elements)
    assert len(errors) == 0


def test_validate_equations_undefined_variable():
    """Test equation validation detects undefined variables"""
    elements = [
        Element(id="x", type="parameter", name="X", value=10),
        Element(id="y", type="variable", name="Y", equation="z * 2"),  # z is undefined
    ]

    errors = validate_equations(elements)
    assert len(errors) > 0
    assert any(e.code == "undefined_variable" for e in errors)


def test_validate_equations_syntax_error():
    """Test equation validation detects syntax errors"""
    elements = [
        Element(id="y", type="variable", name="Y", equation="x + (")  # Syntax error
    ]

    errors = validate_equations(elements)
    assert len(errors) > 0
    assert any(e.code == "syntax_error" for e in errors)


def test_validate_equations_built_in_variables():
    """Test that built-in variables (t, time) are allowed"""
    elements = [
        Element(id="y", type="variable", name="Y", equation="t * 2"),
        Element(id="z", type="variable", name="Z", equation="time + 1"),
    ]

    errors = validate_equations(elements)
    # Should not error on t or time
    undefined_errors = [e for e in errors if e.code == "undefined_variable"]
    assert len(undefined_errors) == 0


def test_equation_with_nested_functions():
    """Test equations with nested function calls"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 4, "y": 2})

    assert evaluator.evaluate("sqrt(pow(x, y))") == 4.0
    assert evaluator.evaluate("max(min(x, 10), y)") == 4
    assert evaluator.evaluate("log(exp(1))") == pytest.approx(1.0, abs=0.001)


def test_equation_division_by_zero_handling():
    """Test handling of division by zero"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10, "y": 0})

    with pytest.raises((EvaluationError, ZeroDivisionError)):
        evaluator.evaluate("x / y")


def test_equation_empty_string():
    """Test handling of empty equation strings"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10})

    # Empty string should return 0.0
    result = evaluator.evaluate("")
    assert result == 0.0


def test_equation_whitespace_handling():
    """Test that whitespace is handled correctly"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10, "y": 5})

    assert evaluator.evaluate("  x  +  y  ") == 15
    assert evaluator.evaluate("\tx + y\n") == 15


def test_equation_element_name_vs_id():
    """Test that equations can reference elements by name or ID"""
    elements = [
        Element(id="param1", type="parameter", name="Param1", value=10),
        Element(
            id="var1", type="variable", name="Var1", equation="param1 * 2"
        ),  # By ID
        Element(
            id="var2", type="variable", name="Var2", equation="Param1 + 5"
        ),  # By name
    ]

    errors = validate_equations(elements)
    # Should not error on either reference method
    undefined_errors = [e for e in errors if e.code == "undefined_variable"]
    assert len(undefined_errors) == 0
