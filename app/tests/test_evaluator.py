"""
Tests for equation evaluator
"""

import pytest
from app.evaluator import (
    SafeEquationEvaluator,
    extract_variable_references,
    build_dependency_graph,
    topological_sort,
)
from app.models import Element, Link
import ast


def test_evaluator_basic_arithmetic():
    """Test basic arithmetic operations"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10, "y": 5})

    assert evaluator.evaluate("x + y") == 15
    assert evaluator.evaluate("x - y") == 5
    assert evaluator.evaluate("x * y") == 50
    assert evaluator.evaluate("x / y") == 2.0
    assert evaluator.evaluate("x ** 2") == 100


def test_evaluator_functions():
    """Test mathematical functions"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 4})

    assert evaluator.evaluate("sqrt(x)") == 2.0
    assert evaluator.evaluate("abs(-x)") == 4
    assert evaluator.evaluate("min(x, 10)") == 4
    assert evaluator.evaluate("max(x, 10)") == 10


def test_evaluator_undefined_variable():
    """Test error handling for undefined variables"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10})

    with pytest.raises(Exception):  # EvaluationError
        evaluator.evaluate("y + 5")


def test_evaluator_ternary():
    """Test ternary conditional expressions"""
    evaluator = SafeEquationEvaluator()
    evaluator.set_variables({"x": 10})

    assert evaluator.evaluate("x if x > 5 else 0") == 10
    assert evaluator.evaluate("x if x < 5 else 0") == 0


def test_extract_variable_references():
    """Test variable reference extraction"""
    tree = ast.parse("x + y * z", mode="eval")
    vars = extract_variable_references(tree)
    assert vars == {"x", "y", "z"}


def test_build_dependency_graph():
    """Test dependency graph building"""
    elements = [
        Element(id="v1", type="variable", name="Var1", equation="v2 + 1"),
        Element(id="v2", type="variable", name="Var2", equation="p1"),
        Element(id="p1", type="parameter", name="Param1", value=5.0),
    ]
    links = [Link(id="l1", source="p1", target="v2")]

    deps = build_dependency_graph(elements, links)
    assert "v2" in deps["v1"]
    assert "p1" in deps["v2"]


def test_topological_sort():
    """Test topological sorting"""
    dependencies = {"a": [], "b": ["a"], "c": ["b"], "d": ["a"]}

    order = topological_sort(dependencies)
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_topological_sort_circular():
    """Test topological sort detects cycles"""
    dependencies = {"a": ["b"], "b": ["a"]}

    with pytest.raises(ValueError) as exc_info:
        topological_sort(dependencies)

    # Verify error message contains cycle information
    error_msg = str(exc_info.value)
    assert "Circular dependency" in error_msg
    assert "a" in error_msg or "b" in error_msg


def test_topological_sort_multilevel_dependencies():
    """Test topological sort with multilevel dependencies"""
    dependencies = {"a": [], "b": ["a"], "c": ["b"], "d": ["c", "a"], "e": ["d"]}

    order = topological_sort(dependencies)
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")
    assert order.index("c") < order.index("d")
    assert order.index("d") < order.index("e")


def test_topological_sort_complex_cycle():
    """Test topological sort detects complex cycles"""
    dependencies = {"a": ["b"], "b": ["c"], "c": ["d"], "d": ["a"], "e": ["a"]}

    with pytest.raises(ValueError) as exc_info:
        topological_sort(dependencies)

    error_msg = str(exc_info.value)
    assert "Circular dependency" in error_msg
    # Should mention cycle elements
    assert any(elem in error_msg for elem in ["a", "b", "c", "d"])
