"""
Tests for validation module
"""

from app.models import Element, Link, SimulationConfig
from app.validation import (
    validate_model,
    validate_simulation_config,
    validate_elements,
    validate_links,
    validate_equations,
    detect_circular_dependencies,
)


def test_validate_simulation_config_valid():
    """Test valid simulation config"""
    config = SimulationConfig(
        start_time=0.0, end_time=100.0, time_step=1.0, method="euler"
    )
    errors = validate_simulation_config(config)
    assert len(errors) == 0


def test_validate_simulation_config_invalid_time_step():
    """Test invalid time step"""
    # Use model_construct to bypass Pydantic validation for testing
    config = SimulationConfig.model_construct(
        start_time=0.0, end_time=100.0, time_step=0.0, method="euler"
    )
    errors = validate_simulation_config(config)
    assert len(errors) > 0
    assert any(e.code == "invalid_time_step" for e in errors)


def test_validate_simulation_config_invalid_time_range():
    """Test invalid time range"""
    config = SimulationConfig(
        start_time=100.0, end_time=0.0, time_step=1.0, method="euler"
    )
    errors = validate_simulation_config(config)
    assert len(errors) > 0
    assert any(e.code == "invalid_time_range" for e in errors)


def test_validate_simulation_config_invalid_method():
    """Test invalid integration method"""
    config = SimulationConfig(
        start_time=0.0, end_time=100.0, time_step=1.0, method="invalid"
    )
    errors = validate_simulation_config(config)
    assert len(errors) > 0
    assert any(e.code == "invalid_method" for e in errors)


def test_validate_elements_valid():
    """Test valid elements"""
    elements = [
        Element(id="s1", type="stock", name="Stock1", initial=100.0),
        Element(id="f1", type="flow", name="Flow1", equation="10"),
        Element(id="p1", type="parameter", name="Param1", value=5.0),
        Element(id="v1", type="variable", name="Var1", equation="p1 * 2"),
    ]
    errors = validate_elements(elements)
    assert len(errors) == 0


def test_validate_elements_missing_initial():
    """Test stock without initial value"""
    # Use model_construct to bypass Pydantic default for testing
    elements = [
        Element.model_construct(id="s1", type="stock", name="Stock1", initial=None)
    ]
    errors = validate_elements(elements)
    assert len(errors) > 0
    assert any(e.code == "missing_initial_value" for e in errors)


def test_validate_elements_missing_equation():
    """Test flow without equation"""
    elements = [Element(id="f1", type="flow", name="Flow1")]
    errors = validate_elements(elements)
    assert len(errors) > 0
    assert any(e.code == "missing_equation" for e in errors)


def test_validate_links_valid():
    """Test valid links"""
    elements = [
        Element(id="s1", type="stock", name="Stock1", initial=100.0),
        Element(id="f1", type="flow", name="Flow1", equation="10"),
    ]
    links = [Link(id="l1", source="s1", target="f1")]
    errors = validate_links(links, elements)
    assert len(errors) == 0


def test_validate_links_invalid_source():
    """Test link with invalid source"""
    elements = [Element(id="s1", type="stock", name="Stock1", initial=100.0)]
    links = [Link(id="l1", source="nonexistent", target="s1")]
    errors = validate_links(links, elements)
    assert len(errors) > 0
    assert any(e.code == "invalid_link_source" for e in errors)


def test_validate_equations_syntax_error():
    """Test equation with syntax error"""
    elements = [
        Element(id="v1", type="variable", name="Var1", equation="invalid syntax ((")
    ]
    errors = validate_equations(elements)
    assert len(errors) > 0
    assert any(e.code == "syntax_error" for e in errors)


def test_validate_equations_undefined_variable():
    """Test equation with undefined variable"""
    elements = [
        Element(id="v1", type="variable", name="Var1", equation="nonexistent * 2")
    ]
    errors = validate_equations(elements)
    assert len(errors) > 0
    assert any(e.code == "undefined_variable" for e in errors)


def test_detect_circular_dependencies():
    """Test circular dependency detection"""
    elements = [
        Element(id="v1", type="variable", name="Var1", equation="v2"),
        Element(id="v2", type="variable", name="Var2", equation="v1"),
    ]
    links = []
    errors = detect_circular_dependencies(elements, links)
    assert len(errors) > 0
    assert any(e.code == "circular_dependency" for e in errors)


def test_validate_model_complete():
    """Test complete model validation"""
    elements = [
        Element(
            id="pop",
            type="stock",
            name="Population",
            initial=1000.0,
            equation="births - deaths",
        ),
        Element(
            id="births", type="flow", name="Births", equation="birth_rate * Population"
        ),
        Element(
            id="deaths", type="flow", name="Deaths", equation="death_rate * Population"
        ),
        Element(id="birth_rate", type="parameter", name="Birth Rate", value=0.03),
        Element(id="death_rate", type="parameter", name="Death Rate", value=0.01),
    ]
    links = [
        Link(id="l1", source="birth_rate", target="births"),
        Link(id="l2", source="pop", target="births"),
        Link(id="l3", source="death_rate", target="deaths"),
        Link(id="l4", source="pop", target="deaths"),
    ]
    config = SimulationConfig(
        start_time=0.0, end_time=10.0, time_step=1.0, method="euler"
    )

    result = validate_model(elements, links, config)
    assert result.valid
