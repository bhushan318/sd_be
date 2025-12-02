"""
Tests for Euler integration method
"""

import pytest
from app.models import Element, Link, SimulationConfig
from app.simulation import SystemDynamicsModel


def test_simple_exponential_growth():
    """Test simple exponential growth model"""
    elements = [
        Element(
            id="pop", type="stock", name="Population", initial=100.0, equation="growth"
        ),
        Element(id="growth", type="flow", name="Growth", equation="0.1 * Population"),
    ]
    links = [Link(id="l1", source="pop", target="growth")]
    config = SimulationConfig(
        start_time=0.0, end_time=10.0, time_step=1.0, method="euler", verbose=False
    )

    model = SystemDynamicsModel(elements, links, verbose=False)
    result = model.simulate_euler(config)

    assert result["time"][0] == 0.0
    assert result["time"][-1] == 10.0
    assert result["results"]["pop"][0] == 100.0
    # Population should grow exponentially
    assert result["results"]["pop"][-1] > result["results"]["pop"][0]


def test_population_model():
    """Test population growth model with births and deaths"""
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
        start_time=0.0, end_time=10.0, time_step=1.0, method="euler", verbose=False
    )

    model = SystemDynamicsModel(elements, links, verbose=False)
    result = model.simulate_euler(config)

    assert len(result["time"]) == 11  # 0 to 10 inclusive
    assert result["results"]["pop"][0] == 1000.0
    # Net growth rate is 0.02, so population should increase
    assert result["results"]["pop"][-1] > result["results"]["pop"][0]


def test_multiple_stocks():
    """Test model with multiple stocks"""
    elements = [
        Element(id="s1", type="stock", name="Stock1", initial=100.0, equation="f1"),
        Element(id="s2", type="stock", name="Stock2", initial=50.0, equation="f2"),
        Element(id="f1", type="flow", name="Flow1", equation="0.1 * Stock1"),
        Element(id="f2", type="flow", name="Flow2", equation="0.2 * Stock2"),
    ]
    links = [
        Link(id="l1", source="s1", target="f1"),
        Link(id="l2", source="s2", target="f2"),
    ]
    config = SimulationConfig(
        start_time=0.0, end_time=5.0, time_step=0.5, method="euler", verbose=False
    )

    model = SystemDynamicsModel(elements, links, verbose=False)
    result = model.simulate_euler(config)

    assert len(result["time"]) == 11
    assert "s1" in result["results"]
    assert "s2" in result["results"]


def test_variables_and_parameters():
    """Test model with variables and parameters"""
    elements = [
        Element(id="stock", type="stock", name="Stock", initial=50.0, equation="rate"),
        Element(id="rate", type="flow", name="Rate", equation="k * sqrt(Stock)"),
        Element(id="k", type="parameter", name="k", value=0.5),
        Element(id="multiplier", type="variable", name="Multiplier", equation="k * 2"),
    ]
    links = [
        Link(id="l1", source="k", target="rate"),
        Link(id="l2", source="stock", target="rate"),
    ]
    config = SimulationConfig(
        start_time=0.0, end_time=10.0, time_step=1.0, method="euler", verbose=False
    )

    model = SystemDynamicsModel(elements, links, verbose=False)
    result = model.simulate_euler(config)

    assert "stock" in result["results"]
    assert "rate" in result["results"]
    assert "multiplier" in result["results"]
    assert result["results"]["multiplier"][0] == 1.0  # k * 2 = 0.5 * 2 = 1.0


def test_circular_dependency_raises_error():
    """Test that circular dependencies raise clear error in simulation"""
    from app.exceptions import SimulationError

    elements = [
        Element(id="v1", type="variable", name="Var1", equation="v2 + 1"),
        Element(id="v2", type="variable", name="Var2", equation="v1 + 1"),
        Element(id="stock", type="stock", name="Stock", initial=100.0, equation="v1"),
    ]
    links = []

    with pytest.raises(SimulationError) as exc_info:
        SystemDynamicsModel(elements, links, verbose=False)

    # Verify error message is clear
    assert exc_info.value.code == "circular_dependency"
    assert "circular dependency" in exc_info.value.message.lower()
    assert (
        "v1" in exc_info.value.message
        or "v2" in exc_info.value.message
        or "Var1" in exc_info.value.message
        or "Var2" in exc_info.value.message
    )
