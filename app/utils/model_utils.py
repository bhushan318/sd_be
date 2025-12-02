"""
Utility functions for model manipulation
Provides reusable functions for common model operations
"""

from typing import List, Dict
from app.models import Element


def apply_parameter_values(
    elements: List[Element], 
    parameter_values: Dict[str, float]
) -> List[Element]:
    """
    Create modified elements with updated parameter values.
    
    This function is used when running simulations with modified parameter
    values (e.g., in optimization, calibration, or sensitivity analysis).
    
    Args:
        elements: Original elements from the model
        parameter_values: Dictionary mapping element IDs to new parameter values
        
    Returns:
        List of modified elements with updated parameter values
        
    Example:
        >>> elements = [Element(id="param1", type="parameter", name="Param 1", value=10.0)]
        >>> modified = apply_parameter_values(elements, {"param1": 20.0})
        >>> modified[0].value
        20.0
    """
    modified_elements = []
    for elem in elements:
        if elem.id in parameter_values:
            # Create modified element with new parameter value
            modified_elem = Element(
                id=elem.id,
                type=elem.type,
                name=elem.name,
                initial=elem.initial,
                equation=elem.equation,
                value=parameter_values[elem.id],
                lookup_table=elem.lookup_table,
            )
            modified_elements.append(modified_elem)
        else:
            # Keep original element unchanged
            modified_elements.append(elem)
    return modified_elements

