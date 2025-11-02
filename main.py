"""
System Dynamics Simulation Backend
FastAPI-based simulation engine for system dynamics models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.integrate import odeint
import ast
import operator
import math

app = FastAPI(title="System Dynamics Simulation API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Models
# ============================================================================

class Element(BaseModel):
    id: str
    type: str  # 'stock', 'flow', 'parameter', 'variable'
    name: str
    initial: Optional[float] = 0.0
    equation: Optional[str] = ""
    value: Optional[float] = None

class Link(BaseModel):
    id: str
    source: str
    target: str

class SimulationConfig(BaseModel):
    start_time: float = 0.0
    end_time: float = 100.0
    time_step: float = 1.0
    method: str = "euler"  # 'euler' or 'rk4'
    verbose: bool = True  # Enable detailed logging

class SimulationRequest(BaseModel):
    elements: List[Element]
    links: List[Link]
    config: SimulationConfig

class SimulationResponse(BaseModel):
    success: bool
    time: List[float]
    results: Dict[str, List[float]]
    error: Optional[str] = None

# ============================================================================
# Safe Equation Evaluator
# ============================================================================

class SafeEquationEvaluator:
    """Safely evaluate mathematical equations with controlled namespace"""
    
    # Allowed operations
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Allowed functions
    SAFE_FUNCTIONS = {
        'abs': abs,
        'min': min,
        'max': max,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'pow': pow,
    }
    
    def __init__(self):
        self.variables = {}
    
    def set_variables(self, variables: Dict[str, float]):
        """Set available variables for evaluation"""
        self.variables = variables.copy()
    
    def parse_equation(self, equation: str) -> ast.Expression:
        """Parse equation string into AST"""
        try:
            return ast.parse(equation, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Syntax error in equation: {equation}") from e
    
    def eval_node(self, node):
        """Recursively evaluate AST node"""
        if isinstance(node, ast.Expression):
            return self.eval_node(node.body)
        
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        
        elif isinstance(node, ast.Num):  # Python 3.7
            return node.n
        
        elif isinstance(node, ast.Name):
            var_name = node.id
            if var_name in self.variables:
                return self.variables[var_name]
            else:
                raise ValueError(f"Undefined variable: {var_name}")
        
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = self.eval_node(node.left)
            right = self.eval_node(node.right)
            return self.SAFE_OPERATORS[op_type](left, right)
        
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self.eval_node(node.operand)
            return self.SAFE_OPERATORS[op_type](operand)
        
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self.SAFE_FUNCTIONS:
                raise ValueError(f"Function not allowed: {func_name}")
            args = [self.eval_node(arg) for arg in node.args]
            return self.SAFE_FUNCTIONS[func_name](*args)
        
        elif isinstance(node, ast.IfExp):
            # Handle ternary operator: value_if_true if condition else value_if_false
            condition = self.eval_node(node.test)
            if condition:
                return self.eval_node(node.body)
            else:
                return self.eval_node(node.orelse)
        
        elif isinstance(node, ast.Compare):
            # Handle comparisons
            left = self.eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = self.eval_node(comparator)
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
                    raise ValueError(f"Unsupported comparison: {type(op).__name__}")
                
                if not result:
                    return False
                left = right
            return True
        
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")
    
    def evaluate(self, equation: str) -> float:
        """Evaluate equation string with current variables"""
        if not equation or equation.strip() == "":
            return 0.0
        
        try:
            tree = self.parse_equation(equation)
            result = self.eval_node(tree)
            return float(result)
        except Exception as e:
            raise ValueError(f"Error evaluating '{equation}': {str(e)}")

# ============================================================================
# Simulation Engine
# ============================================================================

class SystemDynamicsModel:
    """System dynamics model executor"""
    
    def __init__(self, elements: List[Element], links: List[Link], verbose: bool = True):
        self.elements = {e.id: e for e in elements}
        self.links = links
        self.evaluator = SafeEquationEvaluator()
        self.verbose = verbose
        
        # Categorize elements
        self.stocks = {e.id: e for e in elements if e.type == 'stock'}
        self.flows = {e.id: e for e in elements if e.type == 'flow'}
        self.parameters = {e.id: e for e in elements if e.type == 'parameter'}
        self.variables = {e.id: e for e in elements if e.type == 'variable'}
        
        if self.verbose:
            print("\n" + "="*60)
            print("MODEL INITIALIZATION")
            print("="*60)
            print(f"Stocks: {len(self.stocks)}")
            for stock_id, stock in self.stocks.items():
                print(f"  - {stock.name} (ID: {stock_id})")
                print(f"    Initial: {stock.initial}")
                print(f"    Equation: '{stock.equation}'")
            
            print(f"\nFlows: {len(self.flows)}")
            for flow_id, flow in self.flows.items():
                print(f"  - {flow.name} (ID: {flow_id})")
                print(f"    Equation: '{flow.equation}'")
            
            print(f"\nParameters: {len(self.parameters)}")
            for param_id, param in self.parameters.items():
                print(f"  - {param.name} (ID: {param_id})")
                print(f"    Value: {param.value}")
            
            print(f"\nVariables: {len(self.variables)}")
            for var_id, var in self.variables.items():
                print(f"  - {var.name} (ID: {var_id})")
                print(f"    Equation: '{var.equation}'")
            
            print(f"\nConnections: {len(self.links)}")
            for link in self.links:
                source_name = self.elements[link.source].name if link.source in self.elements else link.source
                target_name = self.elements[link.target].name if link.target in self.elements else link.target
                print(f"  - {source_name} → {target_name}")
            print("="*60 + "\n")
        
        self._validate_model()
        self._build_dependency_graph()
    
    def _validate_model(self):
        """Validate model structure"""
        if not self.stocks:
            raise ValueError("Model must have at least one stock")
        
        # Check all equations reference valid elements
        for elem in self.elements.values():
            if elem.equation:
                # This is a basic check - detailed validation happens during evaluation
                pass
    
    def _build_dependency_graph(self):
        """Build dependency graph from links"""
        self.dependencies = {elem_id: [] for elem_id in self.elements}
        for link in self.links:
            if link.target in self.dependencies:
                self.dependencies[link.target].append(link.source)
    
    def _compute_state_variables(self, stock_values: Dict[str, float], t: float) -> Dict[str, float]:
        """Compute all variables, parameters, and flows given current stock values"""
        state = {}
        
        # Add time
        state['t'] = t
        state['time'] = t
        
        # Add stock values
        for stock_id, value in stock_values.items():
            state[stock_id] = value
            if stock_id in self.elements:
                state[self.elements[stock_id].name] = value
        
        # Add parameters (constant values)
        for param_id, param in self.parameters.items():
            value = param.value if param.value is not None else param.initial
            state[param_id] = value
            state[param.name] = value
        
        self.evaluator.set_variables(state)
        
        # Compute variables (may depend on stocks and parameters)
        for var_id, var in self.variables.items():
            try:
                value = self.evaluator.evaluate(var.equation)
                state[var_id] = value
                state[var.name] = value
                self.evaluator.set_variables(state)
            except Exception as e:
                raise ValueError(f"Error computing variable '{var.name}': {str(e)}")
        
        # Compute flows (may depend on stocks, parameters, and variables)
        for flow_id, flow in self.flows.items():
            try:
                value = self.evaluator.evaluate(flow.equation)
                state[flow_id] = value
                state[flow.name] = value
                self.evaluator.set_variables(state)
            except Exception as e:
                raise ValueError(f"Error computing flow '{flow.name}': {str(e)}")
        
        return state
    
    def compute_derivatives(self, stock_values: Dict[str, float], t: float) -> Dict[str, float]:
        """Compute derivatives (rates of change) for all stocks"""
        state = self._compute_state_variables(stock_values, t)
        
        derivatives = {}
        self.evaluator.set_variables(state)
        
        for stock_id, stock in self.stocks.items():
            try:
                # Evaluate stock equation (typically sum of inflows - outflows)
                deriv = self.evaluator.evaluate(stock.equation)
                derivatives[stock_id] = deriv
                
                if self.verbose and t == 0:  # Print only at t=0 to avoid spam
                    print(f"  Stock '{stock.name}' derivative calculation:")
                    print(f"    Equation: '{stock.equation}'")
                    print(f"    Current value: {stock_values[stock_id]:.4f}")
                    print(f"    Derivative (rate of change): {deriv:.4f}")
                    
            except Exception as e:
                raise ValueError(f"Error computing derivative for stock '{stock.name}': {str(e)}")
        
        return derivatives
    
    def simulate_euler(self, config: SimulationConfig) -> Dict[str, Any]:
        """Run simulation using Euler method"""
        # Time points
        t = np.arange(config.start_time, config.end_time + config.time_step, config.time_step)
        n_steps = len(t)
        
        if self.verbose:
            print("\n" + "="*60)
            print("SIMULATION START")
            print("="*60)
            print(f"Time range: {config.start_time} to {config.end_time}")
            print(f"Time step: {config.time_step}")
            print(f"Number of steps: {n_steps}")
            print(f"Integration method: {config.method}")
            print("="*60 + "\n")
        
        # Initialize results
        results = {stock_id: np.zeros(n_steps) for stock_id in self.stocks}
        results.update({flow_id: np.zeros(n_steps) for flow_id in self.flows})
        results.update({var_id: np.zeros(n_steps) for var_id in self.variables})
        results.update({param_id: np.zeros(n_steps) for param_id in self.parameters})
        
        # Set initial values for stocks
        stock_values = {stock_id: stock.initial for stock_id, stock in self.stocks.items()}
        
        if self.verbose:
            print("INITIAL CONDITIONS:")
            for stock_id, stock in self.stocks.items():
                print(f"  {stock.name}: {stock_values[stock_id]:.4f}")
            print()
        
        # Simulation loop
        for i, time in enumerate(t):
            # Compute current state
            state = self._compute_state_variables(stock_values, time)
            
            # Store results
            for stock_id in self.stocks:
                results[stock_id][i] = stock_values[stock_id]
            for flow_id in self.flows:
                results[flow_id][i] = state.get(flow_id, 0)
            for var_id in self.variables:
                results[var_id][i] = state.get(var_id, 0)
            for param_id in self.parameters:
                results[param_id][i] = state.get(param_id, 0)
            
            # Verbose output for first few steps
            if self.verbose and i < 3:
                print(f"TIME STEP {i} (t = {time:.2f}):")
                print("  Stock values:")
                for stock_id, stock in self.stocks.items():
                    print(f"    {stock.name}: {stock_values[stock_id]:.4f}")
                
                print("  Flow values:")
                for flow_id, flow in self.flows.items():
                    print(f"    {flow.name}: {state.get(flow_id, 0):.4f}")
                    print(f"      (Equation: '{flow.equation}')")
            
            # Compute derivatives
            if i < n_steps - 1:  # Don't update on last step
                if self.verbose and i == 0:
                    print("\n  Computing derivatives:")
                
                derivatives = self.compute_derivatives(stock_values, time)
                
                if self.verbose and i < 3:
                    print("  Derivatives (rates of change):")
                    for stock_id, stock in self.stocks.items():
                        print(f"    d({stock.name})/dt = {derivatives[stock_id]:.4f}")
                
                # Update stocks using Euler method
                for stock_id in self.stocks:
                    old_value = stock_values[stock_id]
                    stock_values[stock_id] += derivatives[stock_id] * config.time_step
                    
                    if self.verbose and i < 3:
                        print(f"  Updating {self.stocks[stock_id].name}:")
                        print(f"    Old value: {old_value:.4f}")
                        print(f"    Change: {derivatives[stock_id] * config.time_step:.4f}")
                        print(f"    New value: {stock_values[stock_id]:.4f}")
                
                if self.verbose and i < 3:
                    print()
        
        if self.verbose:
            print("="*60)
            print("SIMULATION COMPLETE")
            print("="*60)
            print("\nFINAL VALUES:")
            for stock_id, stock in self.stocks.items():
                print(f"  {stock.name}: {stock_values[stock_id]:.4f}")
            print("\n")
        
        return {
            'time': t.tolist(),
            'results': {elem_id: vals.tolist() for elem_id, vals in results.items()}
        }

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "System Dynamics Simulation API",
        "version": "1.0.0",
        "endpoints": {
            "simulate": "/simulate",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/simulate", response_model=SimulationResponse)
def simulate(request: SimulationRequest):
    """
    Run system dynamics simulation
    
    Request body should contain:
    - elements: List of model elements (stocks, flows, parameters, variables)
    - links: List of connections between elements
    - config: Simulation configuration (time range, step size, method)
    
    Returns time series data for all elements
    """
    try:
        # Create model with verbose logging
        model = SystemDynamicsModel(request.elements, request.links, verbose=request.config.verbose)
        
        # Run simulation
        if request.config.method == "euler":
            result = model.simulate_euler(request.config)
        else:
            raise ValueError(f"Unsupported integration method: {request.config.method}")
        
        return SimulationResponse(
            success=True,
            time=result['time'],
            results=result['results']
        )
    
    except Exception as e:
        print(f"\n❌ SIMULATION ERROR: {str(e)}\n")
        return SimulationResponse(
            success=False,
            time=[],
            results={},
            error=str(e)
        )

@app.post("/validate")
def validate_model(request: SimulationRequest):
    """
    Validate a model without running simulation
    """
    try:
        verbose = request.config.verbose if hasattr(request.config, 'verbose') else False
        model = SystemDynamicsModel(request.elements, request.links, verbose=verbose)
        return {
            "valid": True,
            "message": "Model is valid",
            "stocks": len(model.stocks),
            "flows": len(model.flows),
            "parameters": len(model.parameters),
            "variables": len(model.variables)
        }
    except Exception as e:
        print(f"\n❌ VALIDATION ERROR: {str(e)}\n")
        return {
            "valid": False,
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)