"""
FastAPI application and endpoints for System Dynamics Backend
Includes security features, CORS, request size limits, and error handling
"""

# Standard library imports
import os
import asyncio
import ast
import csv
import io
import json
from typing import Dict, Any, Optional, List

# Third-party imports
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError

# Local application imports
from app.config import get_settings
from app.exceptions import SimulationError, EvaluationError
from app.models import (
    SimulationRequest,
    SimulationResponse,
    ValidationResponse,
    Element,
    CompareRequest,
    CompareResponse,
    SensitivityRequest,
    SensitivityResponse,
    OptimizationRequest,
    OptimizationResponse,
    CalibrationRequest,
    CalibrationResponse,
    ObservedDataPoint,
)
from app.optimization import (
    ObjectiveFunction,
    CalibrationFunction,
    OptimizationEngine,
    compute_error_metric,
)
from app.simulation import SystemDynamicsModel
from app.validation import validate_model
from app.utils.cache import get_cache
from app.utils.logging_config import (
    setup_logging,
    get_logger,
    set_request_id,
)
from app.utils.model_utils import apply_parameter_values
from app.utils.result_store import get_result_store, SimulationResult

logger = get_logger(__name__)

# Get configuration
settings = get_settings()

# Setup logging
setup_logging(
    level=settings.log_level,
    json_format=settings.log_format_json,
    log_file=settings.log_file,
)

# Create FastAPI app
app = FastAPI(
    title="System Dynamics Simulation API",
    version="1.0.0",
    description="FastAPI-based simulation engine for system dynamics models",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration constants (from settings)
MAX_REQUEST_SIZE = settings.max_request_size
MAX_AST_DEPTH = settings.max_ast_depth
SIMULATION_TIMEOUT = settings.simulation_timeout


def check_request_size(request: Request) -> None:
    """Check if request size exceeds limit"""
    content_length = request.headers.get("content-length")
    if content_length:
        size = int(content_length)
        if size > MAX_REQUEST_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request size ({size} bytes) exceeds maximum ({MAX_REQUEST_SIZE} bytes)",
            )


def count_ast_nodes(node: ast.AST) -> int:
    """
    Count total number of nodes in AST tree
    
    Args:
        node: AST node to count
        
    Returns:
        Total number of nodes in the tree
    """
    count = 1  # Count this node
    
    # Recursively count child nodes
    for child in ast.iter_child_nodes(node):
        count += count_ast_nodes(child)
    
    return count


def check_ast_depth(node: ast.AST, current_depth: int = 0) -> int:
    """
    Check maximum depth of AST tree
    
    Validates equation complexity to prevent DoS attacks.
    
    Args:
        node: AST node to check
        current_depth: Current depth in the tree
        
    Returns:
        Maximum depth found
        
    Raises:
        HTTPException: If depth exceeds maximum
    """
    if current_depth > MAX_AST_DEPTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Equation AST depth ({current_depth}) exceeds maximum ({MAX_AST_DEPTH})",
        )

    max_depth = current_depth

    if isinstance(node, ast.Expression):
        max_depth = max(max_depth, check_ast_depth(node.body, current_depth + 1))
    elif isinstance(node, ast.BinOp):
        max_depth = max(
            max_depth,
            check_ast_depth(node.left, current_depth + 1),
            check_ast_depth(node.right, current_depth + 1),
        )
    elif isinstance(node, ast.UnaryOp):
        max_depth = max(max_depth, check_ast_depth(node.operand, current_depth + 1))
    elif isinstance(node, ast.Call):
        for arg in node.args:
            max_depth = max(max_depth, check_ast_depth(arg, current_depth + 1))
    elif isinstance(node, ast.IfExp):
        max_depth = max(
            max_depth,
            check_ast_depth(node.test, current_depth + 1),
            check_ast_depth(node.body, current_depth + 1),
            check_ast_depth(node.orelse, current_depth + 1),
        )
    elif isinstance(node, ast.Compare):
        max_depth = max(max_depth, check_ast_depth(node.left, current_depth + 1))
        for comparator in node.comparators:
            max_depth = max(max_depth, check_ast_depth(comparator, current_depth + 1))
    elif isinstance(node, ast.BoolOp):  # Handle 'and', 'or'
        for value in node.values:
            max_depth = max(max_depth, check_ast_depth(value, current_depth + 1))
    elif isinstance(node, (ast.Tuple, ast.List)):  # Handle tuples/lists in function args
        for elt in node.elts:
            max_depth = max(max_depth, check_ast_depth(elt, current_depth + 1))

    return max_depth


def validate_equation_complexity(elements: List[Element]) -> None:
    """
    Validate equation complexity (AST depth and node count)
    
    Args:
        elements: List of elements to validate
        
    Raises:
        HTTPException: If equation complexity exceeds limits
    """
    for element in elements:
        if element.equation and element.equation.strip():
            try:
                tree = ast.parse(element.equation, mode="eval")
                
                # Check AST depth
                check_ast_depth(tree)
                
                # Check total node count (DoS protection)
                node_count = count_ast_nodes(tree)
                if node_count > settings.max_ast_nodes:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"Equation for element '{element.id}' has {node_count} AST nodes, "
                            f"exceeding maximum of {settings.max_ast_nodes}"
                        ),
                    )
            except HTTPException:
                raise
            except Exception:
                # Syntax errors are handled by validation
                pass


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Middleware to add request ID to all requests"""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = set_request_id()
    else:
        set_request_id(request_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def request_size_middleware(request: Request, call_next):
    """Middleware to check request size"""
    check_request_size(request)
    response = await call_next(request)
    return response


def add_debug_info(error_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add debug information (traceback) to error dict if DEBUG mode is enabled
    
    Args:
        error_dict: Error dictionary to add debug info to
        
    Returns:
        Modified error dictionary with debug info if enabled
    """
    if settings.debug and "traceback" not in error_dict.get("details", {}):
        import traceback
        if "details" not in error_dict:
            error_dict["details"] = {}
        error_dict["details"]["traceback"] = traceback.format_exc()
    return error_dict


@app.exception_handler(SimulationError)
async def simulation_error_handler(request: Request, exc: SimulationError):
    """Handle simulation errors with structured format"""
    logger.error(f"Simulation error: {exc.message}", extra={"code": exc.code})
    error_dict = add_debug_info(exc.to_dict())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_dict,
    )


@app.exception_handler(EvaluationError)
async def evaluation_error_handler(request: Request, exc: EvaluationError):
    """Handle evaluation errors with structured format"""
    logger.error(f"Evaluation error: {exc.message}", extra={"code": exc.code})
    error_dict = add_debug_info(exc.to_dict())
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_dict,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with structured format"""
    logger.warning(f"Validation error: {exc.errors()}")

    # Format validation errors into a readable message
    error_messages = []
    for error in exc.errors():
        loc = " -> ".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Validation error")
        error_messages.append(f"{loc}: {msg}")

    error_response: Dict[str, Any] = {
        "code": "validation_error",
        "message": "; ".join(error_messages),
        "details": {
            "errors": exc.errors(),
        },
    }

    error_response = add_debug_info(error_response)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured format"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")

    error_response: Dict[str, Any] = {
        "code": f"http_{exc.status_code}",
        "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        "details": {
            "status_code": exc.status_code,
        },
    }

    error_response = add_debug_info(error_response)

    return JSONResponse(status_code=exc.status_code, content=error_response)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions with structured format"""
    logger.exception("Unhandled exception", exc_info=exc)

    error_response: Dict[str, Any] = {
        "code": "internal_error",
        "message": str(exc) if settings.debug else "An internal error occurred",
        "details": {
            "exception_type": type(exc).__name__,
        },
    }

    error_response = add_debug_info(error_response)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response,
    )


@app.get("/")
def read_root():
    """Root endpoint with API information"""
    logger.info("Root endpoint accessed")
    return {
        "message": "System Dynamics Simulation API",
        "version": "1.0.0",
        "endpoints": {
            "simulate": "/simulate",
            "simulate_ws": "/simulate/ws",
            "validate": "/validate",
            "health": "/health",
            "cache_stats": "/cache/stats",
            "export_csv": "/simulate/{id}/export/csv",
            "export_json": "/simulate/{id}/export/json",
            "compare": "/compare",
            "sensitivity": "/sensitivity",
            "optimize": "/optimize",
            "calibrate": "/calibrate",
            "templates": "/templates",
            "template": "/templates/{id}",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {"status": "healthy"}


@app.get("/cache/stats")
def cache_stats():
    """Get cache statistics"""
    logger.debug("Cache stats requested")
    cache = get_cache()
    stats = cache.get_stats()
    return stats


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    """
    Run system dynamics simulation
    
    Returns time series data for all elements.
    """
    logger.info(
        f"Simulation request received: method={request.config.method}, "
        f"time_range=[{request.config.start_time}, {request.config.end_time}], "
        f"elements={len(request.elements)}"
    )
    
    try:
        # Validate equation complexity
        validate_equation_complexity(request.elements)

        # Create model
        model = SystemDynamicsModel(
            request.elements, request.links, verbose=request.config.verbose
        )

        # Run simulation with timeout
        try:
            if request.config.method == "euler":
                result = await asyncio.wait_for(
                    asyncio.to_thread(model.simulate_euler, request.config),
                    timeout=SIMULATION_TIMEOUT,
                )
            elif request.config.method == "rk4":
                result = await asyncio.wait_for(
                    asyncio.to_thread(model.simulate_rk4, request.config),
                    timeout=SIMULATION_TIMEOUT,
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported integration method: {request.config.method}",
                )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=(
                    f"Simulation ({request.config.method}) exceeded timeout of "
                    f"{SIMULATION_TIMEOUT} seconds. Consider reducing time range or "
                    f"increasing time_step."
                ),
            )

        logger.info(f"Simulation completed: {len(result['time'])} time points")

        # Store result for export
        result_store = get_result_store()
        stored_result = SimulationResult(
            time=result["time"],
            results=result["results"]
        )
        result_id = result_store.store(stored_result)
        logger.debug(f"Stored simulation result with ID: {result_id}")

        return SimulationResponse(
            success=True,
            time=result["time"],
            results=result["results"],
            result_id=result_id,
        )

    except HTTPException:
        # Re-raise HTTP exceptions (timeout, invalid method, etc.)
        raise
    except SimulationError:
        # Re-raise SimulationError - it will be handled by the exception handler
        raise
    except Exception as e:
        # Convert unexpected exceptions to SimulationError for consistent handling
        logger.exception("Unexpected error in simulation")
        from app.exceptions import SimulationError as SimError
        raise SimError(
            code="internal_error",
            message=f"Unexpected error during simulation: {str(e)}",
            details={"exception_type": type(e).__name__},
        ) from e


@app.post("/validate", response_model=ValidationResponse)
def validate_model_endpoint(request: SimulationRequest):
    """
    Validate a model without running simulation
    
    Returns structured validation results with errors and warnings.
    """
    logger.info(
        f"Validation request received: elements={len(request.elements)}, "
        f"links={len(request.links)}"
    )
    
    try:
        # Validate equation complexity
        validate_equation_complexity(request.elements)

        result = validate_model(request.elements, request.links, request.config)

        # Add summary if valid
        summary = None
        if result.valid:
            stocks = [e for e in request.elements if e.type == "stock"]
            flows = [e for e in request.elements if e.type == "flow"]
            parameters = [e for e in request.elements if e.type == "parameter"]
            variables = [e for e in request.elements if e.type == "variable"]
            events = [e for e in request.elements if e.type == "event"]

            summary = {
                "stocks": len(stocks),
                "flows": len(flows),
                "parameters": len(parameters),
                "variables": len(variables),
                "events": len(events),
                "links": len(request.links),
            }

        if result.valid:
            logger.info(f"Validation passed: {summary}")
        else:
            logger.warning(f"Validation failed: {len(result.errors)} errors found")

        # Handle warnings if present in result
        warnings = getattr(result, 'warnings', [])

        return ValidationResponse(
            valid=result.valid,
            errors=result.errors,
            warnings=warnings,
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in validation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation error: {str(e)}",
        )


def _generate_csv(data: Dict[str, Any]) -> str:
    """Generate CSV string from simulation results"""
    time_data = data["time"]
    results = data["results"]

    output = io.StringIO()
    # Use newline='' to prevent extra line endings on Windows
    # lineterminator='\n' ensures consistent line endings across platforms
    writer = csv.writer(output, lineterminator='\n')

    # Write header
    header = ["time"] + sorted(results.keys())
    writer.writerow(header)

    # Write data rows
    n_points = len(time_data)
    for i in range(n_points):
        row = [time_data[i]] + [results[elem_id][i] for elem_id in sorted(results.keys())]
        writer.writerow(row)

    # Get value and strip any trailing whitespace/newlines for clean output
    csv_content = output.getvalue()
    return csv_content.rstrip('\r\n')


@app.get("/simulate/{result_id}/export/csv")
def export_csv(
    result_id: str,
    element: Optional[List[str]] = Query(None, description="Filter by element IDs"),
):
    """Export simulation results as CSV"""
    logger.info(f"CSV export requested for result: {result_id}")

    result_store = get_result_store()
    result = result_store.get(result_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation result not found or expired: {result_id}",
        )

    filtered_data = result.filter_elements(element)
    csv_content = _generate_csv(filtered_data)

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=simulation_{result_id}.csv"},
    )


@app.get("/simulate/{result_id}/export/json")
def export_json(
    result_id: str,
    element: Optional[List[str]] = Query(None, description="Filter by element IDs"),
):
    """Export simulation results as JSON"""
    logger.info(f"JSON export requested for result: {result_id}")

    result_store = get_result_store()
    result = result_store.get(result_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation result not found or expired: {result_id}",
        )

    filtered_data = result.filter_elements(element)
    return JSONResponse(content=filtered_data)


@app.websocket("/simulate/ws")
async def simulate_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for simulation with progress tracking
    
    Protocol:
    - Client sends: SimulationRequest as JSON
    - Server sends: { "type": "progress", "progress": 0.0-1.0 }
    - Server sends: { "type": "result", "data": SimulationResponse }
    - Server sends: { "type": "error", "message": str }
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        # Receive simulation request
        data = await websocket.receive_json()
        logger.info("Received simulation request via WebSocket")

        # Parse request
        try:
            request = SimulationRequest(**data)
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Invalid request format: {str(e)}"
            })
            await websocket.close()
            return

        # Validate equation complexity
        try:
            validate_equation_complexity(request.elements)
        except HTTPException as e:
            await websocket.send_json({"type": "error", "message": e.detail})
            await websocket.close()
            return

        # Create model
        model = SystemDynamicsModel(
            request.elements, request.links, verbose=request.config.verbose
        )

        # Progress tracking
        import threading
        progress_list: List[float] = []
        progress_lock = threading.Lock()

        def progress_callback(progress: float):
            # Called from thread context, so use threading.Lock
            with progress_lock:
                progress_list.append(round(progress, 4))

        # Task to send progress updates
        async def send_progress():
            last_sent = -1
            while True:
                try:
                    await asyncio.sleep(settings.progress_update_interval)
                    with progress_lock:
                        if len(progress_list) > last_sent + 1:
                            latest = progress_list[-1]
                            last_sent = len(progress_list) - 1
                            await websocket.send_json({
                                "type": "progress",
                                "progress": latest
                            })
                except Exception:
                    break

        progress_task = asyncio.create_task(send_progress())

        # Run simulation
        try:
            if request.config.method == "euler":
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        model.simulate_euler, request.config, progress_callback
                    ),
                    timeout=SIMULATION_TIMEOUT,
                )
            elif request.config.method == "rk4":
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        model.simulate_rk4, request.config, progress_callback
                    ),
                    timeout=SIMULATION_TIMEOUT,
                )
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unsupported method: {request.config.method}"
                })
                await websocket.close()
                return

            logger.info(f"Simulation completed via WebSocket: {len(result['time'])} points")

            # Store result
            result_store = get_result_store()
            stored_result = SimulationResult(
                time=result["time"],
                results=result["results"]
            )
            result_id = result_store.store(stored_result)

            # Cancel progress task
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

            # Send final result
            response = SimulationResponse(
                success=True,
                time=result["time"],
                results=result["results"],
                result_id=result_id,
            )
            await websocket.send_json({
                "type": "result",
                "data": response.model_dump()
            })

        except asyncio.TimeoutError:
            progress_task.cancel()
            await websocket.send_json({
                "type": "error",
                "message": (
                    f"Simulation ({request.config.method}) exceeded timeout of "
                    f"{SIMULATION_TIMEOUT} seconds. Consider reducing time range or "
                    f"increasing time_step."
                )
            })
        except SimulationError as e:
            progress_task.cancel()
            logger.error(f"Simulation error via WebSocket: {e.message}")
            await websocket.send_json({"type": "error", "message": e.message})
        except Exception as e:
            progress_task.cancel()
            logger.exception("Unexpected error in WebSocket simulation")
            await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("Error in WebSocket handler")
        try:
            await websocket.send_json({"type": "error", "message": f"Internal error: {str(e)}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/compare", response_model=CompareResponse)
async def compare_simulations(request: CompareRequest):
    """Compare two simulation results"""
    logger.info(f"Comparison requested: {request.result_id_1} vs {request.result_id_2}")

    result_store = get_result_store()
    result1 = result_store.get(request.result_id_1)
    result2 = result_store.get(request.result_id_2)

    if result1 is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation result not found: {request.result_id_1}",
        )

    if result2 is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation result not found: {request.result_id_2}",
        )

    # Filter elements if specified
    result1_data = result1.filter_elements(request.elements)
    result2_data = result2.filter_elements(request.elements)

    # Ensure time arrays match
    if len(result1_data["time"]) != len(result2_data["time"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Results have different time lengths",
        )

    # Compute differences
    differences: Dict[str, Dict[str, float]] = {}
    all_elements = set(result1_data["results"].keys()) | set(result2_data["results"].keys())

    for elem_id in all_elements:
        if elem_id not in result1_data["results"] or elem_id not in result2_data["results"]:
            continue

        values1 = result1_data["results"][elem_id]
        values2 = result2_data["results"][elem_id]

        if len(values1) != len(values2):
            continue

        arr1 = np.array(values1)
        arr2 = np.array(values2)
        diff = arr2 - arr1
        abs_diff = np.abs(diff)
        
        # Handle relative difference with zero check
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.where(arr1 != 0, diff / arr1 * 100, np.nan)

        valid_rel = rel_diff[~np.isnan(rel_diff)]
        
        # Check if all values are zero (edge case)
        all_zero = np.all(arr1 == 0) and np.all(arr2 == 0)
        if all_zero:
            # Both series are zero - no meaningful difference
            mean_rel_diff = 0.0
            max_rel_diff = 0.0
        elif len(valid_rel) == 0:
            # No valid relative differences (all zeros in arr1)
            mean_rel_diff = float("inf") if np.any(arr2 != 0) else 0.0
            max_rel_diff = float("inf") if np.any(arr2 != 0) else 0.0
        else:
            mean_rel_diff = float(np.nanmean(valid_rel))
            max_rel_diff = float(np.nanmax(np.abs(valid_rel)))

        differences[elem_id] = {
            "mean_difference": float(np.nanmean(diff)),
            "max_difference": float(np.nanmax(abs_diff)),
            "mean_absolute_difference": float(np.nanmean(abs_diff)),
            "mean_relative_difference_percent": mean_rel_diff,
            "max_relative_difference_percent": max_rel_diff,
            "rms_difference": float(np.sqrt(np.nanmean(diff ** 2))),
        }

    summary = {
        "elements_compared": len(differences),
        "time_points": len(result1_data["time"]),
        "mean_absolute_difference_all": float(np.nanmean([d["mean_absolute_difference"] for d in differences.values()])) if differences else 0.0,
        "max_difference_all": float(np.nanmax([d["max_difference"] for d in differences.values()])) if differences else 0.0,
    }

    logger.info(f"Comparison completed: {len(differences)} elements compared")
    return CompareResponse(differences=differences, summary=summary)


@app.post("/sensitivity", response_model=SensitivityResponse)
async def sensitivity_analysis(request: SensitivityRequest):
    """Perform sensitivity analysis by varying parameters"""
    logger.info(
        f"Sensitivity analysis: {len(request.parameters)} parameters, "
        f"{sum(len(p.values) for p in request.parameters)} variations"
    )

    validate_equation_complexity(request.elements)

    # Validate parameters exist
    element_ids = {e.id for e in request.elements}
    for param in request.parameters:
        if param.element_id not in element_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Parameter element not found: {param.element_id}",
            )

    # Generate parameter combinations
    import itertools
    param_keys = [p.element_id for p in request.parameters]
    param_value_lists = [p.values for p in request.parameters]
    param_combinations = [
        dict(zip(param_keys, combo))
        for combo in itertools.product(*param_value_lists)
    ]

    logger.info(f"Running {len(param_combinations)} simulation variations")

    # Run simulations
    runs: List[Dict[str, Any]] = []
    result_store = get_result_store()

    for idx, param_values in enumerate(param_combinations):
        # Create modified elements with parameter values
        modified_elements = apply_parameter_values(request.elements, param_values)

        # Run simulation
        try:
            model = SystemDynamicsModel(modified_elements, request.links, verbose=False)
            
            if request.config.method == "euler":
                result = await asyncio.wait_for(
                    asyncio.to_thread(model.simulate_euler, request.config),
                    timeout=SIMULATION_TIMEOUT,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(model.simulate_rk4, request.config),
                    timeout=SIMULATION_TIMEOUT,
                )

            stored_result = SimulationResult(
                time=result["time"],
                results=result["results"]
            )
            result_id = result_store.store(stored_result)

            runs.append({
                "result_id": result_id,
                "parameter_values": param_values,
                "time": result["time"],
                "results": result["results"],
            })

        except Exception as e:
            logger.error(f"Simulation failed for variation {param_values}: {str(e)}")
            runs.append({"parameter_values": param_values, "error": str(e)})

    # Compute sensitivity metrics
    metrics: Dict[str, Dict[str, float]] = {}
    successful_runs = [r for r in runs if "error" not in r]

    if len(successful_runs) > 1 and request.metrics:
        for metric_elem_id in request.metrics:
            if metric_elem_id not in element_ids:
                continue

            metric_values = []
            for run in successful_runs:
                if metric_elem_id in run.get("results", {}):
                    values = run["results"][metric_elem_id]
                    metric_values.append(float(np.mean(values)))

            if len(metric_values) > 1:
                metric_array = np.array(metric_values)
                mean_val = float(np.mean(metric_array))
                metrics[metric_elem_id] = {
                    "mean": mean_val,
                    "std": float(np.std(metric_array)),
                    "min": float(np.min(metric_array)),
                    "max": float(np.max(metric_array)),
                    "range": float(np.max(metric_array) - np.min(metric_array)),
                    "coefficient_of_variation": float(np.std(metric_array) / mean_val) if mean_val != 0 else 0.0,
                }

    logger.info(f"Sensitivity analysis completed: {len(successful_runs)}/{len(runs)} successful")

    return SensitivityResponse(
        runs=runs,
        metrics=metrics,
        parameter_values=[r["parameter_values"] for r in runs],
    )


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_parameters(request: OptimizationRequest):
    """Optimize model parameters"""
    logger.info(
        f"Optimization: {len(request.parameters)} parameters, "
        f"objective={request.objective}, type={request.objective_type}"
    )

    validate_equation_complexity(request.elements)

    # Validate parameters
    element_ids = {e.id for e in request.elements}
    for param in request.parameters:
        if param.element_id not in element_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Parameter element not found: {param.element_id}",
            )
        if param.min_value >= param.max_value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid bounds for {param.element_id}: min >= max",
            )

    # Validate objective element
    if request.objective not in element_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Objective element not found: {request.objective}",
        )

    # Validate objective type
    if request.objective_type not in ["maximize", "minimize"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid objective_type: {request.objective_type}",
        )

    # Validate metric
    valid_metrics = ["final", "mean", "max", "min", "integral"]
    if request.metric not in valid_metrics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid metric: {request.metric}. Valid: {valid_metrics}",
        )

    try:
        # Create objective function
        objective_func = ObjectiveFunction(
            request.elements,
            request.links,
            request.config,
            request.parameters,
            request.objective,
            request.objective_type,
            request.metric,
        )

        # Run optimization
        optimal_params, optimal_value, iterations, convergence_info = await asyncio.wait_for(
            asyncio.to_thread(
                OptimizationEngine.run_optimization,
                objective_func,
                request.parameters,
                request.algorithm,
                request.max_iterations,
                request.tolerance,
                request.seed,
            ),
            timeout=SIMULATION_TIMEOUT * 10,
        )

        # Adjust optimal value if maximized
        if request.objective_type == "maximize":
            optimal_value = -optimal_value

        # Run simulation with optimal parameters
        result_id = None
        try:
            modified_elements = apply_parameter_values(request.elements, optimal_params)

            model = SystemDynamicsModel(modified_elements, request.links, verbose=False)
            
            if request.config.method == "euler":
                result = await asyncio.wait_for(
                    asyncio.to_thread(model.simulate_euler, request.config),
                    timeout=SIMULATION_TIMEOUT,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(model.simulate_rk4, request.config),
                    timeout=SIMULATION_TIMEOUT,
                )

            stored_result = SimulationResult(
                time=result["time"],
                results=result["results"]
            )
            result_store = get_result_store()
            result_id = result_store.store(stored_result)

        except Exception as e:
            logger.warning(f"Failed to store optimal result: {str(e)}")

        # Handle inf/nan for JSON
        if optimal_value is not None and not np.isfinite(optimal_value):
            optimal_value = None

        logger.info(f"Optimization completed: {iterations} evaluations, value={optimal_value}")

        return OptimizationResponse(
            success=True,
            optimal_parameters=optimal_params,
            optimal_value=optimal_value,
            iterations=iterations,
            convergence_info=convergence_info,
            result_id=result_id,
        )

    except asyncio.TimeoutError:
        logger.error("Optimization timed out")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Optimization timed out",
        )
    except ValueError as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Optimization error: {str(e)}",
        )
    except Exception as e:
        logger.exception("Unexpected error in optimization")
        return OptimizationResponse(
            success=False,
            optimal_parameters={},
            optimal_value=None,
            iterations=0,
            convergence_info={},
            error=str(e),
        )


@app.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_model_endpoint(request: CalibrationRequest):
    """Calibrate model parameters to fit observed data"""
    logger.info(
        f"Calibration: {len(request.parameters)} parameters, "
        f"{len(request.observed_data)} data points"
    )

    validate_equation_complexity(request.elements)

    # Validate parameters
    element_ids = {e.id for e in request.elements}
    for param in request.parameters:
        if param.element_id not in element_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Parameter element not found: {param.element_id}",
            )
        if param.min_value >= param.max_value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid bounds for {param.element_id}: min >= max",
            )

    # Validate elements_to_fit
    for elem_id in request.elements_to_fit:
        if elem_id not in element_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Element to fit not found: {elem_id}",
            )

    # Validate observed data
    if not request.observed_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No observed data provided",
        )
    
    # Validate that observed data has values for elements_to_fit
    for i, data_point in enumerate(request.observed_data):
        missing_elements = []
        for elem_id in request.elements_to_fit:
            if elem_id not in data_point.values:
                missing_elements.append(elem_id)
        if missing_elements:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Observed data point {i} (time={data_point.time}) missing values for elements: {', '.join(missing_elements)}",
            )

    # Validate error metric
    valid_metrics = ["rmse", "mae", "mse", "mape", "r2"]
    if request.error_metric not in valid_metrics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid error_metric: {request.error_metric}. Valid: {valid_metrics}",
        )

    # Validate interpolation
    if request.interpolation not in ["linear", "step"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid interpolation: {request.interpolation}",
        )

    try:
        # Create calibration function
        # CalibrationFunction expects List[ObservedDataPoint]
        calibration_func = CalibrationFunction(
            request.elements,
            request.links,
            request.config,
            request.parameters,
            request.observed_data,  # Already List[ObservedDataPoint] from Pydantic
            request.elements_to_fit,
            request.error_metric,
            request.interpolation,
        )

        # Run calibration
        calibrated_params, error_value, iterations, convergence_info = await asyncio.wait_for(
            asyncio.to_thread(
                OptimizationEngine.run_calibration,
                calibration_func,
                request.parameters,
                request.algorithm,
                request.max_iterations,
                request.tolerance,
                request.seed,
            ),
            timeout=SIMULATION_TIMEOUT * 10,
        )

        # Run simulation with calibrated parameters
        result_id = None
        fitted_results: Dict[str, List[float]] = {}
        error_by_element: Dict[str, float] = {}

        try:
            modified_elements = apply_parameter_values(request.elements, calibrated_params)

            model = SystemDynamicsModel(modified_elements, request.links, verbose=False)
            
            if request.config.method == "euler":
                result = await asyncio.wait_for(
                    asyncio.to_thread(model.simulate_euler, request.config),
                    timeout=SIMULATION_TIMEOUT,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(model.simulate_rk4, request.config),
                    timeout=SIMULATION_TIMEOUT,
                )

            # Store result
            stored_result = SimulationResult(
                time=result["time"],
                results=result["results"]
            )
            result_store = get_result_store()
            result_id = result_store.store(stored_result)

            # Compute fitted results and per-element errors
            from scipy.interpolate import interp1d
            
            model_time = np.array(result["time"])
            observed_times = np.array([d.time for d in request.observed_data])

            for elem_id in request.elements_to_fit:
                if elem_id not in result["results"]:
                    continue
                    
                model_values_series = np.array(result["results"][elem_id])
                observed_values = np.array([
                    d.values.get(elem_id, np.nan) for d in request.observed_data
                ])

                # Interpolate model to observed times
                kind = "linear" if request.interpolation == "linear" else "nearest"
                if len(model_time) > 1:
                    interp_func = interp1d(
                        model_time,
                        model_values_series,
                        kind=kind,
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    fitted_values = interp_func(observed_times).tolist()
                else:
                    fitted_values = [float(model_values_series[0])] * len(observed_times)

                fitted_results[elem_id] = fitted_values

                # Compute per-element error
                valid_mask = ~np.isnan(observed_values)
                if np.any(valid_mask):
                    element_error = compute_error_metric(
                        np.array(fitted_values)[valid_mask],
                        observed_values[valid_mask],
                        request.error_metric,
                    )
                    error_by_element[elem_id] = element_error

        except Exception as e:
            logger.warning(f"Failed to compute fitted results: {str(e)}")

        logger.info(f"Calibration completed: {iterations} evaluations, error={error_value:.6f}")

        return CalibrationResponse(
            success=True,
            calibrated_parameters=calibrated_params,
            error_value=error_value,
            iterations=iterations,
            convergence_info=convergence_info,
            fitted_results=fitted_results,
            error_by_element=error_by_element,
            result_id=result_id,
        )

    except asyncio.TimeoutError:
        logger.error("Calibration timed out")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Calibration timed out",
        )
    except ValueError as e:
        logger.error(f"Calibration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Calibration error: {str(e)}",
        )
    except Exception as e:
        logger.exception("Unexpected error in calibration")
        return CalibrationResponse(
            success=False,
            calibrated_parameters={},
            error_value=float("inf"),
            iterations=0,
            convergence_info={},
            fitted_results={},
            error_by_element={},
            error=str(e),
        )


@app.get("/templates")
def list_templates():
    """List available model templates"""
    templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    templates = []

    if os.path.exists(templates_dir):
        for filename in os.listdir(templates_dir):
            if filename.endswith(".json"):
                template_path = os.path.join(templates_dir, filename)
                try:
                    with open(template_path, "r") as f:
                        template_data = json.load(f)
                        templates.append({
                            "id": filename[:-5],
                            "name": template_data.get("name", filename[:-5]),
                            "description": template_data.get("description", ""),
                            "filename": filename,
                        })
                except Exception as e:
                    logger.warning(f"Failed to load template {filename}: {str(e)}")

    logger.info(f"Listed {len(templates)} templates")
    return {"templates": templates}


@app.get("/templates/{template_id}")
def get_template(template_id: str):
    """Load a model template by ID"""
    # Security: Validate template_id to prevent path traversal
    if ".." in template_id or "/" in template_id or "\\" in template_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid template ID: path traversal not allowed",
        )
    
    templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    template_path = os.path.join(templates_dir, f"{template_id}.json")
    
    # Security: Ensure resolved path is within templates directory
    resolved_path = os.path.realpath(template_path)
    resolved_dir = os.path.realpath(templates_dir)
    if not resolved_path.startswith(resolved_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid template ID: path outside templates directory",
        )

    if not os.path.exists(template_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found: {template_id}",
        )

    try:
        with open(template_path, "r") as f:
            template_data = json.load(f)
            logger.info(f"Loaded template: {template_id}")
            return template_data
    except Exception as e:
        logger.error(f"Failed to load template {template_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load template: {str(e)}",
        )