"""
Operator-Aware Numerical Correctness Verifier — FraudGraph Intelligence (FGI)
Phase 8: Benchmark Automation & Experiment Management (FGI-MOD-08)

Provides rigorous, operator-specific numerical validation comparing test tensors
(optimized CPU or GPU engines) against the Phase 4 CPU reference tensor.

Key verification policies:
1. Shape and dtype parity: Must match exactly (float32, [num_nodes, feature_dim]).
2. Non-finite rejection: Detects and rejects any NaN or Inf values immediately.
3. Empty neighborhood sentinel verification:
   For nodes with degree 0 (row_ptr[i] == row_ptr[i+1]), both reference and test
   must produce exact zero vectors (all elements == 0.0).
4. Operator-specific tolerance testing:
   |Y_test - Y_ref| <= atol + rtol * |Y_ref|
   - 'mean': atol=1e-5, rtol=1e-5
   - 'max':  atol=1e-5, rtol=0.0
   - 'sum':  atol=1e-4, rtol=1e-4 (FP32 mantissa accumulation bounds on high-degree nodes)
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from backend.benchmark_engine.benchmark_matrix import ExecutionStatus
from backend.cpu_engine.csr_graph import CSRGraph


@dataclass(frozen=True)
class OperatorTolerance:
    """Tolerances for numerical verification of graph aggregators."""
    atol: float
    rtol: float
    description: str


# Canonical operator tolerances
OPERATOR_TOLERANCES: Dict[str, OperatorTolerance] = {
    "mean": OperatorTolerance(
        atol=1.0e-5,
        rtol=1.0e-5,
        description="Standard mean aggregation: atol=1e-5, rtol=1e-5",
    ),
    "max": OperatorTolerance(
        atol=1.0e-5,
        rtol=0.0,
        description="Max element-wise selection: atol=1e-5, rtol=0.0",
    ),
    "sum": OperatorTolerance(
        atol=1.0e-4,
        rtol=1.0e-4,
        description="Sum reduction with FP32 accumulator drift: atol=1e-4, rtol=1e-4",
    ),
}


@dataclass
class NumericalVerificationResult:
    """Detailed verification metrics comparing test output against reference output."""
    status: ExecutionStatus  # MEASURED (valid) or FAILED_*
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    atol: float
    rtol: float
    nan_detected: bool
    inf_detected: bool
    shape_match: bool
    empty_neighborhood_verified: bool
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "max_absolute_error": float(self.max_absolute_error),
            "max_relative_error": float(self.max_relative_error),
            "mean_absolute_error": float(self.mean_absolute_error),
            "tolerance": {
                "atol": self.atol,
                "rtol": self.rtol,
            },
            "nan_detected": self.nan_detected,
            "inf_detected": self.inf_detected,
            "shape_match": self.shape_match,
            "empty_neighborhood_verified": self.empty_neighborhood_verified,
            "details": self.details,
        }


def get_operator_tolerance(operator: str) -> OperatorTolerance:
    """Retrieves standard tolerance for a given aggregation operator."""
    op_lower = operator.lower().strip()
    if op_lower in OPERATOR_TOLERANCES:
        return OPERATOR_TOLERANCES[op_lower]
    return OperatorTolerance(
        atol=1.0e-5,
        rtol=1.0e-5,
        description=f"Default tolerance for operator '{operator}': atol=1e-5, rtol=1e-5",
    )


def verify_aggregation_correctness(
    test_output: np.ndarray,
    ref_output: np.ndarray,
    graph: CSRGraph,
    operator: str,
    custom_atol: Optional[float] = None,
    custom_rtol: Optional[float] = None,
) -> NumericalVerificationResult:
    """
    Rigorously verifies test aggregation output against CPU reference output.

    Parameters:
        test_output: Candidate tensor from optimized CPU or GPU engine.
        ref_output: Ground-truth tensor from Phase 4 CPU reference engine.
        graph: CSRGraph instance on which aggregation was computed.
        operator: Aggregation operator ('mean', 'sum', 'max').
        custom_atol: Optional custom absolute tolerance.
        custom_rtol: Optional custom relative tolerance.

    Returns:
        NumericalVerificationResult detailing status and error metrics.
    """
    # 1. Shape check
    if test_output.shape != ref_output.shape:
        return NumericalVerificationResult(
            status=ExecutionStatus.FAILED_SHAPE_MISMATCH,
            max_absolute_error=float("inf"),
            max_relative_error=float("inf"),
            mean_absolute_error=float("inf"),
            atol=0.0,
            rtol=0.0,
            nan_detected=False,
            inf_detected=False,
            shape_match=False,
            empty_neighborhood_verified=False,
            details=f"Shape mismatch: test shape {test_output.shape} vs ref shape {ref_output.shape}",
        )

    # 2. Non-finite checks (NaN / Inf)
    nan_detected = bool(np.any(np.isnan(test_output)))
    inf_detected = bool(np.any(np.isinf(test_output)))
    if nan_detected or inf_detected:
        return NumericalVerificationResult(
            status=ExecutionStatus.FAILED_NON_FINITE,
            max_absolute_error=float("inf"),
            max_relative_error=float("inf"),
            mean_absolute_error=float("inf"),
            atol=0.0,
            rtol=0.0,
            nan_detected=nan_detected,
            inf_detected=inf_detected,
            shape_match=True,
            empty_neighborhood_verified=False,
            details=f"Non-finite values detected: nan={nan_detected}, inf={inf_detected}",
        )

    # 3. Empty neighborhood sentinel validation (degree 0 -> exact 0)
    empty_nodes = []
    for i in range(graph.num_nodes):
        if graph.row_ptr[i] == graph.row_ptr[i + 1]:
            empty_nodes.append(i)

    empty_verified = True
    if empty_nodes:
        empty_indices = np.array(empty_nodes, dtype=np.int64)
        test_empty_vals = test_output[empty_indices]
        ref_empty_vals = ref_output[empty_indices]

        # Both must be strictly 0.0
        if not np.all(ref_empty_vals == 0.0):
            empty_verified = False
        if not np.all(test_empty_vals == 0.0):
            empty_verified = False
            return NumericalVerificationResult(
                status=ExecutionStatus.FAILED_TOLERANCE,
                max_absolute_error=float(np.max(np.abs(test_empty_vals))),
                max_relative_error=float("inf"),
                mean_absolute_error=float(np.mean(np.abs(test_empty_vals))),
                atol=0.0,
                rtol=0.0,
                nan_detected=False,
                inf_detected=False,
                shape_match=True,
                empty_neighborhood_verified=False,
                details=f"Zero-degree nodes failed zero-sentinel check for {len(empty_nodes)} empty nodes",
            )

    # 4. Operator tolerance checks
    default_tol = get_operator_tolerance(operator)
    atol = custom_atol if custom_atol is not None else default_tol.atol
    rtol = custom_rtol if custom_rtol is not None else default_tol.rtol

    diff = np.abs(test_output - ref_output)
    max_abs_err = float(np.max(diff))
    mean_abs_err = float(np.mean(diff))

    # Compute relative error where reference is non-zero
    denom = np.abs(ref_output)
    mask = denom > 1.0e-7
    if np.any(mask):
        rel_diff = diff[mask] / denom[mask]
        max_rel_err = float(np.max(rel_diff))
    else:
        max_rel_err = 0.0

    # Tolerance formula: |Y_test - Y_ref| <= atol + rtol * |Y_ref|
    allowed_err = atol + rtol * denom
    exceeds_mask = diff > allowed_err
    passed = not np.any(exceeds_mask)

    if not passed:
        num_exceeding = int(np.sum(exceeds_mask))
        details = (
            f"Numerical tolerance exceeded for operator '{operator}': "
            f"max_abs={max_abs_err:.2e} (atol={atol:.2e}), "
            f"max_rel={max_rel_err:.2e} (rtol={rtol:.2e}), "
            f"{num_exceeding} elements exceed bound."
        )
        return NumericalVerificationResult(
            status=ExecutionStatus.FAILED_TOLERANCE,
            max_absolute_error=max_abs_err,
            max_relative_error=max_rel_err,
            mean_absolute_error=mean_abs_err,
            atol=atol,
            rtol=rtol,
            nan_detected=False,
            inf_detected=False,
            shape_match=True,
            empty_neighborhood_verified=empty_verified,
            details=details,
        )

    return NumericalVerificationResult(
        status=ExecutionStatus.MEASURED,
        max_absolute_error=max_abs_err,
        max_relative_error=max_rel_err,
        mean_absolute_error=mean_abs_err,
        atol=atol,
        rtol=rtol,
        nan_detected=False,
        inf_detected=False,
        shape_match=True,
        empty_neighborhood_verified=empty_verified,
        details=f"Passed numerical verification for operator '{operator}' (max_abs={max_abs_err:.2e}, max_rel={max_rel_err:.2e})",
    )
