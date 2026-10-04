"""TPU/XLA-only patches. Importing this module has no side effects.

Qwen3.5's Gated-DeltaNet layers call ``torch.linalg.solve_triangular(ut_system, rhs,
upper=False, unitriangular=True)`` inside ``torch_chunk_gated_delta_rule``. Under
PyTorch/XLA + GSPMD that op has no partitioning rule, so the operands are all-gathered
(replicated) and the triangular-solve workspace OOMs HBM.

Upstream already contains a matmul-only path, but it is gated on
``is_torchdynamo_exporting()`` (never true on XLA) and is a 63-step Python loop of in-place
row updates. Instead of copying upstream's function body (which would drift with
``transformers``), we shim ``torch.linalg.solve_triangular`` itself: for the one
configuration the model uses, on XLA tensors only, we use a log-depth Neumann/product
expansion that is pure matmul (shardable, no in-place ops, no loop over rows).

Math: for strictly-lower T, (I + T)^-1 = (I - N)^-1 with N = -T nilpotent (N^n = 0), and
(I - N)^-1 = sum_k N^k = (I+N)(I+N^2)(I+N^4)...  -> ceil(log2 n) factors (6 for n=64).
"""

from __future__ import annotations

import torch

# Device types for which the shim takes over. Tests may add "cpu" to exercise it off-TPU.
SHIM_DEVICE_TYPES: set[str] = {"xla"}


def unit_lower_solve_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve ``(I + tril(A, -1)) X = B`` using only matmuls (== solve_triangular(upper=False, unitriangular=True))."""
    n = A.shape[-1]
    N = -A.tril(-1)
    X = B
    P = N
    k = 1
    while k < n:
        X = X + P @ X
        k *= 2
        if k < n:
            P = P @ P
    return X


def install_xla_solve_triangular_shim() -> bool:
    """Install the shim once. Returns True if newly installed, False if already present."""
    current = torch.linalg.solve_triangular
    if getattr(current, "_dental_xla_shim", False):
        return False
    original = current

    def solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None):  # noqa: ANN001
        if (
            out is None
            and left
            and not upper
            and unitriangular
            and A.device.type in SHIM_DEVICE_TYPES
        ):
            return unit_lower_solve_matmul(A, B)
        return original(A, B, upper=upper, left=left, unitriangular=unitriangular, out=out)

    solve_triangular._dental_xla_shim = True  # type: ignore[attr-defined]
    solve_triangular._dental_original = original  # type: ignore[attr-defined]
    torch.linalg.solve_triangular = solve_triangular
    return True


def uninstall_xla_solve_triangular_shim() -> None:
    current = torch.linalg.solve_triangular
    if getattr(current, "_dental_xla_shim", False):
        torch.linalg.solve_triangular = current._dental_original  # type: ignore[attr-defined]
