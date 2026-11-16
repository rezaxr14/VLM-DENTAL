"""Explicit data parallelism for the CUDA path: one process per GPU, each holding a FULL replica, LoRA gradients all-reduced.

This is the ``dual_t4_ddp_fp16`` profile: two T4s, one Qwen3.5-4B fp16 replica on each, so no layer is split across the PCIe link and
nothing crosses it except the LoRA gradients (and a few scalars) once per optimizer update.

Deliberately NOT ``torch.nn.parallel.DistributedDataParallel``: the model is a frozen base plus a small LoRA with non-reentrant
gradient checkpointing, a GradScaler and (GRPO) a rank that may have no usable rollout at all. A plain "sum the trainable gradients"
collective at the update boundary is exact for the token-weighted SFT loss, works with a rank that has nothing to contribute, and
because the summed gradients are identical on every rank the GradScaler's finite check and scale update stay identical too.

Nothing in this module runs unless the launcher asked for it (``--data-parallel``) AND torchrun set ``WORLD_SIZE`` > 1. A mismatch
between the two is an error, never a silent fallback. Rank 0 owns every file, Hub and log write.
"""
from __future__ import annotations

import datetime
import os
from typing import Any, Callable, Iterable

import torch
import torch.distributed as dist


def launched_world_size(environ: dict | None = None) -> int:
    """``WORLD_SIZE`` set by torchrun (1 when absent or malformed)."""
    try:
        return max(int((os.environ if environ is None else environ).get("WORLD_SIZE", "1")), 1)
    except ValueError:
        return 1


def is_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if is_active() else 0


def world_size() -> int:
    return dist.get_world_size() if is_active() else 1


def local_rank(environ: dict | None = None) -> int:
    return int((os.environ if environ is None else environ).get("LOCAL_RANK", "0"))


def is_main() -> bool:
    return rank() == 0


def require_consistent(data_parallel: bool, environ: dict | None = None) -> None:
    """``--data-parallel`` and the torchrun launch must agree. Neither is guessed from the other."""
    ws = launched_world_size(environ)
    if data_parallel and ws < 2:
        raise SystemExit("--data-parallel needs a torchrun launch with at least 2 processes "
                         "(python -m torch.distributed.run --nproc_per_node 2 ...); WORLD_SIZE is not set.")
    if not data_parallel and ws > 1:
        raise SystemExit(f"WORLD_SIZE={ws} (torchrun) but --data-parallel was not passed: refusing to run {ws} independent copies of "
                         "the same training that would overwrite each other's files. Pass --data-parallel or launch with plain python.")


def init_process_group(*, backend: str = "nccl", timeout_minutes: int = 120, environ: dict | None = None) -> tuple[int, int, int]:
    """Join the torchrun group and bind this process to its GPU. Returns ``(rank, local_rank, world_size)``.

    The timeout is long on purpose: rank 0 alone runs validation and writes checkpoints while the others wait at the next
    collective, and a Hub upload can take minutes."""
    ws = launched_world_size(environ)
    lr = local_rank(environ)
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise SystemExit("--data-parallel needs CUDA (one GPU per process).")
        if torch.cuda.device_count() < ws:
            raise SystemExit(f"--data-parallel with {ws} processes needs {ws} visible GPUs; found {torch.cuda.device_count()}.")
        torch.cuda.set_device(lr)
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=timeout_minutes))
    return dist.get_rank(), lr, dist.get_world_size()


def shutdown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def check_replica_fits(precision: str, device_total_gib: float, size: str | None = None) -> None:
    """A full replica must fit on ONE GPU: refuse the clear misfits (9B in 16-bit on a 15 GiB T4) before loading anything.

    Not a proof that it fits: the footprint is an estimate from the config shapes (about 9.5 GiB for 4B), activations come on top,
    and only a measured peak says how much room is left. QLoRA is not checked (its 4-bit footprint has not been measured)."""
    from dental_agent.utils.model_size import spec
    sp = spec(size)
    if precision != "qlora" and device_total_gib < sp["weights_gib_16bit"] + 1.0:
        raise SystemExit(f"--data-parallel keeps a FULL {precision} replica of {sp['folder']} on every GPU (about "
                         f"{sp['weights_gib_16bit']:g} GiB of weights, an estimate) but this GPU has {device_total_gib:.1f} GiB. "
                         "Use DENTAL_MODEL_SIZE=4b, or --precision qlora, or the single-process profile that splits layers over both GPUs.")


def check_replica_fits_here(precision: str) -> None:
    check_replica_fits(precision, torch.cuda.get_device_properties(local_rank()).total_memory / 2**30)


def device_map() -> Any:
    """Whole model on this process's GPU (``"auto"`` would split it across both GPUs again)."""
    return {"": local_rank()} if is_active() else "auto"


def barrier() -> None:
    if is_active():
        dist.barrier()


def in_rank_order(fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on rank 0 first, then rank 1, ...: one 4B load at a time keeps host RAM (Kaggle: about 29 GB) bounded."""
    result = None
    for r in range(world_size()):
        if rank() == r:
            result = fn()
        barrier()
    return result


def all_reduce_sum(values: Iterable[float], device: torch.device) -> list[float]:
    """Sum a few Python scalars across ranks (token counts, loss sums, contribution flags). Identity when not distributed."""
    vals = [float(v) for v in values]
    if not is_active():
        return vals
    t = torch.tensor(vals, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(x) for x in t.tolist()]


def broadcast_trainable(params: Iterable[torch.nn.Parameter]) -> int:
    """Copy rank 0's trainable parameters (the LoRA, the merger LoRA) to every rank; returns how many tensors were synced.

    LoRA A matrices are randomly initialised per process. Gradients are summed, so the replicas only stay identical if they START
    identical; this is that guarantee. Also harmless after a resume (every rank restored the same files)."""
    n = 0
    if not is_active():
        return n
    for p in params:
        if p.requires_grad:
            dist.broadcast(p.data, src=0)
            n += 1
    return n


def sync_trainable_grads(params: Iterable[torch.nn.Parameter], *, contributed: bool = True, average_over_contributors: bool = False) -> int:
    """SUM the gradients of the trainable ``params`` across ranks, in place. Returns how many ranks contributed (>= 1 when anyone did).

    * A parameter without a gradient counts as zeros, so a rank that had nothing to train on (no valid tokens, every rollout dropped)
      still joins the collective instead of hanging the others.
    * ``average_over_contributors`` divides by the number of ranks that actually had gradients (GRPO: a mean over images);
      SFT leaves the sum and divides by the global valid-token count itself (exact token-weighted mean).
    * Identical summed gradients on every rank keep the GradScaler's finite check and scale update identical.
    """
    params = [p for p in params if p.requires_grad]
    if not is_active():
        return 1 if contributed else 0
    by_dtype: dict[torch.dtype, list[torch.nn.Parameter]] = {}
    for p in params:
        by_dtype.setdefault(p.dtype, []).append(p)
    device = params[0].device if params else torch.device("cpu")
    flag = torch.tensor([1.0 if contributed else 0.0], dtype=torch.float32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    contributors = int(flag.item())
    divisor = float(max(contributors, 1)) if average_over_contributors else 1.0
    for dtype, group in sorted(by_dtype.items(), key=lambda kv: str(kv[0])):
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in group])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        if divisor != 1.0:
            flat.div_(divisor)
        offset = 0
        for p in group:
            n = p.numel()
            chunk = flat[offset:offset + n].view_as(p)
            if p.grad is None:
                p.grad = chunk.clone()
            else:
                p.grad.copy_(chunk)
            offset += n
    return contributors


class RankStridedBatches(torch.utils.data.Sampler):
    """Splits one globally shuffled order into per-rank batches: global batch ``b`` goes to rank ``b % world``.

    Every rank must run the SAME number of batches, or one rank would wait forever in the next collective: the ragged tail
    (fewer than ``world`` full batches) is dropped, and which samples that is changes every epoch (the order is reshuffled).
    ``base`` is a ``ResumableShuffleSampler`` (or anything iterable with ``set_epoch`` / ``skip_next``); the resume position is in
    per-rank batches, so ``skip_next(n)`` skips ``n`` batches on THIS rank (= ``n * world`` global batches).
    """

    def __init__(self, base: Any, batch_size: int, rank_: int, world: int):
        self.base, self.batch_size, self.rank, self.world = base, int(batch_size), int(rank_), int(world)
        self._skip = 0

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def skip_next(self, n_samples: int) -> None:
        # the caller passes per-rank batches * loader_batch_size samples
        self._skip = int(n_samples)

    def batches_per_rank(self) -> int:
        return len(self.base) // (self.batch_size * self.world)

    def __len__(self) -> int:
        return self.batches_per_rank() * self.batch_size

    def __iter__(self):
        order = list(iter(self.base))     # the full seeded order of this epoch; the resume skip is applied per rank below
        usable = self.batches_per_rank() * self.batch_size * self.world
        order = order[:usable]
        mine: list[int] = []
        for b in range(self.rank, usable // self.batch_size, self.world):
            mine.extend(order[b * self.batch_size:(b + 1) * self.batch_size])
        skip, self._skip = self._skip, 0
        return iter(mine[skip:])
