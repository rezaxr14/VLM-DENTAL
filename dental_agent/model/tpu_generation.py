"""Opt-in, static-shape text generation for TPU (PyTorch/XLA) -- the GRPO rollout and evaluation generator.

Why it exists
-------------
``model.generate`` grows its KV cache and attention mask by one position per token, so under XLA every decode step is a
NEW graph (and every prompt length a new prefill graph). On a TPU that means a compile per generated token. This module
replaces ``generate`` *on TPU only* with a loop whose graphs never depend on the prompt length or on the number of
generated tokens:

* ONE fixed ``StaticCache`` of ``C`` positions (``C`` is chosen from ``cache_tiers``: the smallest tier that holds
  ``prompt + max_new_tokens``; a request that fits no tier raises ``GenerationTooLong`` -- nothing is truncated).
* Chunked prefill: the left-padded prompt is fed ``chunk_tokens`` at a time. The chunk offset, positions and masks are
  DEVICE tensors, so every chunk after the first reuses one graph.
* One decode graph: the cache write index, the rotary position and the attention mask all derive from device counters.
* Sampling (greedy, temperature, top-k, top-p) runs on the device; one tiny ``[batch]`` token tensor is read back per
  step to decide when every row has finished.

Per ``(batch, tier)`` this compiles four programs (first chunk, later chunk, decode, prefill sampler) and nothing else.

Numerics follow stock ``generate`` for the hybrid Qwen3.5 stack: left padding (the recurrent state is still zero where the
padding sits, and padded positions are zeroed before the linear-attention layers), M-RoPE positions computed by the model's
own ``get_rope_index`` on the host, tool images through the same canonical static slots as training (real images first,
empty slots encoded and ignored, so the vision graph is also one shape).

Batch rows are padded to a multiple of ``batch_multiple`` (the number of cores) by repeating row 0, because every row costs
the same per-step weight traffic under FSDP and a batch that divides the mesh shards cleanly. Not imported by CUDA/CPU code
paths unless a ``TPUGenerationConfig`` is installed on the model.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from dental_agent.utils.canonical import (
    PATCH_FEATURE_DIM,
    SLOT_BUDGET,
    MERGE_AREA,
    family_from_grid,
    grid_thw as canonical_grid_thw,
    patches_per_image,
    slot_totals,
    tokens_per_image,
)

ATTR = "_dental_tpu_generation"


class GenerationTooLong(ValueError):
    """The prompt plus ``max_new_tokens`` does not fit the largest cache tier. Never truncated; callers drop and count."""


class SlotOverflow(ValueError):
    """The prompt carries more images of a family than the static vision budget. Never silently dropped."""


@dataclass
class TPUGenerationConfig:
    chunk_tokens: int = 512                       # prefill chunk (multiple of 64: the delta-rule kernel pads to 64)
    cache_tiers: tuple[int, ...] = (8192, 16384, 32768)
    slot_budget: Mapping[str, int] = field(default_factory=lambda: dict(SLOT_BUDGET))
    batch_multiple: int = 1                       # set to the number of cores under SPMD
    mesh: Any = None                              # optional xs.Mesh with axes ('fsdp', 'model'); batch is sharded on 'fsdp'
    stats: dict = field(default_factory=lambda: {"calls": 0, "prefill_chunks": 0, "decode_steps": 0, "too_long": 0,
                                                 "slot_overflow": 0, "rollouts_dropped": 0, "tiers": {}})

    def __post_init__(self):
        self.cache_tiers = tuple(sorted(int(t) for t in self.cache_tiers))
        if self.chunk_tokens <= 0 or self.chunk_tokens % 64:
            raise ValueError("chunk_tokens must be a positive multiple of 64")
        if not self.cache_tiers or any(t % self.chunk_tokens for t in self.cache_tiers):
            raise ValueError(f"every cache tier must be a multiple of chunk_tokens={self.chunk_tokens}: {self.cache_tiers}")
        if self.batch_multiple < 1:
            raise ValueError("batch_multiple must be >= 1")


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------
def find_attr(model: Any, name: str) -> Any:
    """Read ``name`` through PEFT / FSDP wrappers (their ``__getattr__`` delegates to the wrapped module); ``None`` if absent."""
    try:
        return getattr(model, name)
    except AttributeError:
        return None


def unwrap_causal_lm(model: Any) -> Any:
    """The object that owns ``.lm_head`` and a ``.model`` whose ``.language_model`` is the text decoder."""
    obj, seen = model, set()
    while id(obj) not in seen:
        seen.add(id(obj))
        if hasattr(obj, "lm_head") and hasattr(getattr(obj, "model", None), "language_model"):
            return obj
        gb = getattr(obj, "get_base_model", None)
        nxt = gb() if callable(gb) else None
        if nxt is None:
            nxt = getattr(obj, "_modules", {}).get("module")
        if nxt is None:
            break
        obj = nxt
    raise ValueError("TPU generation needs a Qwen3.5 conditional-generation model (lm_head + model.language_model)")


def install_tpu_generation(model: Any, config: TPUGenerationConfig) -> Any:
    """Mark ``model`` so ``local_runtime.generate_local`` uses the static-shape generator. Other forwards are untouched."""
    if find_attr(model, ATTR) is not None:
        raise ValueError("TPU generation is already installed on this model")
    unwrap_causal_lm(model)  # validate the architecture before any weights are used
    object.__setattr__(model, ATTR, config)
    print(f"[TPU GEN] static-shape generation: chunk={config.chunk_tokens} tiers={config.cache_tiers} "
          f"slots={dict(config.slot_budget)} batch_multiple={config.batch_multiple}", flush=True)
    return model


def note_dropped_rollouts(model: Any, n: int = 1) -> None:
    """Count rollouts a caller dropped because generation refused them (GenerationTooLong / SlotOverflow)."""
    config = find_attr(model, ATTR)
    if config is not None:
        config.stats["rollouts_dropped"] += int(n)


def validate_tpu_generation_args(*, chunk_tokens: int, tiers: Sequence[int], is_tpu: bool, spmd: bool, fsdp: bool,
                                 pad_vision_to_slots: bool, canonical_resize: bool) -> None:
    """Fail before the 9B weights load. ``chunk_tokens == 0`` means the feature is off (stock ``generate``)."""
    if chunk_tokens < 0:
        raise ValueError("--tpu-generation-chunk-tokens must be nonnegative")
    if not chunk_tokens:
        return
    if not is_tpu:
        raise ValueError("--tpu-generation-chunk-tokens requires a real TPU; GPU/CPU generation (model.generate) is unchanged")
    if not (spmd and fsdp and pad_vision_to_slots and canonical_resize):
        raise ValueError("TPU static generation requires --spmd --fsdp --canonical-resize --pad-vision-to-slots "
                         "(canonical letterbox views and fixed vision slots)")
    TPUGenerationConfig(chunk_tokens=chunk_tokens, cache_tiers=tuple(tiers))   # validates multiples of 64 / of the chunk


def _is_xla(device: torch.device) -> bool:
    return device.type == "xla"


def _step(device: torch.device) -> None:
    if _is_xla(device):
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def _shard(t: torch.Tensor, mesh: Any, spec: tuple) -> torch.Tensor:
    if mesh is None or not _is_xla(t.device):
        return t
    import torch_xla.distributed.spmd as xs
    xs.mark_sharding(t, mesh, spec)
    return t


def _cache_tensors(cache: Any):
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if isinstance(t, torch.Tensor) and t.ndim >= 2:
                yield t
        for name in ("conv_states", "recurrent_states"):
            for t in (getattr(layer, name, None) or []):
                if isinstance(t, torch.Tensor) and t.ndim >= 2:
                    yield t


def _pin_cache(cache: Any, mesh: Any) -> None:
    """Annotate every cache tensor as batch-sharded INSIDE the graph that updates it, so the output sharding is the same
    after every program (otherwise XLA lets the first program choose one and the next programs see another)."""
    if mesh is None:
        return
    for t in _cache_tensors(cache):
        if _is_xla(t.device):
            _shard(t, mesh, ("fsdp",) + (None,) * (t.ndim - 1))


def _to(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device)


def pick_tier(config: TPUGenerationConfig, prompt_len: int, max_new_tokens: int) -> tuple[int, int]:
    """``(prompt_len padded to a whole number of chunks, cache tier)``; raises GenerationTooLong when nothing fits."""
    q = config.chunk_tokens
    padded = math.ceil(max(prompt_len, 1) / q) * q
    need = padded + int(max_new_tokens)
    for tier in config.cache_tiers:
        if need <= tier:
            return padded, tier
    config.stats["too_long"] += 1
    raise GenerationTooLong(
        f"prompt {prompt_len} tokens (padded {padded}) + max_new_tokens {max_new_tokens} = {need} exceeds the largest "
        f"static cache tier {config.cache_tiers[-1]}. Not truncated: raise the tier or drop this rollout.")


def attention_masks(pad: torch.Tensor, start: torch.Tensor, q_len: int, cache_len: int, *, float_mask: bool,
                    dtype: torch.dtype, linear: bool) -> dict:
    """Mask dict for ``Qwen3_5TextModel`` (keys ``full_attention`` / ``linear_attention``), built from DEVICE tensors.

    ``pad`` is the per-row left-pad length ``[B]``; ``start`` the absolute cache index of the first query (scalar).
    A key is visible iff it is real (``>= pad``) and not in the future. Padded queries see only themselves, so no softmax
    row is empty (their outputs are never read by a real token)."""
    device = pad.device
    q = start + torch.arange(q_len, device=device)                      # [Q]
    k = torch.arange(cache_len, device=device)                          # [C]
    real = k[None, None, :] >= pad[:, None, None]                       # [B,1,C]
    causal = k[None, None, :] <= q[None, :, None]                       # [1,Q,C]
    allowed = (real & causal) | (k[None, None, :] == q[None, :, None])  # [B,Q,C]
    allowed = allowed[:, None]                                          # [B,1,Q,C]
    if float_mask:
        full = torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(~allowed, torch.finfo(dtype).min)
    else:
        full = allowed
    lin = (q[None, :] >= pad[:, None]).to(dtype) if linear else None    # [B,Q]; None for single-token decode
    return {"full_attention": full, "linear_attention": lin}


def _penalize(logits: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    """transformers' ``RepetitionPenaltyLogitsProcessor``: every token already in the sequence is divided (positive logit) or
    multiplied (negative logit) by ``penalty``. ``seen`` is a ``[B,V]`` bool mask carried through the loop."""
    return torch.where(seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits)


def _sample(logits: torch.Tensor, *, temperature: float, top_k: int | None, top_p: float | None,
            seen: torch.Tensor | None = None, repetition_penalty: float | None = None) -> torch.Tensor:
    """Next token ``[B]`` from last-position logits ``[B,V]``. ``temperature <= 0`` is greedy."""
    logits = logits.float()
    if seen is not None and repetition_penalty not in (None, 1.0):
        logits = _penalize(logits, seen, float(repetition_penalty))
    if temperature is None or temperature <= 0:
        return logits.argmax(-1)
    vocab = logits.shape[-1]
    k = vocab if not top_k or top_k <= 0 else min(int(top_k), vocab)
    values, indices = torch.topk(logits, k, dim=-1)                     # descending
    values = values / float(temperature)
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = values.softmax(-1)
        before = probs.cumsum(-1) - probs
        values = values.masked_fill(before >= top_p, float("-inf"))     # always keeps the first (largest) token
    u = torch.rand(values.shape, device=values.device).clamp_(1e-20, 1.0)
    gumbel = -torch.log(-torch.log(u))
    choice = (values + gumbel).argmax(-1, keepdim=True)
    return indices.gather(-1, choice).squeeze(-1)


# ---------------------------------------------------------------------------------------------------------------------
# Host-side preparation (CPU tensors only; nothing here is traced)
# ---------------------------------------------------------------------------------------------------------------------
def _split_images_by_row(ids: list[torch.Tensor], grid: torch.Tensor | None, image_token_id: int, merge_area: int):
    """Assign the processor's flat ``image_grid_thw`` rows to batch rows via each row's placeholder count."""
    per_row: list[list[int]] = [[] for _ in ids]
    if grid is None or grid.shape[0] == 0:
        return per_row
    cursor = 0
    for b, row_ids in enumerate(ids):
        remaining = int((row_ids == image_token_id).sum())
        while remaining > 0:
            if cursor >= grid.shape[0]:
                raise ValueError("image_grid_thw has fewer images than the prompt has <|image_pad|> runs")
            n = int(grid[cursor].prod()) // merge_area
            per_row[b].append(cursor)
            remaining -= n
            cursor += 1
        if remaining != 0:
            raise ValueError("<|image_pad|> placeholders do not match image_grid_thw")
    if cursor != grid.shape[0]:
        raise ValueError("image_grid_thw has more images than the prompt has placeholders")
    return per_row


def _pad_slots(pixel_rows: list[torch.Tensor], grid_rows: list[torch.Tensor], budget: Mapping[str, int]):
    """Real images first (original order), then zero patches for each unused slot; returns pixels, grid, real token count."""
    used = {fam: 0 for fam in budget}
    for g in grid_rows:
        fam = family_from_grid(tuple(int(x) for x in g.tolist()))
        if fam is None:
            raise ValueError(f"image grid {tuple(g.tolist())} is not a canonical family (enable canonical letterbox views)")
        if fam not in used:
            raise SlotOverflow(f"image family {fam} is not in the static budget {dict(budget)}")
        used[fam] += 1
    over = {f: n - budget[f] for f, n in used.items() if n > budget[f]}
    if over:
        raise SlotOverflow(f"prompt needs more images than the static vision budget {dict(budget)}: over by {over}")
    pix = [p for p in pixel_rows]
    grid = [g.view(1, 3) for g in grid_rows]
    dtype = pixel_rows[0].dtype if pixel_rows else torch.float32
    for fam, total in budget.items():
        for _ in range(total - used[fam]):
            pix.append(torch.zeros((patches_per_image(fam), PATCH_FEATURE_DIM), dtype=dtype))
            grid.append(torch.tensor([canonical_grid_thw(fam)], dtype=torch.long))
    n_real_tokens = sum(int(g.prod()) for g in grid_rows) // MERGE_AREA
    return torch.cat(pix, 0), torch.cat(grid, 0), n_real_tokens


@dataclass
class _Prepared:
    ids: torch.Tensor          # [B,C] long, left-padded prompt then zeros
    pad: torch.Tensor          # [B] left-pad lengths
    pos: torch.Tensor          # [3,B,C] long M-RoPE positions (zeros past the prompt)
    base_pos: torch.Tensor     # [B] rotary position of the first generated token
    vis_index: torch.Tensor    # [B,C] long index into the row's vision features
    is_vis: torch.Tensor       # [B,C] bool
    vision: list[torch.Tensor | None]   # per UNIQUE row group: pooled features or None (filled later)
    row_group: list[int]       # batch row -> index in `groups`
    groups: list[int]          # representative batch row of each group
    prompt_len: int            # padded prompt length P
    cache_len: int             # tier C
    real_len: list[int]


def _prepare(base, config: TPUGenerationConfig, inputs: Mapping[str, Any], max_new_tokens: int, pad_token_id: int):
    q = config.chunk_tokens
    ids_in = inputs["input_ids"].cpu()
    B0 = ids_in.shape[0]
    mask_in = inputs.get("attention_mask")
    mask_in = torch.ones_like(ids_in) if mask_in is None else mask_in.cpu()
    mm_in = inputs.get("mm_token_type_ids")
    mm_in = torch.zeros_like(ids_in) if mm_in is None else mm_in.cpu()
    rows = [ids_in[b][mask_in[b].bool()] for b in range(B0)]
    mm_rows = [mm_in[b][mask_in[b].bool()] for b in range(B0)]
    real_len = [int(r.numel()) for r in rows]
    P, C = pick_tier(config, max(real_len), max_new_tokens)

    # pad the batch to a multiple of the core count by repeating row 0
    B = math.ceil(B0 / config.batch_multiple) * config.batch_multiple
    order = list(range(B0)) + [0] * (B - B0)

    ids = torch.full((B, C), int(pad_token_id), dtype=torch.long)
    mmt = torch.zeros((B, C), dtype=torch.long)
    pad = torch.zeros(B, dtype=torch.long)
    maskp = torch.zeros((B, P), dtype=torch.long)
    for out_b, src in enumerate(order):
        n = real_len[src]
        pad[out_b] = P - n
        ids[out_b, P - n:P] = rows[src]
        mmt[out_b, P - n:P] = mm_rows[src].long()
        maskp[out_b, P - n:P] = 1

    cfg = base.config
    image_token_id = int(getattr(cfg, "image_token_id", getattr(getattr(cfg, "text_config", cfg), "image_token_id", -1)))
    grid = inputs.get("image_grid_thw")
    grid = grid.cpu() if grid is not None else None
    pix = inputs.get("pixel_values")
    pix = pix.cpu() if pix is not None else None
    merge_area = int(getattr(cfg.vision_config, "spatial_merge_size", 2)) ** 2
    per_row_imgs = _split_images_by_row([r for r in rows], grid, image_token_id, merge_area)
    # patch offset of every image inside pixel_values
    offsets, acc = [], 0
    if grid is not None:
        for g in grid:
            n = int(g.prod())
            offsets.append((acc, n))
            acc += n

    # positions: the model's own M-RoPE on the left-padded real tokens
    has_images = any(per_row_imgs)
    if has_images:
        grid_in_order = torch.cat([grid[per_row_imgs[src]] for src in order if per_row_imgs[src]], 0)
        position_ids, deltas = base.model.get_rope_index(
            ids[:, :P], mm_token_type_ids=mmt[:, :P], image_grid_thw=grid_in_order, attention_mask=maskp)
        position_ids = position_ids.cpu()
        base_pos = (position_ids.amax(dim=(0, 2)) + 1).long()
    else:
        p1 = (maskp.cumsum(-1) - 1).clamp_(min=0)
        position_ids = p1.unsqueeze(0).expand(3, -1, -1).contiguous()
        base_pos = (p1.amax(-1) + 1).long()
    pos = torch.zeros((3, B, C), dtype=torch.long)
    pos[:, :, :P] = position_ids
    # padded query positions must not feed NaN/garbage into rotary: they stay 0 (set above by get_rope_index/cumsum)

    # vision gather indices and unique row groups (identical rows share one vision pass)
    vis_index = torch.zeros((B, C), dtype=torch.long)
    is_vis = torch.zeros((B, C), dtype=torch.bool)
    for out_b, src in enumerate(order):
        sel = (ids[out_b, :P] == image_token_id)
        n = int(sel.sum())
        if n:
            vis_index[out_b, :P][sel] = torch.arange(n)
            is_vis[out_b, :P] = sel
    def same_images(a: list[int], b: list[int]) -> bool:
        if len(a) != len(b):
            return False
        for ia, ib in zip(a, b):
            if not torch.equal(grid[ia], grid[ib]):
                return False
            (oa, na), (ob, nb) = offsets[ia], offsets[ib]
            if na != nb or not torch.equal(pix[oa:oa + na], pix[ob:ob + nb]):
                return False
        return True

    groups: list[int] = []
    row_group: list[int] = []
    for out_b, src in enumerate(order):
        for gi, rep in enumerate(groups):
            if same_images(per_row_imgs[order[rep]], per_row_imgs[src]):
                row_group.append(gi)
                break
        else:
            groups.append(out_b)
            row_group.append(len(groups) - 1)
    prepared = _Prepared(ids=ids, pad=pad, pos=pos, base_pos=base_pos, vis_index=vis_index, is_vis=is_vis,
                         vision=[None] * len(groups), row_group=row_group, groups=groups, prompt_len=P, cache_len=C,
                         real_len=real_len)
    prepared.order = order  # type: ignore[attr-defined]
    prepared.per_row_imgs = per_row_imgs  # type: ignore[attr-defined]
    prepared.offsets = offsets  # type: ignore[attr-defined]
    return prepared, B, B0, pix, grid


def _encode_vision(base, config, prepared, pix, grid, device, dtype, mesh) -> torch.Tensor | None:
    """Pooled vision features ``[B, NV, H]`` (NV = static budget tokens), one pass per unique row group."""
    if not any(prepared.per_row_imgs):
        return None
    nv = slot_totals(config.slot_budget)["tokens"]
    per_group = []
    for rep in prepared.groups:
        src = prepared.order[rep]
        imgs = prepared.per_row_imgs[src]
        pix_rows = [pix[prepared.offsets[i][0]:prepared.offsets[i][0] + prepared.offsets[i][1]] for i in imgs]
        grid_rows = [grid[i] for i in imgs]
        if not imgs:
            per_group.append(torch.zeros((nv, base.config.text_config.hidden_size), dtype=dtype, device=device))
            continue
        p, g, n_real = _pad_slots(pix_rows, grid_rows, config.slot_budget)
        feats = base.model.get_image_features(_to(p, device), _to(g, device), return_dict=True).pooler_output
        feats = torch.cat(list(feats), 0).to(dtype)
        if feats.shape[0] != nv:
            raise RuntimeError(f"vision features {feats.shape[0]} != static budget tokens {nv}")
        per_group.append(feats)
    stacked = torch.stack([per_group[gi] for gi in prepared.row_group], 0)
    return _shard(stacked, mesh, ("fsdp", None, None))


# ---------------------------------------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------------------------------------
def _resolve_sampling(model, temperature, do_sample, top_k, top_p):
    gc = getattr(model, "generation_config", None) or getattr(unwrap_causal_lm(model), "generation_config", None)
    if do_sample is False:
        return 0.0, None, None
    if temperature is None:
        temperature = getattr(gc, "temperature", None) or 1.0
    if top_k is None:
        top_k = getattr(gc, "top_k", None)
    if top_p is None:
        top_p = getattr(gc, "top_p", None)
    return float(temperature), top_k, top_p


def _eos_list(model, eos_token_id) -> list[int]:
    if eos_token_id is None:
        gc = getattr(model, "generation_config", None) or getattr(unwrap_causal_lm(model), "generation_config", None)
        eos_token_id = getattr(gc, "eos_token_id", None)
    if eos_token_id is None:
        return []
    return [int(x) for x in (eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])]


def tpu_generate(model: Any, inputs: Mapping[str, Any], *, max_new_tokens: int, temperature: float | None = 0.0,
                 do_sample: bool | None = None, top_k: int | None = None, top_p: float | None = None,
                 eos_token_id: int | Sequence[int] | None = None, pad_token_id: int | None = None,
                 repetition_penalty: float | None = None, config: TPUGenerationConfig | None = None) -> torch.Tensor:
    """Generate for the rows of ``inputs`` (a processor output on the CPU). Returns the new tokens ``[B, T]`` on the CPU:
    each row up to and including its first EOS then ``pad_token_id``; ``T`` is the step at which the last row finished
    (at most ``max_new_tokens``), matching ``generate``."""
    config = config or find_attr(model, ATTR)
    if config is None:
        raise ValueError("TPU generation is not installed on this model (install_tpu_generation)")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    base = unwrap_causal_lm(model)
    body, head = base.model.language_model, base.lm_head
    device = head.weight.device
    dtype = head.weight.dtype
    mesh = config.mesh
    eos = _eos_list(model, eos_token_id)
    if pad_token_id is None:
        pad_token_id = eos[0] if eos else 0
    temperature, top_k, top_p = _resolve_sampling(model, temperature, do_sample, top_k, top_p)
    if repetition_penalty is None:
        gc = getattr(model, "generation_config", None) or getattr(base, "generation_config", None)
        repetition_penalty = getattr(gc, "repetition_penalty", None)
    use_penalty = repetition_penalty not in (None, 1.0)
    if use_penalty and float(repetition_penalty) <= 0:
        raise ValueError("repetition_penalty must be positive")
    q = config.chunk_tokens
    attn_impl = getattr(base.config.text_config if hasattr(base.config, "text_config") else base.config,
                        "_attn_implementation", "sdpa")
    float_mask = attn_impl == "eager"

    prepared, B, B0, pix, grid = _prepare(base, config, inputs, max_new_tokens, pad_token_id)
    P, C = prepared.prompt_len, prepared.cache_len
    config.stats["calls"] += 1
    config.stats["tiers"][C] = config.stats["tiers"].get(C, 0) + 1

    from transformers import StaticCache
    text_config = base.config.get_text_config(decoder=True) if hasattr(base.config, "get_text_config") else base.config

    ctx = torch.no_grad()
    with ctx:
        ids = _shard(_to(prepared.ids, device), mesh, ("fsdp", None))
        pad = _shard(_to(prepared.pad, device), mesh, ("fsdp",))
        pos = _shard(_to(prepared.pos, device), mesh, (None, "fsdp", None))
        vis_index = _shard(_to(prepared.vis_index, device), mesh, ("fsdp", None))
        is_vis = _shard(_to(prepared.is_vis, device), mesh, ("fsdp", None))
        base_pos = _shard(_to(prepared.base_pos, device), mesh, ("fsdp",))
        try:
            vision = _encode_vision(base, config, prepared, pix, grid, device, dtype, mesh)
        except SlotOverflow:
            config.stats["slot_overflow"] += 1
            raise
        _step(device)
        cache = StaticCache(config=text_config, max_cache_len=C)
        arange_q = torch.arange(q, device=device)
        zero = torch.zeros((), dtype=torch.long, device=device)
        prompt_dev = zero + P

        def embed_chunk(start):
            idx = start + arange_q
            tok = body.embed_tokens(ids.index_select(1, idx))
            if vision is None:
                return tok, idx
            vi = vis_index.index_select(1, idx)
            iv = is_vis.index_select(1, idx)
            gathered = vision.gather(1, vi.unsqueeze(-1).expand(-1, -1, vision.shape[-1]))
            return torch.where(iv.unsqueeze(-1), gathered, tok), idx

        last_hidden = None
        for j in range(P // q):
            start = zero + j * q
            embeds, idx = embed_chunk(start)
            masks = attention_masks(pad, start, q, C, float_mask=float_mask, dtype=dtype, linear=True)
            out = body(inputs_embeds=embeds, position_ids=pos.index_select(2, idx), attention_mask=masks,
                       past_key_values=cache, use_cache=True)
            last_hidden = out.last_hidden_state[:, -1]
            _pin_cache(cache, mesh)
            _step(device)
            config.stats["prefill_chunks"] += 1
        seen = None
        if use_penalty:
            # Every real prompt token counts as "seen", exactly like input_ids in generate (host-built, uploaded once).
            host_seen = torch.zeros((B, head.out_features), dtype=torch.bool)
            for b in range(B):
                host_seen[b, prepared.ids[b, int(prepared.pad[b]):P]] = True
            seen = _shard(_to(host_seen, device), mesh, ("fsdp", None))
        sample_kw = dict(temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
        tok = _sample(head(last_hidden), seen=seen, **sample_kw)
        if use_penalty:
            seen = seen.scatter(1, tok.unsqueeze(1), torch.ones((B, 1), dtype=torch.bool, device=device))
        _step(device)

        eos_dev = torch.tensor(eos if eos else [-1], dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        pad_dev = zero + int(pad_token_id)
        generated: list[torch.Tensor] = []
        step = zero.clone()
        finished_host = torch.zeros(B0, dtype=torch.bool)
        for s in range(max_new_tokens):
            emitted = torch.where(done, pad_dev.expand(B), tok)
            done = done | (tok.unsqueeze(-1) == eos_dev).any(-1)
            host = emitted.cpu()
            generated.append(host[:B0])
            finished_host = done.cpu()[:B0]
            config.stats["decode_steps"] += 1
            if bool(finished_host.all()) or s == max_new_tokens - 1:
                break
            cur = prompt_dev + step                                       # cache index of the token we feed now
            embeds = body.embed_tokens(emitted.unsqueeze(1))
            rope = (base_pos + step).view(1, B, 1).expand(3, B, 1)
            masks = attention_masks(pad, cur, 1, C, float_mask=float_mask, dtype=dtype, linear=False)
            out = body(inputs_embeds=embeds, position_ids=rope, attention_mask=masks, past_key_values=cache,
                       use_cache=True)
            tok = _sample(head(out.last_hidden_state[:, -1]), seen=seen, **sample_kw)
            if use_penalty:
                seen = seen.scatter(1, tok.unsqueeze(1), torch.ones((B, 1), dtype=torch.bool, device=device))
            step = step + 1
            _pin_cache(cache, mesh)
            _step(device)
    return torch.stack(generated, 1).long()


def tpu_generate_local(model: Any, **kwargs) -> Any:
    """Drop-in for ``local_runtime.generate_local`` (same keyword arguments as ``model.generate``).

    Returns ``[B, prompt + T]`` sequences like ``generate``; with ``return_dict_in_generate`` an object exposing
    ``.sequences`` and ``.past_key_values`` (always ``None``: the static cache is per call, nothing is reused across turns)."""
    config = find_attr(model, ATTR)
    ids = kwargs["input_ids"]
    inputs = {k: kwargs[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
              if k in kwargs and kwargs[k] is not None}
    unsupported = [k for k in ("pixel_values_videos", "video_grid_thw") if kwargs.get(k) is not None]
    if unsupported:
        raise ValueError(f"TPU generation does not support {unsupported} (static canonical image slots only)")
    if kwargs.get("past_key_values") is not None:
        raise ValueError("TPU generation re-prefills every turn; pass past_key_values=None")
    new = tpu_generate(model, inputs, max_new_tokens=int(kwargs.get("max_new_tokens", 512)),
                       temperature=kwargs.get("temperature"), do_sample=kwargs.get("do_sample"),
                       top_k=kwargs.get("top_k"), top_p=kwargs.get("top_p"),
                       eos_token_id=kwargs.get("eos_token_id"), pad_token_id=kwargs.get("pad_token_id"),
                       repetition_penalty=kwargs.get("repetition_penalty"), config=config)
    # ``streamer`` (live token printing) is accepted and ignored: tokens are read back once per step, not streamed.
    sequences = torch.cat([ids.cpu(), new], dim=1)
    if kwargs.get("return_dict_in_generate"):
        return SimpleNamespace(sequences=sequences, past_key_values=None)
    return sequences


# ---------------------------------------------------------------------------------------------------------------------
# Start-up warm-up
# ---------------------------------------------------------------------------------------------------------------------
def warmup_tpu_generation(model: Any, processor: Any, *, batches: Sequence[int] = (1,), decode_steps: int = 6,
                          max_new_tokens: int = 0, repetition_penalty: float | None = None, log=print) -> dict:
    """Compile every generation program once, per cache tier, before the first rollout (a real 32k-class prefill at the top
    tier, so an out-of-memory shows up in the first minutes of the run report, not after hours of queueing).

    The warm-up prompt is one canonical FULL view (zero pixels) plus filler text sized to land in the tier with whole
    chunks. A tier that fails (``RuntimeError``, e.g. XLA RESOURCE_EXHAUSTED) is REMOVED from the config, so later
    prompts that need it are dropped and counted (``GenerationTooLong``) instead of crashing the run. Returns per-tier
    seconds / status."""
    import time
    config = find_attr(model, ATTR)
    if config is None:
        raise ValueError("TPU generation is not installed on this model")
    base = unwrap_causal_lm(model)
    cfg = base.config
    image_token_id = int(cfg.image_token_id)
    start_id, end_id = int(cfg.vision_start_token_id), int(cfg.vision_end_token_id)
    filler = int(getattr(processor.tokenizer, "convert_tokens_to_ids", lambda t: 1)("a") or 1)
    fam = "FULL" if config.slot_budget.get("FULL", 0) else next(f for f, n in config.slot_budget.items() if n)
    n_vis = tokens_per_image(fam)
    q = config.chunk_tokens
    report: dict[int, dict] = {}
    for tier in list(config.cache_tiers):
        if tier < q + max_new_tokens:
            report[tier] = {"status": f"skipped (cannot hold a prompt chunk plus max_new_tokens={max_new_tokens}; never selected)"}
            continue
        length = tier - q                       # padded prompt = tier - q, so prompt + decode_steps always fits the tier
        n_text = length - n_vis - 2
        if n_text < 1 or length // q < 4:
            report[tier] = {"status": "skipped (tier too small to hold a vision slot and 4 chunks)"}
            continue
        ids = torch.cat([torch.full((n_text // 2,), filler), torch.tensor([start_id]), torch.full((n_vis,), image_token_id),
                         torch.tensor([end_id]), torch.full((n_text - n_text // 2,), filler)]).long()
        mm = torch.zeros_like(ids)
        mm[n_text // 2 + 1:n_text // 2 + 1 + n_vis] = 1
        pixels = torch.zeros((patches_per_image(fam), PATCH_FEATURE_DIM))
        grid = torch.tensor([canonical_grid_thw(fam)])
        for b in batches:
            inputs = {"input_ids": ids.unsqueeze(0).repeat(b, 1), "attention_mask": torch.ones((b, ids.numel()), dtype=torch.long),
                      "pixel_values": pixels.repeat(b, 1), "image_grid_thw": grid.repeat(b, 1),
                      "mm_token_type_ids": mm.unsqueeze(0).repeat(b, 1)}
            t0 = time.perf_counter()
            try:
                tpu_generate(model, inputs, max_new_tokens=decode_steps, temperature=0.0, eos_token_id=[], pad_token_id=0,
                             repetition_penalty=repetition_penalty, config=config)
                dt = time.perf_counter() - t0
                report.setdefault(tier, {})[f"batch{b}_s"] = round(dt, 1)
                log(f"[TPU GEN WARMUP] tier {tier} batch {b}: ok in {dt:.1f}s", flush=True)
            except Exception as exc:    # noqa: BLE001 -- XLA raises RuntimeError / XlaRuntimeError for HBM exhaustion
                report[tier] = {"status": f"FAILED: {type(exc).__name__}: {str(exc)[:300]}"}
                log(f"[TPU GEN WARMUP FAILED] tier {tier} batch {b}: {type(exc).__name__}: {str(exc)[:300]}. This tier is "
                    f"disabled: prompts that need it are dropped and counted, the run continues.", flush=True)
                config.cache_tiers = tuple(t for t in config.cache_tiers if t != tier)
                break
    if not any(t >= q + max_new_tokens for t in config.cache_tiers):
        raise RuntimeError("No TPU generation cache tier is left that can hold max_new_tokens; see the [TPU GEN WARMUP FAILED] lines")
    return report
