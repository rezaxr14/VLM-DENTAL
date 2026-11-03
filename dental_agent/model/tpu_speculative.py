"""Speculative decoding on the TPU static-shape generator (prompt-lookup / n-gram drafts, greedy, exact).

Why this is not "just batch the drafts": Qwen3.5 is hybrid. Its Gated-DeltaNet layers carry a recurrent state and a conv state that a
rejected draft token would have corrupted, and transformers' caches can only crop full-attention layers. So one verification forward
records, for every linear layer, the recurrent state after EACH of the ``k + 1`` fed tokens and the conv window, and the same XLA
graph then commits the state of the last accepted token (and rewinds the full-attention write position) from a DEVICE scalar.
Nothing depends on how many drafts were accepted, so there is one verification program per ``k`` and no host round trip for state.

Exactness (greedy): the tokens emitted are ``t_0 .. t_a`` where ``t_i`` is the target model's own argmax after the ``i``-th fed
token and ``a`` is the number of leading drafts equal to it. The linear layers run the per-token recurrent rule that single-token decode
uses (never the chunked rule), so their states and outputs are the ones sequential decode would produce; the full-attention layers see
a ``k + 1``-wide query block instead of one query, which XLA may reduce in a different order. Equality with non-speculative decoding is
therefore CHECKED (``docs/TPU_SPECULATIVE.md``), never assumed: the benchmark gate compares decoded text, tool calls and answers.

Draft source: ``ngram_k`` = vLLM's ``method=ngram`` configuration (``num_speculative_tokens=k``, ``prompt_lookup_min=1``,
``prompt_lookup_max=k``): the longest suffix of the context (``k`` down to 1 tokens) that occurred earlier, most recent occurrence
first, proposes the tokens that followed it. No draft = an ordinary decode step. ``mtp_k`` (the native MTP head) lives in ``tpu_mtp``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

CONDITION = re.compile(r"^(ngram|mtp)_(\d+)$")


@dataclass(frozen=True)
class SpeculativeConfig:
    method: str = "ngram"
    k: int = 3                      # draft tokens verified per round (the verification block is k + 1 tokens wide)
    prompt_lookup_min: int = 1
    prompt_lookup_max: int = 3

    def __post_init__(self):
        if self.method not in {"ngram", "mtp"}:
            raise ValueError(f"unsupported speculative method {self.method!r}: 'ngram' or 'mtp'")
        if not 1 <= self.k <= 8:
            raise ValueError("speculative k must be in 1..8")
        if not 1 <= self.prompt_lookup_min <= self.prompt_lookup_max:
            raise ValueError("need 1 <= prompt_lookup_min <= prompt_lookup_max")

    @property
    def width(self) -> int:
        return self.k + 1

    def record(self) -> dict:
        if self.method == "mtp":
            return dict(method="mtp", num_speculative_tokens=self.k)
        return dict(method=self.method, num_speculative_tokens=self.k, prompt_lookup_min=self.prompt_lookup_min,
                    prompt_lookup_max=self.prompt_lookup_max)


def parse_condition(name: str) -> SpeculativeConfig | None:
    """``baseline`` -> None; ``ngram_3`` -> SpeculativeConfig(k=3, lookup 1..3) and ``mtp_2`` -> SpeculativeConfig('mtp', k=2), like
    ``specdec.speculative_config``."""
    if name == "baseline":
        return None
    match = CONDITION.match(name)
    if not match:
        raise ValueError(f"unknown TPU speculation condition {name!r} (baseline, ngram_<k>, mtp_<k>)")
    method, k = match.group(1), int(match.group(2))
    return SpeculativeConfig(method=method, k=k, prompt_lookup_min=1, prompt_lookup_max=k)


def ngram_draft(history: list[int], k: int, min_n: int, max_n: int) -> list[int]:
    """Up to ``k`` tokens that followed the most recent earlier occurrence of the longest matching suffix of ``history``."""
    n_hist = len(history)
    for n in range(min(max_n, n_hist - 1), min_n - 1, -1):
        suffix = history[n_hist - n:]
        for start in range(n_hist - n - 1, -1, -1):          # most recent earlier occurrence first
            if history[start:start + n] == suffix:
                follow = history[start + n:start + n + k]
                if follow:
                    return follow
    return []


class StateRecorder:
    """Inside one verification forward: (a) run the per-token recurrent rule and keep the state after every token, and (b) keep each
    linear layer's full conv window. Linear layers are visited in order, so the i-th call belongs to the i-th linear cache layer."""

    def __init__(self, cache: Any, width: int):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
        self.modeling, self.width = modeling, width
        self.linear = [layer for layer in cache.layers if getattr(layer, "is_recurrent_states_initialized", {}).get(0)]
        self.states: list[torch.Tensor] = []     # per linear layer [W, B, H, dk, dv]
        self.convs: list[torch.Tensor] = []      # per linear layer [B, conv_dim, K + W]
        self._saved: Any = None

    def _rule(self, query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, **kw):
        step = self._recurrent
        outs, states, state = [], [], initial_state
        for i in range(self.width):
            out, state = step(query[:, i:i + 1], key[:, i:i + 1], value[:, i:i + 1], g=g[:, i:i + 1], beta=beta[:, i:i + 1],
                              initial_state=state, output_final_state=True,
                              use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)
            outs.append(out)
            states.append(state)
        self.states.append(torch.stack(states))
        return torch.cat(outs, 1), states[-1]

    def __enter__(self):
        m = self.modeling
        self._saved = m.torch_chunk_gated_delta_rule
        self._recurrent = m.torch_recurrent_gated_delta_rule
        m.torch_chunk_gated_delta_rule = self._rule
        for layer in self.linear:
            original = layer.update_conv_state

            def wrapped(conv_states, *args, _orig=original, **kwargs):
                full = _orig(conv_states, *args, **kwargs)
                self.convs.append(full)
                return full
            layer.update_conv_state = wrapped
        return self

    def __exit__(self, *exc):
        self.modeling.torch_chunk_gated_delta_rule = self._saved
        for layer in self.linear:
            layer.__dict__.pop("update_conv_state", None)
        return False

    def commit(self, cache: Any, n: torch.Tensor, write_position: torch.Tensor) -> None:
        """Keep the state after ``n`` accepted tokens (``n`` a device scalar, 1..W) and put every full-attention write position back to
        ``write_position`` (device scalar). Called inside the verification graph."""
        if len(self.states) != len(self.linear) or len(self.convs) != len(self.linear):
            raise RuntimeError(f"recorded {len(self.states)} recurrent / {len(self.convs)} conv states for {len(self.linear)} linear layers: "
                               "the model did not run the per-token rule (unsupported transformers version?)")
        for layer, states, full in zip(self.linear, self.states, self.convs):
            kernel = int(layer.conv_kernel_size[0])
            window = n + torch.arange(kernel, device=n.device)
            layer.conv_states[0].copy_(full.index_select(-1, window))
            layer.recurrent_states[0].copy_(states.index_select(0, (n - 1).reshape(1)).squeeze(0))
        for layer in cache.layers:
            length = getattr(layer, "cumulative_length", None)
            if isinstance(length, torch.Tensor):
                length.copy_(write_position)


def speculative_decode(*, spec: SpeculativeConfig, body: Any, head: Any, cache: Any, mesh: Any, pin_cache, step_fn, shard, attention_masks,
                       float_mask: bool, dtype: torch.dtype, device: torch.device, pad: torch.Tensor, base_pos: torch.Tensor,
                       prompt_dev: torch.Tensor, zero: torch.Tensor, cache_len: int, tok: torch.Tensor, seen: torch.Tensor | None,
                       repetition_penalty: float | None, sample, eos: list[int], max_new_tokens: int, history: list[int],
                       stats: dict, batch: int) -> tuple[list[int], dict]:
    """Greedy speculative decode of a single sequence (all ``batch`` rows are identical replicas). ``tok`` is the already sampled first
    token. Returns ``(emitted token ids including the first, counters)``."""
    W, k = spec.width, spec.k
    use_penalty = seen is not None
    arange_w = torch.arange(W, device=device)
    eos_set = set(eos)
    step = zero.clone()                                   # tokens fed after the prompt (device scalar)

    emitted = [int(tok[0].cpu())]
    history = list(history) + emitted
    counters = dict(rounds=0, plain_steps=0, drafted=0, accepted=0)
    done = emitted[-1] in eos_set or len(emitted) >= max_new_tokens

    def plain_step(tok, seen, step):
        cur = prompt_dev + step
        embeds = body.embed_tokens(tok.unsqueeze(1))
        rope = (base_pos + step).view(1, batch, 1).expand(3, batch, 1)
        masks = attention_masks(pad, cur, 1, cache_len, float_mask=float_mask, dtype=dtype, linear=False)
        out = body(inputs_embeds=embeds, position_ids=rope, attention_mask=masks, past_key_values=cache, use_cache=True)
        new = sample(head(out.last_hidden_state[:, -1]), seen)
        if use_penalty:
            seen = seen.scatter(1, new.unsqueeze(1), torch.ones((batch, 1), dtype=torch.bool, device=device))
        pin_cache(cache, mesh)
        return new, seen, step + 1

    def verify_round(feed, seen, step):
        cur = prompt_dev + step
        embeds = body.embed_tokens(feed)
        rope = ((base_pos + step).view(1, batch, 1) + arange_w.view(1, 1, W)).expand(3, batch, W)
        masks = attention_masks(pad, cur, W, cache_len, float_mask=float_mask, dtype=dtype, linear=False)
        with StateRecorder(cache, W) as recorder:
            out = body(inputs_embeds=embeds, position_ids=rope, attention_mask=masks, past_key_values=cache, use_cache=True)
        logits = head(out.last_hidden_state)                                   # [B, W, V]
        running, picks = seen, []
        for i in range(W):                                                     # each position sees the tokens before it, drafts assumed accepted
            picks.append(sample(logits[:, i], running))
            if use_penalty and i + 1 < W:
                running = running.scatter(1, feed[:, i + 1:i + 2], torch.ones((batch, 1), dtype=torch.bool, device=device))
        t = torch.stack(picks, 1)                                              # [B, W]
        match = (feed[:, 1:] == t[:, :-1]).long()                              # [B, k]
        run, lead = torch.ones_like(match[:, 0]), torch.zeros_like(match[:, 0])
        for j in range(k):                                                     # leading-match count without cumprod (CPU fallback on XLA)
            run = run * match[:, j]
            lead = lead + run
        a = lead[0]                                                            # rows are replicas: row 0 decides
        n = a + 1
        recorder.commit(cache, n, prompt_dev + step + n)
        new = t.gather(1, a.reshape(1, 1).expand(batch, 1)).squeeze(1)
        if use_penalty:
            for j in range(W):
                column = t[:, j:j + 1]
                seen = seen.scatter(1, column, seen.gather(1, column) | (j <= a).reshape(1, 1).expand(batch, 1))
        pin_cache(cache, mesh)
        pack = torch.cat([t, a.reshape(1, 1).expand(batch, 1)], 1)             # one readback per round
        return new, seen, step + n, pack

    while not done:
        draft = ngram_draft(history, k, spec.prompt_lookup_min, spec.prompt_lookup_max)
        if not draft:
            tok, seen, step = plain_step(tok, seen, step)
            step_fn(device)
            new_tokens = [int(tok[0].cpu())]
            counters["plain_steps"] += 1
        else:
            m = len(draft)
            feed_row = [emitted[-1]] + draft + [draft[-1]] * (k - m)
            feed = shard(torch.tensor([feed_row] * batch, dtype=torch.long).to(device), mesh, ("fsdp", None))
            tok, seen, step, pack = verify_round(feed, seen, step)
            step_fn(device)
            host = pack[0].cpu().tolist()
            a = int(host[W])
            new_tokens = host[:a + 1]
            counters["rounds"] += 1
            counters["drafted"] += m
            counters["accepted"] += min(a, m)
        stats["decode_steps"] += 1
        for token in new_tokens:
            if len(emitted) >= max_new_tokens:
                done = True
                break
            emitted.append(int(token))
            history.append(int(token))
            if int(token) in eos_set:
                done = True
                break
        if len(emitted) >= max_new_tokens:
            done = True
    for key, value in counters.items():
        stats["spec_" + key] = stats.get("spec_" + key, 0) + value
    return emitted, counters
