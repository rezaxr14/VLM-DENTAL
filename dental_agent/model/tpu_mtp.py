"""Native multi-token-prediction (MTP) speculation on the TPU static-shape generator (``mtp_1`` .. ``mtp_8``).

Hugging Face's Qwen3.5 drops the checkpoint's ``mtp.*`` tensors, so this module rebuilds the head from the Qwen3.5 decoder layer class
and loads those tensors from the local base snapshot. The head (one full-attention layer + dense MLP, the architecture vLLM serves)
for a pair (target hidden state ``h_i`` after the final norm, embedding of the token ``x_{i+1}`` that followed it) predicts ``x_{i+2}``:

    x   = fc( concat[ pre_fc_norm_embedding(embed(x_{i+1})),  pre_fc_norm_hidden(h_i) ] )      # [2H -> H], embedding first
    x   = decoder_layer(x)            # full attention with its OWN KV cache, rotary position of the hidden state's position
    out = norm(x);   logits = lm_head(out)   # the target's own lm_head (shared); ``out`` is also the hidden state fed to the next draft step

Drafts chain: the first draft comes from the pair (``h`` of the last accepted token, that token), the next from (``out``, previous
draft) at the next position, ``k`` times. Everything stays on the device: the drafts of a round are the output of the previous round's
graph, so a decode round is ONE program (verify ``k + 1`` tokens, roll the hybrid state back, refill the head's KV cache from the
verified tokens, draft ``k`` more) and one host readback.

Exactness is the same as for n-gram speculation (``tpu_speculative``): the emitted tokens are always the target model's own greedy
choices, the drafts only decide how many are emitted per forward. A wrong draft head therefore lowers acceptance and can never change
the output; whether the head is wired exactly like vLLM's is visible in the measured acceptance rate, not assumed.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn


class MTPHead(nn.Module):
    def __init__(self, text_config: Any):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
        cfg = head_config(text_config)
        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, 0)])
        self.norm = Qwen3_5RMSNorm(h, eps=eps)

    def forward(self, hidden: torch.Tensor, embeds: torch.Tensor, position_embeddings, attention_mask, cache) -> torch.Tensor:
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden)], dim=-1))
        x = self.layers[0](x, position_embeddings=position_embeddings, attention_mask=attention_mask, position_ids=None,
                           past_key_values=cache, use_cache=True)
        return self.norm(x)


def head_config(text_config: Any) -> Any:
    """The text config of a one-layer, full-attention stack (the head's own KV cache and layer)."""
    cfg = copy.deepcopy(text_config)
    cfg.num_hidden_layers = 1
    cfg.layer_types = ["full_attention"]
    return cfg


def load_mtp_state(folder: str | Path) -> dict[str, torch.Tensor]:
    """The ``mtp.*`` tensors of a local safetensors snapshot, prefix stripped (``fc.weight``, ``layers.0.self_attn.q_proj.weight`` ...)."""
    from safetensors import safe_open
    from dental_agent.harness.checkpoints import tensor_headers
    headers = tensor_headers(folder)
    names = sorted(n for n in headers if n.startswith("mtp."))
    if not names:
        raise ValueError(f"{folder} has no mtp.* tensors: the native MTP head is not part of this checkpoint")
    by_file: dict[Path, list[str]] = {}
    for name in names:
        by_file.setdefault(headers[name]["file"], []).append(name)
    state = {}
    for file, group in by_file.items():
        with safe_open(str(file), framework="pt") as handle:
            for name in group:
                state[name[len("mtp."):]] = handle.get_tensor(name)
    return state


def build_mtp_head(folder: str | Path, text_config: Any, *, device: Any, dtype: torch.dtype) -> MTPHead:
    """Instantiate, load (strictly: every tensor must match by name and shape) and freeze the head on ``device``."""
    state = load_mtp_state(folder)
    head = MTPHead(text_config)
    expected = head.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    wrong = {n: (tuple(state[n].shape), tuple(expected[n].shape)) for n in expected if n in state and state[n].shape != expected[n].shape}
    if missing or unexpected or wrong:
        raise ValueError("the checkpoint's mtp.* tensors do not match the head built from the text config "
                         f"(this layout was never verified against the real 9B checkpoint):\n  missing in checkpoint: {missing}\n"
                         f"  unexpected in checkpoint: {unexpected}\n  shape mismatch (checkpoint, expected): {wrong}")
    head.load_state_dict(state, strict=True)
    head.requires_grad_(False).eval()
    return head.to(device=device, dtype=dtype)


def attach_mtp_head(model: Any, folder: str | Path) -> Any:
    """Load the base snapshot's MTP head next to an installed TPU generator (``find_attr(model, ATTR).mtp_head``)."""
    from dental_agent.model import tpu_generation as tg
    config = tg.find_attr(model, tg.ATTR)
    if config is None:
        raise ValueError("install_tpu_generation first")
    base = tg.unwrap_causal_lm(model)
    text_config = base.config.get_text_config(decoder=True) if hasattr(base.config, "get_text_config") else base.config
    device, dtype = base.lm_head.weight.device, base.lm_head.weight.dtype
    config.mtp_head = build_mtp_head(folder, text_config, device=device, dtype=dtype)
    n = sum(p.numel() for p in config.mtp_head.parameters())
    print(f"[TPU GEN] MTP head attached from {folder}: {n / 1e6:.1f}M parameters, replicated on every core", flush=True)
    return config.mtp_head


class MTPDrafter:
    """The head plus its own static KV cache (one full-attention layer, same length as the target's) for ONE generate call."""

    def __init__(self, head: MTPHead, *, text_config: Any, body: Any, lm_head: Any, mesh: Any, pin_cache, attention_masks, float_mask: bool,
                 dtype: torch.dtype, device: torch.device, pad: torch.Tensor, base_pos: torch.Tensor, prompt_dev: torch.Tensor,
                 cache_len: int, batch: int, k: int):
        from transformers import StaticCache
        self.head, self.body, self.lm_head, self.mesh, self.pin_cache = head, body, lm_head, mesh, pin_cache
        self.attention_masks, self.float_mask, self.dtype, self.device = attention_masks, float_mask, dtype, device
        self.pad, self.base_pos, self.prompt_dev, self.cache_len, self.batch, self.k = pad, base_pos, prompt_dev, cache_len, batch, k
        self.cache = StaticCache(config=head_config(text_config), max_cache_len=cache_len)

    # -- cache position -------------------------------------------------------------------------------------------------
    def set_len(self, length: torch.Tensor) -> None:
        """Move the write position (device scalar). Entries beyond it are overwritten before anything attends to them."""
        for layer in self.cache.layers:
            current = getattr(layer, "cumulative_length", None)
            if isinstance(current, torch.Tensor):
                current.copy_(length)

    # -- one forward over L consecutive slots starting at ``slot`` --------------------------------------------------------
    def run(self, hidden: torch.Tensor, token_embeds: torch.Tensor, rope: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        masks = self.attention_masks(self.pad, slot, hidden.shape[1], self.cache_len, float_mask=self.float_mask, dtype=self.dtype, linear=False)
        position_embeddings = self.body.rotary_emb(hidden, rope)
        out = self.head(hidden, token_embeds, position_embeddings, masks["full_attention"], self.cache)
        self.pin_cache(self.cache, self.mesh)
        return out

    def propose(self, out: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Draft token from a head output ``[B, H]``. ``index`` (device scalar): position of that token in the generated sequence --
        unused here, it is the seam tests use to substitute an oracle."""
        return self.lm_head(out).argmax(-1)

    def _rope(self, offset: torch.Tensor, length: int = 1) -> torch.Tensor:
        r = (self.base_pos + offset).view(1, self.batch, 1) + torch.arange(length, device=self.device).view(1, 1, length)
        return r.expand(3, self.batch, length)

    # -- prefill: fill the head's cache from the prompt's hidden states ------------------------------------------------------
    def prefill_chunk(self, hidden: torch.Tensor, ids: torch.Tensor, rope: torch.Tensor, idx: torch.Tensor, start: torch.Tensor,
                      prompt_len: int) -> None:
        """Pairs (h_i, embed(x_{i+1})) for the chunk's positions ``idx``. The last prompt position has no successor yet (its pair is
        rewritten by ``start``); it reads its own token, which is harmless because that slot is overwritten before use."""
        nxt = ids.index_select(1, (idx + 1).clamp(max=prompt_len - 1))
        self.run(hidden, self.body.embed_tokens(nxt), rope, start)

    def start(self, last_hidden: torch.Tensor, first_token: torch.Tensor, last_rope: torch.Tensor) -> torch.Tensor:
        """After the first token is sampled: rewrite the last prompt pair with its real successor and draft ``k`` tokens. -> ``[B, k]``."""
        P = self.prompt_dev
        self.set_len(P - 1)
        out = self.run(last_hidden.unsqueeze(1), self.body.embed_tokens(first_token.unsqueeze(1)), last_rope, P - 1)
        self.set_len(P)
        return self._chain(out[:, 0], P, self.base_pos, first_index=1)

    def _chain(self, out: torch.Tensor, slot: torch.Tensor, rope_offset: torch.Tensor, *, first_index) -> torch.Tensor:
        """``out`` = head output of the last verified pair. Draft 1 is its prediction; draft i >= 2 comes from (previous out, previous draft)
        written at ``slot + i - 2`` (rope ``rope_offset + i - 2``)."""
        drafts = [self.propose(out, first_index)]
        for i in range(2, self.k + 1):
            token = drafts[-1].unsqueeze(1)
            rope = (rope_offset + (i - 2)).view(1, self.batch, 1).expand(3, self.batch, 1)
            out = self.run(out.unsqueeze(1), self.body.embed_tokens(token), rope, slot + (i - 2))[:, 0]
            drafts.append(self.propose(out, first_index + i - 1))
        self.set_len(slot)
        return torch.stack(drafts, 1)

    def after_round(self, block_hidden: torch.Tensor, t: torch.Tensor, a: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Refill the head's cache from one verified block and draft for the next round.

        ``block_hidden`` [B, W, H]: the target's hidden states of the fed block; ``t`` [B, W]: the target's own token after each fed
        token; ``a``: accepted drafts (device scalar); ``step``: tokens fed before this block. Pairs (h_j, embed(t_j)) are valid for
        j <= a; the rest is rewound. -> next drafts ``[B, k]``."""
        P, W = self.prompt_dev, block_hidden.shape[1]
        n = a + 1
        out = self.run(block_hidden, self.body.embed_tokens(t), self._rope(step, W), P + step)         # writes slots P+step .. P+step+W-1
        self.set_len(P + step + n)
        out_a = out.gather(1, a.reshape(1, 1, 1).expand(self.batch, 1, out.shape[-1])).squeeze(1)
        return self._chain(out_a, P + step + n, self.base_pos + step + n, first_index=step + n + 1)


def mtp_decode(*, spec, drafter: MTPDrafter, body: Any, head: Any, cache: Any, mesh: Any, pin_cache, step_fn, device: torch.device,
               attention_masks, float_mask: bool, dtype: torch.dtype, pad: torch.Tensor, base_pos: torch.Tensor, prompt_dev: torch.Tensor,
               zero: torch.Tensor, cache_len: int, tok: torch.Tensor, seen: torch.Tensor | None, drafts: torch.Tensor, sample, eos: list[int],
               max_new_tokens: int, stats: dict, batch: int) -> tuple[list[int], dict]:
    """Greedy decode of one sequence with a device-resident draft per round. ``tok`` is the first sampled token and ``drafts`` the head's
    first ``k`` drafts after it. Returns ``(emitted ids incl. the first, counters)``."""
    from dental_agent.model.tpu_speculative import StateRecorder
    W, k = spec.width, spec.k
    use_penalty = seen is not None
    arange_w = torch.arange(W, device=device)
    step = zero.clone()
    eos_set = set(eos)
    emitted = [int(tok[0].cpu())]
    counters = dict(rounds=0, plain_steps=0, drafted=0, accepted=0)
    done = emitted[-1] in eos_set or len(emitted) >= max_new_tokens

    def verify_round(tok, drafts, seen, step):
        feed = torch.cat([tok.unsqueeze(1), drafts], 1)                                        # [B, W]
        cur = prompt_dev + step
        embeds = body.embed_tokens(feed)
        rope = ((base_pos + step).view(1, batch, 1) + arange_w.view(1, 1, W)).expand(3, batch, W)
        masks = attention_masks(pad, cur, W, cache_len, float_mask=float_mask, dtype=dtype, linear=False)
        with StateRecorder(cache, W) as recorder:
            out = body(inputs_embeds=embeds, position_ids=rope, attention_mask=masks, past_key_values=cache, use_cache=True)
        hidden = out.last_hidden_state
        logits = head(hidden)
        running, picks = seen, []
        for i in range(W):
            picks.append(sample(logits[:, i], running))
            if use_penalty and i + 1 < W:
                running = running.scatter(1, feed[:, i + 1:i + 2], torch.ones((batch, 1), dtype=torch.bool, device=device))
        t = torch.stack(picks, 1)
        match = (feed[:, 1:] == t[:, :-1]).long()
        run, lead = torch.ones_like(match[:, 0]), torch.zeros_like(match[:, 0])
        for j in range(k):
            run = run * match[:, j]
            lead = lead + run
        a = lead[0]
        n = a + 1
        recorder.commit(cache, n, prompt_dev + step + n)
        new = t.gather(1, a.reshape(1, 1).expand(batch, 1)).squeeze(1)
        if use_penalty:
            for j in range(W):
                column = t[:, j:j + 1]
                seen = seen.scatter(1, column, seen.gather(1, column) | (j <= a).reshape(1, 1).expand(batch, 1))
        new_drafts = drafter.after_round(hidden, t, a, step)
        pin_cache(cache, mesh)
        pack = torch.cat([t, a.reshape(1, 1).expand(batch, 1)], 1)
        return new, new_drafts, seen, step + n, pack

    while not done:
        tok, drafts, seen, step, pack = verify_round(tok, drafts, seen, step)
        step_fn(device)
        host = pack[0].cpu().tolist()
        a = int(host[W])
        counters["rounds"] += 1
        counters["drafted"] += k
        counters["accepted"] += a
        stats["decode_steps"] += 1
        for token in host[:a + 1]:
            if len(emitted) >= max_new_tokens:
                break
            emitted.append(int(token))
            if int(token) in eos_set:
                done = True
                break
        if len(emitted) >= max_new_tokens:
            done = True
    for key, value in counters.items():
        stats["spec_" + key] = stats.get("spec_" + key, 0) + value
    return emitted, counters
