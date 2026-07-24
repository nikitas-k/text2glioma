"""Inspect U-Net cross-attention allocation on the molecular token position.

Diagnostic that answers: "Is the U-Net actually reading the IDH conditioning
token, or is it drowning under the 128 RadBERT text tokens?"

The molecular token is appended as token index 128 (129th total) of the
cross-attention context. If the U-Net cross-attention allocates less than
its fair share of attention mass to that position, the IDH class simply
can't propagate into the generated pixels regardless of how discriminative
the trained embeddings are. This exactly matches the failure observed in
the synth-only classifier grid at CFG=1.0 (val AUROC ~0.5) and gives us a
concrete lever to pull (token position, replication count, FiLM injection,
etc.) if a higher-CFG regeneration still doesn't lift downstream accuracy.

Method
------
1. Load stage-1 VAE + stage-2 LDM + molecular head via the standard engine.
2. Build a controlled batch of B contexts:
     - Empty text prompt for all B samples (isolates the IDH-token effect).
     - Half of B tagged IDH=WT, half tagged IDH=MUT.
3. Run ONE U-Net forward pass at a mid-noise timestep on a random-latent
   input (no real data required; we only care about the attention pattern
   as a function of the input context, not the denoised output).
4. Override ``CrossAttention._attention`` on every cross-attention module
   to stash the softmax output ``attention_probs`` (B*heads, N_query, N_key).
5. Distinguish cross-attn vs self-attn calls by inspecting the key length:
   only calls with ``key.shape[1] == expected_context_len`` are cross-attn
   over the [text || molecular] context.
6. For each cross-attention layer, compute per-token attention mass
   (mean over batch*heads*queries) and split into three groups:
     - ``text_mass_avg``: mean attention per RadBERT text token (positions 0..127)
     - ``idh_mass``:      attention on the IDH token (position 128)
     - ``ratio``:         ``idh_mass / text_mass_avg`` — the null hypothesis
                          for uniform attention is 1.0; << 1 means the IDH
                          token is being starved.
7. Also compute the WT-vs-MUT difference in the per-position attention
   pattern. If the U-Net changes its attention when the IDH label flips,
   at least *some* signal is propagating.

Outputs
-------
* ``unet_attention_report.json``: per-layer stats keyed by module path.
* ``unet_attention_report.png``:
    - Per-layer bar of ``idh_mass`` vs ``text_mass_avg`` (log-y).
    - Per-layer WT-vs-MUT delta on the IDH token position.

Usage
-----
::

    python scripts/analyse_unet_attention.py \\
        --stage1_config configs/stage1.yaml \\
        --stage2_config configs/ldm_radbert.yaml \\
        --stage1_ckpt   /path/to/stage1/checkpoint.pth \\
        --stage2_ckpt   /path/to/ldm_stage2_molecular_idh_only/best_model.pth \\
        --out_dir       results/unet_attn_idh_only \\
        --batch_size    8 \\
        --timestep      500 \\
        --device        cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# Local imports
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from text2glioma.inference.engine import Text2GliomaEngine  # noqa: E402
from text2glioma.utils import load_config  # noqa: E402
from text2glioma.training.molecular_conditioning import (  # noqa: E402
    IDH_WILDTYPE, IDH_MUTANT,
)


# ---------------------------------------------------------------------
# Attention capture: override _attention on every CrossAttention module
# ---------------------------------------------------------------------

def _make_capturing_attention(expected_ctx_len: int):
    """Return a replacement for ``CrossAttention._attention`` that stashes
    the softmax attention probabilities on the module *only when* the
    key length matches the expected cross-attention context length.

    Self-attention calls (where key comes from the spatial input) will
    have key lengths much larger than expected_ctx_len, so this filter
    reliably keeps only the cross-context calls we care about.
    """
    def _capturing_attention(self, query: torch.Tensor, key: torch.Tensor,
                              value: torch.Tensor) -> torch.Tensor:
        # Recomputed inline instead of super()-calling so we don't have
        # to depend on the exact torch.baddbmm signature of the
        # upstream _attention (which changes across MONAI versions).
        dtype = query.dtype
        if getattr(self, "upcast_attention", False):
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1],
                        dtype=query.dtype, device=query.device),
            query, key.transpose(-1, -2),
            beta=0, alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype=dtype)

        if key.shape[1] == expected_ctx_len:
            # (B*heads, N_query, N_key). Detach + CPU to avoid growing
            # the graph across dozens of layers on the GPU.
            self._last_cross_attn_probs = attention_probs.detach().cpu()

        x = torch.bmm(attention_probs, value)
        return x
    return _capturing_attention


def _install_hooks(unet: torch.nn.Module, expected_ctx_len: int) -> list:
    """Walk the U-Net, override ``_attention`` on every ``CrossAttention``,
    register a forward pre-hook on ``to_v`` to also capture its input
    (the per-layer projected context, whose channel dim may differ from
    the raw 768-D embedding), and return the list of (name, module)
    tuples in traversal order."""
    from generative.networks.nets.diffusion_model_unet import CrossAttention
    replacement = _make_capturing_attention(expected_ctx_len)
    hooked: list[tuple[str, CrossAttention]] = []

    def _make_v_input_hook(mod: CrossAttention):
        def _hook(_module, args, _kwargs=None):
            # to_v is nn.Linear(cross_attention_dim, inner_dim). Its input
            # is (B, N_key, cross_attention_dim). We stash the tensor so
            # the V-projection probe can compare WT vs MUT at the actual
            # per-layer projected representation instead of the raw
            # 768-D molecular-head embeddings (which fail with a shape
            # mismatch on layers that project context first).
            if args and args[0].shape[1] == expected_ctx_len:
                mod._last_to_v_input = args[0].detach().cpu()
        return _hook

    for name, m in unet.named_modules():
        if isinstance(m, CrossAttention):
            m._attention = types.MethodType(replacement, m)
            m._last_cross_attn_probs = None
            m._last_to_v_input = None
            # Forward pre-hook on the Linear to_v. Modern torch versions
            # accept the with_kwargs kwarg; older ones don't, so fall
            # back to args-only.
            try:
                m.to_v.register_forward_pre_hook(
                    _make_v_input_hook(m), with_kwargs=True,
                )
            except TypeError:
                m.to_v.register_forward_pre_hook(_make_v_input_hook(m))
            hooked.append((name, m))
    return hooked


# ---------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------

def _build_batch(engine: Text2GliomaEngine, batch_size: int,
                 timestep: int, seed: int) -> dict:
    """Build a controlled forward-pass batch that isolates the IDH
    conditioning-token effect.

    * Text: empty prompt for every sample (removes text-dependent variance).
    * IDH:  half the batch tagged WT, half tagged MUT.
    * MGMT: none (IDH-only head).
    * Latent: random noise (attention pattern is a function of context, not
      of the noisy input's fine structure).
    * Mask condition: zeros (unconditional-mask direction; the mask channels
      are then a constant offset shared by every sample in the batch).
    """
    device = engine.device
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    B = int(batch_size)
    if B % 2 != 0:
        raise ValueError(f"batch_size must be even to split WT/MUT (got {B})")
    half = B // 2

    # Context: empty prompt encoded once, then tiled.
    tok = engine.tokenizer("", return_tensors="pt", padding="max_length",
                            truncation=True, max_length=128).to(device)
    with torch.no_grad():
        text_emb = engine.text_encoder(**tok).last_hidden_state  # (1, 128, D)
    text_emb = text_emb.repeat(B, 1, 1)                          # (B, 128, D)

    if engine.molecular_head is None:
        raise ValueError("engine has no molecular head; nothing to inspect")

    # IDH tokens: WT for first half, MUT for second half.
    idh_vals = torch.tensor(
        [IDH_WILDTYPE] * half + [IDH_MUTANT] * half,
        dtype=torch.long, device=device,
    )
    mol_kwargs = {"idh": idh_vals}
    if "mgmt" in engine.molecular_head.fields:
        from text2glioma.training.molecular_conditioning import MGMT_UNKNOWN
        mol_kwargs["mgmt"] = torch.full((B,), MGMT_UNKNOWN,
                                         dtype=torch.long, device=device)
    with torch.no_grad():
        mol_tokens = engine.molecular_head(**mol_kwargs).to(text_emb.dtype)
    ctx = torch.cat([text_emb, mol_tokens], dim=1)  # (B, 128 + n_fields, D)

    # Latent + mask channels.
    latent = torch.randn((B, engine.stage1_latent_ch) + engine.latent_spatial,
                          device=device)
    mask   = torch.zeros((B, engine.num_mask_classes) + engine.latent_spatial,
                          device=device)
    x_in   = torch.cat([latent, mask], dim=1)
    ts     = torch.full((B,), int(timestep), dtype=torch.long, device=device)

    return {
        "x_in": x_in,
        "ts":   ts,
        "ctx":  ctx,
        "half": half,
        "ctx_len": ctx.shape[1],
    }


# ---------------------------------------------------------------------
# V-projection probe (static, no forward pass)
# ---------------------------------------------------------------------

def _v_projection_probe(engine: Text2GliomaEngine,
                         hooked: list,
                         idh_token_pos: int,
                         batch_half: int) -> dict:
    """Per-layer cosine similarity between ``to_v(WT_ctx)`` and
    ``to_v(MUT_ctx)`` at the IDH token position, using the *actual
    projected* context each layer received during the forward pass.

    Rationale: cross-attention output is ``softmax(QK^T) @ V``. Even when
    the softmax weights on the IDH position differ across WT / MUT batches
    (as confirmed by the attention-mass diagnostic above), if the ``to_v``
    projection collapses the two class embeddings onto near-identical
    vectors then the attention-weighted sum is nearly identical and the
    class distinction never reaches the pixel decoder.

    Why we use the captured pre-projected context instead of the raw 768-D
    embeddings:
      * Some U-Net layers have their own upstream projection (e.g. from
        768 down to 512) applied before cross-attention. Pushing the raw
        head embedding through ``to_v`` fails with a shape mismatch on
        those layers.
      * The captured tensor is exactly what fed into ``to_v`` in that
        specific layer during the forward pass, so the probe measures the
        real behaviour at the exact channel dim used by that block.

    Assumes the forward-pass batch was structured as [WT half | MUT half]
    by ``_build_batch`` (default). Skips any layer whose ``to_v`` pre-hook
    didn't fire (e.g. layers unreachable in this forward pass).

    Read the ``value_cos`` column:
      * < 0.5  : V pathway differentiates WT and MUT strongly. If the
                 downstream classifier still fails, the fault is later
                 (post-attention MLP, or CFG=1 masking the small delta).
      * 0.5-0.9: mild collapse; class info is preserved but subtle.
      * > 0.9  : V pathway has effectively collapsed the two classes.
                 Even perfect attention allocation cannot rescue this;
                 needs an auxiliary contrastive loss on ``to_v`` outputs
                 or a FiLM-style bypass.
    """
    if engine.molecular_head is None or "idh" not in engine.molecular_head.fields:
        return {}
    from torch.nn.functional import cosine_similarity

    probe: dict = {}
    for name, mod in hooked:
        ctx_in = getattr(mod, "_last_to_v_input", None)
        if ctx_in is None:
            continue
        # ctx_in: (B, ctx_len, per_layer_dim). Slice the IDH token
        # position and average within each class half so we get one
        # vector per class regardless of batch size.
        if ctx_in.shape[1] <= idh_token_pos:
            continue
        idh_ctx = ctx_in[:, idh_token_pos, :]           # (B, dim)
        wt_ctx  = idh_ctx[:batch_half].mean(dim=0, keepdim=True)   # (1, dim)
        mut_ctx = idh_ctx[batch_half:].mean(dim=0, keepdim=True)   # (1, dim)

        # Push through the same to_v that ran during the forward pass.
        # to_v is on-device; move inputs to match.
        target_device = next(mod.to_v.parameters()).device
        with torch.no_grad():
            v_wt  = mod.to_v(wt_ctx.to(target_device))
            v_mut = mod.to_v(mut_ctx.to(target_device))
        cos = float(cosine_similarity(v_wt, v_mut, dim=-1).item())
        # Norms are informative too: a near-zero-norm V collapses to the
        # attention baseline regardless of embedding identity.
        probe[name] = {
            "value_cos":       cos,
            "value_norm_wt":   float(v_wt.norm().item()),
            "value_norm_mut":  float(v_mut.norm().item()),
            "value_delta_l2":  float((v_wt - v_mut).norm().item()),
            "ctx_dim":         int(ctx_in.shape[-1]),
        }
    return probe


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage1_config", type=Path, required=True)
    ap.add_argument("--stage2_config", type=Path, required=True)
    ap.add_argument("--stage1_ckpt",   type=Path, required=True)
    ap.add_argument("--stage2_ckpt",   type=Path, required=True)
    ap.add_argument("--molecular_head_ckpt", type=Path, default=None,
                    help="Explicit molecular head checkpoint. Default: look "
                         "for best_molecular_head.pth next to --stage2_ckpt.")
    ap.add_argument("--out_dir",       type=Path, required=True)
    ap.add_argument("--batch_size",    type=int, default=8,
                    help="Even; half tagged IDH=WT, half IDH=MUT.")
    ap.add_argument("--timestep",      type=int, default=500,
                    help="Diffusion timestep at which to probe attention. "
                         "Mid-schedule (~500) is representative; at t=0 the "
                         "denoiser is near-deterministic and at t=999 it is "
                         "unconditional-noise-dominated.")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--device",        type=str, default="cuda")
    ap.add_argument("--cache_dir",     type=str, default=None,
                    help="HuggingFace cache directory (for RadBERT).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load engine ---------------------------------------
    print(f"[load] engine on {args.device} ...", file=sys.stderr, flush=True)
    engine = Text2GliomaEngine(
        stage1_config=load_config(str(args.stage1_config)),
        stage2_config=load_config(str(args.stage2_config)),
        stage1_ckpt=str(args.stage1_ckpt),
        stage2_ckpt=str(args.stage2_ckpt),
        molecular_head_ckpt=(str(args.molecular_head_ckpt)
                              if args.molecular_head_ckpt else None),
        device=args.device,
        cache_dir=args.cache_dir,
    )
    if engine.molecular_head is None:
        raise SystemExit(
            "engine has no molecular_head loaded; nothing to analyse. Pass "
            "--molecular_head_ckpt or place best_molecular_head.pth next to "
            "the stage-2 checkpoint."
        )
    print(f"[load] molecular_head.fields={engine.molecular_head.fields}",
          file=sys.stderr, flush=True)

    # ---------- Build batch ---------------------------------------
    batch = _build_batch(engine, args.batch_size, args.timestep, args.seed)
    B    = args.batch_size
    half = batch["half"]
    ctx_len = batch["ctx_len"]
    n_text  = ctx_len - len(engine.molecular_head.fields)
    print(f"[batch] B={B} (half={half})  ctx_len={ctx_len}  "
          f"(text={n_text}, molecular={len(engine.molecular_head.fields)})  "
          f"timestep={args.timestep}",
          file=sys.stderr, flush=True)

    # ---------- Install attention-capture hooks -------------------
    hooked = _install_hooks(engine.model, expected_ctx_len=ctx_len)
    print(f"[hook] {len(hooked)} CrossAttention modules instrumented",
          file=sys.stderr, flush=True)

    # ---------- Forward pass --------------------------------------
    engine.model.eval()
    with torch.no_grad():
        _ = engine.model(x=batch["x_in"], timesteps=batch["ts"],
                          context=batch["ctx"])

    # ---------- Aggregate ----------------------------------------
    fields = list(engine.molecular_head.fields)
    idh_token_pos = n_text  # first molecular token
    field_positions = {f: n_text + i for i, f in enumerate(fields)}

    report: dict = {
        "batch_size":     B,
        "timestep":       int(args.timestep),
        "seed":           int(args.seed),
        "n_text":         int(n_text),
        "field_positions": {k: int(v) for k, v in field_positions.items()},
        "ctx_len":        int(ctx_len),
        "layers":         OrderedDict(),
        "summary":        {},
    }

    layer_names: list[str] = []
    per_layer_text_avg: list[float] = []
    per_layer_field_masses: dict[str, list[float]] = {f: [] for f in fields}
    per_layer_ratios:      dict[str, list[float]] = {f: [] for f in fields}
    per_layer_wt_mut_delta: dict[str, list[float]] = {f: [] for f in fields}

    n_heads = int(getattr(hooked[0][1], "num_heads", 1)) if hooked else 1

    for name, mod in hooked:
        probs = mod._last_cross_attn_probs
        if probs is None:
            # Layer wasn't reached in this forward pass (rare — e.g.
            # conditional-only branches) — skip cleanly.
            continue
        # probs: (B*heads, N_query, N_key). Reshape to (B, heads, Nq, Nk)
        # then mean over heads & queries to get per-token mass, keeping B.
        BH, Nq, Nk = probs.shape
        H = BH // B
        p = probs.view(B, H, Nq, Nk).mean(dim=(1, 2))    # (B, Nk)

        text_mass_avg = p[:, :n_text].mean(dim=1)         # (B,)
        # WT and MUT halves (order defined in _build_batch: [WT half, MUT half])
        text_avg_all = float(text_mass_avg.mean().item())

        layer_stats: dict = {
            "n_query":       int(Nq),
            "n_key":         int(Nk),
            "n_heads":       int(H),
            "text_mass_avg": text_avg_all,
        }
        for f in fields:
            pos = field_positions[f]
            f_mass = p[:, pos]                            # (B,)
            f_wt   = float(f_mass[:half].mean().item())
            f_mut  = float(f_mass[half:].mean().item())
            f_all  = float(f_mass.mean().item())
            ratio  = f_all / max(text_avg_all, 1e-12)
            layer_stats[f] = {
                "mass_all":  f_all,
                "mass_wt":   f_wt,
                "mass_mut":  f_mut,
                "wt_mut_delta_abs": abs(f_wt - f_mut),
                "ratio_vs_text": ratio,
            }
            per_layer_field_masses[f].append(f_all)
            per_layer_ratios[f].append(ratio)
            per_layer_wt_mut_delta[f].append(abs(f_wt - f_mut))

        report["layers"][name] = layer_stats
        layer_names.append(name)
        per_layer_text_avg.append(text_avg_all)

    # ---------- Global summary -----------------------------------
    def _stat(vals: list[float]) -> dict:
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return {"n": 0}
        return {
            "n":     int(arr.size),
            "mean":  float(arr.mean()),
            "std":   float(arr.std()),
            "min":   float(arr.min()),
            "max":   float(arr.max()),
            "median":float(np.median(arr)),
        }

    report["summary"]["text_mass_avg"] = _stat(per_layer_text_avg)
    for f in fields:
        report["summary"][f] = {
            "mass":            _stat(per_layer_field_masses[f]),
            "ratio_vs_text":   _stat(per_layer_ratios[f]),
            "wt_mut_delta_abs":_stat(per_layer_wt_mut_delta[f]),
        }

    # ---------- V-projection probe (IDH only) --------------------
    v_probe = _v_projection_probe(engine, hooked,
                                   idh_token_pos=idh_token_pos,
                                   batch_half=half)
    # Fold into per-layer records so both signals live side-by-side
    # in the JSON.
    for name, stats in v_probe.items():
        if name in report["layers"]:
            report["layers"][name]["v_probe"] = stats

    if v_probe:
        v_cos_all = np.asarray(
            [v_probe[n]["value_cos"] for n in layer_names if n in v_probe],
            dtype=np.float64,
        )
        report["summary"]["value_cos_idh_wt_vs_mut"] = _stat(v_cos_all.tolist())

    # Uniform-attention reference: 1 / ctx_len per token.
    report["uniform_reference_mass_per_token"] = 1.0 / ctx_len

    out_json = args.out_dir / "unet_attention_report.json"
    out_json.write_text(json.dumps(report, indent=2))

    # ---------- Pretty print -------------------------------------
    print(f"\nAnalysed {len(layer_names)} cross-attention layers "
          f"(context length {ctx_len} = {n_text} text + {len(fields)} molecular).")
    print(f"Uniform reference: {1.0 / ctx_len:.5f} mass per token.\n")

    print(f"Text tokens: mean mass per token across layers = "
          f"{np.mean(per_layer_text_avg):.5f}")
    for f in fields:
        ratios = per_layer_ratios[f]
        deltas = per_layer_wt_mut_delta[f]
        print(f"\n[{f.upper()} @ position {field_positions[f]}]")
        print(f"  mean mass (per layer):       "
              f"{np.mean(per_layer_field_masses[f]):.5f} "
              f"(vs text-per-token {np.mean(per_layer_text_avg):.5f})")
        print(f"  ratio vs text-per-token:     "
              f"mean={np.mean(ratios):.3f}  median={np.median(ratios):.3f}  "
              f"min={np.min(ratios):.3f}  max={np.max(ratios):.3f}")
        print(f"  |WT-MUT| delta (per layer):  "
              f"mean={np.mean(deltas):.6f}  max={np.max(deltas):.6f}")
        print(f"  Interpretation:")
        r_mean = np.mean(ratios)
        if r_mean < 0.5:
            print(f"    ratio << 1: token is STARVED. Attention prepend / "
                  f"replicate / FiLM likely the fix.")
        elif r_mean < 1.5:
            print(f"    ratio ~ 1: token gets its uniform share of attention. "
                  f"If downstream classifier still fails, attention exists but "
                  f"the value vector is ignored -> auxiliary bottleneck loss "
                  f"or FiLM.")
        else:
            print(f"    ratio > 1: token is over-attended. Problem is not "
                  f"attention starvation; look at gradient flow or value "
                  f"projection.")
        d_mean = np.mean(deltas)
        if d_mean < 1e-4:
            print(f"    |WT-MUT| ~ 0: attention pattern does NOT change when "
                  f"the IDH label flips -> the token identity is being ignored.")
        else:
            print(f"    |WT-MUT| > 0: attention pattern shifts with IDH class "
                  f"-> at least some signal is propagating through K/Q.")

    # ---------- Top-K layer table --------------------------------
    # Rank layers by the primary field's ratio_vs_text so the operator can
    # eyeball which specific cross-attn blocks are attending most to the
    # molecular token. Late (higher-index) hot layers are more likely to
    # influence the final pixel output than early ones.
    if fields:
        primary = fields[0]
        rows = [(name, report["layers"][name][primary]["ratio_vs_text"],
                 report["layers"][name][primary]["wt_mut_delta_abs"],
                 (report["layers"][name].get("v_probe") or {}).get("value_cos"))
                for name in layer_names]
        rows.sort(key=lambda r: r[1], reverse=True)
        print(f"\nTop-K layers by {primary.upper()} attention ratio:")
        print(f"  {'idx':>3}  {'layer':<70}  {'ratio':>7}  "
              f"{'|WT-MUT|':>10}  {'V_cos(WT,MUT)':>13}")
        for i, (name, r, d, vc) in enumerate(rows[:min(10, len(rows))]):
            vc_str = f"{vc:>13.4f}" if vc is not None else "           n/a"
            traversal_idx = layer_names.index(name)
            print(f"  {traversal_idx:>3}  {name:<70}  {r:>7.3f}  "
                  f"{d:>10.5f}  {vc_str}")

    # ---------- V-projection probe summary -----------------------
    if v_probe:
        v_cos_vals = np.asarray(
            [v_probe[n]["value_cos"] for n in layer_names if n in v_probe],
            dtype=np.float64,
        )
        print(f"\n[V-projection probe (IDH: WT vs MUT after to_v)]")
        print(f"  cos(V(WT), V(MUT)) per layer: "
              f"mean={v_cos_vals.mean():.4f}  median={np.median(v_cos_vals):.4f}  "
              f"min={v_cos_vals.min():.4f}  max={v_cos_vals.max():.4f}")
        print(f"  Interpretation:")
        v_mean = v_cos_vals.mean()
        if v_mean > 0.9:
            print(f"    cos > 0.9: V pathway has COLLAPSED WT/MUT into the "
                  f"same vector. Even perfect attention allocation cannot "
                  f"rescue this — needs auxiliary contrastive loss on to_v "
                  f"outputs or a FiLM-style bypass.")
        elif v_mean > 0.5:
            print(f"    0.5 < cos < 0.9: mild V-collapse. Class info is "
                  f"preserved but subtle. CFG amplification (higher CFG) "
                  f"may still be enough to make it visible in pixels.")
        else:
            print(f"    cos < 0.5: V pathway CLEARLY differentiates WT and "
                  f"MUT. If the downstream classifier still fails, the fault "
                  f"is later (post-attention MLP filtering, or CFG=1 masking "
                  f"the delta at sampling time).")
        # Highlight worst offenders
        rows = [(name, v_probe[name]["value_cos"]) for name in layer_names
                if name in v_probe]
        rows.sort(key=lambda r: r[1], reverse=True)
        print(f"  Top-5 most-collapsed layers (highest V-cos):")
        for name, vc in rows[:5]:
            traversal_idx = layer_names.index(name)
            print(f"    idx={traversal_idx:>3}  cos={vc:>7.4f}  {name}")

    # ---------- Figure -------------------------------------------
    try:
        import matplotlib.pyplot as plt

        n_layers = len(layer_names)
        # 3 panels if V-probe data available, else 2.
        n_panels = 3 if v_probe else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.8),
                                  squeeze=False)
        axes = axes[0]
        x = np.arange(n_layers)

        # Panel 1: per-layer mass on IDH vs text-per-token, log-y
        ax = axes[0]
        ax.axhline(1.0 / ctx_len, color="grey", linestyle=":",
                    label=f"uniform (1/{ctx_len}={1.0/ctx_len:.4f})")
        ax.plot(x, per_layer_text_avg, "o-", color="#888888",
                 label="text mass / token", markersize=3)
        for f, col in zip(fields, ["#1f77b4", "#d62728"]):
            ax.plot(x, per_layer_field_masses[f], "o-", color=col,
                     label=f"{f.upper()} token mass", markersize=4)
        ax.set_yscale("log")
        ax.set_xlabel("cross-attention layer (traversal order)")
        ax.set_ylabel("mean attention mass per token")
        ax.set_title("Per-layer attention mass allocation")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.3)

        # Panel 2: WT-vs-MUT attention difference on molecular token
        ax = axes[1]
        for f, col in zip(fields, ["#1f77b4", "#d62728"]):
            ax.plot(x, per_layer_wt_mut_delta[f], "o-", color=col,
                     label=f"|WT-MUT| on {f.upper()} token", markersize=4)
        ax.set_xlabel("cross-attention layer (traversal order)")
        ax.set_ylabel("|attention(WT) - attention(MUT)|")
        ax.set_title("Class-flip sensitivity on molecular token")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # Panel 3: V-projection cosine similarity (if available).
        if v_probe:
            ax = axes[2]
            v_cos_series = [v_probe[n]["value_cos"]
                            if n in v_probe else np.nan
                            for n in layer_names]
            v_delta_l2 = [v_probe[n]["value_delta_l2"]
                          if n in v_probe else np.nan
                          for n in layer_names]
            ax.axhline(1.0, color="red",  linestyle="--", linewidth=0.8,
                        label="V-collapse (cos=1)")
            ax.axhline(0.9, color="orange", linestyle=":", linewidth=0.8,
                        label="strong collapse threshold")
            ax.plot(x, v_cos_series, "o-", color="#2ca02c", markersize=5,
                     label="cos(V(WT), V(MUT))")
            ax.set_xlabel("cross-attention layer (traversal order)")
            ax.set_ylabel("cosine similarity")
            ax.set_ylim(-0.05, 1.10)
            ax.set_title("V-projection collapse probe (IDH: WT vs MUT)")
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.3)
            # Twin y-axis for delta L2 norm — gives absolute scale
            # alongside the cosine.
            ax2 = ax.twinx()
            ax2.plot(x, v_delta_l2, "s--", color="#7f7f7f", markersize=3,
                      alpha=0.6, label="||V(WT) - V(MUT)||_2")
            ax2.set_ylabel("||V(WT) - V(MUT)||_2", color="#7f7f7f")
            ax2.tick_params(axis="y", labelcolor="#7f7f7f")

        fig.tight_layout()
        out_png = args.out_dir / "unet_attention_report.png"
        fig.savefig(str(out_png), dpi=150)
        print(f"\nWrote figure: {out_png}")
    except ImportError:
        print("\n[warn] matplotlib not available, skipping figure")

    print(f"Wrote JSON:   {out_json}")


if __name__ == "__main__":
    main()
