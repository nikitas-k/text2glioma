#!/usr/bin/env python3
"""Find max LDM batch size on H200 by progressively increasing bs until OOM.

Simulates one LDM training step (forward + backward) with the actual model
architecture. Reports peak GPU memory at each batch size.

Usage on Gadi (single GPU, interactive or PBS):
    python scripts/find_max_bs_ldm.py \
        --stage1_config configs/stage1.yaml \
        --ldm_config configs/ldm.yaml \
        --latent_channels 3   # override to test different values
"""
from __future__ import annotations
import argparse
import gc
import torch
import torch.nn.functional as F
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_config", type=str, default="configs/stage1.yaml")
    p.add_argument("--ldm_config", type=str, default="configs/ldm.yaml")
    p.add_argument("--latent_channels", type=int, default=None,
                   help="Override latent_channels (default: from stage1 config)")
    p.add_argument("--max_bs", type=int, default=16,
                   help="Max batch size to try")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="HuggingFace cache dir for text encoder")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")

    with open(args.stage1_config) as f:
        s1_cfg = yaml.safe_load(f)
    with open(args.ldm_config) as f:
        ldm_cfg = yaml.safe_load(f)

    lc = args.latent_channels or s1_cfg["model"]["params"]["latent_channels"]
    mask_ch = ldm_cfg.get("mask", {}).get("num_classes", 4)
    in_ch = lc + mask_ch
    out_ch = lc

    # Update LDM config
    ldm_params = dict(ldm_cfg["model"]["params"])
    ldm_params["in_channels"] = in_ch
    ldm_params["out_channels"] = out_ch

    print(f"Latent channels: {lc}")
    print(f"LDM in_channels: {in_ch} ({lc} latent + {mask_ch} mask)")
    print(f"LDM out_channels: {out_ch}")
    print(f"LDM config: {ldm_params}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Build LDM
    try:
        from generative.networks.nets import DiffusionModelUNet
    except ImportError:
        from monai.networks.nets import DiffusionModelUNet

    model = DiffusionModelUNet(**ldm_params).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LDM parameters: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Also load frozen VAE to account for its memory footprint
    from text2glioma.utils import get_model
    vae = get_model(s1_cfg["model"]["name"], s1_cfg)
    if args.latent_channels:
        # Rebuild with overridden latent_channels
        s1_override = dict(s1_cfg)
        s1_override["model"] = dict(s1_cfg["model"])
        s1_override["model"]["params"] = dict(s1_cfg["model"]["params"])
        s1_override["model"]["params"]["latent_channels"] = lc
        vae = get_model(s1_cfg["model"]["name"], s1_override)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE parameters (frozen): {vae_params/1e6:.1f}M")

    # Text encoder — load the real model from config
    from text2glioma.utils import load_text_encoder_and_tokenizer
    cond_cfg = ldm_cfg.get("conditioning", {})
    cache_dir = args.cache_dir
    try:
        tokenizer, text_encoder = load_text_encoder_and_tokenizer(
            cond_cfg, cache_dir=cache_dir, local_files_only=True,
        )
        text_encoder = text_encoder.to(device).eval()
        for p in text_encoder.parameters():
            p.requires_grad_(False)
        te_params = sum(p.numel() for p in text_encoder.parameters())
        print(f"Text encoder ({cond_cfg.get('text_encoder', '?')}): {te_params/1e6:.1f}M params, "
              f"{te_params * 2 / 1024**3:.2f}GB (bf16)")
        seq_len = cond_cfg.get("max_length", 77)
        # Pre-encode a dummy text to get hidden dim
        dummy_tokens = tokenizer(
            ["test"] * 2, padding="max_length", max_length=seq_len,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        with torch.no_grad():
            te_out = text_encoder(dummy_tokens)
            if hasattr(te_out, "last_hidden_state"):
                text_hidden = te_out.last_hidden_state
            else:
                text_hidden = te_out[0]
        print(f"Text encoder output shape: {list(text_hidden.shape)} (seq_len={seq_len})")
        del dummy_tokens, te_out, text_hidden
        use_real_te = True
    except Exception as e:
        print(f"Could not load text encoder: {e}")
        print("Falling back to random embeddings (text encoder memory NOT counted)")
        text_encoder = None
        tokenizer = None
        use_real_te = False
        seq_len = cond_cfg.get("max_length", 77)

    torch.cuda.reset_peak_memory_stats()
    baseline_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"\nBaseline GPU memory (models loaded): {baseline_mem:.2f} GB")

    latent_spatial = (20, 28, 20)  # 160/8, 224/8, 160/8
    full_spatial = (160, 224, 160)
    cross_attn_dim = ldm_params.get("cross_attention_dim", 1024)
    seq_len = 77  # CLIP token length

    print(f"\n{'='*70}")
    print(f"{'BS':>4s}  {'Peak GPU':>10s}  {'Delta':>8s}  {'%80GB':>6s}  {'Status':>10s}")
    print(f"{'='*70}")

    last_peak = 0.0
    for bs in range(1, args.max_bs + 1):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Simulate one training step
            with torch.no_grad():
                # VAE encode (frozen) — need full-res input
                images = torch.randn(bs, 4, *full_spatial, device=device, dtype=torch.bfloat16)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z_mu, z_sigma = vae.encode(images)
                    z = vae.sampling(z_mu, z_sigma)
                del images, z_mu, z_sigma

            # LDM forward + backward
            latents = z.detach()  # (bs, lc, 20, 28, 20)
            del z

            # Random mask conditioning (one-hot)
            mask = torch.zeros(bs, mask_ch, *latent_spatial, device=device, dtype=torch.bfloat16)
            mask[:, 0] = 1.0  # all background for memory test

            # Text embeddings — real encoder or fallback
            if use_real_te:
                dummy_text = ["A glioblastoma with ring enhancement and surrounding edema."] * bs
                tokens = tokenizer(
                    dummy_text, padding="max_length", max_length=seq_len,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(device)
                with torch.no_grad():
                    te_out = text_encoder(tokens)
                    if hasattr(te_out, "last_hidden_state"):
                        text_emb = te_out.last_hidden_state.detach()
                    else:
                        text_emb = te_out[0].detach()
                del tokens, te_out
            else:
                text_emb = torch.randn(bs, seq_len, cross_attn_dim, device=device, dtype=torch.bfloat16)

            # Random timesteps
            timesteps = torch.randint(0, 1000, (bs,), device=device)

            # Noise
            noise = torch.randn_like(latents)

            # Noisy latents (simulate scheduler.add_noise)
            noisy = latents + noise * 0.1  # simplified

            # Concat mask
            ldm_input = torch.cat([noisy, mask], dim=1)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(ldm_input, timesteps=timesteps, context=text_emb)
                loss = F.mse_loss(pred, noise)

            loss.backward()
            optimizer.step()

            peak = torch.cuda.max_memory_allocated() / 1024**3
            delta = peak - last_peak if last_peak > 0 else 0
            pct = peak / 80 * 100
            status = "OK" if pct < 95 else "TIGHT"
            print(f"{bs:>4d}  {peak:>9.2f}G  {delta:>7.2f}G  {pct:>5.0f}%  {status:>10s}")
            last_peak = peak

            # Clean up
            del latents, mask, text_emb, timesteps, noise, noisy, ldm_input, pred, loss

        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"{bs:>4d}  {peak:>9.2f}G  {'':>7s}  {'':>5s}  {'OOM':>10s}")
            print(f"\n  Max batch size: {bs - 1}")
            break

        except Exception as e:
            print(f"{bs:>4d}  ERROR: {e}")
            break
    else:
        print(f"\n  All batch sizes up to {args.max_bs} fit in memory!")

    print()


if __name__ == "__main__":
    main()
