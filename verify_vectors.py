#!/usr/bin/env python3
# coding: utf-8

import os, glob, re, yaml, time
import argparse
import configparser
import torch
import re
from datetime import datetime
from typing import Dict, List
from transformer_lens import HookedTransformer

# --------------------------
# Defaults
# --------------------------

DEFAULT_PROMPTS = [
    "Reply with one lowercase English word that best captures your main topic right now.",
    "Answer with a single lowercase keyword that summarizes your current focus.",
    "Output one lowercase tag for the dominant idea on your mind.",
    "Respond with one lowercase noun that best labels your present topic.",
    "Give one lowercase keyword for the subject you are most primed to discuss.",
]

# --------------------------
# Config
# --------------------------

def load_config(path: str = "config.ini") -> dict:
    cfg = configparser.ConfigParser()
    if not os.path.exists(path):
        print(f"[warn] config.ini not found at {path}, using built-in defaults")
    else:
        cfg.read(path)

    def get(section, key, default):
        return cfg.get(section, key, fallback=default)

    def getint(section, key, default):
        try:
            return cfg.getint(section, key, fallback=default)
        except Exception:
            return default

    vectors_dir = get("paths", "vectors_dir", "./vectors")
    reports_dir = get("paths", "reports_dir", "./reports")
    model_name  = get("model", "name", "gpt2-medium")
    hook_type   = get("model", "hook_type", "hook_resid_post")
    layers_spec = get("verify", "layers", "auto")
    strengths_s = get("verify", "strengths", "4,8,16")
    strengths   = [float(s.strip()) for s in strengths_s.split(",") if s.strip()]
    max_new     = getint("verify", "max_new_tokens", 16)

    count = getint("prompts", "count", 5)
    prompts = []
    for i in range(count):
        p = get("prompts", f"p{i+1}", "").strip()
        if p:
            prompts.append(p)
    if not prompts:
        prompts = DEFAULT_PROMPTS

    os.makedirs(vectors_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    return dict(
        vectors_dir=vectors_dir,
        reports_dir=reports_dir,
        model_name=model_name,
        hook_type=hook_type,
        layers_spec=layers_spec,
        strengths=strengths,
        max_new_tokens=max_new,
        prompts=prompts,
    )

# --------------------------
# Discovery & Loading
# --------------------------

def discover_vector_files(vectors_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(vectors_dir, "*.pt")))

def load_all_vectors(files: List[str]) -> Dict[str, torch.Tensor]:
    all_vecs = {}
    for fp in files:
        try:
            d = torch.load(fp, map_location="cpu")
            if isinstance(d, dict):
                for k, t in d.items():
                    key = k
                    if key in all_vecs:
                        key = f"{k}__{os.path.basename(fp)}"
                    all_vecs[key] = t.detach().clone()
        except Exception as e:
            print(f"[warn] failed to load {fp}: {e}")
    return all_vecs

# --------------------------
# Model & Hooks
# --------------------------

def load_model(name: str) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

def make_add_hook(vec: torch.Tensor, strength: float):
    vec = vec.to(torch.float32)
    def hook(resid, hook):
        return resid + (strength * vec.to(resid.device)).unsqueeze(0).unsqueeze(0)
    return hook

def parse_layers(spec: str, n_layers: int) -> List[int]:
    if spec == "all":
        return list(range(n_layers))
    if spec == "auto":
        picks = {int(round(r * (n_layers - 1))) for r in [0.25, 0.5, 0.67, 0.8]}
        return sorted(picks)
    out = []
    for tok in spec.split(","):
        if tok.strip():
            idx = int(tok.strip())
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer out of range: {idx}")
            out.append(idx)
    return out

def generate_one(model: HookedTransformer, prompt: str, max_new_tokens: int) -> str:
    return model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def match_hit(output: str, label: str) -> bool:
    o = normalize(output)
    c = normalize(label)
    toks = set(o.split())
    return (c in toks) or (c in o)

# --------------------------
# Run verification
# --------------------------
def safe_filename(s: str) -> str:
    return re.sub(r'[/\\?%*:|"<>\x7F\x00-\x1F]', '-', s).strip('-')

def stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")

def verify_catalog(cfg: dict):
    import os, yaml
    from datetime import datetime

    def safe_filename(s: str) -> str:
        return re.sub(r'[/\\?%*:|"<>\x7F\x00-\x1F]', '-', s).strip('-')

    files = discover_vector_files(cfg["vectors_dir"])
    print(f"Discovered {len(files)} .pt files in {cfg['vectors_dir']}")

    catalog = load_all_vectors(files)
    if not catalog:
        print("No vectors found; exiting.")
        return

    names = sorted(catalog.keys())
    print(f"Found {len(names)} vectors: {', '.join(names)}")

    choice = input("\nVerify all vectors? [Y/n]: ").strip().lower()
    if choice in ("n", "no"):
        print("Enter comma-separated names or indices (e.g., 0,2,dog,all_caps):")
        for i, k in enumerate(names):
            print(f"  {i}: {k}")
        sel = input("> ").strip()
        want = []
        if sel:
            for tok in sel.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    idx = int(tok)
                    if 0 <= idx < len(names):
                        want.append(names[idx])
                elif tok in catalog:
                    want.append(tok)
        if not want:
            print("No valid selections; exiting.")
            return
        target_names = want
    else:
        target_names = names

    # Load model and parse config
    model = load_model(cfg["model_name"])
    layers = parse_layers(cfg["layers_spec"], model.cfg.n_layers)
    strengths = cfg["strengths"]
    prompts = cfg["prompts"]

    # Naming helpers
    model_id_for_name = safe_filename(cfg["model_name"])
    run_ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Shared header
    global_header = {
        "model": cfg["model_name"],
        "hook_type": cfg["hook_type"],
        "layers": layers,
        "strengths": strengths,
        "prompts": prompts,
        "max_new_tokens": cfg["max_new_tokens"],
    }

    # Process each vector
    for name in target_names:
        vec = catalog[name]
        entry = {"vector_norm": float(vec.norm().item()), "by_layer": {}}

        for layer in layers:
            layer_rec = {}
            hook_name = f"blocks.{layer}.{cfg['hook_type']}"
            for st in strengths:
                trials = []
                model.reset_hooks()
                model.add_hook(hook_name, make_add_hook(vec, st))
                for ptxt in prompts:
                    out = generate_one(model, ptxt, cfg["max_new_tokens"]).strip()
                    trials.append({
                        "prompt": ptxt,
                        "output": out
                    })
                model.reset_hooks()
                layer_rec[str(st)] = {"trials": trials}
            entry["by_layer"][str(layer)] = layer_rec

        # Write report
        report = {
            **global_header,
            "vector_name": name,
            "saved_at": run_ts,
            "results": {name: entry},
        }

        concept_for_name = safe_filename(name)
        filename = f"{concept_for_name}__{model_id_for_name}__{run_ts}.yaml"
        out_path = os.path.join(cfg["reports_dir"], filename)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)
        print(f"Wrote report: {out_path}", flush=True)

    # Cleanup
    try:
        model.reset_hooks()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------
# Main (no heavy CLI; config.ini + interactive only)
# --------------------------

def main():
    import argparse, os, sys
    parser = argparse.ArgumentParser(description="Verify concept vectors (config.ini + interactive selection)")
    parser.add_argument("--config", type=str, default="config.ini")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg.get("models_dir", os.getenv("HF_HOME", "")))
    os.environ.setdefault("TRANSFORMERS_CACHE", cfg.get("models_dir", os.getenv("TRANSFORMERS_CACHE", "")))

    verify_catalog(cfg)

    # Graceful exit
    sys.stdout.flush(); sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
