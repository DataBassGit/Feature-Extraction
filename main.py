#!/usr/bin/env python3
# coding: utf-8

import os, glob, json, argparse
from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
import yaml
import configparser
from transformer_lens import HookedTransformer
import re, time
from datetime import datetime

# ----------------------------
# Config
# ----------------------------

def load_config(path: str = "config.ini") -> dict:
    import configparser, os

    # Enable inline comments with ; and #
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cfg.read(path)

    def get(section, key, default):
        val = cfg.get(section, key, fallback=default)
        # Expand ~ and %ENV% on Windows
        return os.path.expandvars(os.path.expanduser(val))

    def getbool(section, key, default):
        try:
            return cfg.getboolean(section, key, fallback=default)
        except Exception:
            return default

    models_dir   = get("paths", "models_dir", "./models")
    examples_dir = get("paths", "examples_dir", "./tests")
    vectors_dir  = get("paths", "vectors_dir", "./vectors")
    model_name   = get("model", "name", "gpt2-medium")
    hook_type    = get("model", "hook_type", "hook_resid_post")
    default_layer = get("extract", "default_layer", "auto")
    normalize     = getbool("extract", "normalize", False)
    ask_examples  = getbool("console", "ask_examples", True)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)
    os.makedirs(vectors_dir, exist_ok=True)

    # Ensure HF caches go where you expect
    os.environ.setdefault("HF_HOME", models_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", models_dir)

    return dict(
        models_dir=models_dir,
        examples_dir=examples_dir,
        vectors_dir=vectors_dir,
        model_name=model_name,
        hook_type=hook_type,
        default_layer=default_layer,
        normalize=normalize,
        ask_examples=ask_examples,
    )

# ----------------------------
# TL helpers
# ----------------------------

def load_model(name: str) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

def find_delimiter_position(model: HookedTransformer, prompt: str) -> int:
    toks = model.to_str_tokens(prompt)
    for i in range(len(toks)-1, -1, -1):
        if toks[i] == ":" or toks[i].endswith(":"):
            return i
    return len(toks) - 1

def resid_tensor(cache, layer_idx: int, hook_type: str = "hook_resid_post"):
    return cache[f"blocks.{layer_idx}.{hook_type}"]

# ----------------------------
# Extraction core
# ----------------------------

@dataclass
class ContrastivePair:
    name: str
    with_concept: str
    without_concept: str

def load_pairs_from_yaml(path: str) -> List[ContrastivePair]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pairs = []
    for item in data.get("concepts", []):
        pairs.append(ContrastivePair(
            name=item["name"],
            with_concept=item["with"],
            without_concept=item["without"],
        ))
    return pairs


def compute_layer(model, default_layer: str) -> list:
    """
    Returns a list of layer indices to extract from.
    - 'auto': [first, middle, 2/3, last]
    - 'all': every layer
    - integer: single layer
    - comma-separated: explicit list
    """
    n_layers = model.cfg.n_layers

    if default_layer == "auto":
        # Four-point sweep: first, middle, 2/3, last
        first = 0
        middle = n_layers // 2
        two_thirds = int(round(2 * n_layers / 3))
        last = n_layers - 1
        # Use set to deduplicate if model is very small
        return sorted(set([first, middle, two_thirds, last]))

    if default_layer == "all":
        return list(range(n_layers))

    # Try parsing as integer
    try:
        idx = int(default_layer)
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer {idx} out of range (model has {n_layers} layers)")
        return [idx]
    except ValueError:
        pass

    # Try parsing as comma-separated list
    layers = []
    for tok in default_layer.split(","):
        tok = tok.strip()
        if tok:
            idx = int(tok)
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer {idx} out of range")
            layers.append(idx)
    return sorted(set(layers))


def extract_contrastive_vector_for_pair(
        model: HookedTransformer,
        pair: ContrastivePair,
        layer_idx: int,
        hook_type: str,
        l2_normalize: bool = False,
):
    _, cache_with = model.run_with_cache(pair.with_concept)
    _, cache_without = model.run_with_cache(pair.without_concept)

    # Use mean pooling across all tokens instead of single position
    r_with_full = resid_tensor(cache_with, layer_idx, hook_type)[0, :, :]  # [seq, d_model]
    r_without_full = resid_tensor(cache_without, layer_idx, hook_type)[0, :, :]  # [seq, d_model]

    # Mean pool to get sequence-level representation
    r_with = r_with_full.mean(dim=0)  # [d_model]
    r_without = r_without_full.mean(dim=0)  # [d_model]

    vec = (r_with - r_without).detach().cpu()

    if l2_normalize:
        vec = vec / (vec.norm(p=2) + 1e-8)

    return vec


# ----------------------------
# Console flow
# ----------------------------

def choose_examples(examples_dir: str) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(examples_dir, "*.yml")) +
                   glob.glob(os.path.join(examples_dir, "*.yaml")))
    if not files:
        print(f"No YAML files in {examples_dir}.")
        return None
    print("\nAvailable example YAML files:")
    for i, p in enumerate(files):
        print(f"  {i}: {p}")
    sel = input("Type an index or path (or Enter to cancel): ").strip()
    if not sel:
        return None
    if sel.isdigit():
        idx = int(sel)
        if 0 <= idx < len(files):
            return files[idx]
        return None
    return sel if os.path.exists(sel) else None

def enter_manual_pair() -> Optional[ContrastivePair]:
    name = input("\nConcept name: ").strip() or "manual_concept"
    print("\nEnter prompt WITH concept, end with a line containing only ###")
    w_lines = []
    while True:
        line = input()
        if line.strip() == "###": break
        w_lines.append(line)
    print("\nEnter prompt WITHOUT concept, end with a line containing only ###")
    wo_lines = []
    while True:
        line = input()
        if line.strip() == "###": break
        wo_lines.append(line)
    with_c = "\n".join(w_lines)
    without_c = "\n".join(wo_lines)
    if not with_c or not without_c:
        return None
    return ContrastivePair(name=name, with_concept=with_c, without_concept=without_c)

# ----------------------------
# Save
# ----------------------------

def save_vectors(out_dir: str, filename: str, vectors: Dict[str, torch.Tensor], meta: Dict):
    os.makedirs(out_dir, exist_ok=True)
    pt_path = os.path.join(out_dir, filename)
    torch.save(vectors, pt_path)
    with open(pt_path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(vectors)} vectors to {pt_path}")

# ----------------------------
# Main
# ----------------------------
def safe_filename(s: str) -> str:
    # Replace Windows-illegal chars and control codes with "-"
    return re.sub(r'[/\\?%*:|"<>\x7F\x00-\x1F]', '-', s).strip('-')

def stamp() -> str:
    # Windows-safe, sortable timestamp
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def main():
    import argparse, os, sys, re
    from datetime import datetime

    def safe_filename(s: str) -> str:
        return re.sub(r'[/\\?%*:|"<>\x7F\x00-\x1F]', '-', s).strip('-')

    def stamp() -> str:
        return datetime.now().strftime("%Y%m%dT%H%M%S")

    ap = argparse.ArgumentParser(description="Contrastive concept extractor (config-driven)")
    ap.add_argument("--config", type=str, default="config.ini")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = load_model(cfg["model_name"])

    # Get list of layers to extract from
    layer_indices = compute_layer(model, cfg["default_layer"])
    print(f"Extracting from layers: {layer_indices}")

    # Get contrastive pairs
    pairs: List[ContrastivePair] = []
    selected_yaml = None
    if cfg["ask_examples"]:
        path = choose_examples(cfg["examples_dir"])
        if path:
            selected_yaml = path
            pairs = load_pairs_from_yaml(path)
        else:
            p = enter_manual_pair()
            if p:
                pairs = [p]
    else:
        path = choose_examples(cfg["examples_dir"])
        if path:
            selected_yaml = path
            pairs = load_pairs_from_yaml(path)

    if not pairs:
        print("No pairs provided; exiting.")
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
        sys.stdout.flush();
        sys.stderr.flush()
        sys.exit(0)

    # Extract and save: one file per (concept, layer) combination
    model_id_for_name = safe_filename(cfg["model_name"])
    ts = stamp()
    os.makedirs(cfg["vectors_dir"], exist_ok=True)

    for pair in pairs:
        for layer_idx in layer_indices:
            vec = extract_contrastive_vector_for_pair(
                model=model,
                pair=pair,
                layer_idx=layer_idx,
                hook_type=cfg["hook_type"],
                l2_normalize=bool(cfg["normalize"]),
            )

            concept_name = safe_filename(pair.name)
            # Include layer in filename: concept__model__layer_N__timestamp
            base = f"{concept_name}__{model_id_for_name}__layer_{layer_idx}__{ts}"
            out_pt = os.path.join(cfg["vectors_dir"], base + ".pt")

            payload = {pair.name: vec}
            torch.save(payload, out_pt)

            meta = {
                "concept": pair.name,
                "model_name": cfg["model_name"],
                "layer_idx": layer_idx,
                "hook_type": cfg["hook_type"],
                "l2_normalize": bool(cfg["normalize"]),
                "vector_norm": float(vec.norm().item()),
                "source_yaml": selected_yaml,
                "saved_at": ts,
            }
            with open(out_pt + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"Saved layer {layer_idx} vector to {out_pt}", flush=True)

    # Graceful shutdown
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

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # Windows
    main()
