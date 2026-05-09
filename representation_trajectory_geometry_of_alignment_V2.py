#!/usr/bin/env python3
"""
================================================================================
representation_trajectory_geometry_of_alignment.py   (v3.0)
================================================================================

Representation Trajectory Geometry of Alignment — a multi-lens characterization
of how DPO alignment reshapes per-prompt trajectories in transformer hidden
states, with controls to distinguish alignment-specific geometry from lexical
propagation.

For a given (Base, IT, DPO) model triple and a stratified safe/unsafe prompt
set, this pipeline:
  1. Extracts per-layer last-token hidden-state trajectories under each model.
  2. Computes comparative geometry (IT↔DPO, Base↔IT) with 7 metrics.
  3. Computes intrinsic geometry (per-model) with 3 additional metrics.
  4. Runs controls:
        • Base→IT comparison (to distinguish fine-tuning-in-general from
          safety-specific alignment)
        • Two alternative safe-prompt sets (Litmus Information-Seeking;
          JailbreakBench content-matched benign behaviors)
  5. Runs analyses:
        • Per-layer linear probe (logistic regression on activations)
        • TTV null distribution (is TTV signal meaningfully non-random?)
        • Per-axiom breakdown at peak layer
  6. Produces publication-quality 2D figures and interactive 3D HTML plots.

--------------------------------------------------------------------------------
THE 7 COMPARATIVE METRICS  (need both IT & DPO, per prompt)
--------------------------------------------------------------------------------
  θ    Step Direction Angle         angle(Δh_A, Δh_B)                    local · direction
  SR   Scale Ratio                  ‖Δh_B‖ / ‖Δh_A‖                      local · magnitude
  NDM  Normalized Deflection Mag.   ‖Δh_B − Δh_A‖ / ‖Δh_A‖                local · combined
  CTD  Cumulative Traj. Divergence  ‖h_B_ℓ − h_A_ℓ‖ / ‖h_A_ℓ‖             global · position
  PL   Path Length                  Σ ‖Δh_ℓ‖  (per model)                 global · distance
  TEA  Trajectory Endpoint Angle    angle(h_A_final−h_0, h_B_final−h_0)   global · direction
  TTV  Trajectory Twist Volume      √det(Gram(Δ_{ℓ-1},Δ_ℓ,Δ_{ℓ+1}))/∏‖Δ‖  local · geometry

(A, B) is the model pair — for the primary experiment A=IT, B=DPO; for the
control A=Base, B=IT.

--------------------------------------------------------------------------------
THE 3 INTRINSIC METRICS  (one model only, for model-anchored plots)
--------------------------------------------------------------------------------
  speed   Step magnitude              ‖Δh_ℓ‖ per layer
  turn    Turning angle              angle(Δh_{ℓ-1}, Δh_ℓ) within one trajectory
  disp    Cumulative displacement    ‖h_ℓ − h_0‖ per layer

These answer: "on a given prompt, how does IT's trajectory differ from DPO's
trajectory, individually?" Rather than "how does their joint deflection differ
between safe and unsafe prompts?"

--------------------------------------------------------------------------------
EXPERIMENT MATRIX (configured; toggles which to execute)
--------------------------------------------------------------------------------
  primary       : your-IT → your-DPO, Litmus-original prompts        [runs]
  control_base  : your-Base → your-IT, Litmus-original prompts       [runs]
  safe_A        : IT → DPO, Litmus Information-Seeking safe prompts  [runs]
  safe_B        : IT → DPO, JailbreakBench matched benign vs harmful [runs]
  tulu3         : Tulu-3-IT → Tulu-3-DPO                              [wired, not run]
  olmo3         : OLMo-3-Instruct-IT → OLMo-3-Instruct-DPO            [wired, not run]

Execute with --experiments primary,control_base,safe_A,safe_B (default).
To add Tulu-3 / OLMo-3 later: --experiments primary,tulu3  etc.

--------------------------------------------------------------------------------
FUTURE WORK (explicitly parked, see reference doc §8)
--------------------------------------------------------------------------------
  • Sparse autoencoders at L14 (Phase 2, ~2 week project)
  • Arditi refusal-direction projection
  • Skew-symmetric cross-covariance (v4.1/v5 formulation)
  • Frenet-Serret τ (numerically fragile — TTV replaces it robustly)
  • Parallel transport / Schild's ladder
  • DPO training-dynamics (needs intermediate checkpoints)
  • Last-token vs mean-pooled sensitivity check
  • Causal intervention (the headline experiment for the eventual paper)

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python representation_trajectory_geometry_of_alignment.py \\
      --base-model   meta-llama/Meta-Llama-3-8B \\
      --sft-model    sirius5005/SFT-and-DPO --sft-subfolder SFT_merged \\
      --dpo-model    sirius5005/SFT-and-DPO --dpo-subfolder DPO_merged \\
      --pair-name    pair1_llama8b_v3 \\
      --n-safe 3000 --n-unsafe 3000 \\
      --experiments  primary,control_base,safe_A,safe_B

================================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import signal
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np


# ===========================================================================
#  SECTION 1  —  CONSTANTS, REGISTRIES, COLORS
# ===========================================================================

SCRIPT_VERSION = "3.0.0"

# Six unsafe axioms drawn from Litmus (excludes Information-Seeking, which is
# mostly safe)
UNSAFE_AXIOMS = [
    "Wisdom & Knowledge",
    "Well-being & Peace",
    "Justice & Rights",
    "Duty & Accountability",
    "Civility & Tolerance",
    "Empathy & Helpfulness",
]

PROMPT_SEED = 42

# ---- metric registry ----
COMPARATIVE_METRICS = ["theta_deg", "sr", "ndm", "ctd", "ttv"]   # per-layer
COMPARATIVE_GLOBAL  = ["tea_deg"]                                 # scalar
INTRINSIC_METRICS   = ["speed", "turn_deg", "disp"]               # per-model

METRIC_LABELS = {
    "theta_deg": "θ  Step Direction Angle (deg)",
    "sr":        "SR  Scale Ratio (‖Δh_B‖/‖Δh_A‖)",
    "ndm":       "NDM  Normalized Deflection Magnitude",
    "ctd":       "CTD  Cumulative Trajectory Divergence",
    "ttv":       "TTV  Trajectory Twist Volume",
    "pl_a":      "PL_A  Cumulative path length (model A)",
    "pl_b":      "PL_B  Cumulative path length (model B)",
    "tea_deg":   "TEA  Trajectory Endpoint Angle (deg)",
    "speed":     "Speed  ‖Δh_ℓ‖",
    "turn_deg":  "Turn  angle(Δh_{ℓ-1}, Δh_ℓ) (deg)",
    "disp":      "Disp  ‖h_ℓ − h_0‖",
}

METRIC_BLURBS = {
    "theta_deg": "Angle between the two models' step vectors. Did the second model point the step differently?",
    "sr":        "Ratio of step magnitudes (B/A). Did the second model push harder (>1) or softer (<1)?",
    "ndm":       "Size of the second model's correction, normalized by natural step size.",
    "ctd":       "How far apart the two trajectories have drifted in absolute position.",
    "ttv":       "Does the trajectory twist out-of-plane (→1) or bend flatly (→0)?",
    "pl_a":      "Total distance traveled through the network under model A.",
    "pl_b":      "Total distance traveled through the network under model B.",
    "tea_deg":   "Overall direction from start to end: model A vs model B.",
    "speed":     "Per-layer step magnitude. How fast is the representation moving at each layer?",
    "turn_deg":  "Per-layer turning angle within one trajectory. How sharply does the path bend?",
    "disp":      "How far the representation has moved from its layer-0 position.",
}

# ---- experiment registry ----
@dataclass
class ExperimentSpec:
    name: str
    model_a_role: str             # "base" | "sft" | "dpo"  (for naming)
    model_b_role: str
    prompt_set: str               # "litmus_original" | "litmus_infoseeking" | "jailbreakbench"
    description: str

EXPERIMENTS = {
    "primary": ExperimentSpec(
        name="primary",
        model_a_role="sft",  model_b_role="dpo",
        prompt_set="litmus_original",
        description="your-IT → your-DPO on Litmus-original (the main finding)",
    ),
    "control_base": ExperimentSpec(
        name="control_base",
        model_a_role="base", model_b_role="sft",
        prompt_set="litmus_original",
        description="Base-Llama-3 → your-IT (isolates fine-tuning-in-general from safety-specific)",
    ),
    "safe_A": ExperimentSpec(
        name="safe_A",
        model_a_role="sft",  model_b_role="dpo",
        prompt_set="litmus_infoseeking",
        description="IT → DPO on Litmus Information-Seeking safe prompts (better-matched safe set)",
    ),
    "safe_B": ExperimentSpec(
        name="safe_B",
        model_a_role="sft",  model_b_role="dpo",
        prompt_set="jailbreakbench",
        description="IT → DPO on JailbreakBench matched harmful/benign pairs (content-matched)",
    ),
    # --- wired but not executed today ---
    "tulu3": ExperimentSpec(
        name="tulu3",
        model_a_role="tulu3_sft", model_b_role="tulu3_dpo",
        prompt_set="litmus_original",
        description="Tulu-3-IT → Tulu-3-DPO (cross-model replication)",
    ),
    "olmo3": ExperimentSpec(
        name="olmo3",
        model_a_role="olmo3_sft", model_b_role="olmo3_dpo",
        prompt_set="litmus_original",
        description="OLMo-3-Instruct-IT → OLMo-3-Instruct-DPO (cross-architecture robustness)",
    ),
}

# ---- colors ----
COL_SAFE   = "#2E86AB"
COL_UNSAFE = "#C73E1D"
COL_ACCENT = "#6A4C93"
COL_GRID   = "#E8E8E8"
COL_MODEL_A = "#1A7F5A"   # green: model A (IT in primary, Base in control)
COL_MODEL_B = "#9B2D5E"   # magenta: model B (DPO in primary, IT in control)

# ---- display labels ----
# On disk we still use "sft" as the role name for cache compatibility, but in
# all USER-FACING output (figure titles, legends, console messages, summaries)
# we display "IT" (Instruction Tuning). Per supervisor's directive after the
# April advisor meeting.
ROLE_DISPLAY = {
    "sft":       "IT",
    "dpo":       "DPO",
    "base":      "Base",
    "tulu3_sft": "TULU3-IT",
    "tulu3_dpo": "TULU3-DPO",
    "olmo3_sft": "OLMO3-IT",
    "olmo3_dpo": "OLMO3-DPO",
}


def display_role(role: str) -> str:
    """Map an internal role name to the user-facing display label."""
    return ROLE_DISPLAY.get(role, role.upper())


# ---- dataclasses ----
@dataclass
class GPUConfig:
    tier: str
    dtype: str
    batch_size: int
    clear_cache_every: int
    use_sdpa: bool
    vram_gb: float


@dataclass
class PipelineConfig:
    base_model: Optional[str]
    base_subfolder: Optional[str]
    sft_model: str
    sft_subfolder: Optional[str]
    dpo_model: str
    dpo_subfolder: Optional[str]
    tulu3_sft_model: Optional[str]
    tulu3_dpo_model: Optional[str]
    olmo3_sft_model: Optional[str]
    olmo3_dpo_model: Optional[str]
    pair_name: str
    output_dir: str
    n_safe: int
    n_unsafe: int
    experiments: list[str]
    max_prompt_length_chars: int = 800
    max_tokens: int = 512
    seed: int = PROMPT_SEED
    # Set at runtime
    gpu: Optional[GPUConfig] = None
    script_version: str = SCRIPT_VERSION

    def role_to_model(self, role: str) -> tuple[Optional[str], Optional[str]]:
        """Resolve a role name to (model_id, subfolder)."""
        m = {
            "base":      (self.base_model,       self.base_subfolder),
            "sft":       (self.sft_model,        self.sft_subfolder),
            "dpo":       (self.dpo_model,        self.dpo_subfolder),
            "tulu3_sft": (self.tulu3_sft_model,  None),
            "tulu3_dpo": (self.tulu3_dpo_model,  None),
            "olmo3_sft": (self.olmo3_sft_model,  None),
            "olmo3_dpo": (self.olmo3_dpo_model,  None),
        }
        if role not in m:
            raise ValueError(f"unknown role: {role}")
        return m[role]
# ===========================================================================
#  SECTION 2  —  LOGGING, PROGRESS, TIMING, GPU PROFILING
# ===========================================================================

class _Ansi:
    B, DIM, R = "\033[1m", "\033[2m", "\033[0m"
    CYAN, GREEN, YELLOW, RED, MAGENTA, BLUE = (
        "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[35m", "\033[34m"
    )


def _color_ok() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def banner(msg: str) -> None:
    bar = "=" * 78
    if _color_ok():
        print(f"\n{_Ansi.B}{_Ansi.CYAN}{bar}\n  {msg}\n{bar}{_Ansi.R}")
    else:
        print(f"\n{bar}\n  {msg}\n{bar}")


def subbanner(msg: str) -> None:
    bar = "─" * 78
    if _color_ok():
        print(f"\n{_Ansi.B}{_Ansi.BLUE}{bar}\n  {msg}\n{bar}{_Ansi.R}")
    else:
        print(f"\n{bar}\n  {msg}\n{bar}")


def ok(msg: str) -> None:
    tag = f"{_Ansi.GREEN}[OK]{_Ansi.R}" if _color_ok() else "[OK]"
    print(f"  {tag} {msg}", flush=True)


def info(msg: str) -> None:
    tag = f"{_Ansi.CYAN}[..]{_Ansi.R}" if _color_ok() else "[..]"
    print(f"  {tag} {msg}", flush=True)


def warn(msg: str) -> None:
    tag = f"{_Ansi.YELLOW}[!!]{_Ansi.R}" if _color_ok() else "[!!]"
    print(f"  {tag} {msg}", flush=True)


def fail(msg: str) -> None:
    tag = f"{_Ansi.RED}[XX]{_Ansi.R}" if _color_ok() else "[XX]"
    print(f"  {tag} {msg}", file=sys.stderr, flush=True)


def step(msg: str) -> None:
    """Labeled step marker used for major progress announcements."""
    tag = f"{_Ansi.B}{_Ansi.MAGENTA}[>>]{_Ansi.R}" if _color_ok() else "[>>]"
    print(f"  {tag} {msg}", flush=True)


# ------------------------- progress reporter --------------------------

class Progress:
    """Prints rich progress lines like:
         [extract:IT]  1234/3000  41.1%  rate=28.3/s  ETA=01:02  elapsed=00:44

    Use as:
        p = Progress("extract:IT", total=3000)
        for i, x in enumerate(items):
            ...
            p.update(i+1)       # or p.tick()
        p.done()

    Only reprints on meaningful change (no more than every 1s, and key
    milestones: first 5, every 5%, last).
    """
    def __init__(self, label: str, total: int, min_interval_s: float = 1.0):
        self.label = label
        self.total = max(1, total)
        self.min_interval = min_interval_s
        self.t0 = time.time()
        self.t_last = 0.0
        self.n = 0
        self._milestones = set()
        # Record a 5%-bucket set so we always print at each bucket crossing
        for pct in range(5, 101, 5):
            self._milestones.add(int(total * pct / 100))

    def _fmt_dur(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        if m >= 60:
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def update(self, n: int, force: bool = False) -> None:
        self.n = n
        now = time.time()
        should_print = force or (now - self.t_last) >= self.min_interval \
            or n <= 5 or n == self.total or n in self._milestones
        if not should_print:
            return
        self.t_last = now
        elapsed = now - self.t0
        rate = n / max(elapsed, 1e-6)
        remaining = (self.total - n) / max(rate, 1e-6)
        pct = 100.0 * n / self.total
        line = (f"  [{self.label}]  {n}/{self.total}  {pct:5.1f}%  "
                f"rate={rate:5.1f}/s  "
                f"ETA={self._fmt_dur(remaining)}  "
                f"elapsed={self._fmt_dur(elapsed)}")
        print(line, flush=True)

    def tick(self) -> None:
        self.update(self.n + 1)

    def done(self) -> None:
        self.update(self.total, force=True)


# ------------------------- structured logger --------------------------

class RunLogger:
    def __init__(self, out_dir: Path):
        self.jsonl_path = out_dir / "run_log.jsonl"
        self.txt_path = out_dir / "run_log.txt"
        self._jsonl = open(self.jsonl_path, "a", buffering=1)
        self._txt = open(self.txt_path, "a", buffering=1)

    def event(self, kind: str, **fields) -> None:
        rec = {"t": time.time(), "kind": kind, **fields}
        self._jsonl.write(json.dumps(rec, default=str) + "\n")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        payload = " ".join(f"{k}={v}" for k, v in fields.items())
        self._txt.write(f"[{ts}] {kind}: {payload}\n")

    def close(self) -> None:
        self._jsonl.close()
        self._txt.close()


class Stopwatch:
    def __init__(self):
        self.timings: dict[str, float] = {}
        self._stack: list[tuple[str, float]] = []

    def start(self, name: str) -> None:
        self._stack.append((name, time.time()))
        info(f"starting: {name}")

    def stop(self) -> None:
        name, t0 = self._stack.pop()
        dt = time.time() - t0
        self.timings[name] = self.timings.get(name, 0.0) + dt
        ok(f"finished: {name}  ({dt:.2f}s)")

    def summary(self) -> str:
        lines = ["", "Timing breakdown:"]
        total = 0.0
        for k, v in self.timings.items():
            lines.append(f"  {k:<46s}  {v:>9.2f} s")
            total += v
        lines.append(f"  {'TOTAL':<46s}  {total:>9.2f} s  ({total/60:.1f} min)")
        return "\n".join(lines)


# ------------------------- GPU profiling --------------------------

def profile_gpu() -> GPUConfig:
    """Detect GPU VRAM and pick a safe configuration.

    <50 GB tier  : fp16, batch=1, clear cache every 4 prompts.
    >=50 GB tier : bf16, batch=4, clear cache every 16 prompts.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    name = props.name

    if vram_gb < 35:
        raise RuntimeError(
            f"Detected '{name}' with {vram_gb:.1f} GB VRAM. "
            "Need >=40 GB for an 8B model in fp16/bf16."
        )

    if vram_gb < 50:
        cfg = GPUConfig("small", "float16", 1, 4, True, round(vram_gb, 1))
    else:
        cfg = GPUConfig("large", "bfloat16", 4, 16, True, round(vram_gb, 1))

    ok(f"GPU: {name}  ({cfg.vram_gb} GB)  ->  tier={cfg.tier}")
    ok(f"dtype={cfg.dtype}  batch={cfg.batch_size}  "
       f"cache-clear-every={cfg.clear_cache_every}")
    return cfg
# ===========================================================================
#  SECTION 3  —  PROMPT PREPARATION
#
#  We support three prompt sources, each returning a DataFrame with columns:
#      prompt_id, text, safety_label, axiom, source_dataset, char_len
#
#  1. "litmus_original"     : stratified safe/unsafe from Litmus, length-matched
#  2. "litmus_infoseeking"  : only Information-Seeking axiom safe + stratified
#                             unsafe (addresses content-mismatch confound)
#  3. "jailbreakbench"      : 100 matched harmful + 100 matched benign pairs
# ===========================================================================

def _balanced_sample(df, n: int, seed: int):
    """Sample up to n rows, uniformly with a fixed seed. Shorter helper."""
    import pandas as pd
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _length_match(safe_pool, target_lens, seed: int):
    """For each target length, pick a prompt from safe_pool within ±20% that
    hasn't been used yet. Fall back to closest by |Δlen| if none match."""
    import pandas as pd
    safe_pool = safe_pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    picked_idx = []
    used = set()
    for tl in target_lens:
        low, high = tl * 0.8, tl * 1.2
        cands = safe_pool[
            (safe_pool["char_len"] >= low)
            & (safe_pool["char_len"] <= high)
            & (~safe_pool.index.isin(used))
        ]
        if len(cands) == 0:
            cands = safe_pool[~safe_pool.index.isin(used)]
            if len(cands) == 0:
                break
            idx = (cands["char_len"] - tl).abs().idxmin()
        else:
            idx = cands.sample(n=1, random_state=seed).index[0]
        picked_idx.append(idx)
        used.add(idx)
    return safe_pool.loc[picked_idx].reset_index(drop=True)


def _prepare_litmus_original(cfg, rng):
    """Our standard Litmus prompt set: stratified unsafe, length-matched safe."""
    import pandas as pd
    from datasets import load_dataset
    info("loading Litmus from HuggingFace...")
    ds = load_dataset("hasnat79/litmus", split="train")
    df = ds.to_pandas()
    df["char_len"] = df["input"].str.len()
    df = df[df["char_len"] <= cfg.max_prompt_length_chars].reset_index(drop=True)

    # Unsafe: stratified across 6 axioms
    per_axiom = cfg.n_unsafe // len(UNSAFE_AXIOMS)
    remainder = cfg.n_unsafe - per_axiom * len(UNSAFE_AXIOMS)
    unsafe_chunks = []
    for i, ax in enumerate(UNSAFE_AXIOMS):
        k = per_axiom + (1 if i < remainder else 0)
        pool = df[(df["safety_label"] == "unsafe") & (df["axiom"] == ax)]
        if len(pool) < k:
            warn(f"axiom '{ax}': only {len(pool)} unsafe available, requested {k}")
            k = len(pool)
        picks = pool.sample(n=k, random_state=int(rng.integers(0, 2**31)))
        unsafe_chunks.append(picks)
    unsafe_df = pd.concat(unsafe_chunks, ignore_index=True)

    # Safe: length-matched
    safe_pool = df[df["safety_label"] == "safe"].copy()
    safe_df = _length_match(safe_pool, unsafe_df["char_len"].to_numpy(),
                             seed=cfg.seed)

    return pd.concat([
        unsafe_df.assign(side="unsafe"),
        safe_df.assign(side="safe"),
    ], ignore_index=True)


def _prepare_litmus_infoseeking(cfg, rng):
    """Alternative safe-prompt source: Litmus Information-Seeking axiom.

    Rationale: IS prompts are first-person-style questions that better match
    the phrasing of unsafe prompts. Reduces the lexical confound observed at
    layer 0 (d≈0.52 in the original run)."""
    import pandas as pd
    from datasets import load_dataset
    info("loading Litmus (Information-Seeking safe variant)...")
    ds = load_dataset("hasnat79/litmus", split="train")
    df = ds.to_pandas()
    df["char_len"] = df["input"].str.len()
    df = df[df["char_len"] <= cfg.max_prompt_length_chars].reset_index(drop=True)

    # Unsafe: same stratified as original
    per_axiom = cfg.n_unsafe // len(UNSAFE_AXIOMS)
    remainder = cfg.n_unsafe - per_axiom * len(UNSAFE_AXIOMS)
    unsafe_chunks = []
    for i, ax in enumerate(UNSAFE_AXIOMS):
        k = per_axiom + (1 if i < remainder else 0)
        pool = df[(df["safety_label"] == "unsafe") & (df["axiom"] == ax)]
        if len(pool) < k:
            k = len(pool)
        picks = pool.sample(n=k, random_state=int(rng.integers(0, 2**31)))
        unsafe_chunks.append(picks)
    unsafe_df = pd.concat(unsafe_chunks, ignore_index=True)

    # Safe: Information-Seeking only, length-matched
    safe_pool = df[(df["safety_label"] == "safe")
                    & (df["axiom"] == "Information Seeking")].copy()
    if len(safe_pool) < cfg.n_safe:
        warn(f"Only {len(safe_pool)} Information-Seeking safe prompts available; "
             f"requested {cfg.n_safe}. Using all available.")
    safe_df = _length_match(safe_pool, unsafe_df["char_len"].to_numpy(),
                             seed=cfg.seed)

    return pd.concat([
        unsafe_df.assign(side="unsafe"),
        safe_df.assign(side="safe"),
    ], ignore_index=True)


def _prepare_jailbreakbench(cfg, rng):
    """JailbreakBench: 100 harmful + 100 content-matched benign behaviors.

    Rationale: JBB is the only safety dataset with explicit matched-topic
    benign counterparts ("write a tutorial on hacking" ↔ "write a tutorial on
    network security"). Much tighter content match than length-matching can
    ever achieve. Smaller n (~100/100) but dramatically reduced lexical
    confound.
    """
    import pandas as pd
    from datasets import load_dataset
    info("loading JailbreakBench (matched harmful/benign behaviors)...")
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    except Exception as e:
        warn(f"could not load JailbreakBench 'behaviors' config: {e}")
        warn("falling back to default config")
        ds = load_dataset("JailbreakBench/JBB-Behaviors")

    # The dataset has 'harmful' and 'benign' splits (or columns, depending
    # on version). Try both patterns.
    harmful_df = None
    benign_df = None
    if isinstance(ds, dict) or hasattr(ds, 'keys'):
        keys = list(ds.keys())
        for k in keys:
            sub = ds[k].to_pandas()
            if "harmful" in k.lower() or sub.get("Category", pd.Series()).notna().any():
                # detection heuristic
                pass
        if "harmful" in keys:
            harmful_df = ds["harmful"].to_pandas()
        if "benign" in keys:
            benign_df = ds["benign"].to_pandas()
        if harmful_df is None or benign_df is None:
            # try first split as harmful, last as benign if two splits
            if len(keys) == 2:
                harmful_df = ds[keys[0]].to_pandas()
                benign_df  = ds[keys[1]].to_pandas()
            else:
                raise RuntimeError(f"Unexpected JBB structure; splits={keys}")
    else:
        # single-split dataset
        full = ds.to_pandas()
        if "harmful" in full.columns and "benign" in full.columns:
            # paired-column format: reshape
            harmful_df = full[["Behavior", "harmful", "Category"]].rename(
                columns={"harmful": "Goal"})
            benign_df  = full[["Behavior", "benign", "Category"]].rename(
                columns={"benign": "Goal"})
        else:
            raise RuntimeError("Unexpected JBB format")

    # Normalize column names — JBB uses 'Goal' or 'Behavior' for the prompt text
    def _get_text_col(df):
        for c in ["Goal", "goal", "Behavior", "behavior", "prompt", "Prompt"]:
            if c in df.columns:
                return c
        raise RuntimeError(f"No prompt-text column; got {list(df.columns)}")

    h_text_col = _get_text_col(harmful_df)
    b_text_col = _get_text_col(benign_df)
    h_cat_col = "Category" if "Category" in harmful_df.columns else None
    b_cat_col = "Category" if "Category" in benign_df.columns else None

    harmful = harmful_df[[h_text_col] + ([h_cat_col] if h_cat_col else [])].copy()
    harmful.columns = ["text"] + (["axiom"] if h_cat_col else [])
    if "axiom" not in harmful.columns:
        harmful["axiom"] = "JBB-Harmful"
    harmful["safety_label"] = "unsafe"
    harmful["source_dataset"] = "JailbreakBench"
    harmful["char_len"] = harmful["text"].str.len()

    benign = benign_df[[b_text_col] + ([b_cat_col] if b_cat_col else [])].copy()
    benign.columns = ["text"] + (["axiom"] if b_cat_col else [])
    if "axiom" not in benign.columns:
        benign["axiom"] = "JBB-Benign"
    benign["safety_label"] = "safe"
    benign["source_dataset"] = "JailbreakBench"
    benign["char_len"] = benign["text"].str.len()

    # Optionally subsample (JBB has 100 per side; allow --n-safe 50 to halve)
    if cfg.n_safe < len(benign):
        benign = _balanced_sample(benign, cfg.n_safe, cfg.seed)
    if cfg.n_unsafe < len(harmful):
        harmful = _balanced_sample(harmful, cfg.n_unsafe, cfg.seed + 1)

    # Rename for consistency ('input' not needed; we renamed to 'text')
    return pd.concat([harmful, benign], ignore_index=True)


def prepare_prompts(cfg: PipelineConfig, which: str, out_csv: Path):
    """Build or load the prompt set specified by `which`.

    Resume-safe: if out_csv already matches the requested sizes, reuse it.
    """
    import pandas as pd

    if out_csv.exists():
        df = pd.read_csv(out_csv)
        n_safe_on_disk = int((df["safety_label"] == "safe").sum())
        n_unsafe_on_disk = int((df["safety_label"] == "unsafe").sum())
        expected_safe = min(cfg.n_safe, 100 if which == "jailbreakbench" else 10_000_000)
        expected_unsafe = min(cfg.n_unsafe, 100 if which == "jailbreakbench" else 10_000_000)
        if n_safe_on_disk == expected_safe and n_unsafe_on_disk == expected_unsafe:
            ok(f"reusing prompt set ({which}): {out_csv}  "
               f"(safe={n_safe_on_disk}, unsafe={n_unsafe_on_disk})")
            return df
        warn(f"prompt set '{which}' size mismatch on disk "
             f"(on disk {n_safe_on_disk}/{n_unsafe_on_disk}, "
             f"expected {expected_safe}/{expected_unsafe}). Resampling.")

    rng = np.random.default_rng(cfg.seed + hash(which) % 1000)

    if which == "litmus_original":
        df = _prepare_litmus_original(cfg, rng)
    elif which == "litmus_infoseeking":
        df = _prepare_litmus_infoseeking(cfg, rng)
    elif which == "jailbreakbench":
        df = _prepare_jailbreakbench(cfg, rng)
    else:
        raise ValueError(f"unknown prompt_set: {which}")

    # Normalize: ensure all expected columns present
    if "input" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"input": "text"})
    for col in ["prompt_id", "text", "safety_label", "axiom", "source_dataset", "char_len"]:
        if col == "prompt_id":
            df["prompt_id"] = [f"{which[:3]}_{i:05d}" for i in range(len(df))]
        elif col == "char_len":
            if "char_len" not in df.columns:
                df["char_len"] = df["text"].str.len()
        elif col == "source_dataset" and col not in df.columns:
            df["source_dataset"] = "unknown"
        elif col == "axiom" and col not in df.columns:
            df["axiom"] = "n/a"
    df = df[["prompt_id", "text", "safety_label", "axiom",
             "source_dataset", "char_len"]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_safe = int((df["safety_label"] == "safe").sum())
    n_unsafe = int((df["safety_label"] == "unsafe").sum())
    ok(f"wrote prompt set ({which}): {out_csv}  "
       f"(safe={n_safe}, unsafe={n_unsafe})")

    # Length summary
    safe_lens = df[df["safety_label"] == "safe"]["char_len"]
    unsafe_lens = df[df["safety_label"] == "unsafe"]["char_len"]
    ok(f"char length: safe median={safe_lens.median():.0f}  "
       f"unsafe median={unsafe_lens.median():.0f}")
    return df
# ===========================================================================
#  SECTION 4  —  MODEL LOADING, PRE-FLIGHT, EXTRACTION
# ===========================================================================

def load_model_and_tokenizer(model_id: str, subfolder: Optional[str],
                              gpu: GPUConfig):
    """Load a causal LM and tokenizer, applying the adaptive config.

    Handles:
      - subfolder loading (our HF repo nests IT and DPO in subfolders)
      - pad_token_id=None fallback (Llama-3 ships with this)
      - SDPA attention (fast + memory-efficient)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, gpu.dtype)
    tok_kwargs = {"trust_remote_code": True}
    if subfolder:
        tok_kwargs["subfolder"] = subfolder
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if gpu.use_sdpa:
        model_kwargs["attn_implementation"] = "sdpa"
    if subfolder:
        model_kwargs["subfolder"] = subfolder

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return model, tokenizer, model.config.num_hidden_layers, model.config.hidden_size


def free_model(model) -> None:
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------- pre-flight --------------------------

def _format_chat_prompt(tokenizer, text: str):
    """Apply chat template with add_generation_prompt=True. Falls back if
    the tokenizer doesn't support chat templates (e.g., a raw base model)."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        # Base models without chat templates: use the raw text with a BOS token.
        # This is OK for the Base→IT control — we want to see the trajectory
        # the *base* model produces for the same prompt text, no chat framing.
        return text


def preflight(cfg: PipelineConfig, prompts_df, out_root: Path,
              roles_to_check: list[str]) -> None:
    """Catch problems before the long run: disk, each model loads + shapes."""
    import torch
    banner("PRE-FLIGHT VALIDATION")

    out_root.mkdir(parents=True, exist_ok=True)
    test = out_root / ".write_test"
    try:
        test.write_text("ok"); test.unlink()
        ok("output directory writable")
    except Exception as e:
        fail(f"output directory not writable: {e}")
        raise

    free_gb = shutil.disk_usage(out_root).free / (1024 ** 3)
    # 6000 prompts * 3 models * ~270KB ≈ 5GB for the primary experiment;
    # variants add ~2-3GB more. Warn if < 10 GB.
    if free_gb < 10:
        warn(f"only {free_gb:.1f} GB free; may be tight")
    else:
        ok(f"free disk: {free_gb:.1f} GB")

    dummy_prompt = prompts_df.iloc[0]["text"]
    L_ref, D_ref = None, None
    for role in roles_to_check:
        model_id, sub = cfg.role_to_model(role)
        if model_id is None:
            warn(f"role '{role}' has no model configured; skipping preflight")
            continue
        info(f"loading {role}: {model_id} (subfolder={sub})")
        model, tokenizer, L, D = load_model_and_tokenizer(model_id, sub, cfg.gpu)
        ok(f"{role} loaded: n_layers={L}  d_model={D}  "
           f"vocab={model.config.vocab_size}")
        if L_ref is None:
            L_ref, D_ref = L, D
        elif (L, D) != (L_ref, D_ref):
            warn(f"{role} has n_layers={L} d_model={D}, differs from "
                 f"reference ({L_ref}, {D_ref}) — cross-model comparison "
                 f"will need alignment")

        chat_input = _format_chat_prompt(tokenizer, dummy_prompt)
        batch = tokenizer([chat_input], return_tensors="pt",
                           truncation=True, max_length=cfg.max_tokens)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True)
        hs = out.hidden_states
        if len(hs) != L + 1:
            fail(f"{role} returned {len(hs)} hidden states, expected {L+1}")
            raise RuntimeError("hidden_states mismatch")
        ok(f"{role} forward pass OK  "
           f"(hidden_states: {len(hs)} x {tuple(hs[0].shape)})")
        free_model(model)

    ok("pre-flight passed — safe to start extraction")


# ------------------------- extraction --------------------------

def _state_path(d: Path, role: str) -> Path:
    return d / f"state_{role}.json"


def _load_state(d: Path, role: str) -> dict:
    p = _state_path(d, role)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            warn(f"state_{role}.json corrupt; starting fresh")
    return {"done": []}


def _save_state_atomic(d: Path, role: str, state: dict) -> None:
    p = _state_path(d, role)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def extract_activations_for_role(
    role: str,
    prompts_df,
    act_root: Path,
    cfg: PipelineConfig,
    logger: RunLogger,
) -> None:
    """Extract per-prompt (L+1, D) last-token activations under `role`.

    Stores .npy per prompt in act_root/<role>/. Resume-safe via state file.
    Prints rich per-prompt progress via the Progress class.
    """
    import torch

    model_id, subfolder = cfg.role_to_model(role)
    if model_id is None:
        raise ValueError(f"role '{role}' has no model configured")

    act_dir = act_root / role
    act_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(act_root, role)
    done = set(state.get("done", []))

    todo = [
        (pid, txt)
        for pid, txt in zip(prompts_df["prompt_id"], prompts_df["text"])
        if pid not in done and not (act_dir / f"{pid}.npy").exists()
    ]
    already = len(prompts_df) - len(todo)
    info(f"{display_role(role)}: {already}/{len(prompts_df)} already extracted, "
         f"{len(todo)} to process")
    if not todo:
        ok(f"{display_role(role)}: nothing to do")
        return

    info(f"loading {display_role(role)}: {model_id}")
    model, tokenizer, L, D = load_model_and_tokenizer(model_id, subfolder, cfg.gpu)
    logger.event("model_loaded", role=role, n_layers=L, d_model=D)

    progress = Progress(f"extract:{display_role(role)}", total=len(todo))
    try:
        with torch.no_grad():
            for i, (pid, text) in enumerate(todo):
                chat_input = _format_chat_prompt(tokenizer, text)
                batch = tokenizer([chat_input], return_tensors="pt",
                                   truncation=True, max_length=cfg.max_tokens)
                attn = batch["attention_mask"]
                batch = {k: v.to(model.device) for k, v in batch.items()}

                out = model(**batch, output_hidden_states=True, use_cache=False)
                seq_len = int(attn.sum().item())
                last_idx = seq_len - 1
                traj = torch.stack(
                    [h[0, last_idx, :] for h in out.hidden_states],
                    dim=0,
                ).to(torch.float16).cpu().numpy()
                np.save(act_dir / f"{pid}.npy", traj)

                done.add(pid)
                state["done"] = sorted(done)
                _save_state_atomic(act_root, role, state)

                progress.update(i + 1)

                if (i + 1) % cfg.gpu.clear_cache_every == 0:
                    torch.cuda.empty_cache()
                if (i + 1) % 500 == 0:
                    logger.event("extraction_progress", role=role,
                                 done=i + 1, total=len(todo))
    finally:
        free_model(model)

    progress.done()
    logger.event("model_done", role=role, processed_this_run=len(todo))
    ok(f"{display_role(role)}: extraction complete ({len(todo)} processed)")


def load_trajectories_for_roles(act_root: Path, prompts_df,
                                  roles: list[str]) -> dict:
    """Load (N, L+1, D) for each role. Returns {role: array}."""
    out = {}
    for role in roles:
        act_dir = act_root / role
        arrs = []
        for pid in prompts_df["prompt_id"]:
            arrs.append(np.load(act_dir / f"{pid}.npy").astype(np.float32))
        out[role] = np.stack(arrs, axis=0)
        ok(f"loaded {role}: {out[role].shape}")
    return out
# ===========================================================================
#  SECTION 5  —  METRICS
#
#  compute_comparative_metrics(h_A, h_B)  -> dict of arrays
#     Comparative metrics: θ, SR, NDM, CTD, PL_A, PL_B, TEA, TTV (DPO-side by
#     default — but we also compute TTV for both just in case).
#
#  compute_intrinsic_metrics(h)            -> dict of arrays
#     Intrinsic metrics for a single model: speed, turn, disp (+ PL as
#     companion). Used for the model-anchored plots E and F.
# ===========================================================================

def compute_comparative_metrics(h_A: np.ndarray, h_B: np.ndarray,
                                  eps: float = 1e-8) -> dict:
    """Compute all 7 comparative metrics for a (model_A, model_B) pair.

    Inputs:
      h_A, h_B : (N, L+1, D) — per-prompt per-layer hidden states

    Returns dict with:
      theta_deg, sr, ndm, ctd, ttv        : (N, L) per-layer
      pl_a, pl_b                          : (N, L) cumulative
      tea_deg                             : (N,)
      ttv_A, ttv_B                        : (N, L) per-model twist (bonus)
    """
    N, Lp1, D = h_A.shape
    L = Lp1 - 1

    d_A = np.diff(h_A, axis=1)
    d_B = np.diff(h_B, axis=1)
    n_A = np.linalg.norm(d_A, axis=-1) + eps
    n_B = np.linalg.norm(d_B, axis=-1) + eps

    # θ — step direction angle
    dot = np.sum(d_A * d_B, axis=-1)
    cos = np.clip(dot / (n_A * n_B), -1.0, 1.0)
    theta = np.degrees(np.arccos(cos))

    # SR — scale ratio
    sr = n_B / n_A

    # NDM — normalized deflection magnitude
    ndm = np.linalg.norm(d_B - d_A, axis=-1) / n_A

    # CTD — global position divergence (align to per-step shape by dropping ℓ=0)
    pos_diff_norm = np.linalg.norm(h_B - h_A, axis=-1)
    pos_A_norm = np.linalg.norm(h_A, axis=-1) + eps
    ctd = (pos_diff_norm / pos_A_norm)[:, 1:]

    # PL — path length
    pl_a = np.cumsum(n_A, axis=1)
    pl_b = np.cumsum(n_B, axis=1)

    # TEA — endpoint angle
    start = h_A[:, 0, :]
    v_A = h_A[:, -1, :] - start
    v_B = h_B[:, -1, :] - start
    n_vA = np.linalg.norm(v_A, axis=-1) + eps
    n_vB = np.linalg.norm(v_B, axis=-1) + eps
    cos_g = np.clip(np.sum(v_A * v_B, axis=-1) / (n_vA * n_vB), -1.0, 1.0)
    tea_deg = np.degrees(np.arccos(cos_g))

    # TTV — trajectory twist volume (computed per-model)
    def _ttv(d: np.ndarray) -> np.ndarray:
        N_, L_, D_ = d.shape
        out = np.full((N_, L_), np.nan, dtype=np.float32)
        if L_ < 3:
            return out
        v1, v2, v3 = d[:, :-2, :], d[:, 1:-1, :], d[:, 2:, :]
        nv1 = np.linalg.norm(v1, axis=-1) + eps
        nv2 = np.linalg.norm(v2, axis=-1) + eps
        nv3 = np.linalg.norm(v3, axis=-1) + eps
        G = np.zeros((N_, L_ - 2, 3, 3), dtype=np.float32)
        G[..., 0, 0] = np.sum(v1 * v1, axis=-1)
        G[..., 1, 1] = np.sum(v2 * v2, axis=-1)
        G[..., 2, 2] = np.sum(v3 * v3, axis=-1)
        G[..., 0, 1] = G[..., 1, 0] = np.sum(v1 * v2, axis=-1)
        G[..., 0, 2] = G[..., 2, 0] = np.sum(v1 * v3, axis=-1)
        G[..., 1, 2] = G[..., 2, 1] = np.sum(v2 * v3, axis=-1)
        det = np.linalg.det(G)
        vol = np.sqrt(np.clip(det, 0.0, None))
        out[:, 1:-1] = (vol / (nv1 * nv2 * nv3)).astype(np.float32)
        return out

    ttv_A = _ttv(d_A)
    ttv_B = _ttv(d_B)

    return {
        "theta_deg": theta,
        "sr": sr,
        "ndm": ndm,
        "ctd": ctd,
        "pl_a": pl_a,
        "pl_b": pl_b,
        "ttv": ttv_B,         # default to model-B's twist as headline
        "ttv_A": ttv_A,
        "ttv_B": ttv_B,
        "tea_deg": tea_deg,
    }


def compute_intrinsic_metrics(h: np.ndarray, eps: float = 1e-8) -> dict:
    """Per-model trajectory quantities for the model-anchored plots.

    Inputs:
      h : (N, L+1, D)

    Returns:
      speed    : (N, L)   step magnitudes
      turn_deg : (N, L)   turning angle between consecutive steps (NaN at ℓ=0)
      disp     : (N, L)   cumulative displacement from layer 0
      pl       : (N, L)   cumulative path length
    """
    N, Lp1, D = h.shape
    L = Lp1 - 1
    d = np.diff(h, axis=1)                       # (N, L, D)
    n = np.linalg.norm(d, axis=-1) + eps         # (N, L)
    speed = n.astype(np.float32)

    # Turning angle between Δh_{ℓ-1} and Δh_ℓ
    turn = np.full((N, L), np.nan, dtype=np.float32)
    if L >= 2:
        v1 = d[:, :-1, :]; v2 = d[:, 1:, :]
        nv1 = np.linalg.norm(v1, axis=-1) + eps
        nv2 = np.linalg.norm(v2, axis=-1) + eps
        c = np.clip(np.sum(v1 * v2, axis=-1) / (nv1 * nv2), -1.0, 1.0)
        turn[:, 1:] = np.degrees(np.arccos(c)).astype(np.float32)

    # Cumulative displacement from layer 0
    disp = np.linalg.norm(h[:, 1:, :] - h[:, :1, :], axis=-1).astype(np.float32)

    # Path length
    pl = np.cumsum(n, axis=1).astype(np.float32)

    return {"speed": speed, "turn_deg": turn, "disp": disp, "pl": pl}


# ===========================================================================
#  Stat helpers (no scipy dep)
# ===========================================================================

def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000,
                  alpha: float = 0.05, seed: int = 0):
    """Per-layer mean + bootstrap CI, NaN-safe."""
    rng = np.random.default_rng(seed)
    if x.ndim == 1:
        m = ~np.isnan(x); xv = x[m]
        if len(xv) == 0: return np.nan, np.nan, np.nan
        idx = rng.integers(0, len(xv), size=(n_boot, len(xv)))
        boot = xv[idx].mean(axis=1)
        return xv.mean(), np.percentile(boot, 100*alpha/2), np.percentile(boot, 100*(1-alpha/2))
    N, L = x.shape
    idx = rng.integers(0, N, size=(n_boot, N))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        boot = np.nanmean(x[idx], axis=1)
        mean = np.nanmean(x, axis=0)
        lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    return mean, lo, hi


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    if pooled < 1e-12: return 0.0
    return float((a.mean() - b.mean()) / pooled)


def permutation_test(a: np.ndarray, b: np.ndarray,
                     n_perm: int = 10_000, seed: int = 0) -> float:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0: return 1.0
    rng = np.random.default_rng(seed)
    combined = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    na = len(a); count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        if abs(combined[:na].mean() - combined[na:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)
# ===========================================================================
#  SECTION 6  —  ANALYSES: TTV null + linear probe
# ===========================================================================

def ttv_null_distribution(observed_ttv: np.ndarray, D: int = 4096,
                           n_samples: int = 6000, seed: int = 0) -> dict:
    """Is the observed TTV signal meaningfully different from random?

    Generate n_samples random trajectories of the same shape (L+1 points in
    R^D), compute their TTV values, and compare distributions.

    Returns summary statistics for both distributions.
    """
    N, L = observed_ttv.shape
    rng = np.random.default_rng(seed)
    # Build random "trajectories": accumulate random Gaussian steps so the
    # points have the right structural relationship.
    info(f"generating {n_samples} random trajectories of shape "
         f"({L+1}, {D}) for TTV null distribution ...")
    progress = Progress("TTV-null", total=n_samples, min_interval_s=2.0)

    # Compute TTV for random trajectories in batches to avoid a giant alloc
    batch = 200
    random_ttvs = []
    done = 0
    for b0 in range(0, n_samples, batch):
        b1 = min(b0 + batch, n_samples)
        h_rand = rng.standard_normal(size=(b1 - b0, L + 1, D)).astype(np.float32)
        d = np.diff(h_rand, axis=1)
        # Reuse the Gram-determinant inline (don't import compute_comparative)
        from_ttv = _ttv_gram(d)
        random_ttvs.append(from_ttv)
        done = b1
        progress.update(done)
    progress.done()

    random_ttv = np.concatenate(random_ttvs, axis=0)   # (n_samples, L)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        obs_mean = np.nanmean(observed_ttv, axis=0)
        rand_mean = np.nanmean(random_ttv, axis=0)
        obs_std = np.nanstd(observed_ttv, axis=0)
        rand_std = np.nanstd(random_ttv, axis=0)

    return {
        "observed_per_layer_mean":   obs_mean.astype(float).tolist(),
        "random_per_layer_mean":     rand_mean.astype(float).tolist(),
        "observed_per_layer_std":    obs_std.astype(float).tolist(),
        "random_per_layer_std":      rand_std.astype(float).tolist(),
        "observed_values":           observed_ttv,   # for plotting
        "random_values":             random_ttv,     # for plotting
        "interpretation": (
            "If observed_per_layer_mean ≈ random_per_layer_mean at most layers, "
            "TTV is noise-dominated and should not be used as a headline metric. "
            "If observed ≠ random systematically, TTV measures real structure."
        ),
    }


def _ttv_gram(d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Duplicate of the TTV inner function from section 5, for use by
    the null-distribution code without circular imports.

    d : (N, L, D)  step vectors
    returns TTV : (N, L) with NaN at boundaries.
    """
    N_, L_, D_ = d.shape
    out = np.full((N_, L_), np.nan, dtype=np.float32)
    if L_ < 3: return out
    v1, v2, v3 = d[:, :-2, :], d[:, 1:-1, :], d[:, 2:, :]
    nv1 = np.linalg.norm(v1, axis=-1) + eps
    nv2 = np.linalg.norm(v2, axis=-1) + eps
    nv3 = np.linalg.norm(v3, axis=-1) + eps
    G = np.zeros((N_, L_ - 2, 3, 3), dtype=np.float32)
    G[..., 0, 0] = np.sum(v1 * v1, axis=-1)
    G[..., 1, 1] = np.sum(v2 * v2, axis=-1)
    G[..., 2, 2] = np.sum(v3 * v3, axis=-1)
    G[..., 0, 1] = G[..., 1, 0] = np.sum(v1 * v2, axis=-1)
    G[..., 0, 2] = G[..., 2, 0] = np.sum(v1 * v3, axis=-1)
    G[..., 1, 2] = G[..., 2, 1] = np.sum(v2 * v3, axis=-1)
    det = np.linalg.det(G)
    vol = np.sqrt(np.clip(det, 0.0, None))
    out[:, 1:-1] = (vol / (nv1 * nv2 * nv3)).astype(np.float32)
    return out


# ===========================================================================
#  Linear probe
#
#  Intuition: A linear probe is a minimal classifier (logistic regression)
#  trained to predict a label from a model's internal representations. If
#  activations at a given layer *contain* information that distinguishes
#  unsafe from safe prompts, a linear probe can find that direction.
#
#  Two outputs:
#    1. Probe accuracy per layer — tells us where harmfulness information
#       emerges. If layer 0 already has high accuracy, the information is
#       purely lexical. If it emerges progressively and peaks around L14,
#       that's consistent with an alignment mechanism.
#    2. Direction-alignment between probe direction and deflection direction
#       at L14 — tells us whether the geometric deflection we measured goes
#       in the direction that encodes harmfulness.
# ===========================================================================

def linear_probe_per_layer(h: np.ndarray, labels: np.ndarray,
                             test_frac: float = 0.2, seed: int = 0) -> dict:
    """Train a logistic regression probe at each layer. Returns test accuracy
    per layer and the learned weight vector at each layer.

    h      : (N, L+1, D)  activations (typically from one model, e.g. DPO)
    labels : (N,) array of 0/1 labels
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError:
        warn("sklearn not installed — skipping linear probe")
        return {"accuracy": [], "weights": None}

    N, Lp1, D = h.shape
    y = labels.astype(int)

    # Split once so all layers use the same train/test prompts
    idx_train, idx_test = train_test_split(
        np.arange(N), test_size=test_frac, random_state=seed, stratify=y,
    )
    accs = []
    weights = []
    progress = Progress("probe", total=Lp1, min_interval_s=0.5)
    for l in range(Lp1):
        X = h[:, l, :]
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accs.append(float(acc))
        weights.append(clf.coef_[0].astype(np.float32))
        progress.update(l + 1)
    progress.done()

    return {
        "accuracy": accs,                    # len Lp1
        "weights": np.stack(weights, axis=0),  # (Lp1, D)
        "idx_train": idx_train,
        "idx_test": idx_test,
    }


def probe_direction_alignment(probe_weights: np.ndarray,
                               h_A: np.ndarray, h_B: np.ndarray,
                               layer: int,
                               unsafe_mask: np.ndarray,
                               eps: float = 1e-8) -> dict:
    """Cosine similarity between the probe direction at `layer` and the mean
    deflection direction (Δh_B − Δh_A, averaged over unsafe prompts) at the
    corresponding step boundary.

    Step `layer` sits between hidden states at positions `layer` and `layer+1`.
    For the probe, we want the direction at hidden-state position `layer` —
    but we compare against the deflection *into* that position, i.e. the
    step from layer-1 to layer.

    Returns cosine similarity in [-1, 1].
    """
    if probe_weights is None or len(probe_weights) == 0:
        return {"cos_sim": None}

    # Probe direction at the chosen hidden-state index
    w = probe_weights[layer].astype(np.float32)
    w_unit = w / (np.linalg.norm(w) + eps)

    # Deflection direction averaged over unsafe prompts at the matching step
    if layer == 0:
        warn("probe_direction_alignment: layer=0 has no incoming step; "
             "using step 0→1 instead")
        step_idx = 0
    else:
        step_idx = layer - 1   # step ending at hidden-state position `layer`
    d_A = h_A[:, step_idx + 1, :] - h_A[:, step_idx, :]
    d_B = h_B[:, step_idx + 1, :] - h_B[:, step_idx, :]
    defl = (d_B - d_A)[unsafe_mask].mean(axis=0)   # (D,)
    d_unit = defl / (np.linalg.norm(defl) + eps)

    cos_sim = float(np.dot(w_unit, d_unit))
    return {
        "cos_sim": cos_sim,
        "layer": layer,
        "step_idx": step_idx,
        "interpretation": (
            f"cos_sim={cos_sim:+.3f}. Values near +1 mean the DPO-induced "
            "deflection points in the same direction the probe uses to "
            "classify harmfulness — i.e., DPO is nudging representations "
            "*along the harmfulness axis*. Near 0 means deflection and "
            "harmfulness are encoded in different subspaces."
        ),
    }
# ===========================================================================
#  SECTION 7  —  2D PLOTTING
#
#  Figures produced:
#    A  per-layer comparative metrics (2×3 grid)
#    B  effect-size + significance summary
#    C  TEA violin (global endpoint angle)
#    D  per-axiom bar chart at peak θ layer
#    E  unsafe-only, model-anchored intrinsic metrics
#    F  safe-only, model-anchored intrinsic metrics
#    G  safe-variant comparison (original vs InfoSeeking vs JailbreakBench)
#    H  experiment comparison (primary vs control_base)
#    I  linear probe per-layer accuracy
# ===========================================================================

def _mpl_setup():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COL_GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _panel_safe_unsafe(ax, layers, safe_vals, unsafe_vals, metric_key: str,
                        label_A: str = "A", label_B: str = "B",
                        n_boot: int = 5_000):
    """One panel: safe vs unsafe curves with CI bands and peak-Δ marker."""
    mean_s, lo_s, hi_s = bootstrap_ci(safe_vals, n_boot=n_boot, seed=1)
    mean_u, lo_u, hi_u = bootstrap_ci(unsafe_vals, n_boot=n_boot, seed=2)
    ax.fill_between(layers, lo_s, hi_s, color=COL_SAFE, alpha=0.2, linewidth=0)
    ax.plot(layers, mean_s, color=COL_SAFE, linewidth=2.0,
            label=f"safe ({label_A}→{label_B})", zorder=3)
    ax.fill_between(layers, lo_u, hi_u, color=COL_UNSAFE, alpha=0.2, linewidth=0)
    ax.plot(layers, mean_u, color=COL_UNSAFE, linewidth=2.0,
            label=f"unsafe ({label_A}→{label_B})", zorder=3)

    sep = np.abs(np.asarray(mean_u) - np.asarray(mean_s))
    if np.any(~np.isnan(sep)):
        peak = int(np.nanargmax(sep))
        ax.axvline(peak, color=COL_ACCENT, linestyle="--", linewidth=1.0,
                   alpha=0.6, zorder=2)
        mid_y = (mean_s[peak] + mean_u[peak]) / 2
        xo = 12 if peak < len(layers) / 2 else -12
        ha = "left" if xo > 0 else "right"
        ax.annotate(
            f"peak Δ(gap) @ L{peak}",
            xy=(peak, mid_y), xytext=(xo, 0), textcoords="offset points",
            fontsize=8, color=COL_ACCENT, weight="bold",
            ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=COL_ACCENT, alpha=0.85),
        )

    ax.set_xlabel("Layer transition ℓ")
    ax.set_ylabel(METRIC_LABELS[metric_key])
    ax.set_title(METRIC_LABELS[metric_key].split("  ")[0],
                 loc="left", weight="bold")
    ax.text(0.02, 0.98, METRIC_BLURBS[metric_key],
            transform=ax.transAxes, fontsize=8, color="#555",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#DDD", alpha=0.8))
    ax.legend(loc="lower right", fontsize=9)


def _panel_model_anchored(ax, layers, vals_A, vals_B, metric_key: str,
                            role_A: str, role_B: str, prompt_side: str):
    """One panel: two lines (model A vs model B) on a fixed prompt subset."""
    mean_a, lo_a, hi_a = bootstrap_ci(vals_A, n_boot=5_000, seed=1)
    mean_b, lo_b, hi_b = bootstrap_ci(vals_B, n_boot=5_000, seed=2)

    ax.fill_between(layers, lo_a, hi_a, color=COL_MODEL_A, alpha=0.18, linewidth=0)
    ax.plot(layers, mean_a, color=COL_MODEL_A, linewidth=2.0,
            label=f"{display_role(role_A)} · {prompt_side}", zorder=3)
    ax.fill_between(layers, lo_b, hi_b, color=COL_MODEL_B, alpha=0.18, linewidth=0)
    ax.plot(layers, mean_b, color=COL_MODEL_B, linewidth=2.0,
            label=f"{display_role(role_B)} · {prompt_side}", zorder=3)

    ax.set_xlabel("Layer transition ℓ")
    ax.set_ylabel(METRIC_LABELS[metric_key])
    ax.set_title(METRIC_LABELS[metric_key].split("  ")[0],
                 loc="left", weight="bold")
    ax.text(0.02, 0.98, METRIC_BLURBS[metric_key],
            transform=ax.transAxes, fontsize=8, color="#555",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#DDD", alpha=0.8))
    ax.legend(loc="lower right", fontsize=9)


# -------------------------- Figure A --------------------------

def plot_figure_A(pair_dir: Path, metrics: dict, prompts_df,
                    label_A: str, label_B: str, pair_name: str) -> dict:
    """Per-layer comparative metrics — 2×3 grid, safe vs unsafe curves."""
    import matplotlib.pyplot as plt
    _mpl_setup()
    plot_dir = pair_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    safe = (prompts_df["safety_label"] == "safe").to_numpy()
    unsafe = ~safe

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panel_keys = ["theta_deg", "sr", "ndm", "ctd", "ttv"]
    for i, key in enumerate(panel_keys):
        ax = axes.flat[i]
        m = metrics[key]
        L = m.shape[1]
        _panel_safe_unsafe(ax, np.arange(L), m[safe], m[unsafe], key,
                            label_A=label_A, label_B=label_B)

    # 6th panel: PL — 4 lines (model A × safe/unsafe, model B × safe/unsafe)
    ax = axes.flat[5]
    L_pl = metrics["pl_a"].shape[1]
    layers = np.arange(L_pl)
    for data, lab, color, ls in [
        (metrics["pl_a"][safe],   f"{label_A} · safe",   COL_MODEL_A, "-"),
        (metrics["pl_a"][unsafe], f"{label_A} · unsafe", COL_MODEL_A, "--"),
        (metrics["pl_b"][safe],   f"{label_B} · safe",   COL_MODEL_B, "-"),
        (metrics["pl_b"][unsafe], f"{label_B} · unsafe", COL_MODEL_B, "--"),
    ]:
        mean, lo, hi = bootstrap_ci(data, n_boot=3_000,
                                      seed=hash(lab) % 2**31)
        ax.fill_between(layers, lo, hi, color=color, alpha=0.1, linewidth=0)
        ax.plot(layers, mean, color=color, linestyle=ls, linewidth=2, label=lab)
    ax.set_xlabel("Layer transition ℓ")
    ax.set_ylabel("Cumulative path length ‖Σ Δh‖")
    ax.set_title("PL  Cumulative path length", loc="left", weight="bold")
    ax.text(0.02, 0.98,
            f"Total distance traveled through the network. "
            f"{label_A} vs {label_B} × safe vs unsafe.",
            transform=ax.transAxes, fontsize=8, color="#555",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#DDD", alpha=0.8))
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"Figure A — Per-layer comparative metrics   "
        f"[{pair_name}: {label_A}→{label_B}]     "
        f"n_safe={safe.sum()}  n_unsafe={unsafe.sum()}",
        x=0.02, ha="left", y=1.00, weight="bold", fontsize=14,
    )
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figureA_per_layer_metrics.{ext}")
    plt.close(fig)
    ok("wrote figureA_per_layer_metrics.{png,pdf}")

    # Build summary stats for this experiment's metrics
    summary = {"label_A": label_A, "label_B": label_B, "metrics": {}}
    for key in ["theta_deg", "sr", "ndm", "ctd", "ttv"]:
        m = metrics[key]
        L = m.shape[1]
        ds = np.array([cohens_d(m[unsafe, l], m[safe, l]) for l in range(L)])
        if np.all(np.isnan(ds)):
            continue
        peak = int(np.nanargmax(np.abs(ds)))
        summary["metrics"][key] = {
            "peak_layer": peak,
            "cohens_d_at_peak": float(ds[peak]),
            "mean_safe_at_peak": float(np.nanmean(m[safe, peak])),
            "mean_unsafe_at_peak": float(np.nanmean(m[unsafe, peak])),
        }
    # TEA
    tea = metrics["tea_deg"]
    summary["metrics"]["tea_deg"] = {
        "mean_safe": float(tea[safe].mean()),
        "mean_unsafe": float(tea[unsafe].mean()),
        "cohens_d": cohens_d(tea[unsafe], tea[safe]),
    }
    return summary


# -------------------------- Figure B --------------------------

def plot_figure_B(pair_dir: Path, metrics: dict, prompts_df,
                    label_A: str, label_B: str, pair_name: str) -> None:
    """Cohen's d and −log10 p per layer for each comparative per-layer metric."""
    import matplotlib.pyplot as plt
    _mpl_setup()
    plot_dir = pair_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe = (prompts_df["safety_label"] == "safe").to_numpy()
    unsafe = ~safe

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    colors = {"theta_deg": COL_UNSAFE, "sr": "#8D6E63", "ndm": COL_ACCENT,
              "ctd": "#1565C0", "ttv": "#2E7D32"}
    markers = {"theta_deg": "o", "sr": "^", "ndm": "s", "ctd": "D", "ttv": "*"}
    labels = {"theta_deg": "θ", "sr": "SR", "ndm": "NDM",
              "ctd": "CTD", "ttv": "TTV"}

    info("computing per-layer Cohen's d + permutation p (2k perms each)...")
    per_layer_stats = {}
    progress = Progress("fig-B:stats", total=5 * 32, min_interval_s=1.0)
    done = 0
    for key in ["theta_deg", "sr", "ndm", "ctd", "ttv"]:
        m = metrics[key]
        L = m.shape[1]
        ds, ps = [], []
        for l in range(L):
            ds.append(cohens_d(m[unsafe, l], m[safe, l]))
            ps.append(permutation_test(m[unsafe, l], m[safe, l],
                                        n_perm=2_000, seed=l))
            done += 1
            progress.update(done)
        per_layer_stats[key] = {"d": np.array(ds), "p": np.array(ps)}
    progress.done()

    layers = np.arange(len(per_layer_stats["theta_deg"]["d"]))
    for key in ["theta_deg", "sr", "ndm", "ctd", "ttv"]:
        axes[0].plot(layers, per_layer_stats[key]["d"],
                     color=colors[key], marker=markers[key], ms=5,
                     linewidth=1.5, label=labels[key])
    axes[0].axhline(0, color="#888", linewidth=0.8)
    axes[0].axhline(0.5, color="#AAA", linewidth=0.5, linestyle=":")
    axes[0].axhline(0.8, color="#AAA", linewidth=0.5, linestyle=":")
    axes[0].set_xlabel("Layer transition ℓ")
    axes[0].set_ylabel("Cohen's d  (unsafe − safe)")
    axes[0].set_title("Effect size by layer", loc="left", weight="bold")
    axes[0].legend(loc="best", ncol=2, fontsize=9)

    for key in ["theta_deg", "sr", "ndm", "ctd", "ttv"]:
        p = per_layer_stats[key]["p"]
        axes[1].plot(np.arange(len(p)), -np.log10(p),
                     color=colors[key], marker=markers[key], ms=5,
                     linewidth=1.5, label=labels[key])
    axes[1].axhline(-np.log10(0.05), color="#888", linewidth=0.8,
                     linestyle="--", label="p=0.05")
    axes[1].set_xlabel("Layer transition ℓ")
    axes[1].set_ylabel("−log₁₀ p (permutation, 2k)")
    axes[1].set_title("Significance by layer", loc="left", weight="bold")
    axes[1].legend(loc="best", ncol=2, fontsize=9)

    fig.suptitle(
        f"Figure B — Statistical summary   "
        f"[{pair_name}: {label_A}→{label_B}]",
        x=0.02, ha="left", weight="bold", fontsize=14,
    )
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figureB_statistical_summary.{ext}")
    plt.close(fig)
    ok("wrote figureB_statistical_summary.{png,pdf}")


# -------------------------- Figure C --------------------------

def plot_figure_C(pair_dir: Path, metrics: dict, prompts_df,
                    label_A: str, label_B: str, pair_name: str) -> None:
    """TEA distribution — safe vs unsafe violins."""
    import matplotlib.pyplot as plt
    _mpl_setup()
    plot_dir = pair_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe = (prompts_df["safety_label"] == "safe").to_numpy()
    unsafe = ~safe
    tea = metrics["tea_deg"]

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [tea[safe], tea[unsafe]]
    parts = ax.violinplot(data, showmeans=True, showmedians=True,
                           positions=[0, 1], widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([COL_SAFE, COL_UNSAFE][i])
        pc.set_alpha(0.5)
    for k in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if k in parts: parts[k].set_edgecolor("#444")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"safe  (n={safe.sum()})",
                         f"unsafe  (n={unsafe.sum()})"])
    ax.set_ylabel(METRIC_LABELS["tea_deg"])
    ax.set_title(f"Figure C — Trajectory Endpoint Angle ({label_A}→{label_B})",
                 loc="left", weight="bold")
    ax.text(0.02, 0.98, METRIC_BLURBS["tea_deg"],
            transform=ax.transAxes, fontsize=9, color="#555",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#DDD", alpha=0.85))
    d = cohens_d(tea[unsafe], tea[safe])
    p = permutation_test(tea[unsafe], tea[safe], n_perm=10_000, seed=11)
    ax.text(0.98, 0.02, f"d = {d:+.3f}\np = {p:.1e}",
            transform=ax.transAxes, fontsize=10, color="#222",
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#BBB", alpha=0.9))
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figureC_tea_distribution.{ext}")
    plt.close(fig)
    ok("wrote figureC_tea_distribution.{png,pdf}")


# -------------------------- Figure D --------------------------

def plot_figure_D(pair_dir: Path, metrics: dict, prompts_df,
                    pair_name: str) -> None:
    """Per-axiom bar chart at peak θ layer."""
    import matplotlib.pyplot as plt
    _mpl_setup()
    plot_dir = pair_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe = (prompts_df["safety_label"] == "safe").to_numpy()
    unsafe = ~safe
    axioms_arr = prompts_df["axiom"].to_numpy()

    # Find peak θ layer by Cohen's d
    theta = metrics["theta_deg"]
    L = theta.shape[1]
    ds = np.array([cohens_d(theta[unsafe, l], theta[safe, l]) for l in range(L)])
    peak = int(np.nanargmax(np.abs(ds)))

    # Compute mean θ at peak layer per axiom
    unique_axioms = sorted(set(axioms_arr[unsafe]))
    rows = []
    for ax_name in unique_axioms:
        m = (axioms_arr == ax_name) & unsafe
        if m.sum() < 5: continue
        rows.append((ax_name, m.sum(), float(theta[m, peak].mean())))
    safe_mean = float(theta[safe, peak].mean())

    if not rows:
        warn("Figure D: no axioms with ≥5 samples — skipping")
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    names = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    vals = [r[2] for r in rows]
    xpos = np.arange(len(names))
    bars = ax.bar(xpos, vals, color=COL_UNSAFE, alpha=0.7,
                   edgecolor="white", linewidth=1.2)
    ax.axhline(safe_mean, color=COL_SAFE, linestyle="--", linewidth=2,
                label=f"safe reference  θ={safe_mean:.2f}°")
    for b, v, n in zip(bars, vals, counts):
        ax.annotate(f"{v:.2f}°\nn={n}",
                     xy=(b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8, color="#333")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel(f"Mean θ at L{peak}  (degrees)")
    ax.set_title(f"Figure D — θ by unsafe-axiom at peak Cohen's d layer (L{peak})   "
                  f"[{pair_name}]", loc="left", weight="bold")
    ax.text(0.02, 0.98,
            "Each bar = mean θ at the peak layer for unsafe prompts of that axiom. "
            "Dashed line = θ for safe prompts at the same layer. "
            "A flat bar profile means DPO treats all harm types similarly; "
            "tall variation would mean axiom-specific mechanisms.",
            transform=ax.transAxes, fontsize=8, color="#555",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#DDD", alpha=0.85))
    ax.legend(loc="lower right", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figureD_per_axiom.{ext}")
    plt.close(fig)
    ok("wrote figureD_per_axiom.{png,pdf}")


# -------------------------- Figures E, F --------------------------

def plot_figures_EF(pair_dir: Path, intrinsic_A: dict, intrinsic_B: dict,
                      prompts_df, role_A: str, role_B: str,
                      pair_name: str) -> None:
    """Model-anchored intrinsic trajectories: one figure each for unsafe-only
    and safe-only. 2×2 grid per figure: speed, turn, disp, PL.
    """
    import matplotlib.pyplot as plt
    _mpl_setup()
    plot_dir = pair_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe = (prompts_df["safety_label"] == "safe").to_numpy()
    unsafe = ~safe

    for side_name, mask, fname in [
        ("unsafe", unsafe, "figureE_unsafe_model_anchored"),
        ("safe",   safe,   "figureF_safe_model_anchored"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        keys = ["speed", "turn_deg", "disp", "pl"]
        display_keys = ["speed", "turn_deg", "disp", "pl_a"]  # use pl_a label for pl panel
        for i, (key_internal, key_label) in enumerate(zip(keys, display_keys)):
            ax = axes.flat[i]
            vals_A = intrinsic_A[key_internal][mask]
            vals_B = intrinsic_B[key_internal][mask]
            L = vals_A.shape[1]
            # use a metric-key that exists in METRIC_LABELS
            display_key = key_internal if key_internal in METRIC_LABELS else "pl_a"
            _panel_model_anchored(
                ax, np.arange(L), vals_A, vals_B,
                display_key, role_A=role_A, role_B=role_B,
                prompt_side=side_name,
            )
        fig.suptitle(
            f"Figure {'E' if side_name == 'unsafe' else 'F'} — "
            f"{side_name.capitalize()}-only trajectories: "
            f"{display_role(role_A)} vs {display_role(role_B)}   "
            f"[{pair_name}]   n={mask.sum()}",
            x=0.02, ha="left", weight="bold", fontsize=14,
        )
        for ext in ("png", "pdf"):
            fig.savefig(plot_dir / f"{fname}.{ext}")
        plt.close(fig)
        ok(f"wrote {fname}.{{png,pdf}}")


# -------------------------- Figure G --------------------------

def plot_figure_G(pair_dir: Path, variant_metrics: dict,
                    variant_prompts: dict, pair_name: str) -> None:
    """Safe-prompt variant comparison. For each comparative metric, overlay
    the unsafe-minus-safe Cohen's d curve under each variant.

    variant_metrics: {"litmus_original": metrics_dict, "litmus_infoseeking": ...,
                       "jailbreakbench": ...}
    """
    import matplotlib.pyplot as plt
    _mpl_setup()
    plot_dir = pair_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    variants = [v for v in ["litmus_original", "litmus_infoseeking", "jailbreakbench"]
                if v in variant_metrics]
    if len(variants) < 2:
        warn("Figure G: need at least 2 safe-prompt variants — skipping")
        return

    variant_colors = {
        "litmus_original":     "#1F4E79",
        "litmus_infoseeking":  "#E8871E",
        "jailbreakbench":      "#2E7D32",
    }
    variant_labels = {
        "litmus_original":     "Litmus (original, length-matched)",
        "litmus_infoseeking":  "Litmus (Information-Seeking safe)",
        "jailbreakbench":      "JailbreakBench (content-matched)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panel_keys = ["theta_deg", "sr", "ndm", "ctd", "ttv"]
    for i, key in enumerate(panel_keys):
        ax = axes.flat[i]
        for variant in variants:
            m = variant_metrics[variant][key]
            df = variant_prompts[variant]
            safe = (df["safety_label"] == "safe").to_numpy()
            unsafe = ~safe
            L = m.shape[1]
            ds = np.array([cohens_d(m[unsafe, l], m[safe, l]) for l in range(L)])
            ax.plot(np.arange(L), ds, color=variant_colors[variant],
                    linewidth=2, label=variant_labels[variant], marker="o", ms=3)
        ax.axhline(0, color="#888", linewidth=0.7)
        ax.axhline(0.8, color="#AAA", linewidth=0.4, linestyle=":")
        ax.set_xlabel("Layer transition ℓ")
        ax.set_ylabel("Cohen's d  (unsafe − safe)")
        ax.set_title(METRIC_LABELS[key].split("  ")[0], loc="left", weight="bold")
        ax.legend(loc="best", fontsize=8)

    # 6th panel: text summary
    ax = axes.flat[5]
    ax.axis("off")
    ax.text(0.02, 0.95,
            "Reading the overlay:\n"
            "  • All curves roughly flat near 0 → no safe-vs-unsafe signal\n"
            "  • Curve peaks drop going from 'Litmus original' → 'Info-Seeking'\n"
            "    → 'JailbreakBench' : lexical confound was doing most of the work\n"
            "  • Curve peaks survive across variants → robust alignment signal\n\n"
            "Layer-0 Cohen's d is the cleanest lexical-confound diagnostic:\n"
            "  if it's near 0 under JailbreakBench, content-matching worked.",
            transform=ax.transAxes, fontsize=10, color="#333",
            va="top", ha="left")

    fig.suptitle(f"Figure G — Safe-prompt variant comparison   [{pair_name}]",
                 x=0.02, ha="left", weight="bold", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figureG_safe_prompt_variants.{ext}")
    plt.close(fig)
    ok("wrote figureG_safe_prompt_variants.{png,pdf}")


# -------------------------- Figure H --------------------------

def plot_figure_H(pair_dir: Path, experiment_metrics: dict,
                    experiment_prompts: dict, pair_name: str) -> None:
    """Experiment comparison: primary (IT→DPO) vs control_base (Base→IT).

    Answers: "is the L14 deflection pattern specific to DPO, or does it also
    appear under generic instruction-tuning?"
    """
    import matplotlib.pyplot as plt
    _mpl_setup()
    plot_dir = pair_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    experiments = [e for e in ["primary", "control_base", "safe_A", "safe_B"]
                    if e in experiment_metrics]
    if len(experiments) < 2:
        warn("Figure H: need ≥2 experiments — skipping")
        return

    exp_colors = {
        "primary":       "#C73E1D",
        "control_base":  "#1A7F5A",
        "safe_A":        "#E8871E",
        "safe_B":        "#2E7D32",
    }
    exp_labels = {
        "primary":      "IT → DPO (Litmus)",
        "control_base": "Base → IT (control)",
        "safe_A":       "IT → DPO (InfoSeeking safe)",
        "safe_B":       "IT → DPO (JailbreakBench)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panel_keys = ["theta_deg", "sr", "ndm", "ctd", "ttv"]
    for i, key in enumerate(panel_keys):
        ax = axes.flat[i]
        for exp in experiments:
            m = experiment_metrics[exp][key]
            df = experiment_prompts[exp]
            safe = (df["safety_label"] == "safe").to_numpy()
            unsafe = ~safe
            L = m.shape[1]
            ds = np.array([cohens_d(m[unsafe, l], m[safe, l]) for l in range(L)])
            ax.plot(np.arange(L), ds, color=exp_colors[exp],
                    linewidth=2, label=exp_labels[exp], marker="o", ms=3)
        ax.axhline(0, color="#888", linewidth=0.7)
        ax.axhline(0.8, color="#AAA", linewidth=0.4, linestyle=":")
        ax.set_xlabel("Layer transition ℓ")
        ax.set_ylabel("Cohen's d  (unsafe − safe)")
        ax.set_title(METRIC_LABELS[key].split("  ")[0], loc="left", weight="bold")
        ax.legend(loc="best", fontsize=8)

    ax = axes.flat[5]
    ax.axis("off")
    ax.text(0.02, 0.95,
            "This plot is the decisive disambiguation:\n\n"
            "  If Base→IT shows the same L14 peak as IT→DPO, the signature\n"
            "  is generic fine-tuning-meets-lexical-difference, NOT alignment-\n"
            "  specific. The paper would need to be reframed.\n\n"
            "  If Base→IT is flat and only IT→DPO peaks at L14, the signature\n"
            "  IS alignment-specific. We have a defensible main finding.\n\n"
            "Safe-variant curves (orange, green) show whether reducing the\n"
            "lexical confound makes the effect shrink — telling us how much\n"
            "of the signal is alignment vs. word-level propagation.",
            transform=ax.transAxes, fontsize=10, color="#333",
            va="top", ha="left")

    fig.suptitle(f"Figure H — Experiment comparison (primary vs control)   "
                 f"[{pair_name}]",
                 x=0.02, ha="left", weight="bold", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figureH_experiment_comparison.{ext}")
    plt.close(fig)
    ok("wrote figureH_experiment_comparison.{png,pdf}")


# -------------------------- Figure I --------------------------

def plot_figure_I(pair_dir: Path, probe_result: dict, alignment_result: dict,
                    pair_name: str, role_for_probe: str) -> None:
    """Linear probe: accuracy per layer + direction alignment readout."""
    import matplotlib.pyplot as plt
    _mpl_setup()
    plot_dir = pair_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not probe_result.get("accuracy"):
        warn("Figure I: no probe accuracy — skipping")
        return

    acc = probe_result["accuracy"]
    layers = np.arange(len(acc))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(layers, acc, color=COL_ACCENT, linewidth=2.2, marker="o", ms=4,
            label=f"probe on {role_for_probe.upper()} activations")
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.8,
                label="chance (50%)")
    ax.set_xlabel("Hidden-state layer index")
    ax.set_ylabel("Test accuracy (unsafe vs safe)")
    ax.set_ylim(0.45, 1.02)
    ax.set_title(f"Figure I — Linear probe accuracy by layer   [{pair_name}]",
                 loc="left", weight="bold")
    ax.text(0.02, 0.98,
            "A linear probe trained on activations at each layer to classify\n"
            "unsafe vs safe. If layer 0 is already high, harmfulness is\n"
            "encoded lexically in the embeddings. A later peak indicates\n"
            "progressively built-up harmfulness representation.",
            transform=ax.transAxes, fontsize=8, color="#555",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#DDD", alpha=0.85))
    # Alignment readout text
    if alignment_result.get("cos_sim") is not None:
        cs = alignment_result["cos_sim"]
        layer = alignment_result["layer"]
        ax.text(0.98, 0.02,
                f"Direction alignment (L{layer}):\n"
                f"  cos(probe_dir, deflection_dir) = {cs:+.3f}",
                transform=ax.transAxes, fontsize=9, color="#222",
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#BBB", alpha=0.9))
    ax.legend(loc="lower right", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figureI_linear_probe.{ext}")
    plt.close(fig)
    ok("wrote figureI_linear_probe.{png,pdf}")


# -------------------------- TTV null plot --------------------------

def plot_ttv_null(pair_dir: Path, null_result: dict, pair_name: str) -> None:
    """Side-by-side: observed vs random TTV distribution per layer."""
    import matplotlib.pyplot as plt
    _mpl_setup()
    plot_dir = pair_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    L = len(null_result["observed_per_layer_mean"])
    obs_mean = np.array(null_result["observed_per_layer_mean"])
    rand_mean = np.array(null_result["random_per_layer_mean"])
    obs_std = np.array(null_result["observed_per_layer_std"])
    rand_std = np.array(null_result["random_per_layer_std"])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    layers = np.arange(L)
    ax.fill_between(layers, obs_mean - obs_std, obs_mean + obs_std,
                     color=COL_UNSAFE, alpha=0.15, label="observed ±1σ")
    ax.plot(layers, obs_mean, color=COL_UNSAFE, linewidth=2, marker="o", ms=4,
            label="observed mean")
    ax.fill_between(layers, rand_mean - rand_std, rand_mean + rand_std,
                     color="#888", alpha=0.15, label="random-trajectory ±1σ")
    ax.plot(layers, rand_mean, color="#555", linewidth=2, linestyle="--",
            marker="s", ms=4, label="random-trajectory mean")
    ax.set_xlabel("Layer transition ℓ")
    ax.set_ylabel("TTV (Trajectory Twist Volume)")
    ax.set_title(f"TTV null distribution check   [{pair_name}]",
                 loc="left", weight="bold")
    ax.text(0.02, 0.98,
            "If observed TTV tracks the random baseline closely, TTV is "
            "noise-dominated at this granularity. Wide separation would "
            "indicate TTV captures real trajectory structure.",
            transform=ax.transAxes, fontsize=9, color="#555",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#DDD", alpha=0.85))
    ax.legend(loc="best", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_TTV_null.{ext}")
    plt.close(fig)
    ok("wrote figure_TTV_null.{png,pdf}")
# ===========================================================================
#  SECTION 8  —  3D INTERACTIVE HTML PLOTS (plotly)
#
#  Per the user's explicit instruction, we do NOT speed up D2 at the cost of
#  quality. Each individual trajectory is its own trace. Slower but faithful.
# ===========================================================================

def _pca_fit(X: np.ndarray, n_components: int = 3):
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:n_components].T, mu, (S ** 2 / (X.shape[0] - 1))


def _pca_project(X: np.ndarray, components, mean) -> np.ndarray:
    return (X - mean) @ components


def plot_3d_figures(pair_dir: Path, h_A: np.ndarray, h_B: np.ndarray,
                     prompts_df, pair_name: str,
                     role_A: str, role_B: str,
                     sample_n: int = 20) -> None:
    """Four plotly HTML plots: D1 mean trajectories, D2 individual,
    D3 deflection field, D4 per-axiom."""
    import plotly.graph_objects as go

    out_dir = pair_dir / "plots_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe = (prompts_df["safety_label"] == "safe").to_numpy()
    unsafe = ~safe
    axioms_arr = prompts_df["axiom"].to_numpy()

    N, Lp1, D = h_A.shape
    layers = np.arange(Lp1)

    # Fit PCA on combined points
    stacked = np.concatenate([h_A.reshape(-1, D), h_B.reshape(-1, D)], axis=0)
    if len(stacked) > 50_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(stacked), size=50_000, replace=False)
        stacked_sub = stacked[idx]
    else:
        stacked_sub = stacked
    info(f"fitting PCA on {len(stacked_sub)} points ...")
    components, pca_mean, var_explained = _pca_fit(stacked_sub, 3)
    total_var = var_explained.sum()
    ev = [v / total_var * 100 for v in var_explained[:3]] if total_var > 0 else [0, 0, 0]

    def proj(X):
        return _pca_project(X.reshape(-1, D), components, pca_mean).reshape(*X.shape[:-1], 3)

    p_A = proj(h_A)
    p_B = proj(h_B)

    A_up, B_up = display_role(role_A), display_role(role_B)

    # D1 mean trajectories
    fig = go.Figure()
    for name, traj, color, dash in [
        (f"{A_up} · safe",   p_A[safe].mean(axis=0),   COL_MODEL_A, "solid"),
        (f"{A_up} · unsafe", p_A[unsafe].mean(axis=0), COL_MODEL_A, "dash"),
        (f"{B_up} · safe",   p_B[safe].mean(axis=0),   COL_MODEL_B, "solid"),
        (f"{B_up} · unsafe", p_B[unsafe].mean(axis=0), COL_MODEL_B, "dash"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode="lines+markers",
            line=dict(color=color, width=4, dash=dash),
            marker=dict(size=3, color=layers, colorscale="Viridis", showscale=False),
            name=name,
        ))
    fig.update_layout(
        title=(f"D1 — Mean trajectories ({A_up} vs {B_up} × safe vs unsafe)"
                f"<br><sub>PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%  PC3={ev[2]:.1f}%  "
                f"· gradient = layer 0→{Lp1-1}</sub>"),
        scene=dict(
            xaxis_title=f"PC1  ({ev[0]:.1f}%)",
            yaxis_title=f"PC2  ({ev[1]:.1f}%)",
            zaxis_title=f"PC3  ({ev[2]:.1f}%)",
            bgcolor="#FAFAFA",
        ),
        width=1100, height=750,
        legend=dict(x=0.02, y=0.98),
    )
    fig.write_html(out_dir / "D1_mean_trajectories.html", include_plotlyjs="cdn")
    ok("wrote D1_mean_trajectories.html")

    # D2 individual trajectories (intentionally per-trajectory for fidelity)
    fig = go.Figure()
    rng = np.random.default_rng(0)
    safe_idx = rng.choice(np.where(safe)[0],
                           size=min(sample_n, safe.sum()), replace=False)
    unsafe_idx = rng.choice(np.where(unsafe)[0],
                             size=min(sample_n, unsafe.sum()), replace=False)
    info(f"rendering D2 with {len(safe_idx)} + {len(unsafe_idx)} individual trajectories")
    progress = Progress("D2-render", total=2 * (len(safe_idx) + len(unsafe_idx)),
                         min_interval_s=2.0)
    done = 0
    for group_name, idx_list, color_A, color_B in [
        ("safe", safe_idx, "#90C2DC", "#E8A49A"),
        ("unsafe", unsafe_idx, "#1A5F8E", "#8F1E0E"),
    ]:
        for j, i in enumerate(idx_list):
            showleg = (j == 0)
            fig.add_trace(go.Scatter3d(
                x=p_A[i, :, 0], y=p_A[i, :, 1], z=p_A[i, :, 2],
                mode="lines", line=dict(color=color_A, width=1.5), opacity=0.5,
                name=f"{A_up} · {group_name}" if showleg else None,
                showlegend=showleg, hoverinfo="skip",
            ))
            done += 1; progress.update(done)
            fig.add_trace(go.Scatter3d(
                x=p_B[i, :, 0], y=p_B[i, :, 1], z=p_B[i, :, 2],
                mode="lines", line=dict(color=color_B, width=1.5), opacity=0.5,
                name=f"{B_up} · {group_name}" if showleg else None,
                showlegend=showleg, hoverinfo="skip",
            ))
            done += 1; progress.update(done)
    progress.done()
    fig.update_layout(
        title=(f"D2 — Individual trajectories (sample of {sample_n}+{sample_n})"
                f"<br><sub>Light=safe, dark=unsafe · "
                f"Greenish={A_up}, reddish={B_up}</sub>"),
        scene=dict(
            xaxis_title=f"PC1  ({ev[0]:.1f}%)",
            yaxis_title=f"PC2  ({ev[1]:.1f}%)",
            zaxis_title=f"PC3  ({ev[2]:.1f}%)",
            bgcolor="#FAFAFA",
        ),
        width=1100, height=750,
        legend=dict(x=0.02, y=0.98),
    )
    fig.write_html(out_dir / "D2_individual_trajectories.html", include_plotlyjs="cdn")
    ok("wrote D2_individual_trajectories.html")

    # D3 deflection field
    fig = go.Figure()
    for group_name, mask, color in [("safe", safe, COL_SAFE),
                                     ("unsafe", unsafe, COL_UNSAFE)]:
        A_means = p_A[mask].mean(axis=0)
        B_means = p_B[mask].mean(axis=0)
        fig.add_trace(go.Scatter3d(
            x=A_means[:, 0], y=A_means[:, 1], z=A_means[:, 2],
            mode="lines+markers",
            line=dict(color=color, width=3, dash="solid"),
            marker=dict(size=3, color=color),
            name=f"{A_up} mean · {group_name}", opacity=0.7,
        ))
        for l in range(Lp1):
            fig.add_trace(go.Scatter3d(
                x=[A_means[l, 0], B_means[l, 0]],
                y=[A_means[l, 1], B_means[l, 1]],
                z=[A_means[l, 2], B_means[l, 2]],
                mode="lines", line=dict(color=color, width=4), opacity=0.5,
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter3d(
            x=B_means[:, 0], y=B_means[:, 1], z=B_means[:, 2],
            mode="markers",
            marker=dict(size=5, color=color, symbol="diamond"),
            name=f"{B_up} mean · {group_name}", opacity=0.9,
        ))
    fig.update_layout(
        title=(f"D3 — Deflection-vector field: {A_up} mean → {B_up} mean, per layer"
                "<br><sub>Line + circles = model-A trajectory. Diamonds = model-B mean. "
                "Thin segments = layer-wise deflection.</sub>"),
        scene=dict(
            xaxis_title=f"PC1  ({ev[0]:.1f}%)",
            yaxis_title=f"PC2  ({ev[1]:.1f}%)",
            zaxis_title=f"PC3  ({ev[2]:.1f}%)",
            bgcolor="#FAFAFA",
        ),
        width=1100, height=750,
        legend=dict(x=0.02, y=0.98),
    )
    fig.write_html(out_dir / "D3_deflection_field.html", include_plotlyjs="cdn")
    ok("wrote D3_deflection_field.html")

    # D4 per-axiom
    fig = go.Figure()
    axiom_palette = {
        "Wisdom & Knowledge":    "#AD1457",
        "Well-being & Peace":    "#1976D2",
        "Justice & Rights":      "#388E3C",
        "Duty & Accountability": "#E65100",
        "Civility & Tolerance":  "#6A1B9A",
        "Empathy & Helpfulness": "#00796B",
    }
    for ax_name, color in axiom_palette.items():
        mask = (axioms_arr == ax_name) & unsafe
        if mask.sum() < 5: continue
        A_mean = p_A[mask].mean(axis=0)
        B_mean = p_B[mask].mean(axis=0)
        fig.add_trace(go.Scatter3d(
            x=A_mean[:, 0], y=A_mean[:, 1], z=A_mean[:, 2],
            mode="lines", line=dict(color=color, width=2, dash="dash"),
            name=f"{A_up} · {ax_name}", opacity=0.7,
        ))
        fig.add_trace(go.Scatter3d(
            x=B_mean[:, 0], y=B_mean[:, 1], z=B_mean[:, 2],
            mode="lines+markers",
            line=dict(color=color, width=3, dash="solid"),
            marker=dict(size=3, color=color),
            name=f"{B_up} · {ax_name}", opacity=0.9,
        ))
    fig.add_trace(go.Scatter3d(
        x=p_A[safe].mean(0)[:, 0], y=p_A[safe].mean(0)[:, 1],
        z=p_A[safe].mean(0)[:, 2],
        mode="lines", line=dict(color="#666", width=4, dash="dot"),
        name=f"{A_up} · safe (reference)", opacity=0.8,
    ))
    fig.update_layout(
        title=(f"D4 — Per-axiom mean trajectories ({B_up} vs {A_up}, unsafe)"
                "<br><sub>Dashed = A per axiom. Solid = B per axiom. "
                "Grey dotted = A safe (reference).</sub>"),
        scene=dict(
            xaxis_title=f"PC1  ({ev[0]:.1f}%)",
            yaxis_title=f"PC2  ({ev[1]:.1f}%)",
            zaxis_title=f"PC3  ({ev[2]:.1f}%)",
            bgcolor="#FAFAFA",
        ),
        width=1100, height=750,
        legend=dict(x=0.02, y=0.98),
    )
    fig.write_html(out_dir / "D4_per_axiom_trajectories.html", include_plotlyjs="cdn")
    ok("wrote D4_per_axiom_trajectories.html")
# ===========================================================================
#  SECTION 9  —  ORCHESTRATION
# ===========================================================================

def install_signal_handler(logger: RunLogger):
    def _handler(signum, frame):
        warn(f"signal {signum} — exiting cleanly (resume-safe)")
        try:
            logger.event("signal", signum=int(signum))
            logger.close()
        except Exception:
            pass
        sys.exit(130)
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except Exception:
            pass


def _print_headline(all_summaries: dict) -> None:
    """One-liner readout per experiment."""
    print("\nHeadline numbers (per-layer metrics — peak by |Cohen's d|):")
    for exp_name, summary in all_summaries.items():
        lA, lB = summary.get("label_A", "?"), summary.get("label_B", "?")
        print(f"\n  [{exp_name}]  {lA} → {lB}")
        m = summary.get("metrics", {})
        for key in ["theta_deg", "sr", "ndm", "ctd", "ttv"]:
            if key not in m: continue
            v = m[key]
            display = {"theta_deg": "θ  ", "sr": "SR ", "ndm": "NDM",
                       "ctd": "CTD", "ttv": "TTV"}[key]
            print(f"    {display}: peak L{v['peak_layer']:>2d}  "
                  f"d={v['cohens_d_at_peak']:+.2f}")
        if "tea_deg" in m:
            v = m["tea_deg"]
            print(f"    TEA: d={v['cohens_d']:+.2f}")


def run_experiment(exp_name: str, cfg: PipelineConfig,
                    pair_dir: Path, logger: RunLogger,
                    sw: Stopwatch,
                    run_3d: bool = False) -> tuple[dict, object, dict, dict]:
    """Run one experiment end-to-end.

    Returns (summary, prompts_df, comparative_metrics, activations_dict).
    """
    spec = EXPERIMENTS[exp_name]
    subbanner(f"EXPERIMENT: {exp_name}  ({spec.description})")

    # ---- prompts for this experiment ----
    prompts_csv = pair_dir / f"prompts__{spec.prompt_set}.csv"
    sw.start(f"{exp_name}:prompts")
    prompts_df = prepare_prompts(cfg, spec.prompt_set, prompts_csv)
    logger.event(f"prompts_ready_{exp_name}",
                  n=len(prompts_df), prompt_set=spec.prompt_set)
    sw.stop()

    # ---- extract activations for both model roles ----
    act_root = pair_dir / f"activations__{exp_name}"
    act_root.mkdir(parents=True, exist_ok=True)
    for role in [spec.model_a_role, spec.model_b_role]:
        sw.start(f"{exp_name}:extract_{role}")
        extract_activations_for_role(role, prompts_df, act_root, cfg, logger)
        sw.stop()

    # ---- load trajectories ----
    sw.start(f"{exp_name}:load")
    hs = load_trajectories_for_roles(act_root, prompts_df,
                                       [spec.model_a_role, spec.model_b_role])
    h_A = hs[spec.model_a_role]
    h_B = hs[spec.model_b_role]
    sw.stop()

    # ---- compute metrics ----
    sw.start(f"{exp_name}:metrics")
    comparative = compute_comparative_metrics(h_A, h_B)
    intrinsic_A = compute_intrinsic_metrics(h_A)
    intrinsic_B = compute_intrinsic_metrics(h_B)
    sw.stop()

    # ---- save metrics ----
    exp_dir = pair_dir / f"experiment__{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    np.savez_compressed(
        exp_dir / "metrics.npz",
        **{k: comparative[k] for k in
           ["theta_deg", "sr", "ndm", "ctd", "ttv", "pl_a", "pl_b",
            "tea_deg", "ttv_A", "ttv_B"]},
        speed_A=intrinsic_A["speed"], turn_A=intrinsic_A["turn_deg"],
        disp_A=intrinsic_A["disp"],
        speed_B=intrinsic_B["speed"], turn_B=intrinsic_B["turn_deg"],
        disp_B=intrinsic_B["disp"],
        prompt_id=np.array(prompts_df["prompt_id"].tolist()),
        safety_label=np.array(prompts_df["safety_label"].tolist()),
        axiom=np.array(prompts_df["axiom"].tolist()),
    )

    # ---- plots ----
    sw.start(f"{exp_name}:plot_A")
    summary = plot_figure_A(exp_dir, comparative, prompts_df,
                              label_A=display_role(spec.model_a_role),
                              label_B=display_role(spec.model_b_role),
                              pair_name=f"{cfg.pair_name}:{exp_name}")
    sw.stop()

    sw.start(f"{exp_name}:plot_B")
    plot_figure_B(exp_dir, comparative, prompts_df,
                    label_A=display_role(spec.model_a_role),
                    label_B=display_role(spec.model_b_role),
                    pair_name=f"{cfg.pair_name}:{exp_name}")
    sw.stop()

    sw.start(f"{exp_name}:plot_C")
    plot_figure_C(exp_dir, comparative, prompts_df,
                    label_A=display_role(spec.model_a_role),
                    label_B=display_role(spec.model_b_role),
                    pair_name=f"{cfg.pair_name}:{exp_name}")
    sw.stop()

    if spec.prompt_set != "jailbreakbench":
        # Figure D requires multiple axioms; JBB has only 1 category for unsafe
        sw.start(f"{exp_name}:plot_D")
        plot_figure_D(exp_dir, comparative, prompts_df,
                        pair_name=f"{cfg.pair_name}:{exp_name}")
        sw.stop()

    sw.start(f"{exp_name}:plots_EF")
    plot_figures_EF(exp_dir, intrinsic_A, intrinsic_B, prompts_df,
                      role_A=spec.model_a_role, role_B=spec.model_b_role,
                      pair_name=f"{cfg.pair_name}:{exp_name}")
    sw.stop()

    # 3D only for the primary experiment (per user: save time, same qualitative story)
    if run_3d:
        sw.start(f"{exp_name}:plots_3d")
        plot_3d_figures(exp_dir, h_A, h_B, prompts_df,
                          pair_name=f"{cfg.pair_name}:{exp_name}",
                          role_A=spec.model_a_role,
                          role_B=spec.model_b_role)
        sw.stop()

    # Save summary
    (exp_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    return summary, prompts_df, comparative, hs


def run_pipeline(cfg: PipelineConfig) -> None:
    out_root = Path(cfg.output_dir)
    pair_dir = out_root / cfg.pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(pair_dir)
    install_signal_handler(logger)
    sw = Stopwatch()

    banner(f"REPRESENTATION TRAJECTORY GEOMETRY OF ALIGNMENT  v{SCRIPT_VERSION}")
    info(f"pair: {cfg.pair_name}")
    info(f"output: {pair_dir}")
    info(f"experiments: {cfg.experiments}")

    # GPU profile
    sw.start("gpu_profile")
    cfg.gpu = profile_gpu()
    logger.event("gpu_profile", **asdict(cfg.gpu))
    sw.stop()

    (pair_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str))

    # Collect roles needed across all experiments
    roles_needed = set()
    for exp_name in cfg.experiments:
        spec = EXPERIMENTS[exp_name]
        roles_needed.add(spec.model_a_role)
        roles_needed.add(spec.model_b_role)
    info(f"model roles required: {sorted(roles_needed)}")

    # Preflight: load each unique model once to verify
    # Use the primary experiment's prompt set for preflight dummy prompt
    primary_prompt_set = EXPERIMENTS[cfg.experiments[0]].prompt_set
    primary_prompts_csv = pair_dir / f"prompts__{primary_prompt_set}.csv"
    sw.start("prepare_primary_prompts_for_preflight")
    primary_prompts_df = prepare_prompts(cfg, primary_prompt_set,
                                           primary_prompts_csv)
    sw.stop()
    sw.start("preflight")
    preflight(cfg, primary_prompts_df, pair_dir, sorted(roles_needed))
    sw.stop()

    # Run each experiment
    banner("RUNNING EXPERIMENTS")
    all_summaries: dict = {}
    all_prompts: dict = {}
    all_metrics: dict = {}
    all_activations: dict = {}
    run_3d_for_primary = "primary" in cfg.experiments

    for i, exp_name in enumerate(cfg.experiments):
        info(f"experiment {i+1}/{len(cfg.experiments)}: {exp_name}")
        run_3d = (exp_name == "primary" and run_3d_for_primary)
        summary, df, comparative, acts = run_experiment(
            exp_name, cfg, pair_dir, logger, sw, run_3d=run_3d,
        )
        all_summaries[exp_name] = summary
        all_prompts[exp_name] = df
        all_metrics[exp_name] = comparative
        all_activations[exp_name] = acts

    # -------- cross-experiment comparison plots --------
    banner("CROSS-EXPERIMENT COMPARISONS")

    # Figure G: safe-prompt variants (all IT→DPO experiments with different prompts)
    variant_map = {
        "primary":     "litmus_original",
        "safe_A":      "litmus_infoseeking",
        "safe_B":      "jailbreakbench",
    }
    variant_metrics = {}
    variant_prompts = {}
    for exp, pset in variant_map.items():
        if exp in all_metrics:
            variant_metrics[pset] = all_metrics[exp]
            variant_prompts[pset] = all_prompts[exp]
    if len(variant_metrics) >= 2:
        sw.start("figure_G")
        plot_figure_G(pair_dir, variant_metrics, variant_prompts, cfg.pair_name)
        sw.stop()

    # Figure H: experiment comparison
    if len(cfg.experiments) >= 2:
        sw.start("figure_H")
        plot_figure_H(pair_dir, all_metrics, all_prompts, cfg.pair_name)
        sw.stop()

    # -------- analyses on primary experiment --------
    if "primary" in cfg.experiments:
        banner("ANALYSES (on primary experiment)")

        # TTV null distribution
        sw.start("ttv_null")
        primary_ttv = all_metrics["primary"]["ttv"]
        null_result = ttv_null_distribution(primary_ttv, D=4096,
                                              n_samples=6000, seed=0)
        plot_ttv_null(pair_dir, null_result, cfg.pair_name)
        # Save numeric summary
        (pair_dir / "ttv_null_summary.json").write_text(json.dumps({
            "observed_per_layer_mean": null_result["observed_per_layer_mean"],
            "random_per_layer_mean": null_result["random_per_layer_mean"],
            "observed_per_layer_std": null_result["observed_per_layer_std"],
            "random_per_layer_std": null_result["random_per_layer_std"],
            "interpretation": null_result["interpretation"],
        }, indent=2))
        sw.stop()

        # Linear probe at every layer on the DPO-side activations
        sw.start("linear_probe")
        primary_acts = all_activations["primary"]
        primary_df = all_prompts["primary"]
        spec = EXPERIMENTS["primary"]
        h_probe = primary_acts[spec.model_b_role]   # DPO activations
        labels_arr = (primary_df["safety_label"] == "unsafe").to_numpy().astype(int)
        probe_result = linear_probe_per_layer(h_probe, labels_arr)

        # Direction alignment at the peak θ layer (from primary summary)
        theta_peak = all_summaries["primary"]["metrics"]["theta_deg"]["peak_layer"]
        alignment_result = probe_direction_alignment(
            probe_result["weights"],
            primary_acts[spec.model_a_role],
            primary_acts[spec.model_b_role],
            layer=theta_peak + 1,   # probe on hidden state after the step
            unsafe_mask=(labels_arr == 1),
        )

        plot_figure_I(pair_dir, probe_result, alignment_result,
                       pair_name=cfg.pair_name,
                       role_for_probe=spec.model_b_role)

        (pair_dir / "linear_probe_summary.json").write_text(json.dumps({
            "accuracy_per_layer": probe_result["accuracy"],
            "peak_theta_layer": theta_peak,
            "probe_direction_vs_deflection_direction_cos_sim":
                alignment_result.get("cos_sim"),
            "alignment_interpretation": alignment_result.get("interpretation"),
        }, indent=2))
        sw.stop()

    # -------- finalize --------
    banner("DONE")
    overall = {
        "pair_name": cfg.pair_name,
        "experiments": {k: v for k, v in all_summaries.items()},
        "timing_seconds": dict(sw.timings),
        "script_version": SCRIPT_VERSION,
    }
    (pair_dir / "summary.json").write_text(json.dumps(overall, indent=2, default=str))

    print(sw.summary())
    logger.event("pipeline_done", **dict(sw.timings))
    logger.close()

    print()
    print(f"Inspect outputs at: {pair_dir}")
    for exp_name in cfg.experiments:
        exp_dir = pair_dir / f"experiment__{exp_name}"
        print(f"  {exp_dir}/")
        print(f"    plots/figureA..F.{{png,pdf}}")
        if exp_name == "primary":
            print(f"    plots_3d/D1..D4.html")
    if any(e in cfg.experiments for e in ["primary", "safe_A", "safe_B"]):
        print(f"  {pair_dir}/plots/figureG_safe_prompt_variants.{{png,pdf}}")
    if len(cfg.experiments) >= 2:
        print(f"  {pair_dir}/plots/figureH_experiment_comparison.{{png,pdf}}")
    if "primary" in cfg.experiments:
        print(f"  {pair_dir}/plots/figureI_linear_probe.{{png,pdf}}")
        print(f"  {pair_dir}/plots/figure_TTV_null.{{png,pdf}}")

    _print_headline(all_summaries)


# ===========================================================================
#  SECTION 10  —  CLI
# ===========================================================================

def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(
        description="Representation Trajectory Geometry of Alignment (v3.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B",
                    help="Base model HF repo ID (for control_base experiment)")
    p.add_argument("--base-subfolder", default=None)
    p.add_argument("--sft-model", required=True)
    p.add_argument("--sft-subfolder", default=None)
    p.add_argument("--dpo-model", required=True)
    p.add_argument("--dpo-subfolder", default=None)
    p.add_argument("--tulu3-sft-model",
                    default="allenai/Llama-3.1-Tulu-3-8B-SFT",
                    help="(not executed by default)")
    p.add_argument("--tulu3-dpo-model",
                    default="allenai/Llama-3.1-Tulu-3-8B-DPO")
    p.add_argument("--olmo3-sft-model",
                    default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--olmo3-dpo-model",
                    default="allenai/Olmo-3-7B-Instruct-DPO")
    p.add_argument("--pair-name", required=True)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--n-safe", type=int, default=3000)
    p.add_argument("--n-unsafe", type=int, default=3000)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=PROMPT_SEED)
    p.add_argument("--experiments", default="primary,control_base,safe_A,safe_B",
                    help="Comma-separated experiment names. Valid: "
                         + ",".join(EXPERIMENTS.keys()))
    args = p.parse_args()
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    for e in experiments:
        if e not in EXPERIMENTS:
            raise ValueError(f"unknown experiment: {e}. "
                             f"Valid: {list(EXPERIMENTS.keys())}")

    return PipelineConfig(
        base_model=args.base_model, base_subfolder=args.base_subfolder,
        sft_model=args.sft_model, sft_subfolder=args.sft_subfolder,
        dpo_model=args.dpo_model, dpo_subfolder=args.dpo_subfolder,
        tulu3_sft_model=args.tulu3_sft_model,
        tulu3_dpo_model=args.tulu3_dpo_model,
        olmo3_sft_model=args.olmo3_sft_model,
        olmo3_dpo_model=args.olmo3_dpo_model,
        pair_name=args.pair_name, output_dir=args.output_dir,
        n_safe=args.n_safe, n_unsafe=args.n_unsafe,
        experiments=experiments,
        max_tokens=args.max_tokens, seed=args.seed,
    )


def main() -> int:
    cfg = parse_args()
    try:
        run_pipeline(cfg)
        return 0
    except KeyboardInterrupt:
        warn("interrupted — state is safe for resume")
        return 130
    except Exception as e:
        fail(f"pipeline failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
