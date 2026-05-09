# Refusal in the Trajectory: Step-Direction Geometry Exposes and Amplifies DPO's Latent Refusal Alignment via Representation Steering Vectors

> **ICML 2026 · Mechanistic Interpretability Workshop** · Under Review · Anonymous Submission

---

## Overview

Direct Preference Optimization (DPO) improves the safety behavior of language models, but the representational changes that support this behavior remain poorly understood. We introduce a **trajectory-geometry framework** for analyzing alignment — measuring the per-layer step-direction angle (θ) between an instruction-tuned (IT) model and its DPO-aligned counterpart as they process safe and unsafe prompts.

Applied to Llama-3-8B under a controlled training recipe (OpenHermes 2.5 for IT, HH-RLHF for DPO), this analysis reveals a **safety-specific mid-layer directional signature** centered on L12–17. We then turn this diagnostic into an intervention: five-layer steering at L12–16 (α = 1.5) achieves **76% full refusal** on unsafe prompts — substantially exceeding DPO's own 43% — without any weight updates.

---

## Key Results

| Condition | Full Refusal ↑ | Compliance ↓ | Safe Helpfulness ↑ |
|---|---|---|---|
| IT baseline | 39% | 18% | 54% |
| DPO baseline | 43% | 13% | 56% |
| Steered L13–15 α=1.5 | 54% | 9% | 48% |
| **Steered L12–16 α=1.5** | **76%** | **4%** | 34% |
| Steered L11–17 α=1.5 | 80% | 4% | 20% |

*LLM-judge evaluation (Qwen-2.5-7B-Instruct), n=100 unsafe + 50 safe held-out prompts.*

**Headline numbers:**
- Cohen's d = **1.66** at L14 (peak layer)
- Steering surpasses DPO by **+33 percentage points** on full refusal
- Negative steering (α = −1) collapses refusal to **1%**, confirming causal directionality
- DPO deflection direction is **orthogonal** to the harmfulness-classification axis (cosine = +0.052, r² < 0.003)

---

## The Three-Stage Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   1. DIAGNOSE   │    │   2. EXTRACT    │    │    3. STEER     │
│                 │    │                 │    │                 │
│  Measure θ per  │───▶│  Compute D-in-D │───▶│  Inject vectors │
│  layer between  │    │  safety vectors │    │  via forward    │
│  IT and DPO.    │    │  within window  │    │  hooks at infer │
│  Peak d → W     │    │  W (L12–16)     │    │  No weight upd. │
│  (L12–16)       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Core equations:**

$$\theta^\ell(p) = \arccos\!\left(\frac{\Delta h^\ell_A(p) \cdot \Delta h^\ell_B(p)}{\|\Delta h^\ell_A(p)\|\,\|\Delta h^\ell_B(p)\|}\right)$$

$$v^\text{safety}_\ell = \mathbb{E}_\text{unsafe}\!\left[h^\ell_B - h^\ell_A\right] - \mathbb{E}_\text{safe}\!\left[h^\ell_B - h^\ell_A\right]$$

$$\tilde{h}^\ell = h^\ell_A + \alpha \cdot v^\text{safety}_\ell \quad \forall\,\ell \in W$$

---

## Figures

| | |
|---|---|
| ![Fig 1 — θ diagnostic](figures/fig1_theta_diagnostic_v2.pdf) | ![Fig 2 — Steering results](figures/fig2_steering_results_v2.pdf) |
| **Fig 1.** Step-direction angle θ per layer. Gap between safe and unsafe prompts peaks at L14 (Cohen's d = 1.66). Shaded band = steering window L12–16. | **Fig 2.** Multi-layer steering sweep: window × strength × behavioral validation. Only the θ-guided mid-layer band produces selective safety improvement. |
| ![Fig 3 — Selectivity landscape](figures/fig3_selectivity_landscape.pdf) | ![Fig 4 — Headline comparison](figures/fig4_headline_comparison.pdf) |
| **Fig 3.** Selectivity landscape (unsafe refusal vs. safe over-refusal). Only θ-guided mid-layer windows reach the upper-left ideal zone. | **Fig 4.** LLM-judge verified results. L12–16 achieves 76% full refusal, nearly doubling DPO's 43%. L13–15 preserves 48% helpfulness. |

![Fig 5 — Pipeline overview](figures/fig5_pipeline_overview.pdf)

**Fig 5.** The three-stage pipeline. Left: θ Cohen's d localizes the intervention window. Center: safety vector norms. Right: unsafe vs. safe refusal — mid-layer steering uniquely achieves high unsafe refusal with low safe over-refusal.

> **Note:** GitHub does not render PDF previews inline. Clone the repo and open the PDFs directly, or see the paper for full-resolution figures.

---

## Mechanistic Evidence

**Linear probe** — Probe accuracy on DPO activations rises from 50% (chance) at layer 0 to 99%+ by layer 6, confirming harmfulness is linearly encoded. Critically, the DPO deflection direction is **near-orthogonal** to the harmfulness axis (cosine = +0.052 at L15). DPO does not push representations along the harm-classification axis; it encodes a *response-policy* direction.

**Causal ablation** — Replacing DPO's L14 hidden state with IT's L14 state breaks 27.5% of refusals (41/149). Of those, 19 produce clearly operational harm guidance (network exploitation, malware authoring, targeted harassment). L14 is causally necessary, not merely correlated.

**Attention heads** — At L14, heads 29 and 31 show ≈5× the mean per-head symmetric KL divergence between IT and DPO on unsafe prompts vs. safe prompts.

**Logit lens** — Safety-conditional disagreement between IT and DPO token distributions begins specifically at L14 (ΔKLL = +0.007) and grows sharply at L16 (+0.063). The geometric deflection at L14 is *upstream* of the token-level output divergence at L16+.

**Axiom uniformity** — Mean θ at L14 falls within a 1° band across all six harm axioms (12.93°–13.93°, vs. safe baseline 10.31°). DPO learns a general concern signal, not six category-specific responses.

---

## The Latent Safety Hypothesis

> *DPO training imprints a safety-relevant direction in representation space that is stronger than what DPO behaviorally deploys — perhaps because DPO training optimizes for a soft preference signal rather than hard refusal behavior. When we amplify it (α > 1) and inject it uniformly across the band, we push representations further along this direction than DPO itself does during its forward pass — effectively turning up the volume on a signal DPO only whispers.*

Three non-exclusive explanations for why DPO underexpresses its own safety signal:

1. **Soft preference optimisation** — DPO is trained on pairwise preferences, not hard binary refusal labels. The geometry changes more than the behavior because the loss does not directly reward behavior.
2. **Late-layer behavioural override** — Layers L18–32 may partially undo the mid-layer safety signal, converting it back to compliant outputs. This explains why late-layer steering fails.
3. **Residual pretraining attractors** — Llama-3's pretraining creates strong attractors toward completion of harmful instructions. Steering bypasses this by repeatedly reinforcing the direction at each mid-layer step.

---

## Contributions

1. A trajectory-geometry framework centered on step-direction angles (θ, NDM, CTD, TEA)
2. Controlled experiments establishing the mid-layer θ signature is alignment-specific, axiom-uniform, and lexically robust under JailbreakBench content-matched controls
3. Mechanistic evidence: linear probe, causal ablation at L14, attention-head divergence, logit-lens analyses
4. A θ-guided steering pipeline with 20-condition window × strength ablation that surpasses DPO's own refusal rate
5. LLM-judge validation revealing DPO's representational knowledge exceeds its behavioral expression

---

## Installation

```bash
git clone https://github.com/<your-username>/rtga
cd rtga
pip install -r requirements.txt
```

**Dependencies:** `torch`, `transformers`, `numpy`, `pandas`, `datasets`, `scikit-learn`, `matplotlib`, `plotly`

---

## Usage

### 1. Run the θ diagnostic

```bash
python representation_trajectory_geometry_of_alignment_V2.py \
    --base-model   meta-llama/Meta-Llama-3-8B \
    --sft-model    <your-hf-username>/your-model --sft-subfolder SFT_merged \
    --dpo-model    <your-hf-username>/your-model --dpo-subfolder DPO_merged \
    --pair-name    your_pair_name \
    --n-safe 3000 --n-unsafe 3000 \
    --experiments  primary,control_base,safe_A,safe_B
```

### 2. Extract steering vectors

```bash
python rtga_steering.py extract \
    --it-model  <your-hf-username>/your-model --it-subfolder SFT_merged \
    --dpo-model <your-hf-username>/your-model --dpo-subfolder DPO_merged \
    --pair-name your_pair_name --output-dir outputs
```

### 3. Run the steering sweep

```bash
python rtga_steering.py steer \
    --it-model  <your-hf-username>/your-model --it-subfolder SFT_merged \
    --pair-name your_pair_name --output-dir outputs
```

### 4. LLM-judge evaluation

```bash
python rtga_steering.py judge \
    --pair-name your_pair_name --output-dir outputs \
    --judge-model Qwen/Qwen2.5-7B-Instruct
```

### 5. Mechanistic interpretability analyses

```bash
python rtga_perprompt_and_mechinterp.py mechinterp \
    --pair-dir outputs/your_pair_name \
    --sft-model  <your-hf-username>/your-model --sft-subfolder SFT_merged \
    --dpo-model  <your-hf-username>/your-model --dpo-subfolder DPO_merged \
    --n-causal 500
```

---

## Repository Structure

```
rtga/
├── figures/
│   ├── fig1_theta_diagnostic_v2.pdf
│   ├── fig2_steering_results_v2.pdf
│   ├── fig3_selectivity_landscape.pdf
│   ├── fig4_headline_comparison.pdf
│   └── fig5_pipeline_overview.pdf
├── representation_trajectory_geometry_of_alignment_V2.py   # Stage 1: θ diagnostic
├── rtga_steering.py                                         # Stage 2–3: extract + steer + judge
├── rtga_perprompt_and_mechinterp.py                         # Mechanistic analyses
└── README.md
```

---

## Experimental Setup

| Component | Detail |
|---|---|
| Base model | Meta-Llama-3-8B (32 layers, D=4096) |
| IT stage | LoRA fine-tuning on OpenHermes 2.5 (100K samples, <1% safety content) |
| DPO stage | LoRA fine-tuning on Anthropic HH-RLHF harmless-base |
| Extraction set | 3,000 unsafe + 3,000 safe prompts from Litmus (stratified across 6 axioms) |
| Evaluation set | 500 unsafe + 200 safe held-out Litmus prompts |
| Lexical control | JailbreakBench: 100 harmful + 100 content-matched benign prompts |
| Judge | Qwen-2.5-7B-Instruct |

---

## Citation

```bibtex
@inproceedings{anonymous2026rtga,
  title     = {Refusal in the Trajectory: Step-Direction Geometry Exposes and
               Amplifies DPO's Latent Refusal Alignment via Representation
               Steering Vectors},
  author    = {Anonymous Authors},
  booktitle = {ICML Workshop on Mechanistic Interpretability},
  year      = {2026},
  note      = {Under review},
}
```

---

## Related Work

- Arditi et al. (NeurIPS 2024) — Refusal in language models is mediated by a single direction
- Pan et al. (ICML 2025) — The hidden dimensions of LLM alignment: orthogonal safety directions
- Zou et al. (2023) — Representation engineering: A top-down approach to AI transparency
- Zhao et al. (ICML Workshop 2025) — LLMs encode harmfulness and refusal separately
- Li et al. (NeurIPS 2023) — Inference-time intervention: Eliciting truthful answers

---

*Preliminary work. Under review. Do not distribute.*
