# DCPO: Distribution-Centric Policy Optimization

This repository hosts the paper **“Distribution-Centric Policy Optimization Dominates the Exploration–Exploitation Trade-off”**
and (optionally) the accompanying code and artifacts.

---

## What is DCPO?

**Distribution-Centric Policy Optimization (DCPO)** reframes entropy regulation as a *distribution-level* optimization problem:
sustained exploration is governed by the **distribution under which the expected gradient is evaluated**, rather than the
existence of a few “lucky” high-entropy samples.

At a high level, DCPO:
- Maintains **fully on-policy** sampling from the current policy while steering optimization toward a **virtual target
  distribution** (e.g., higher temperature / higher entropy) via **importance sampling**.
- Uses **double importance sampling** with distinct roles:
  - a standard ratio for stable online policy updates (with clipping),
  - a second ratio to shape the regularizer’s expectation toward the target distribution.
- Treats **REINFORCE as regularization**: it provides an exploratory signal while being scaled by a small coefficient
  to control variance.

---

## Method snapshot

Key objects (paper notation):
- Token-level policy ratio:
    $r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}$
- Target-distribution ratio:
    $\rho_{i,t}=\frac{\pi^T_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}$

Intuition: DCPO remains on-policy for sampling, but makes the *regularizer’s gradient expectation* behave as if it were
evaluated under a higher-entropy distribution.

---

## Experimental highlights (from the paper)

### Metrics note (important for interpreting numbers)

For low-cardinality contest benchmarks (e.g., AIME/AMC/HMMT), the paper reports **Avg@32**:
for each problem, sample 32 solutions and average correctness across samples. The paper also reports **Pass@128**
on selected contest benchmarks for Qwen3-4B.

### Main results across three backbones (7-benchmark suites)

The table below summarizes the **average** score across each paper’s 7-benchmark suite for the corresponding backbone.

| Backbone | Base | +GRPO | +AEPO | +DCPO |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 36.24 | 47.70 | 49.57 | **50.94** |
| Qwen2.5-Math-7B | 32.56 | 51.60 | 53.87 | **55.77** |
| Qwen3-4B | 34.33 | 44.63 | 45.31 | **46.41** |

### Selected per-benchmark improvements over GRPO (DCPO vs. GRPO)

Representative gains highlighted in the paper include:
- **Qwen2.5-7B**: AIME25×32 **+7.7**, AMC×32 **+4.1**, Olympiad **+3.4**
- **Qwen2.5-Math-7B**: Minerva **+8.8**, AIME25×32 **+6.8**, GSM8K **+3.3**
- **Qwen3-4B**: AIME24×32 **+3.7**, HMMT25×32 **+1.7**, MMLUpro **+2.0**, GPQAdiamond **+1.0**

### High-budget sampling (Pass@128, Qwen3-4B)

On contest benchmarks with Qwen3-4B:

| Pass@128 | AIME24 | AIME25 | HMMT25 |
|---|---:|---:|---:|
| +GRPO | 83.3 | 76.7 | 66.7 |
| **+DCPO** | **86.7** | **80.0** | **73.3** |

### Ablations: what is necessary for stable exploration?

Removing either **double importance sampling** or the **REINFORCE term** causes **entropy collapse** and reduces
performance by **3.9–4.4** points relative to the full DCPO setting (55.77 average in that configuration).

### Exploration level control (target entropy \(\mathcal{H}_0\))

Varying \(\mathcal{H}_0\) shows that **moderate exploration** achieves the best overall average score, while overly aggressive
exploration reduces the average.

| Setting | Avg |
|---|---:|
| DCPO, $\(\mathcal{H}_0 = 0.25\)$ | **55.77** |
| DCPO, $\(\mathcal{H}_0 = 0.50\)$ | 55.27 |
| DCPO, $\(\mathcal{H}_0 = 0.75\)$ | 54.21 |
| DCPO, $\(\mathcal{H}_0 = 1.00\)$ | 52.43 |

---
## Requirements
### Software
Install via pip:
```bash
conda create -n DCPO python=3.11
conda activate DCPO
git clone https://github.com/597358816/AEPO.git
cd AEPO
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 vllm==0.8.3 transformers==4.51.2 
pip install ray==2.48.0 tensordict==0.9.1 pydantic==2.11.7
pip install flash-attn
pip install -e .
pip install tensorboard
cd example
bash qwen-math-7b-DCPO.sh
```
## Declaration

AI was used only for translation and language polishing in the paper.
