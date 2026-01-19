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


**Qwen2.5-7B (7-benchmark suite)**

| Method | AIME24 | AIME25 | AMC | GSM8K | MATH | Minerva | Olympiad | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 7.91 | 5.31 | 36.2 | 88.5 | 64.4 | 22.0 | 29.3 | 36.24 |
| +GRPO | 17.1 | 7.60 | 65.8 | 92.3 | 75.6 | 36.8 | 38.8 | 47.70 |
| +Entropy-Reg | 13.6 | 8.85 | 67.4 | 92.3 | 76.8 | 35.5 | 39.1 | 47.65 |
| +Entropy-Adv | 14.8 | 8.23 | 67.3 | 91.9 | 76.6 | 38.2 | 37.5 | 47.79 |
| +AEPO | 17.5 | 11.4 | 69.3 | 92.9 | 78.0 | 37.8 | 40.2 | 49.57 |
| +DCPO | 18.8 | 15.3 | 69.9 | 93.0 | 79.2 | 38.2 | 42.2 | **50.94** |
| Δ vs. GRPO | +1.7 | +7.7 | +4.1 | +0.7 | +3.6 | +1.4 | +3.4 | +3.24 (+28.3%) |

**Qwen2.5-Math-7B (7-benchmark suite)**

| Method | AIME24 | AIME25 | AMC | GSM8K | MATH | Minerva | Olympiad | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Math-7B | 15.5 | 7.81 | 42.1 | 65.4 | 59.4 | 11.0 | 26.7 | 32.56 |
| +GRPO | 32.1 | 11.0 | 72.4 | 88.7 | 80.6 | 34.6 | 41.8 | 51.60 |
| +Entropy-Reg | 31.4 | 10.1 | 74.3 | 87.0 | 80.4 | 35.7 | 40.4 | 51.10 |
| +Entropy-Adv | 32.1 | 11.4 | 72.1 | 87.8 | 80.4 | 37.5 | 42.1 | 51.76 |
| +AEPO | 36.4 | 12.6 | 74.8 | 89.5 | 81.6 | 39.0 | 43.0 | 53.87 |
| +DCPO | 35.2 | 17.8 | 76.3 | 92.0 | 82.0 | 43.4 | 43.7 | **55.77** |
| Δ vs. GRPO | +3.1 | +6.8 | +3.9 | +3.3 | +1.4 | +8.8 | +1.9 | +4.17 (+21.9%) |

**Qwen3-4B (7-benchmark suite)**

| Method | AIME24 | AIME25 | HMMT25 | Minerva | Olympiad | GPQA diamond | MMLU pro | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 36.4 | 22.7 | 13.0 | 42.3 | 47.2 | 6.06 | 72.67 | 34.33 |
| +GRPO | 52.9 | 41.5 | 27.1 | 46.7 | 60.0 | 10.1 | 74.1 | 44.63 |
| +Entropy-Reg | 52.4 | 42.6 | 25.3 | 46.3 | 60.1 | 10.6 | 74.1 | 44.48 |
| +Entropy-Adv | 51.6 | 41.7 | 25.5 | 46.0 | 58.4 | 10.6 | 74.8 | 44.08 |
| +AEPO | 54.5 | 43.7 | 26.3 | 47.8 | 60.9 | 10.6 | 73.9 | 45.31 |
| +DCPO | 56.6 | 42.7 | 28.8 | 48.2 | 61.4 | 11.1 | 76.1 | **46.41** |
| Δ vs. GRPO | +3.7 | +1.2 | +1.7 | +1.5 | +1.4 | +1.0 | +2.0 | +1.78 (+17.2%) |
|

### Selected per-benchmark improvements over GRPO (DCPO vs. GRPO)

Representative gains highlighted in the paper include:
- **Qwen2.5-7B**: AIME25×32 **+7.7**, AMC×32 **+4.1**, Olympiad **+3.4**
- **Qwen2.5-Math-7B**: Minerva **+8.8**, AIME25×32 **+6.8**, GSM8K **+3.3**
- **Qwen3-4B**: AIME24×32 **+3.7**, HMMT25×32 **+1.7**, MMLUpro **+2.0**, GPQAdiamond **+1.0**

### High-budget sampling (Pass@128, Qwen3-4B)

On contest benchmarks with Qwen3-4B:

| Method | AIME24 | AIME25 | HMMT25 |
|---|---:|---:|---:|
| Qwen3-4B | 76.7 | 63.3 | 60.0 |
| +GRPO | 83.3 | 76.7 | 66.7 |
| +Entropy-Reg | 83.3 | 73.3 | 66.7 |
| +Entropy-Adv | 83.3 | 73.3 | 70.0 |
| +AEPO | 83.3 | 76.7 | 66.7 |
| **+DCPO** | **86.7** | **80.0** | **73.3** |

### Ablations: what is necessary for stable exploration?

Removing either **double importance sampling** or the **REINFORCE term** causes **entropy collapse** and reduces
performance by **3.9–4.4** points relative to the full DCPO setting (55.77 average in that configuration).

### Exploration level control (target entropy \(\mathcal{H}_0\))

Varying \(\mathcal{H}_0\) shows that **moderate exploration** achieves the best overall average score, while overly aggressive
exploration reduces the average.

### Exploration level control 

| Setting | AIME24×32 | AIME25×32 | AMC×32 | GSM8K | MATH | Minerva | Olympiad | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DCPO, $\mathcal{H}_0 = 0.25$ | 35.2 | 17.8 | 76.3 | 92.0 | 82.0 | 43.4 | 43.7 | **55.77** |
| DCPO, $\mathcal{H}_0 = 0.50$ | 36.4 | 16.7 | 78.5 | 92.0 | 81.6 | 43.9 | 38.8 | 55.27 |
| DCPO, $\mathcal{H}_0 = 0.75$ | 34.5 | 15.6 | 78.5 | 91.7 | 80.5 | 42.5 | 36.2 | 54.21 |
| DCPO, $\mathcal{H}_0 = 1.00$ | 32.4 | 14.9 | 74.9 | 90.8 | 78.7 | 41.2 | 34.1 | 52.43 |


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
