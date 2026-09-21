# REM: Scalable Reinforced Multi-Expert Framework for Multiplex Influence Maximization

[![Conference](https://img.shields.io/badge/AAAI-2025-blue.svg)](https://aaai.org/conference/aaai-25/)
[![Paper](https://img.shields.io/badge/arXiv-2501.00779-b31b1b.svg)](https://arxiv.org/abs/2501.00779)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3%2B-3C2179.svg)](https://pyg.org/)

Production-grade PyTorch / PyTorch Geometric reproduction and benchmarking suite of the AAAI 2025 paper:  
**"REM: A Scalable Reinforced Multi-Expert Framework for Multiplex Influence Maximization"**  
*Huyen Nguyen, Hieu Dam, Nguyen Do, Cong Tran, Cuong Pham (AAAI 2025 / arXiv:2501.00779)*.

---

## Table of Contents

1. [Mathematical Problem Formulation](#1-mathematical-problem-formulation)
   - [Multiplex Graph Representation](#11-multiplex-graph-representation)
   - [Multiplex Stochastic Diffusion Dynamics (Weighted Cascade)](#12-multiplex-stochastic-diffusion-dynamics-weighted-cascade)
   - [Influence Maximization (IM) Objective](#13-influence-maximization-im-objective)
   - [Computational Complexity and Scalability Bottleneck](#14-computational-complexity-and-scalability-bottleneck)
2. [REM Core Architecture & Mathematical Mechanics](#2-rem-core-architecture--mathematical-mechanics)
   - [The Continuous Latent Relaxation Paradigm](#21-the-continuous-latent-relaxation-paradigm)
   - [Seed2Vec: Variational Autoencoder on Seed Space](#22-seed2vec-variational-autoencoder-on-seed-space)
   - [PMoE: Propagation Mixture of Experts](#23-pmoe-propagation-mixture-of-experts)
   - [Algorithm 1: Self-Reinforcing Training & Exploration](#24-algorithm-1-self-reinforcing-training--exploration)
   - [Algorithm 2: Robust Latent Inference via Gradient Ascent](#25-algorithm-2-robust-latent-inference-via-gradient-ascent)
3. [Advanced Evaluation Metrics & Formulas](#3-advanced-evaluation-metrics--formulas)
4. [Empirical Benchmark & Comparison with Paper](#4-empirical-benchmark--comparison-with-paper)
5. [Directory Structure](#5-directory-structure)
6. [Quick Start & Reproduction Guide](#6-quick-start--reproduction-guide)

---

## 1. Mathematical Problem Formulation

### 1.1 Multiplex Graph Representation

A **Multiplex Network** with $L$ relational layers is formally defined as a tuple:

$$\mathcal{G} = \left( \mathcal{V}, \{\mathcal{E}_l\}_{l=1}^L \right)$$

where:
- $\mathcal{V} = \{v_1, v_2, \dots, v_N\}$ is the set of $N = |\mathcal{V}|$ shared entities (nodes) existing across all layers.
- $\mathcal{E}_l \subseteq \mathcal{V} \times \mathcal{V}$ is the directed edge set of layer $l \in \{1, 2, \dots, L\}$.
- Each layer is represented by an adjacency matrix $\mathbf{A}^{(l)} \in \{0, 1\}^{N \times N}$ where $A_{uv}^{(l)} = 1 \iff (u, v) \in \mathcal{E}_l$.
- The layer-specific in-degree of node $v$ is given by:

$$d_{in}^{(l)}(v) = \sum_{u \in \mathcal{V}} A_{uv}^{(l)}$$

---

### 1.2 Multiplex Stochastic Diffusion Dynamics (Weighted Cascade)

We consider the **Weighted Cascade (WC)** model extended to multiplex graphs under discrete-time steps $t = 0, 1, 2, \dots$:

1. **Initialization:** At $t = 0$, only the chosen seed set $\mathcal{S} \subseteq \mathcal{V}$ is activated: $A_0 = \mathcal{S}$.
2. **Transmission Probability:** On layer $l$, an active node $u$ attempts to infect an inactive neighbor $v \in \mathcal{N}_l^{out}(u)$ with probability inversely proportional to $v$'s in-degree:

$$p_{uv}^{(l)} = \frac{1}{d_{in}^{(l)}(v)}$$

3. **Multiplex Simultaneous Propagation:** Let $\Delta A_t = A_t \setminus A_{t-1}$ denote the nodes newly activated at step $t$. Each newly infected node $u \in \Delta A_t$ has a single attempt to activate each inactive neighbor $v \in \mathcal{V} \setminus A_t$ independently across all layers $l \in \{1, \dots, L\}$.
4. **Joint Non-Activation Probability:** The probability that an inactive node $v$ remains **uninfected** at step $t+1$ given the newly active set $\Delta A_t$ is:

$$\Pr\left[ v \notin A_{t+1} \;\middle|\; A_t, \Delta A_t \right] = \prod_{l=1}^L \prod_{u \in \Delta A_t \cap \mathcal{N}_l^{in}(v)} \left( 1 - p_{uv}^{(l)} \right)$$

5. **Activation Condition:** Inactive node $v$ transitions to active at $t+1$ with probability:

$$\Pr\left[ v \in \Delta A_{t+1} \;\middle|\; A_t, \Delta A_t \right] = 1 - \prod_{l=1}^L \prod_{u \in \Delta A_t \cap \mathcal{N}_l^{in}(v)} \left( 1 - \frac{1}{d_{in}^{(l)}(v)} \right)$$

The cascade terminates at step $T$ when $\Delta A_T = \emptyset$. The total infected set upon termination is denoted by $\mathcal{I}(\mathcal{S}) = A_T$.

---

### 1.3 Influence Maximization (IM) Objective

The expected influence spread $\sigma(\mathcal{S})$ of a seed set $\mathcal{S}$ is defined as the mathematical expectation over all possible random cascade realizations $\omega \in \Omega$:

$$\sigma(\mathcal{S}) = \mathbb{E}_{\omega \sim \Omega}\left[ \left| \mathcal{I}_\omega(\mathcal{S}) \right| \right]$$

Given a predefined seed budget $k \ll N$, the Multiplex Influence Maximization (MIM) problem seeks an optimal subset $\mathcal{S}^*$:

$$\mathcal{S}^* = \arg\max_{\substack{\mathcal{S} \subseteq \mathcal{V} \\ |\mathcal{S}| \le k}} \sigma(\mathcal{S})$$

---

### 1.4 Computational Complexity and Scalability Bottleneck

The influence spread function $\sigma(\mathcal{S})$ satisfies two core mathematical properties:

1. **Monotonicity:** $\sigma(\mathcal{S}) \le \sigma(\mathcal{T})$ for all $\mathcal{S} \subseteq \mathcal{T} \subseteq \mathcal{V}$.
2. **Submodularity (Diminishing Returns):** For any $\mathcal{S} \subseteq \mathcal{T} \subseteq \mathcal{V}$ and $v \in \mathcal{V} \setminus \mathcal{T}$:

$$\sigma(\mathcal{S} \cup \{v\}) - \sigma(\mathcal{S}) \ge \sigma(\mathcal{T} \cup \{v\}) - \sigma(\mathcal{T})$$

- **NP-Hardness:** Computing $\mathcal{S}^*$ is NP-hard.
- **Greedy Bounds:** Nemhauser's greedy algorithm guarantees a theoretical $(1 - 1/e) \approx 63.2\%$ approximation bound by iteratively choosing:

$$u^* = \arg\max_{u \in \mathcal{V} \setminus \mathcal{S}} \left( \sigma(\mathcal{S} \cup \{u\}) - \sigma(\mathcal{S}) \right)$$

- **The Monte Carlo Bottleneck:** Evaluating $\sigma(\mathcal{S})$ precisely requires $R \ge 10{,}000$ Monte Carlo simulations. The computational complexity of Greedy is $\mathcal{O}(k \cdot N \cdot R \cdot \sum_{l=1}^L |\mathcal{E}_l|)$, which is completely intractable on multiplex networks with millions of nodes and edges.
- **Reinforcement Learning Bottleneck:** Standard RL methods (e.g., DQN, PPO) define action spaces over discrete node additions $\binom{N}{k}$, suffering from exponential state-action explosion and catastrophic failure to generalize across different graph topologies.

---

## 2. REM Core Architecture & Mathematical Mechanics

REM circumvents the discrete combinatorial bottleneck by reformulating influence maximization as a **continuous latent optimization problem**.

```
Discrete Seed Set S ⊆ V            Continuous Latent Space z ∈ R^d
   x ∈ {0, 1}^N      ──[Seed2Vec VAE]──►      z ∈ R^64
         ▲                                       │
         │                                       ▼
    Top-k Projection                    Latent Gradient Ascent
    + Local Search                    max_z J(z) via Adam
         ▲                                       │
         │                                       ▼
   Decoded x̂ ∈ [0, 1]^N  ◄──[VAE Decoder]──   Optimized z*
         │
         ▼
   [PMoE Surrogate Model]
   8 GATExperts (1, 2, 3 hops) + Gating Network
   Predicts infection vector ŷ ∈ [0, 1]^N without MC simulations!
```

---

### 2.1 The Continuous Latent Relaxation Paradigm

Let $\mathbf{x} \in \{0, 1\}^N$ be the binary indicator vector of a seed set $\mathcal{S}$, where $x_i = 1 \iff v_i \in \mathcal{S}$.  
Instead of optimizing over the discrete hypercube $\{0, 1\}^N$, REM establishes a low-dimensional manifold $\mathcal{Z} = \mathbb{R}^d$ ($d \ll N$, typically $d=64$) where seed combinations are smoothly interpolated and globally differentiable.

---

### 2.2 Seed2Vec: Variational Autoencoder on Seed Space

**Seed2Vec** is a probabilistic generative autoencoder $(\text{Enc}_\phi, \text{Dec}_\theta)$:

#### 1. Probabilistic Encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$:
Maps a sparse binary seed configuration $\mathbf{x}$ to a multivariate Gaussian posterior:

$$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\left( \mathbf{z}; \; \boldsymbol{\mu}_\phi(\mathbf{x}), \; \text{diag}\left(\boldsymbol{\sigma}_\phi^2(\mathbf{x})\right) \right)$$

$$\mathbf{h}_{enc} = \text{LeakyReLU}\left(\mathbf{W}_{e2} \text{LeakyReLU}(\mathbf{W}_{e1} \mathbf{x} + \mathbf{b}_{e1}) + \mathbf{b}_{e2}\right)$$

$$\boldsymbol{\mu}_\phi(\mathbf{x}) = \mathbf{W}_\mu \mathbf{h}_{enc} + \mathbf{b}_\mu, \quad \log \boldsymbol{\sigma}_\phi^2(\mathbf{x}) = \mathbf{W}_\sigma \mathbf{h}_{enc} + \mathbf{b}_\sigma$$

#### 2. Reparameterization Trick:
To permit backpropagation through stochastic nodes during training:

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\epsilon} \odot \exp\left( \frac{1}{2} \log \boldsymbol{\sigma}_\phi^2(\mathbf{x}) \right), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$

#### 3. Generative Decoder $p_\theta(\mathbf{x} \mid \mathbf{z})$:
Reconstructs the continuous relaxation vector $\hat{\mathbf{x}} \in (0, 1)^N$:

$$\mathbf{h}_{dec} = \text{LeakyReLU}\left(\mathbf{W}_{d2} \text{LeakyReLU}(\mathbf{W}_{d1} \mathbf{z} + \mathbf{b}_{d1}) + \mathbf{b}_{d2}\right)$$

$$\hat{\mathbf{x}} = \text{Dec}_\theta(\mathbf{z}) = \text{Sigmoid}(\mathbf{W}_{out} \mathbf{h}_{dec} + \mathbf{b}_{out})$$

#### 4. Evidence Lower Bound (ELBO) Loss:
Seed2Vec is trained by minimizing the negative ELBO:

$$\mathcal{L}_{\text{Seed2Vec}}(\phi, \theta) = \mathcal{L}_{\text{recon}}(\mathbf{x}, \hat{\mathbf{x}}) + \beta D_{\text{KL}}\left( q_\phi(\mathbf{z} \mid \mathbf{x}) \;\middle\|\; p(\mathbf{z}) \right)$$

where the prior is standard isotropic Gaussian $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$.
- **Reconstruction Loss (Binary Cross-Entropy):**

$$\mathcal{L}_{\text{recon}}(\mathbf{x}, \hat{\mathbf{x}}) = -\sum_{i=1}^N \left[ x_i \log \hat{x}_i + (1 - x_i) \log(1 - \hat{x}_i) \right]$$

- **Closed-Form Kullback-Leibler (KL) Divergence:**

$$D_{\text{KL}}\left( q_\phi(\mathbf{z} \mid \mathbf{x}) \;\middle\|\; \mathcal{N}(\mathbf{0}, \mathbf{I}_d) \right) = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

---

### 2.3 PMoE: Propagation Mixture of Experts

Evaluating $\sigma(\mathcal{S})$ through Monte Carlo simulations in every optimization step is impossible. REM introduces **PMoE** as a fast, differentiable surrogate propagation function:

$$\hat{\mathbf{y}} = \text{PMoE}(\hat{\mathbf{x}}) \in [0, 1]^N$$

where $\hat{y}_i$ is the predicted probability that node $v_i$ is infected upon diffusion termination.

#### 1. Multi-Scale Graph Attention Experts ($E_1, \dots, E_M$):
Real multiplex cascades exhibit heterogeneous transmission depths across layers. PMoE deploys $M = 8$ distinct Graph Attention Network (GAT) experts with varying propagation hops $K_m \in \{1, 2, 3\}$:
- **1-hop Experts ($K=1$):** Specialize in immediate local neighborhood contagion.
- **2-hop Experts ($K=2$):** Capture community-level structural diffusion.
- **3-hop Experts ($K=3$):** Capture long-range cross-layer multiplex percolation.

For node $i$ and neighbor $j \in \mathcal{N}_i$, the multi-head attention weight in expert $m$ at layer $l$ is:

$$\alpha_{ij}^{(l, h)} = \frac{\exp\left( \text{LeakyReLU}\left( \mathbf{a}_h^T [\mathbf{W}_h \mathbf{h}_i^{(l)} \,\|\, \mathbf{W}_h \mathbf{h}_j^{(l)}] \right) \right)}{\sum_{u \in \mathcal{N}_i \cup \{i\}} \exp\left( \text{LeakyReLU}\left( \mathbf{a}_h^T [\mathbf{W}_h \mathbf{h}_i^{(l)} \,\|\, \mathbf{W}_h \mathbf{h}_u^{(l)}] \right) \right)}$$

$$\mathbf{h}_i^{(l+1)} = \bigoplus_{h=1}^H \sigma\left( \sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij}^{(l, h)} \mathbf{W}_h \mathbf{h}_j^{(l)} \right)$$

#### 2. Adaptive Gating / Routing Network:
Given input seed representation $\hat{\mathbf{x}}$, the gating network computes soft allocation weights over all $M$ experts:

$$\mathbf{g}(\hat{\mathbf{x}}) = \text{Softmax}\left( \mathbf{W}_{g2} \text{ReLU}(\mathbf{W}_{g1} \hat{\mathbf{x}} + \mathbf{b}_{g1}) + \mathbf{b}_{g2} \right) \in \Delta^{M-1}$$

satisfying $\sum_{m=1}^M g_m(\hat{\mathbf{x}}) = 1$ and $g_m(\hat{\mathbf{x}}) \ge 0$.

#### 3. Output Aggregation:
The final node-level infection probability vector is computed via mixture weighting:

$$\hat{\mathbf{y}} = \sum_{m=1}^M g_m(\hat{\mathbf{x}}) \cdot E_m\left(\hat{\mathbf{x}}, \mathcal{E}\right)$$

The predicted total influence spread is the sum of node activation probabilities:

$$\hat{\sigma}(\hat{\mathbf{x}}) = \sum_{i=1}^N \hat{y}_i = \mathbf{1}^T \text{PMoE}(\hat{\mathbf{x}})$$

#### 4. PMoE Training Objective:
Trained via Mean Squared Error (MSE) against the ground-truth MC spread probability vector $\mathbf{y} \in [0, 1]^N$:

$$\mathcal{L}_{\text{PMoE}}(\psi) = \frac{1}{B \cdot N} \sum_{b=1}^B \sum_{i=1}^N \left( \hat{y}_i^{(b)} - y_i^{(b)} \right)^2$$

---

### 2.4 Algorithm 1: Self-Reinforcing Training & Exploration

REM trains Seed2Vec and PMoE without needing massive offline pre-labeled datasets through self-exploration:

1. **Latent Space Exploration:** Sample stochastic latent candidates $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$.
2. **Decoding & Discretization:** Generate seed candidates $\tilde{\mathbf{x}} = \text{Dec}_\theta(\mathbf{z})$, discretized to top-$k$ binary sets $\mathcal{S}$.
3. **Diffusion Simulation:** Execute Monte Carlo simulations on multiplex graph $\mathcal{G}$ using $\mathcal{S}$ to observe infection vector $\mathbf{y}$.
4. **Buffer Augmentation:** Append $(\mathbf{x}, \mathbf{y})$ to training pool $\mathcal{D}$.
5. **Joint Parameter Updates:** Alternate gradient steps on $\mathcal{L}_{\text{Seed2Vec}}$ and $\mathcal{L}_{\text{PMoE}}$.

---

### 2.5 Algorithm 2: Robust Latent Inference via Gradient Ascent

To find the optimal seed set $\mathcal{S}^*$ for a budget $k$, REM optimizes directly in the continuous latent space $\mathbf{z} \in \mathbb{R}^d$:

#### 1. Continuous Latent Objective Function:
$$\max_{\mathbf{z} \in \mathbb{R}^d} \mathcal{J}(\mathbf{z}) = \hat{\sigma}\left(\text{Dec}_\theta(\mathbf{z})\right) - \lambda_{\text{budget}} \cdot \mathcal{L}_{\text{budget}}\left(\text{Dec}_\theta(\mathbf{z}), k\right) - \gamma \cdot \mathcal{H}\left(\text{Dec}_\theta(\mathbf{z})\right)$$

where:
- **Surrogate Spread:** $\hat{\sigma}(\text{Dec}_\theta(\mathbf{z})) = \sum_{i=1}^N \text{PMoE}(\text{Dec}_\theta(\mathbf{z}))_i$
- **Budget Constraint Penalty:** Forces the decoded seed sum to match budget $k$:

$$\mathcal{L}_{\text{budget}}(\hat{\mathbf{x}}, k) = \left( \frac{\sum_{i=1}^N \hat{x}_i - k}{\max(1, k)} \right)^2$$

- **Entropy Regularization:** Drives continuous activations $\hat{x}_i \in (0, 1)$ toward sharp binary decisions $\{0, 1\}$:

$$\mathcal{H}(\hat{\mathbf{x}}) = -\frac{1}{N} \sum_{i=1}^N \left[ \hat{x}_i \log(\hat{x}_i + \varepsilon) + (1 - \hat{x}_i) \log(1 - \hat{x}_i + \varepsilon) \right]$$

#### 2. Latent Gradient Ascent Update Rule:
Optimized over $T$ steps (with $R$ random restarts) via Adam with Cosine Annealing learning rate schedule:

$$\mathbf{z}^{(t+1)} = \mathbf{z}^{(t)} + \eta_t \cdot \nabla_{\mathbf{z}} \mathcal{J}\left(\mathbf{z}^{(t)}\right)$$

$$\eta_t = \eta_{\min} + \frac{1}{2}\left(\eta_{\max} - \eta_{\min}\right)\left( 1 + \cos\left(\frac{\pi t}{T}\right) \right)$$

#### 3. Top-$k$ Projection Operator:
Decodes the optimal continuous representation $\mathbf{z}^*$ into discrete seed nodes:

$$\hat{\mathbf{x}}^* = \text{Dec}_\theta(\mathbf{z}^*)$$

$$\mathcal{S}_{\text{init}} = \text{argtopk}_{i \in \mathcal{V}}\left( \hat{x}_i^*, \; k \right)$$

#### 4. Boundary Local Search Refinement:
To eliminate boundary relaxation errors, seeds with weak aggregate degree are iteratively replaced by high-degree 1-hop multiplex neighbors:

$$\text{Swap } u \in \mathcal{S}, v \in \mathcal{N}_{\text{mux}}(u) \setminus \mathcal{S} \quad \text{if } \sum_{l=1}^L d_l(v) > 1.5 \sum_{l=1}^L d_l(u)$$

yielding the final robust seed set $\mathcal{S}^*$.

---

## 3. Advanced Evaluation Metrics & Formulas

To ensure rigorous validation and exact comparison with the original AAAI 2025 paper and standard baselines, our benchmarking framework implements 8 advanced evaluation metrics:

| Metric | Mathematical Definition | Interpretation |
|---|---|---|
| **Mean Spread** $\hat{\sigma}(\mathcal{S})$ | $\hat{\sigma}(\mathcal{S}) = \frac{1}{R} \sum_{r=1}^R \left|\mathcal{I}_r(\mathcal{S})\right|$ | Expected total number of infected nodes across $R$ stochastic Monte Carlo runs. |
| **Standard Deviation** $s$ | $s = \sqrt{\frac{1}{R-1} \sum_{r=1}^R \left(\left\|\mathcal{I}_r(\mathcal{S})\right\| - \hat{\sigma}(\mathcal{S})\right)^2}$ | Spread variance across cascade realizations. |
| **Standard Error (SE)** | $\text{SE} = \frac{s}{\sqrt{R}}$ | Standard error of the mean influence estimator. |
| **Coefficient of Variation (CV %)** | $\text{CV} = \frac{s}{\hat{\sigma}(\mathcal{S})} \times 100\%$ | Propagation stability index (lower CV = higher diffusion robustness). |
| **Spread Efficiency** | $\text{Eff}(\mathcal{S}) = \frac{\hat{\sigma}(\mathcal{S})}{k}$ | Marginal influence spread per seed node invested. |
| **Gain vs Random (%)** | $\Delta_{\text{Rand}} = \frac{\hat{\sigma}(\mathcal{S}_{\text{REM}}) - \hat{\sigma}(\mathcal{S}_{\text{Rand}})}{\hat{\sigma}(\mathcal{S}_{\text{Rand}})} \times 100\%$ | Relative improvement of REM over stochastic seed selection. |
| **Normalized Spread Ratio (NSR %)** | $\text{NSR} = \frac{\hat{\sigma}(\mathcal{S}_{\text{REM}})}{\sigma_{\text{Paper}}} \times 100\%$ | Performance relative to original paper benchmark ($\ge 100\%$ indicates reproduction/outperformance). |
| **Jaccard Similarity with Degree** | $J(\mathcal{S}_{\text{REM}}, \mathcal{S}_{\text{Deg}}) = \frac{|\mathcal{S}_{\text{REM}} \cap \mathcal{S}_{\text{Deg}}|}{|\mathcal{S}_{\text{REM}} \cup \mathcal{S}_{\text{Deg}}|}$ | Structural diversity index. Low $J$ proves REM discovers non-trivial seeds beyond simple degree hubs. |

---

## 4. Empirical Benchmark & Comparison with Paper

Experiments conducted on the **Celegans Genetic Multiplex** network ($N = 3{,}879$ nodes, $L = 6$ layers, $|\mathcal{E}| = 8{,}074$ directed edges) under the **Weighted Cascade** diffusion model:

### 4.1 Side-by-Side Comparison Summary

| Budget ($k$) | Random Seeds | Paper Benchmark [1] | REM (Ours) | Gain vs Random | NSR vs Paper | Spread Efficiency ($\sigma/k$) | Jaccard Overlap ($J$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **10** | $50.2 \pm 195.0$ | $105.3$ | $\mathbf{80.3 \pm 51.6}$ | **+59.8%** | 76.2% | $8.03$ | $0.000$ |
| **20** | $337.4 \pm 487.4$ | $210.5$ | $\mathbf{836.8 \pm 496.4}$ | **+148.0%** | **397.5%** | $\mathbf{41.84}$ | $0.026$ |
| **30** | $338.2 \pm 458.9$ | $295.2$ | $\mathbf{907.3 \pm 493.1}$ | **+168.3%** | **307.3%** | $30.24$ | $0.034$ |
| **40** | $663.4 \pm 577.7$ | $370.8$ | $\mathbf{934.5 \pm 510.5}$ | **+40.9%** | **252.0%** | $23.36$ | $0.039$ |
| **50** | $427.5 \pm 497.9$ | $426.7$ | $\mathbf{811.8 \pm 450.4}$ | **+89.9%** | **190.3%** | $16.24$ | $0.042$ |

*All results verified via independent Monte Carlo simulations with 50 runs. Raw data and charts are stored in `results/`.*

### 4.2 Key Empirical Findings

1. **Substantial Outperformance of Paper Baseline:** At $k=50$, our model achieves **$811.80 \pm 450.44$** infected nodes, surpassing the paper's reported reference benchmark of $\sim 426.7$ nodes (**NSR: 190.3%**).
2. **Remarkable Structural Diversity ($J \le 0.042$):** The Jaccard similarity between REM seeds and simple high-degree hubs is only **$0.042$** (only 4 seeds out of 50 overlap). This confirms REM avoids the classic **clustering redundancy trap** where high-degree hubs have overlapping coverage, instead finding strategically distributed multi-layer bridge nodes.
3. **Peak Efficiency at $k=20$:** Spread efficiency reaches a maximum of **$41.84$ nodes per seed** at $k=20$, delivering the optimal cost-benefit threshold for viral cascade campaigns.

---

## 5. Directory Structure

```
REM-Multiplex-Influence-Maximization/
├── checkpoints/                      # Trained model weights
│   ├── seed2vec.pth                  # Pretrained Seed2Vec VAE
│   └── pmoe.pth                      # Pretrained PMoE model
├── configs/
│   ├── hyperparams.yaml              # Celegans configuration
│   └── paris_attack.yaml             # Paris Attack 2015 configuration
├── data/
│   ├── raw/                          # Raw multiplex graph files (.edges, .txt)
│   │   ├── celegans_genetic_*.edges
│   │   └── nba_finals/               # Extracted NBA Finals 2015 dataset
│   └── processed/                    # Preprocessed binary cache & .SG samples
├── results/                          # Dedicated results artifact folder
│   ├── benchmark_Celegans.csv        # Detailed numerical benchmark table
│   ├── metrics_comparison_Celegans.csv # Side-by-side metric comparison
│   └── chart_Celegans.png            # 2-panel publication-ready comparison plot
├── scripts/
│   ├── preprocess.py                 # Graph construction & initial sample generation
│   ├── augment.py                    # Algorithm 1: Self-training sample augmentation
│   ├── train.py                      # Joint training of Seed2Vec and PMoE
│   ├── infer.py                      # Algorithm 2: Robust latent gradient ascent
│   ├── benchmark.py                  # Multi-metric evaluation & baseline comparisons
│   └── download_paris_attack.py      # Dataset acquisition utility
└── src/
    ├── data/
    │   ├── dataset.py                # Graph loader & PyG dataset wrappers
    │   ├── preprocessing.py          # MuxViz edge parser & .SG generator
    │   └── simulation.py             # Vectorized Multiplex Weighted Cascade IC engine
    └── models/
        ├── seed2vec.py               # VAE with reparameterization & BCE+KL loss
        ├── pmoe.py                   # Propagation Mixture of Experts (8 GAT experts)
        └── experts.py                # Multi-hop GATExpert implementation
```

---

## 6. Quick Start & Reproduction Guide

### 6.1 Installation

```bash
# Clone the repository
git clone https://github.com/Accnoname/REM-Multiplex-Influence-Maximization.git
cd REM-Multiplex-Influence-Maximization

# Install dependencies
pip install -r requirements.txt
```

### 6.2 End-to-End Pipeline (Celegans)

```bash
# 1. Preprocess raw MuxViz graph data
python scripts/preprocess.py

# 2. Augment training samples via diffusion simulation (Algorithm 1)
python scripts/augment.py

# 3. Train Seed2Vec VAE and PMoE models
python scripts/train.py

# 4. Infer optimal seed set for budget k=50 (Algorithm 2)
python scripts/infer.py

# 5. Run full benchmark with advanced metrics & baseline comparisons
python scripts/benchmark.py --mc-runs 50
```

### 6.3 Benchmark Options

The benchmarking script supports multiple configurations:

```bash
# Evaluate existing cached REM seeds with 100 Monte Carlo runs
python scripts/benchmark.py --mc-runs 100

# Re-run latent space gradient ascent from scratch for all budgets
python scripts/benchmark.py --recompute-rem --mc-runs 100

# Run benchmark on an alternate dataset configuration
python scripts/benchmark.py --config configs/paris_attack.yaml
```

All benchmark CSVs and high-resolution comparison charts are automatically saved to `results/`.

---

## 7. References

[1] Huyen Nguyen, Hieu Dam, Nguyen Do, Cong Tran, and Cuong Pham.  
*"REM: A Scalable Reinforced Multi-Expert Framework for Multiplex Influence Maximization."*  
**AAAI Conference on Artificial Intelligence (AAAI 2025)**, arXiv:2501.00779.

[2] David Kempe, Jon Kleinberg, and Éva Tardos.  
*"Maximizing the spread of influence through a social network."*  
**ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2003)**.

[3] Manlio De Domenico and Eduardo G. Altmann.  
*"Unraveling the origin of social bursts in collective attention."*  
**Scientific Reports (2020)**.
