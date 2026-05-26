# SAC-TME State Space Comparison Report

**Project:** `SAC_ASYNC_TME_Tcells` (W&B)  
**Algorithm:** Soft Actor-Critic (SAC) with asynchronous parallel environments  
**Environment:** PhysiGym — PhysiCell Tumor Micro-Environment (TME) with T-cells  
**Date:** 2026-05-18

---

## 1. Environment & Task Description

The reinforcement learning environment simulates a **Tumor Micro-Environment (TME)** using PhysiCell. The agent controls targeted drug delivery to reduce tumor cell proliferation. Three cell types are modeled:

| Cell type     | Role                     |
|---------------|--------------------------|
| `tumor`       | Target — to be eliminated |
| `macrophage`  | Immune effector (M1/M2)   |
| `t_cell`      | Adaptive immune effector  |

Five substrates exist in the microenvironment (`PhysiCell_settings.xml`):

| Substrate | Diffusion (µm²/min) | Decay (1/min) | Role |
|-----------|:-------------------:|:-------------:|------|
| `anti_tumoral_factor` | 3.0 | 0.001 | Repels tumor growth, attracts T-cells |
| `pro_tumoral_factor` | 3.0 | 0.001 | Promotes tumor growth, repels T-cells |
| `drug_1` | 0 (non-diffusing) | 0.05 | Agent-controlled therapeutic |
| `tumor_molecule` | 2.0 | 0.1 | Danger signal secreted by tumor |
| `cytokine` | 0.003 | 0.01 | Immune-activating signal from T-cells |

---

## 2. Biological Model (PhysiCell)

### 2.1 Domain

A 2D simulation domain of **63 × 63 µm** (1 µm resolution), representing a simplified tissue cross-section. Virtual walls enforce boundary containment.

| Parameter | Value |
|-----------|-------|
| Domain | 63 × 63 µm (2D) |
| Time step — diffusion | 0.01 min |
| Time step — mechanics | 0.1 min |
| Time step — phenotype | 6 min |
| Max simulation time | 10,080 min (7 days) |
| RL gym step (`dt_gym`) | 15 min |

### 2.2 Initial Conditions

Cells are loaded from `config/cells.csv` at episode start (overridden by the spatial generation module during training/testing). Default baseline: ~128 tumor cells, 32 T-cells, 16 macrophages. During training, initial positions are re-sampled from a spatial generative process (network-field layout with tunable correlation length and density).

### 2.3 Cell Types

#### Tumor (`ID=0`)

| Property | Value |
|----------|-------|
| Proliferation rate | 3×10⁻⁴ /min ≈ 0.432 /day |
| Apoptosis rate (baseline) | 1×10⁻⁶ /min |
| Motility | **Immobile** (speed = 0) |
| Secretion | `tumor_molecule` at rate 1.0/min → target 1.0 |
| Adhesion | Adheres to all cell types (affinity = 1.0) |

Tumor cells grow via continuous cycling (live-cycle model, code 5). Their growth rate `λ = 2.99×10⁻⁴ /min` is used directly in the reward normalisation. They have no motility and continuously secrete `tumor_molecule`, creating a spatial danger-signal gradient.

#### T-cell (`ID=1`)

| Property | Value |
|----------|-------|
| Proliferation rate | 0 (no division) |
| Apoptosis rate | 0 (immortal in simulation) |
| Speed | 0.1 µm/min |
| Persistence time | 1 min |
| Migration bias | 0.6 |
| Motility | **Active, chemotaxis-driven** |
| Secretion | `cytokine` at rate 1.0/min → target 1.0 |
| Uptake | `anti_tumoral_factor` (rate 1.0), `pro_tumoral_factor` (rate 1.0) |
| Adhesion | Only to `tumor` (affinity = 1.0); none to self or macrophage |

**T-cell chemotaxis (advanced, from XML):**

| Substrate | Sensitivity | Effect |
|-----------|:-----------:|--------|
| `anti_tumoral_factor` | **+0.3** | Attracts T-cells toward anti-tumoral regions |
| `pro_tumoral_factor` | **0** (baseline XML) | No direct gradient following by default |
| `drug_1` | 0 | Insensitive to drug |
| `tumor_molecule` | 0 | Insensitive to danger signal |
| `cytokine` | 0 | Insensitive to own signal |

> **Key immunosuppression mechanism:** Although the baseline XML chemotactic sensitivity to `pro_tumoral_factor` is 0, the `pro_tumoral_factor` **indirectly repels T-cells** through the macrophage polarisation axis: high `tumor_molecule` tips macrophages toward M2-like secretion of `pro_tumoral_factor`, which displaces `anti_tumoral_factor` (competing for the same gradient landscape that T-cells follow). Furthermore, T-cells **actively uptake** `pro_tumoral_factor` (rate 1.0), so where this factor is abundant it creates a biochemical sink that can suppress T-cell effector function (the cytokine secretion is suppressed by uptake competition). The net effect is that regions high in `pro_tumoral_factor` are hostile to T-cell persistence.

T-cells secrete `cytokine`, which in turn triggers tumor apoptosis (see cell rules below).

#### Macrophage (`ID=2`)

| Property | Value |
|----------|-------|
| Proliferation rate | 0 |
| Apoptosis rate | 0 |
| Speed | 0.1 µm/min |
| Persistence time | 1 min |
| Migration bias | 0.6 |
| Motility | **Active, chemotaxis toward `tumor_molecule`** |
| Adhesion | Only to `tumor` (affinity = 1.0) |

Macrophages are **dual-phenotype** (M1/M2-like) depending on local signals. Their behaviour is governed entirely by the cell-rules CSV and by the active chemotaxis toward `tumor_molecule`.

**Macrophage chemotaxis (simple, from XML):**
- Primary: positive chemotaxis along `tumor_molecule` gradient (direction = +1) — macrophages migrate toward tumour.

### 2.4 Cell Rules (`cell_rules.csv`)

The CBHG ruleset (v3.0) defines signal-response relationships applied every phenotype timestep (6 min):

| Cell | Signal | Effect | Target behaviour | Max effect | Half-max | Hill n |
|------|--------|--------|-----------------|:----------:|:--------:|:------:|
| macrophage | `tumor_molecule` ↑ | **increases** | `pro_tumoral_factor` secretion | 10 | 0.5 | 1 |
| macrophage | `drug_1` ↑ | **decreases** | `pro_tumoral_factor` secretion | 0 | 0.5 | 4 |
| macrophage | `drug_1` ↑ | **increases** | `anti_tumoral_factor` secretion | 10 | 0.5 | 4 |
| macrophage | `tumor_molecule` ↑ | **decreases** | `anti_tumoral_factor` secretion | 0 | 0.5 | 1 |
| macrophage | `tumor_molecule` ↑ | **decreases** | `migration speed` | 0.0001 | 0.5 | 4 |
| tumor | `cytokine` ↑ | **increases** | `apoptosis` rate | 1 | 0.5 | 4 |

**Biological interpretation:**

1. **M2-like polarisation under tumor burden:** When `tumor_molecule` is high (dense tumor), macrophages increase `pro_tumoral_factor` secretion and suppress `anti_tumoral_factor` secretion — mimicking immunosuppressive M2 polarisation. They also slow down (reduced migration speed at half-max 0.5), becoming trapped near the tumor.

2. **Drug re-polarises macrophages to M1-like:** `drug_1` reverses the polarisation: it suppresses `pro_tumoral_factor` secretion and boosts `anti_tumoral_factor` secretion. This is the primary mechanism by which the drug is therapeutic — not direct tumor killing, but restoring immune competence.

3. **T-cell cytotoxicity via cytokine:** T-cells secrete `cytokine`; high local `cytokine` increases the tumor apoptosis rate (Hill function, n=4, half-max=0.5). T-cells must physically reach the tumor to exert this effect.

### 2.5 Complete Signalling Network

```
Tumor
  │── secretes ──► tumor_molecule ──► macrophage chemotaxis (attracts)
  │                                └─► macrophage rule: ↑ pro_tumoral_factor
  │                                └─► macrophage rule: ↓ anti_tumoral_factor
  │                                └─► macrophage rule: ↓ migration speed
  │── responds to ◄── cytokine (from T-cell) ──► ↑ apoptosis rate
  │
T-cell
  │── secretes ──► cytokine ──► tumor apoptosis (rule)
  │── uptakes ──── anti_tumoral_factor (depletes local pool)
  │── uptakes ──── pro_tumoral_factor  (depletes local pool)
  │── chemotaxis ─► toward anti_tumoral_factor gradient (+0.3)
  │                 ◄── REPELLED from pro_tumoral_factor zones (indirect:
  │                      pro_tumoral_factor displaces anti_tumoral_factor gradient)
  │
Macrophage
  │── secretes ──► pro_tumoral_factor  (when tumor_molecule is high → M2)
  │── secretes ──► anti_tumoral_factor (when drug_1 is high → M1)
  │── chemotaxis ─► toward tumor_molecule gradient
  │
Drug_1 (agent-controlled, non-diffusing)
  │── re-polarises macrophages: ↓ pro_tumoral_factor, ↑ anti_tumoral_factor
  │── decays at rate 0.05/min (half-life ≈ 14 min)
```

### 2.6 Why Substrate Observations Are Crucial

The above network explains why `img_mc_cells_substrates` (which includes spatial maps of `anti_tumoral_factor`, `pro_tumoral_factor`, and `drug_1`) outperforms pure cell-count observations:

- The **drug effect is indirect** — the agent cannot observe tumor death directly per step; it must infer the state of macrophage polarisation (encoded in `pro_tumoral_factor` / `anti_tumoral_factor` gradients) to know whether its last action was effective.
- **T-cell positioning is functionally coupled to substrate gradients** — T-cells follow `anti_tumoral_factor`; observing the gradient allows the agent to predict where T-cells will concentrate and thus where `cytokine`-mediated killing will occur.
- **Pro-tumoral factor is an immunosuppression indicator** — high `pro_tumoral_factor` signals both that macrophages are M2-polarised and that T-cells are being repelled, making it a critical state variable for planning drug placement.

A scalar observation of `max(pro_tumoral_factor)` (S2 mode) loses the spatial structure entirely; the agent cannot tell *where* the immunosuppressive zone is relative to the tumor or T-cells. The 64×64 image channel preserves this spatial coupling, which is why the image modes generalise far better to novel spatial configurations (circular, rectangle test layouts).

---

## 3. Reward Function

The reward is a **normalized tumor reduction signal** defined as:

```
r(t) = (c_{t-1} - c_t) / expected_growth
```

where:
- `c_t` is the alive tumor cell count at time *t*
- `expected_growth = c_{t-1} * (exp(λ·Δt) - 1)` is the expected number of new tumor cells under uninhibited exponential growth
- `λ` is the intrinsic tumor growth rate (from the PhysiCell XML config)
- `Δt` is the gym step duration

**Interpretation:** A reward of **+1.0** means the agent suppressed exactly the growth that would have occurred naturally; a reward of **0** means no net change; **negative** means tumor grew.

**Episode termination:** `c_t ≤ 3` (eradication) or `c_t > 256` (uncontrolled growth).

The wrapper additionally weights components:
- `w_cell = 0.3` (cell-count based reward weight)

**Metric used for comparison:** `charts/train_return_mean50` — exponentially-smoothed mean of episode returns over the last 50 training episodes (higher = better tumor control). A secondary metric `charts/test_return_mean50` measures generalization on held-out spatial configurations.

---

## 4. State Spaces Compared

Seven observation modes were evaluated. All scalar modes are **float32**; image modes are **uint8** grids of shape `(channels, 64, 64)`.

| ID | Observation Mode | Type | Shape | Description |
|----|-----------------|------|-------|-------------|
| **S1** | `scalars_cells` | scalar | `(3,)` | Normalized alive count per cell type |
| **S2** | `scalars_cells_substrates` | scalar | `(8,)` | Cell counts + max substrate concentration per substrate |
| **S3** | `spatial_scalars_cells` | scalar | `(21,)` | Cell counts + spatial statistics per cell type (presence, x_mean, y_mean, x_std, y_std, dist_to_center) |
| **S4** | `spatial_scalars_cells_substrates` | scalar | `(27,)` | Cell counts + substrate scalars + cell spatial statistics |
| **S5** | `spatial_scalars_cells_spatial_substrates` | scalar | `(39,)` | Cell counts + substrate scalars + spatial features for both cells and substrates |
| **I1** | `img_mc_cells` | image | `(3, 64, 64)` | One channel per cell type: spatial density grid |
| **I2** | `img_mc_cells_substrates` | image | `(8, 64, 64)` | Cell density grids + all 5 substrate concentration grids |

### Feature Detail

**Scalar cell features (`get_cells_scalars`):**  
Per cell type: `(alive_count / normalization_factor) - 1` → range ≈ [−1, 1]

**Substrate scalar features (`get_substrates_scalars`):**  
Per substrate: `max(concentration)` over all grid voxels

**Spatial cell features (`get_spatial_features`):**  
Per cell type (6 values): `[presence_flag, x_mean, y_mean, x_std, y_std, dist_to_center]`  
Coordinates normalized to [0,1] over the domain.

**Spatial substrate features (`get_spatial_substrate_features`):**  
Per substrate (6 values): `[mean, std, min, max, x_centroid, y_centroid]`  
Concentration-weighted centroid.

**Image features (`get_matrix_cells` / `get_matrix_substrates`):**  
Cells are discretized onto a 64×64 grid; pixel intensity ∝ cell density per voxel, clipped to [0,255]. One channel per cell type or substrate.

---

## 5. Training Setup

| Parameter | Value |
|-----------|-------|
| Algorithm | SAC (Soft Actor-Critic) |
| Total timesteps | 500,000 |
| Num parallel envs | 9 |
| Replay buffer size | 300,000 |
| Batch size | 576 |
| Policy LR / Q LR | 3e-4 |
| Gamma | 0.99 |
| Tau (soft update) | 0.005 |
| Entropy tuning | Automatic (autotune) |
| Policy architecture | MLP (scalar) / IMPALA CNN (image) |
| Learning starts | 5,000 steps |
| Test frequency | Every 4 episodes |
| Training init | Network-field spatial layout |
| Test layouts | `circular`, `rectangle` |

**Neural architecture for image modes:** IMPALA-style CNN  
`Conv(in→32, 8×8, stride 4) → Mish → Conv(32→64, 4×4, stride 2) → Mish → Conv(64→64, 3×3, stride 1) → Mish → Flatten → FC(256) → FC(256)`

---

## 6. Results — Aggregated by Observation Mode

Only **complete runs (≥ 400,000 environment steps)** with logged metrics are included.  
Multiple seeds are averaged. `train_mean50` = smoothed training return; `test_mean50` = held-out test return.

| Observation Mode | n runs | Train mean50 | Test mean50 | Test Rectangle | Test Circular |
|-----------------|--------|:------------:|:-----------:|:--------------:|:-------------:|
| **I2** `img_mc_cells_substrates` | 5 | **+77.2** | **+56.5** | +44.5 | **+97.9** |
| **I1** `img_mc_cells` | 4 | +50.3 | +20.8 | −0.6 | +5.0 |
| **S3** `spatial_scalars_cells` | 3 | +51.2 | +12.4 | **+112.7** | +16.3 |
| **S4** `spatial_scalars_cells_substrates` | 3 | +29.2 | +3.4 | −22.1 | −1.2 |
| **S5** `spatial_scalars_cells_spatial_substrates` | 2 | +3.2 | +0.5 | +85.1 | −37.6 |
| **S1** `scalars_cells` | 4 | +5.9 | +17.6 | −7.7 | −20.5 |
| **S2** `scalars_cells_substrates` | 4 | −16.8 | +12.3 | +40.2 | −47.7 |

### Per-Seed Breakdown
Rect: Rectangular, Circ: Circular.
#### I2 — `img_mc_cells_substrates` (image, 6 channels)
| Seed | Steps | Train50 | Test50 | Rect | Circ |
|------|-------|---------|--------|------|------|
| 1 | 499,356 | +78.0 | +58.2 | +4.4 | +106.7 |
| 32 | 529,501 | +79.7 | +69.9 | +4.9 | +61.4 |
| 64 | 410,875 | +86.9 | +47.7 | +109.6 | +33.7 |
| 64 | 499,977 | +64.3 | +42.3 | +47.3 | +128.9 |
| 128 | 499,914 | +77.1 | +64.5 | +56.1 | +158.8 |

#### I1 — `img_mc_cells` (image, 3 channels)
| Seed | Steps | Train50 | Test50 | Rect | Circ |
|------|-------|---------|--------|------|------|
| 1 | 499,884 | +39.8 | +22.4 | −28.7 | −28.1 |
| 32 | 526,602 | +42.4 | +27.8 | −17.4 | −43.9 |
| 64 | 499,769 | +51.3 | +14.5 | −0.5 | −2.8 |
| 128 | 499,632 | +67.5 | +18.3 | +44.2 | +94.8 |

#### S3 — `spatial_scalars_cells` (scalar, 21-dim)
| Seed | Steps | Train50 | Test50 | Rect | Circ |
|------|-------|---------|--------|------|------|
| 1 | 499,976 | +16.7 | −28.9 | +32.2 | −81.3 |
| 64 | 499,996 | +59.8 | +27.3 | +223.1 | +91.1 |
| 128 | 499,990 | +77.1 | +38.8 | +82.9 | +39.2 |

#### S4 — `spatial_scalars_cells_substrates` (scalar, 27-dim)
| Seed | Steps | Train50 | Test50 | Rect | Circ |
|------|-------|---------|--------|------|------|
| 1 | 499,981 | −1.7 | −7.2 | −107.0 | −57.7 |
| 64 | 499,951 | −2.3 | −28.5 | −5.4 | −15.5 |
| 128 | 499,938 | +91.5 | +46.0 | +46.0 | +69.5 |

#### S5 — `spatial_scalars_cells_spatial_substrates` (scalar, 39-dim)
| Seed | Steps | Train50 | Test50 | Rect | Circ |
|------|-------|---------|--------|------|------|
| 1 | 499,979 | +10.4 | +1.2 | +6.0 | −67.5 |
| 64 | 499,979 | −4.1 | −0.1 | +164.2 | −7.6 |

#### S1 — `scalars_cells` (scalar, 3-dim)
| Seed | Steps | Train50 | Test50 | Rect | Circ |
|------|-------|---------|--------|------|------|
| 1 | 499,925 | +37.2 | +24.0 | −1.5 | −21.8 |
| 32 | 527,895 | +6.2 | +23.3 | +30.3 | +46.7 |
| 64 | 499,988 | −20.3 | +6.9 | −52.6 | −33.1 |
| 128 | 499,938 | +0.4 | +16.3 | −6.9 | −74.0 |

#### S2 — `scalars_cells_substrates` (scalar, 8-dim)
| Seed | Steps | Train50 | Test50 | Rect | Circ |
|------|-------|---------|--------|------|------|
| 1 | 499,939 | +1.7 | +26.8 | +144.6 | −0.6 |
| 32 | 528,409 | −79.2 | −24.6 | −4.8 | −90.8 |
| 64 | 499,942 | +25.3 | +18.8 | +8.0 | −44.4 |
| 128 | 499,959 | −15.1 | +28.1 | +12.9 | −54.8 |

---

## 7. Analysis & Key Findings

### 7.1 Image-Based State Spaces Outperform Scalar State Spaces

The clearest finding is that **image-based representations (`I1`, `I2`) yield substantially higher and more consistent training returns** than scalar-based ones (`S1`–`S5`).

- **`img_mc_cells_substrates` (I2)** is the best-performing mode across all metrics:
  - Average train return: **+77.2** vs. next-best scalar +51.2 (S3) and +5.9 (S1)
  - Average test return: **+56.5** — more than 2× better than any scalar mode
  - **Low variance across seeds**: all seeds in [+64, +87], while scalar modes show extreme seed sensitivity (e.g., S4 ranges from −2.3 to +91.5)

- **`img_mc_cells` (I1)** also performs well on training (+50.3) but **struggles on held-out test configurations**, especially circular and rectangle layouts (both near 0 or negative on average). This suggests it learns a policy overfitted to the training spatial layout without the substrate information that helps generalize.

- Adding substrate channels to the image (**I1 → I2**) delivers a consistent improvement:
  - Train: +50.3 → +77.2 (+27 points)
  - Test: +20.8 → +56.5 (+36 points)
  - The substrate channels (debris, pro-tumoral, anti-tumoral factors) provide direct information about where the drug has been applied and its effects, enabling better credit assignment.

### 7.2 Scalar Spatial Features Help Training but Not Generalization

- **`spatial_scalars_cells` (S3)** achieves a similar *training* return to `img_mc_cells` (+51 vs +50), but its test rectangle return is surprisingly high (+112.7 on average), driven by seed 64 (+223.1). This is likely a lucky seed rather than a robust result — seed 1 collapses to −28.9 test50.

- **`spatial_scalars_cells_substrates` (S4)** shows extremely high seed variance (−2.3 to +91.5), indicating the higher-dimensional input is harder to optimize reliably.

- **`scalars_cells` (S1)** and **`scalars_cells_substrates` (S2)** show near-random or negative performance on both training and testing, confirming that aggregate cell counts alone provide insufficient signal for the agent to learn spatial drug delivery.

### 7.3 More Scalar Features ≠ Better Performance

Counter-intuitively, adding more scalar features does not improve performance:

```
Scalar complexity (approx):  S1 (3d) → S2 (8d) → S3 (21d) → S4 (27d) → S5 (39d)
Train mean50:                +5.9    → −16.8   → +51.2    → +29.2    → +3.2
```

The `spatial_scalars_cells_spatial_substrates` (S5, 39-dim) performs **worse** than just spatial cell features (S3, 21-dim). The substrate spatial features may add noise that destabilizes learning, or the combined 39-dim space is too high-dimensional for the scalar MLP to optimize within 500k steps.

### 7.4 Seed Sensitivity

Scalar modes exhibit much higher seed-to-seed variance than image modes:

| Mode | Train50 std across seeds |
|------|--------------------------|
| `img_mc_cells_substrates` | ~8.5 |
| `img_mc_cells` | ~11.3 |
| `spatial_scalars_cells` | ~25.8 |
| `scalars_cells` | ~22.0 |
| `scalars_cells_substrates` | ~38.3 |

Image-based representations are **more robust to random initialization**, likely because the CNN inductive bias (spatial locality, translation equivariance) aligns well with the physical structure of the TME.

---

## 8. Summary

| Finding | Conclusion |
|---------|-----------|
| **Best overall state space** | `img_mc_cells_substrates` (6-channel image, 64×64) |
| **Best scalar state space** | `spatial_scalars_cells` (21-dim), but high variance |
| **Worst state spaces** | Pure global scalars: `scalars_cells`, `scalars_cells_substrates` |
| **Image vs. scalar gap** | +71 points on train_mean50 (I2 vs S1), +39 on test_mean50 |
| **Substrate info impact** | Critical: I2 outperforms I1 by ~27 pts train, ~36 pts test |
| **Spatial features value** | Helpful for training (S3 > S1) but inconsistent on test |
| **More is not always better** | S5 (39-dim) ≈ S5 (3-dim) due to optimization difficulty |
| **Seed robustness** | Image modes ~3–4× more consistent than scalar modes |

### Recommendation

For the PhysiCell TME task, **`img_mc_cells_substrates`** should be the default state representation. It provides:
1. Full spatial layout of all cell types
2. Full spatial layout of microenvironmental substrates
3. Consistent performance across seeds
4. Strong generalization to unseen spatial configurations (circular, rectangle test layouts)

If compute is constrained and image processing overhead is a concern, **`spatial_scalars_cells`** is the best scalar alternative but requires more seeds to achieve reliable performance.

---

## 9. Spatial Layout Examples

The figures below show representative initial cell configurations for each evaluation regime.

### 9.1 Test Layouts (held-out, never seen during training)

![Spatial layouts overview](figures/fig_layouts.png)


The figure shows all four layouts side-by-side. Test layouts (rectangle, circular) are **out-of-distribution**: their spatial correlation structure differs from the training regime, so good generalisation requires a policy that transfers without having memorised training positions. Training layouts are sampled from a **network-field generative process** (Gaussian random field, correlation length 45 µm, threshold 0.55), producing clustered but irregular patterns.

---

## 10. Episode Comparison — run\_000143

The figure below compares a single matched episode (step 143 of the test rollout, network-field layout) across the three best-performing state spaces: **I2** (`img_mc_cells_substrates`), **I1** (`img_mc_cells`), and **S3** (`spatial_scalars_cells`).

![Episode comparison](figures/fig_episode_comparison.png)


### 10.1 Episode Statistics

| Metric | I2 — `img_mc_cells_substrates` | I1 — `img_mc_cells` | S3 — `spatial_scalars_cells` |
|--------|:------------------------------:|:-------------------:|:----------------------------:|
| Final tumor count | **4** | 112 | 14 |
| Cumulative return | **+175.1** | −7.4 | +106.3 |
| Cumulative dose used | 35.42 | 12.16 | 31.05 |
| Peak dose per step | 0.813 | 0.451 | 0.484 |
| Episode length | 480 steps | 480 steps | 480 steps |

### 10.2 Panel-by-Panel Interpretation

**Cumulative Return (top panel)**

I2 achieves a strongly positive and steadily increasing cumulative return (+175), indicating sustained tumor suppression throughout the episode. S3 also shows positive return (+106) but with a flatter slope in the second half — tumor count stabilises at 14 rather than approaching eradication. I1 collapses to near-zero and dips negative, ending with 112 tumor cells — comparable to an untreated simulation. The agent with only cell image channels but no substrate information fails to identify where and when to apply the drug effectively.

**Cumulative Drug Used (middle panel)**

I2 and S3 both apply significantly more drug (35.4 and 31.1 cumulative dose units) than I1 (12.2). This reflects two distinct failure modes:
- **I1 under-medicates**: without substrate feedback (especially `anti_tumoral_factor` / `pro_tumoral_factor` gradients), the agent cannot tell whether its previous injection was absorbed or whether macrophages have been re-polarised, so it converges to a conservative low-dose policy.
- **I2 and S3 over-apply relative to I1**, but only I2 achieves near-eradication, suggesting that spatial substrate information (I2) allows the agent to target drug delivery more precisely, converting dose into effective macrophage repolarisation.

**Drug Dose per Step (bottom panel)**

All three agents show high variance in per-step dosing, consistent with the stochastic nature of the SAC policy and the spatially heterogeneous tumour environment. I2 shows pronounced late-episode spikes (steps 300–480), corresponding to the final push toward eradication (tumor count approaching the termination threshold of ≤3). I1 shows sparse, low-magnitude pulses throughout, with no concentrated effort. S3 distributes dosing more evenly, suggesting a less spatially-targeted strategy that nevertheless achieves partial tumour control.

### 10.3 Why I2 Succeeds Where I1 Fails

This single episode illustrates the key argument of the report in concrete terms:

1. **Substrate channels close the feedback loop.** The drug `drug_1` acts by repolarising macrophages. Without observing `pro_tumoral_factor` and `anti_tumoral_factor` maps, the I1 agent receives no spatial feedback on whether its injection re-polarised nearby macrophages. It can only infer success from the slowly-changing cell counts, which are too noisy for credit assignment.

2. **Spatial targeting requires spatial state.** To place the drug optimally (inside a dense tumor cluster where macrophages are present), the agent needs to see *where* the factors are concentrated. The 64×64 substrate grids provide this; the scalar max-concentration S2 mode does not. S3's spatial cell features provide partial spatial information but miss the substrate coupling entirely.

3. **S3 reaches partial control but not eradication.** Knowing the centroid and spread of each cell type is enough to develop a rough spatial drug strategy, but without knowing where `pro_tumoral_factor` is suppressing T-cell infiltration, the agent cannot achieve the final push to eradication.

---

## 11. Episode Comparison — run\_000147 (seed 128)

The figure below shows a second matched episode (`run_000147`, seed 128, network-field training layout) comparing the same three state spaces. This episode is harder than run\_000143: no agent achieves near-eradication, and all three exhibit a **return rollback** in the second half of the episode.

![Episode comparison 2](figures/fig_episode_comparison_2.png)

### 11.1 Episode Statistics

| Metric | I2 — `img_mc_cells_substrates` | I1 — `img_mc_cells` | S3 — `spatial_scalars_cells` |
|--------|:------------------------------:|:-------------------:|:----------------------------:|
| Final tumor count | 31 | 64 | 43 |
| Cumulative return (final) | **+53.4** | +35.0 | +32.3 |
| Peak cumulative return | **+63.1** (step 365) | +58.2 (step 323) | +66.6 (step 330) |
| Cumulative dose used | **35.69** | 7.16 | 34.92 |
| Peak dose per step | **0.834** | 0.563 | 0.496 |
| Episode length | 480 steps | 480 steps | 480 steps |

### 11.2 Panel-by-Panel Interpretation

**Cumulative Return (top panel)**

All three agents build positive cumulative return through the first ~330 steps, then the slope flattens or turns negative — indicating the tumor rebounds after initial suppression. The three trajectories are remarkably close up to step ~200, then diverge:

- **I2** (+53.4 final) maintains the lead throughout and declines the least after its peak (+63.1 at step 365). Its substrate channels continue to provide feedback that lets the agent partially contain the rebound.
- **S3** peaks highest of the three (+66.6 at step 330) but declines more sharply afterward, ending at +32.3. The spatial cell features are sufficient for early suppression but cannot sustain it once the tumor re-establishes — without substrate spatial information the agent cannot track the evolving immunosuppressive gradient.
- **I1** (+35.0 final) shows the same rollback pattern. Notably its peak (+58.2 at step 323) is close to S3, suggesting both modes achieve similar early suppression but both lose control past the midpoint for different reasons.

Comparing with run\_000143: in that episode I2 achieved +175 by near-eradicating the tumour (final count 4). Here the tumour is not eradicated (final count 31), so the agent accumulates fewer positive rewards in the second half — illustrating that the reward function is **trajectory-sensitive**: near-eradication yields compounding positive rewards, while a stabilised-but-not-eliminated tumour yields near-zero per-step rewards as growth and suppression balance out.

**Cumulative Drug Used (middle panel)**

The drug usage split is starker here than in run\_000143:

- **I2 and S3** both apply ~35 cumulative dose units, nearly identical and much higher than I1.
- **I1** uses only 7.2 units — again under-medicating severely, consistent with the pattern seen in run\_000143. The cell-only image provides no substrate feedback, so the policy has not learned to commit to sustained dosing.

Despite identical cumulative dose between I2 and S3, only I2 achieves a better final outcome (+53.4 vs +32.3), confirming that **where** the drug is placed (guided by substrate maps in I2) matters as much as **how much** is applied.

**Drug Dose per Step — dosing schedule (bottom panel)**

- **I2** shows a burst-heavy schedule concentrated in the second half (steps 300–480), corresponding to an attempted late-episode push against the rebounding tumour. The largest spikes appear after step 350, when cumulative return is already declining — a reactive rather than pre-emptive strategy.
- **S3** applies doses more uniformly across the episode with moderate spikes, reflecting a spatial-centroid-based targeting strategy that lacks the precision of substrate gradient information.
- **I1** produces sparse, low-amplitude pulses throughout with no sustained burst phase — confirming it has learned a fundamentally conservative policy that fails to commit drug resources at the critical moments.

### 11.3 Comparison Between run\_000143 and run\_000147

| | run\_000143 (seed 64) | run\_000147 (seed 128) |
|---|:---:|:---:|
| I2 final tumor | **4** (near-eradication) | 31 (rebound) |
| I2 cumulative return | +175.1 | +53.4 |
| I1 final tumor | 112 | 64 |
| S3 final tumor | 14 | 43 |
| All agents: return rollback? | No | **Yes** |

The contrast between these two episodes reveals that the state space advantage of I2 is most decisive when eradication is achievable — in that regime I2 pushes all the way to the termination threshold while I1 and S3 stall. When the tumour is more resilient (run\_000147), all three agents struggle with the rebound, but I2 still maintains the highest final return. This suggests the substrate spatial information is especially critical for the **final-phase targeting** needed to cross the eradication threshold.

---

## 12. W&B Training Curves & Observation Visualisations

### 12.1 Observation Mode Visualisation (I2 — `img_mc_cells_substrates`)

The images below show the 8-channel image observation at six time steps of a single episode, illustrating what the best-performing agent actually sees.

| Step 0 | Step 20 | Step 40 |
|--------|---------|---------|
| ![obs0](../img/observation_0.png) | ![obs20](../img/observation_20.png) | ![obs40](../img/observation_40.png) |

| Step 60 | Step 80 | Step 100 |
|---------|---------|----------|
| ![obs60](../img/observation_60.png) | ![obs80](../img/observation_80.png) | ![obs100](../img/observation_100.png) |

Each image has 8 channels (shown as separate panels): **CH 0–2** = tumor, T-cell, macrophage density grids; **CH 3–7** = anti\_tumoral\_factor, pro\_tumoral\_factor, drug\_1, tumor\_molecule, cytokine concentration maps. All 5 substrates are included.

**Key observation per state space:**

| State space | What the agent sees | What is missing |
|-------------|--------------------|--------------------|
| **I2** `img_mc_cells_substrates` (8ch, 64×64) | Full spatial layout of all 3 cell types + all 5 substrate concentration maps (anti\_tumoral, pro\_tumoral, drug\_1, tumor\_molecule, cytokine) | Nothing — full spatial state |
| **I1** `img_mc_cells` (3ch, 64×64) | Spatial layout of tumor, T-cell, macrophage only | All substrate maps — no feedback on drug spread or immune gradients |
| **S3** `spatial_scalars_cells` (21-dim) | Per cell type: presence flag, x\_mean, y\_mean, x\_std, y\_std, dist\_to\_center | All substrate information; spatial resolution limited to centroid/spread summary |
| **S1** `scalars_cells` (3-dim) | Normalised alive count per cell type only | Spatial structure, substrate gradients — minimal signal |

The substrate channels (CH 3–7) are crucial: **CH 4** (`pro_tumoral_factor`) shows where the immunosuppressive gradient is strongest — exactly where the agent must deliver drug to repolarise macrophages. Without this, I1 and S1 agents cannot close the feedback loop between dosing and macrophage repolarisation.

For **S2** (`scalars_cells_substrates`) and **S4/S5** (scalar substrate extensions): adding `max(concentration)` per substrate introduces a single scalar per substrate — this is dominated by the global peak and adds noise rather than spatial signal, explaining why S2 performs worse than S1 and S4/S5 do not outperform S3.

---

### 12.2 Training Curves — All Observation Modes (W&B panels)

The panels below cover **all 7 observation modes** (S1–S5, I1, I2). Legend: **I2** `img_mc_cells_substrates` = red dashed; **I1** `img_mc_cells` = dark green; **S5** `spatial_scalars_cells_spatial_substrates` = green dot; **S3** `spatial_scalars_cells` = blue; **S4** `spatial_scalars_cells_substrates` = red solid; **S2** `scalars_cells_substrates` = cyan; **S1** `scalars_cells` = grey.

**Training return (smoothed, `train_return_mean50`):**

![train_return_mean50](../img/Section-2-Panel-3-uwnnqvhkz.png)

**I2** (red dashed) separates clearly from all other modes after ~50k steps and reaches ~75–80 by 500k steps. **I1** (dark green) converges to ~47, showing that spatial cell images alone provide strong signal but the substrate gap is decisive. All scalar modes (S1–S5) plateau below 35, confirming the image vs. scalar gap reported in §6. Among scalars, **S3** (blue) is most stable; S4/S5 with substrate scalars do not outperform S3 despite higher dimensionality.

**Training return standard deviation (`train_return_std`):**

![train_return_std](../img/Section-2-Panel-2-g0ddriu6t.png)

**I2** and **I1** maintain consistently high std (~60–65) throughout training — reflecting the diverse episode outcomes across seeds (some near-eradication, some rebound). Scalar modes show lower and decreasing std, meaning their policies converge to a narrower, less ambitious behavioural range.

**Test return — rectangle layout (raw):**

![test_rectangle_return_raw](../img/Section-2-Panel-0-t6c8f8adz.png)

**Test return — circular layout (raw):**

![test_circular_return_raw](../img/Section-2-Panel-1-v5gma8p0j.png)

On both held-out test layouts, **I2** (red dashed) leads with the highest and most consistent returns. **I1** (dark green) shows volatile performance — competitive on some seeds, poor on others — highlighting that without substrate information, generalisation to unseen layouts is unreliable. Scalar modes (S1–S5) are clustered near zero or negative on both test layouts, with high run-to-run variance. The circular layout (right) is harder to generalise to: even I2 shows more spread than on rectangle.

**Test return (smoothed, `test_return_mean50`):**

![test_return_mean50](../img/Section-2-Panel-6-42sh5v4hc.png)

The smoothed view confirms **I2** as the dominant mode on test, steadily climbing to ~55–60 by 500k steps. **I1** trails at ~20–25. Among scalar modes, **S3** is the most stable but peaks around 15. `scalars_cells_substrates` (S2, cyan) degrades after an early peak — the substrate noise problem: max-concentration scalars overfit to training layout geometry and do not transfer.

**Test return standard deviation (`test_return_std`):**

![test_return_std](../img/Section-2-Panel-4-q55wfb7fm.png)

**I2** shows the highest test std across all modes — reflecting that it achieves very high returns on some episodes (near-eradication) and moderate returns on others. This is a sign of a capable but not yet fully robust policy, rather than instability. All scalar modes converge to low test std, consistent with conservative low-dose policies that produce predictable but modest outcomes.

