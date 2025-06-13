# Plan
Goal find a cumulative return improvement <=> a policy learnt is better than a random or dummy one
## Problems
- **Problem 1**: Using images as state space makes the algorithm very slow (>5 seconds per step).
- **Problem 2**: Stochastic environment.
- **Problem 3**: How to compare the current learned policy to a random policy or a simple human-designed strategy.

## Solutions
- **Problem 1**: The slowdown was caused by the Replay Buffer (fixed — 5× speed improvement). Another solution would be to use PPO instead of SAC.
- **Problem 2**: Compute multiple environment dynamics under different strategies (done), and compute a confidence interval.
- **Problem 3**: 
  - Given the multiple dynamics, we can compare the learned policy against them.
  - We can also design dummy strategies, e.g., if exhausted CD8 cells are detected, add the drug that affects CD8; do similarly for M2 cells.
  - If the current policy is less effective than any dummy strategy, take the best dummy strategy (hopefully not random), perform behavior cloning on it, and then fine-tune with PPO or SAC to find a better strategy.

## Add new feature
- PPO (Algorithm)
- Transformer (Neural Architecture)
- [x] Replay Buffer in Rust

## Tasks
- Analyze [result_11](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/code_tests/stochastic_results_sureli11.csv) and [result_9](https://github.com/Dante-Berth/PhysiGym/blob/main/rl/code_tests/stochastic_results_sureli9.csv)

For the current learned policy:
- [x] Based on the results, conclude if the policy learned with scalar state space is one of the best.
- Find a dummy strategy and compare it to other strategies.
- Perform Behavior Cloning (BC) and use PPO/SAC to improve the policy.

For image-based state space:
- [x] Debug the Replay Buffer in Rust and compare it to the pure Python implementation.
- [x] Launch with image state space on Sureli11.
- Integrate a Transformer and launch on Sureli9.

## Tasks done
- [x] Replay Buffer in Rust (same speed as the Python version)
- [x] Launched with the new replay buffer in Python on Sureli11 with image state space
- [x] [Given the results from dummy policies](https://github.com/Dante-Berth/PhysiGym/tree/main/rl/code_tests), the [policy learned from scalar values](https://wandb.ai/corporate-manu-sureli/SAC_IMAGE_COMPLEX_TME/runs/8y6ebe1p?nw=nwuseralexandrebertin) performs better on average

# New Ideas

## Easier Problem
1. Instead of solving a continuous problem, solve a discrete one.  
2. Reduce the number of cells — this could increase stochasticity and requires re-running the [code tests](https://github.com/Dante-Berth/PhysiGym/tree/main/rl/code_tests). This change would also require adding [IQN](https://proceedings.mlr.press/v80/dabney18a/dabney18a.pdf).

## New Features
- Add [Simba](https://arxiv.org/pdf/2410.09754) for concentration of cells as state

# 12 May

## Problems Solved
- [x] Fixed a bug where a `-1` reward was missing for drugs (re-launched on Sureli9 (scalars) and Sureli11 — no improvement observed).
- [x] Re-launched with the corrected reward; no improvements observed (still waiting for image results).

## Rewards Used

Proposed a new reward — each component is scaled between 0 and 1:

$$
r(t) = (1 - \alpha) \left(1 - \frac{d_{1,t} + d_{2,t}}{2} \right) + \alpha \cdot c_{norm,t}
$$

Where:

$$
c_{norm,t} = 1 - \frac{\text{clip}\left( \log\left( \frac{c_t}{c_{t-1}} \right), -0.05, 0.05 \right) - (-0.05)}{0.1}
$$

[Link to reward implementation](https://github.com/Dante-Berth/PhysiGym/blob/main/model/complex_tme/custom_modules/physigym/physicell_model.py) — working on Sureli9.

Also developed a **simple reward** to reduce the number of cancer cells over the episode:

$$
r(t) = (1 - \alpha) \left(1 - \frac{d_{1,t} + d_{2,t}}{2} \right) + \alpha \left(1 - \frac{\min(C_t, \text{maxcells})}{\text{maxcells}} \right)
$$

Where $\text{maxcells} = 1000$ — working on Sureli11.

### Notes
- The weight $\alpha$ equals to $0.8$.
- **Problem**: Lack of resources — requested an account from Genotoul => now i have an account.
- **Problem**: [No improvements observed](https://api.wandb.ai/links/corporate-manu-sureli/ip9ppxmp) after 4 Millions steps.

## New Features Added but Not Launched

To address neuron fading and scale ambiguity in deep learning, added:
- Layer Normalization
- L2 weight norm on actor and critic

References:
- [Normalization and Effective Learning Rates in Reinforcement Learning](https://arxiv.org/pdf/2407.01800)
- [Hyperspherical Normalization for Scalable Deep Reinforcement Learning](https://arxiv.org/pdf/2502.15280)

I added these two features in my code, it is launched on Sureli 9 since Saturday !
## Side Quest
- Thursday and Friday courses

## Conclusion 
Given the previous ideas and given the last results, using PPO or Behavior Cloning are not good ideas because the problem remains the same and adds more difficulty. We also do not change the number of cells to avoid any highest stochasticity. The easiet solution to implement and to accelerate the research is to use discrete actions instead of continous actions.
To simplify the problem:
- Use **categorical actions** instead of continuous actions.  
  Example: \[0, 0.5, 1\] instead of \[0, 1\].  
  This requires changes in `physicell_model.py` from `complex_tme` and adoption of a discrete action RL algorithm like [C51](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py). We begin with C51 and then we can end with [IQN](https://github.com/BY571/IQN-and-Extensions/blob/master/IQN-DQN.ipynb)


## Sunday Results
I changed from continous problem to discrete, there is no improvement  for $\alpha=0.8$!

# 19 May
## Launched 
- [x] C51 launched with $\alpha=0.7$ => i have not seen any improvement
## Ideas
### In general
  - Keep changing $\alpha\in[0.5,0.8]$, $\alpha = 0.5$ means $50\%$ of the reward refers to the drug toxicity while $\alpha = 0.8$ means $20\%$ of the reward refers to the drug toxicity. You have to refer to the rewards proposed.
  - weak reward-action, we can compute $TD(n)$ instead of $TD(1)$
  - Add Transformers, the idea behind it is the fact we use a dataframe like of cells and cells can be seen as Token. The transformer will be used as encoder. Besides, there is an advantage in the neural architecture of a Transformer despite it head of attention, the layer norm used which is now used in Reinforcement Leanring seems to stablilize the learning process - [Normalization and Effective Learning Rates in Reinforcement Learning](https://arxiv.org/pdf/2407.01800) and  [Hyperspherical Normalization for Scalable Deep Reinforcement Learning](https://arxiv.org/pdf/2502.15280).
  - Launch CNN & Transformers
### Continous problem
  - No Idea
### Discrete problem
 - Use [Lazy MDP](https://arxiv.org/pdf/2203.08542) for discrete actions (C51) => Not a great idea, because in the current reward used we have already been encouraging ro reduce the amount of drug by $(1 - \alpha) \left(1 - \frac{d_{1,t} + d_{2,t}}{2} \right)$.
 - Increase the number of actions from  \[0, 0.5, 1\] to $[0,0.1,0.2,...,1]$ for each drugs that implies $11^{2}$ classes instead of $3^{2}$ actions

## Code cleaned
  - Improved the code, more readable (save_img, writer)
  - New reward which combines the both reward
  - Simulations for $\alpha=0.5$ -> not adding as much drugs

## Done
- [x] Save entropy for SAC
- [x] Launched with image
- [x] Launched with C51 from 9 classes to 121 classes
I found [high difference between the Q values](https://wandb.ai/corporate-manu-sureli/SAC_IMAGE_COMPLEX_TME/reports/Layer-Norm-or-Hyperspectral-on-Weights-vs-without--VmlldzoxMjg4Mjk1Nw) using [Normalization and Effective Learning Rates in Reinforcement Learning](https://arxiv.org/pdf/2407.01800) and [Hyperspherical Normalization for Scalable Deep Reinforcement Learning](https://arxiv.org/pdf/2502.15280)

## Ideas to simply the problem
### Modify the environment
  - Reduce the stochasticity by modifying the size of the environment (reducing)
  - Decrease the radius where cancer cells can appear (initial state)
  - Fix the inital state instead to have a uniform distribution
### Improve C51 into IQN
 - C51 was already implemented but can we use [IQN](https://github.com/BY571/IQN-and-Extensions/blob/master/IQN-DQN.ipynb) or [RAINBOW](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py)

## Done
 - [x] Much faster replay buffer with numba and jit, jax replay buffer is slowed (mainly caused by the fact the code implies numpy and torch), besides 0.1sec is lost between CPU and GPU, a sample time in around less than  0.13 seconds in CPU while in GPU is around 0.23 seconds for a batch size equals to 128. The batch size has an impact on the performance on the replay buffer
 - [x] Adding a new state the Transformer state is a dictionnary composed of position, type and if the cell is dead
## In progress
 - [ ] Reading about [Temporal Credit Assignment in DRL](https://arxiv.org/pdf/2312.01072) ( our problem is refering to)
## Idea 
WHat is the impact of set of actions to contribute to a realization ? In our case, the set of actions is the set of drugs introduced and the realization the complete or almost complete eradication of cancer cells. This is the credit assignment, to map actions to an outcome under delay, partial observability, stochasticity from the MDP and the environment. [Phd thesis from Johan Ferret](https://theses.hal.science/tel-03958482/document)
# 26 May
## Done
 - [x] Add Transformers Layers in utils
 - [x] Launch SAC with Transformers (Sureli11)
 - [x] Added tumor immune base
 - [x] Launch SAC with Transfomers with the new tumor immune base (Sureli9)
 - [x] Building the replay buffer for the Transformer state 

## :warning: Important Problems :warning:
Be aware of the path used in the PhysiCell_settings.xml file ! 
Replace **Basic_Agent::release_internalized_substrates** in the file **BioFVM_basic_agent.cpp** by 
```cpp
void Basic_Agent::release_internalized_substrates( void )
{
	Microenvironment* pS = get_default_microenvironment(); 
	
	// change in total in voxel: 
	// total_ext = total_ext + fraction*total_internal 
	// total_ext / vol_voxel = total_ext / vol_voxel + fraction*total_internal / vol_voxel 
	// density_ext += fraction * total_internal / vol_volume 
	
	// std::cout << "\t\t\t" << (*pS)(current_voxel_index) << "\t\t\t" << std::endl; 
	*internalized_substrates /=  pS->voxels(current_voxel_index).volume; // turn to density 
	*internalized_substrates *= *fraction_released_at_death;  // what fraction is released? 
	
	// release this amount into the environment 
	if ((*pS)(current_voxel_index).size() ==6)
	{
		(*pS)(current_voxel_index) += *internalized_substrates; 
	}
	
	// zero out the now-removed substrates 
	
	internalized_substrates->assign( internalized_substrates->size() , 0.0 ); 
	
	return; 
}
```
That avoids a Segmentation Fault!

## Results (new model)
With the new model, the drug transforms M2 (pro tumor) to M1 (anti tumor).
The goal is to reduce the tumor size while minimizing the amount of drug added.
I tested several different reward functions. The plot shared on Slack corresponds to the following reward function:
```math
r(t) = -d_t*(1-\alpha) + \alpha*10 \cdot \mathbb{1}_{\{C_t = 0\}}
```
with $d_{t}$ the drug amount (the action) and $C_t$ the number of cancer cells, $\alpha=0.8$
The agent learns something, but it was not expected.
It fails to discover a good policy that maximizes the expected discounted cumulative return by eliminating all cancer cells while a random policy can sometimes achieve this. The policy found is suboptimal. The learning agent should discover a treatment regime that eliminates all cancer cells while minimizing drug usage, thus earning the final reward of 10 points. However, this objective might be too ambitious.
The agent can earn at most $10*\gamma**(100)$ almost equals to $3.66$ where $\gamma=0.99$ represents the discounted factor and $100$ the number of steps. The agent has to at least to add enough drugs to kill all cancer cells but that may imply a discounted cumulative return related to drugs higher than $10*\gamma**(100)$ even though the terminal reward is missed. 
So, from the agent’s perspective (based on its Q-values), it is better to not administer any drugs, since it can at least aim for the 10-point reward if all cancer cells disappear—despite this being unlikely.

As a result, the agent stucks in a local policy, essentially trading off the cost of adding zero drugs with the low-probability chance of earning a large reward.


A solution to that is to increase the term from 10 to 100 which implies $100*\gamma**(100)$ almost equals to $36.6$.
I also added a new term to help the agent  
```math 
\mu(t) = (-\mathbb{1}_{\left\{ C_t \geq C_{t-1} \right\}} + \mathbb{1}_{\left\{ C_t < C_{t-1} \right\}}).
```
Finally, the reward is:
```math 
r_{1,t} = \alpha(\mu(t) + 100 \cdot \mathbb{1}_{\{C_t = 0\}})+ -d_t*(1-\alpha)
```
With this reward, results can be better. 
Learning agent found a good policy ![strategy.png]: it consists of adding a lot of drugs in the half first steps and then letting M1 macrophages kill the cancer cells.
Why adding a lot of drugs at the beginning and not at the final steps? This is explained by the environment and the reward function. In fact, there is a rule that allows M1 to transform into M2 due to pressure. At the beginning, there are around 512 cancer cells, and globally, there is more pressure in the environment compared to the same environment with fewer cancer cells.

Thus, a good strategy to kill all cancer cells while not adding too many drugs would be to act early in the episode to prevent M1 from transforming into M2, allowing M1 to kill a large number of cancer cells. Then, the treatment can be stopped, letting M1 finish killing the remaining cancer cells, with the advantage that M1 is less likely to transform into M2 due to the lower number of cells.

Even if some M1 macrophages transform into M2, it is not a problem because the killing rate of M1 can be seen higher than the sum of the division rate of the remaining cancer cells and the probability of M1 transforming into M2 (which depends on the pressure).

However, sometimes the discounted cumulative return is not high because a single cancer cell remains alive at the end of the episode. 

Despite this, the curves related ![returns_length.png] to returns (discounted cumulative return and cumulative return) seem flat. However, it is important to keep in mind that the framework (RL) aims to maximize the discounted cumulative return, so it is more relevant to focus on that.

Finally, despite the flat curves, something has been learned. Changing the reward parameters could be a way to obtain a better-shaped curve. Alternatively, we can "sell" our product by saying: "You have an environment, and you can find a policy for your problem that aims to maximize the discounted cumulative return."

I also propose a new reward model without $\alpha$ and which seems relevant in our environment composed at the beggingin of $512$ cancer cells.
```math
r_{2,t}=-\frac{\log(C_{t}+1)}{\log(100)}e^{d_{t}-1}
```
We have a magnitude between 1.5 and 0 for $\frac{\log(C_{t}+1)}{\log(100)}$ and $e^{d_{t}-1}$ a magnitude between 1.0 and 0.36.
I will also launching with the last rewards used. I did not have expected results with $r_{2}$, i may have a problem of magnitude. The policy learnt does not try to kill all cancer cells but it seems keep to a certain number of cancer cells, and avoids to add drug.
# 2 June
## Done
 - [x] Launch SAC with image with the reward called $r_{1}$ => better results in terms of mean episodic return and discounted cumulative return
 - [x] Analysis different policies with $r_{1}$ badly called sparse reward w ehave different policies given different states susch as image and scalars
 - [x] Push on github sac_tib.py one file as pre-tutorial
 - [x] New model added
 - [x] videos created for $r_{3,t}$ and $\alpha = 0.3$ see below
## Simple reward
```math
r_{3,t} = \alpha*\mathbb{1}_{\{C_t\ge C_{t-1}\}} -d_t*(1-\alpha)
```

```math
r_{4,t} = \alpha*(\mathbb{1}_{\{C_t\ge C_{t-1}\}}-\mathbb{1}_{\{C_{t-1} \gt C_{t}\}}) -d_t*(1-\alpha),
```

## To Do
- [ ] Write the tutorial for sac_tib, no details, explain important things, such as explain how to create an account, add pip install for numba, torch, tensorboard, tensordict, wandb
 - [In progress] Launch with different rewards function $r_{1}$ seems a good policy but can be improved
 - [In progress] Analysis different policies
 - [ ] Launch on C51 with the $r_{1}$ (image and concentration)
 - [In progress] (in progress) Try to make more generic, SAC/C51 code into utils and call it ( done for SAC not for C51 yet)
 - [ ] How to add wrapper into physicell_model (specified for each model)
 - [In progress] Add documenation from RL package, for all python functions used
 - [In progress] [Into a new repository](https://github.com/Dante-Berth/rllib)
 - [ ] Add test codes to avoid any problems
 - [ ] Use pip install to install the new lib
 - [ ] Create two tutorials, teach how to use SAC, C51, RL
## PhysiNA (Neural Architectures)
 - [ ] [Add](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html#torch.nn.utils.spectral_norm)
 - [ ] Solve Transformers memory
 ## PhysiTCA (Temporal credit assignment)
 - [ ] Add SAIL: Self-Imitation Advantage Learning into my C51
 - [ ] Adapt the code SAIL+C51

## To Do (not now)
 - [x] Check for Cmake does not work when you use make install_requirement
 - [ ] Add SAIL: Self-Imitation Advantage Learning into my C51
 - [ ] Adapt the code SAIL+C51


## 📋 Tumor Shrinkage Reward Functions with Drug Penalty


---

### 1. **Linear Tumor Shrinkage – Drug Penalty**

```math
r_t = \gamma \cdot \frac{C_{t-1} - C_t}{512} - \alpha \cdot d_t
```

- Encourages absolute reduction in tumor size  

---

### 2. **Exponential Tumor Reduction Bonus**

```math
r_t = \gamma \cdot \left(e^{-C_t / 100} - e^{-C_{t-1} / 100}\right) - \alpha \cdot d_t
```

- More aggressive reward when tumor size is already low

---


---
### 3. **Weighted Linear Tumor Shrinkage - Drug Penalty**

```math
r_t = \alpha \cdot \frac{C_{t-1} - C_t}{\log(C_{t-1}+1)} - (1-\alpha) \cdot d_t
```

- Encourages weighted reduction in tumor size 

### 4. **Relative Tumor Shrinkage - Drug Penalty**

```math
r_t = \alpha \cdot \frac{C_{t-1} - C_t}{C_{t-1}} - (1-\alpha) \cdot d_t
```

# 9 June
 - [x] Launched with Weighted Linear Tumor Shrinkage - Drug Penalty (almost great)
 - [x] Launched with Relative Tumor Shrinkage - Drug Penalty (not great)
 - [x] rename variables from M1, M2, pro-inflammatory factor, anti-inflammatory factor to cell_1, cell_2, anti-tumoral factor, pro-tumoral factor 
 - [x] Tutorial
# To do
Goes to 2 June

## 📘 Reward Functions

Let:

- $C_t$: number of cancer cells  
- $d_t$: quantity of administered drug  
- $d_t \in [0, 1]$  
- $C_t \in [0, 700]$

---

### Reward Function $r_1$ **Sparse Tumor Shrinkage - Drug Penalty**

$r_1(t) = -d_t + \mathbb{1}_{C_{t-1}=0} \cdot \beta$

$\sum_{t=0}^{T} \gamma^t r_1(t) = -\sum_{t=0}^{T} \gamma^t d_t + \beta \sum_{t=0}^{T} \gamma^t \mathbb{1}_{C_{t-1}=0}$

Assume:
- $d_t = 0.5$
- $\beta$ is constant

Then:

$\sum_{t=0}^{T} \gamma^t r_1(t) = -\frac{1}{2} \sum_{t=0}^{T} \gamma^t + \beta \sum_{t=0}^{T} \gamma^t$

$= -\frac{1}{2} \cdot \frac{1 - \gamma^{T+1}}{1 - \gamma} + \beta \cdot \frac{1 - \gamma^{T+1}}{1 - \gamma}$

---

### Numerical Example

Let $\gamma = 0.99 \Rightarrow 1 - \gamma = 0.01$, and $T \in [100, 200]$

Then:

$\gamma^T \in [0.134, 0.366] \Rightarrow \frac{\gamma^T - 1}{2(1 - \gamma)} \in [-43.3, -31.695]$

We want to find a $\beta$ of the same order of magnitude:

$\frac{\gamma^T - 1}{2(1 - \gamma)} + \frac{1}{\gamma^T} \in [-323, -86] \Rightarrow \beta \in [86, 323] \Rightarrow \text{Order of magnitude: } \mathcal{O}\left( \frac{2}{\gamma^T} \right)$

Assume $\beta = 100$ points.

If $d_t$ changes, then:

$[-323, -86] \times \frac{2}{\bar{d}_t}$

Where $\bar{d}_t$ is the mean quantity of drug administered under a random policy.

$\left( \frac{1 - \frac{1}{\gamma^T}}{1 - \gamma} \right) \cdot \frac{1}{d_t}$

Must have the same order of magnitude as $\beta$.

---

## 📙 Reward Function $r_2$

$r_2(t) = \frac{C_{t-1} - C_t}{\log(C_{t-1} + 1)} - d_t$

**Notation:**

$C_{t-1} - C_t = \Delta C_t, \quad C_{t-1} + 1 > 2 \quad \text{(since if } t = 0 \Rightarrow t = T)$

- $\log(C_{t-1} + 1)$ represents tumor mass.
- The difference depends on tumor size. For a tumor with 10,000 cells, a reduction of 3 cells is negligible compared to a tumor with 10 cells.

$\Delta C_t \in [-10, 10], \quad \log(C_{t-1} + 1) \in [0.693, 6.55] \Rightarrow \text{gives an order of magnitude close to 1}$

---

### Reward Function $r_3$

$r_3(t) = \frac{C_{t-1} - C_t}{C_{t-1}} - \beta d_t$

If $\beta = 1$:

$r_3(t) = \frac{\Delta C_t}{C_{t-1}} - d_t$

We know relative variations are bounded by ±50%:

- A tumor with 1 cell can be completely eliminated (100% reduction).
- A larger tumor can be reduced by at most 50%.

We must adjust for $C_{t-1}$, but this leads back to $r_2(t)$:

$r_1(t) \geq 2 (C_{t-1} - C_t) - \beta d_t \quad \text{with } \beta = 1$

$\Rightarrow = -2 \Delta C_t - \beta d_t$

$\Rightarrow \Delta C_t \in [-10, 10]$

$\Rightarrow \text{So } 2 = \frac{1}{10}$

---

## ✅ Conclusion

For $r_1$, $r_2$, and $r_3$, the reward functions appear similar and not particularly difficult to manage.

The main challenge lies in **choosing the right coefficients**.

