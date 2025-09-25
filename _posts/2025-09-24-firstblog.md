---
title: "Form REINFORCE to PPO: Classic Policy Gradient Methods revisited"
date: 2025-09-23
---

## **Stitching the narrative arc**
When I started studying reinforcement learning, I found blogs to be a very good medium for gaining intuition. There are as many ways of presenting ideas as there are bloggers online. Yet I sometimes find myself struggling to see how the various algorithm boxes fit together into a coherent picture. Inspired by other blogs in the area (@Seita's place and @lil'log), and the recognition how a coherent narrative is conductive to deeper understanding, I decided to write down my own learning diary here. My aim is threefold: to fully understand the logic by writing it down, to create a scaffold I can revisit in the future, and to share the process with anyone who might find value in it. I try to keep the flow didactic, the language casual, and the math tight. There are bound to be typøs, despite my best intentions. All notations apart from the typøs follow Andrew Barto and Richard Sutton's Introduction to Reinforcement Learning.

Policy gradients and value-based are two families of methods. Suppose we want to arrive at a place, I tend to think of value-based as drawing a map, whereas policy gradients is like building intuition. With that, policy gradients handle stochastic policy more naturally. Here we focus on the policy gradients family, specifically how the vanilla REINFORCE logically develops into PPO.

The fundamental goal of Policy Gradients is to select a $\theta = \arg \max J_{\pi}(\theta)$ , namely, find a parameter $\theta$ for policy $\pi_\theta$ that maximizes $J_\pi(\theta)$ , which is (on average) how much rewards an episode collects under a policy $\pi_{\theta}$.

$$
J_{\pi}(\theta) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^{t}r(s_t, a_t)\right] = \mathbb{E}_{\tau \sim \pi_\theta}\left[r(\tau)\right]
$$

## **How do we maximize this objective?**

We can use $\theta$ to take gradient ascent steps w.r.t $J_\pi$ : this step can be derived neatly from definition, or in[ Berkeley CS285 Notes.](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/slides/lec-5.pdf)

$$
\begin{align}
\nabla_{\theta} J_{\pi}(\theta) &= \int p_{\theta}(\tau)r(\tau)d\tau\\
&= \mathbb{E}_{\tau\sim\pi_{\theta}}\left[ \left( \sum_{t=0}^{\infty} \gamma ^{t}r_{t+1} \right) \left( \sum_{t=0}^{\infty} \nabla \log \pi _{\theta}(a_t| s_t)\right) \right]\\\\
&(\text{REINFORCE})
\end{align}
$$

This means: to find the direction to improve $J$, we sample some episodes, and take the episodic rewards $\times$ preferred direction for this trajectory. One simply REINFORCES trajectories that leads to higher yields.
## **How to simplify the $r_t$ terms?**

Recall that we have $Q^{\pi}(S_t, A_t)$ defined as discounted sum of rewards after taking $s_t, a_t$ following $\pi$. It'd be nice to plug this in, but right now the summed reward is episode-wide, so some extra work is needed to connect the REINFORCE with value-functions.

$$
\begin{align}
\nabla_{\theta} J_{\pi}(\theta) &= \mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \left( \sum_{t=0}^{\infty} \gamma^{t}r_{t+1} \right) \left( \sum_{t=0}^{\infty} \nabla \log \pi _{\theta}(a_t| s_t)\right) \right] \\ &=
\mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \sum_{t^{\prime}=0}^{\infty} \gamma^{t'}r_{t^{\prime}+1} \sum_{t=0}^{t'} \nabla \log \pi _{\theta}(a_t| s_t) \right] \\
&= \mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \sum_{t^{\prime}=0}^{\infty}\nabla \log \pi _{\theta}(a_t| s_t)\sum_{t=t'}^{\infty}\gamma ^{t}r_{t+1} \right]  
\end{align}
$$

This is a trick using the fact: $ \mathbb{E}_{s_{0:\infty}, a_{0:\infty}}[r(t)] = \mathbb{E}_{s_{0:t}, a_{0:t}}[r(t)] $, namely: the expectation of a certain reward step does not depend on the transition steps beyond it. So for each $r_t$ we may only keep the trajectory transition leading up to it. This helps us rewrite the sum of $r_t+$ as an estimated $t$-th step Monte Carlo estimate $G_t$:

$$

\begin{align}

\nabla_{\theta} J_{\pi}(\theta) &=

\mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \left( \sum_{t=0}^{\infty} r_{t+1} \right) \left( \sum_{t=0}^{\infty} \nabla \log \pi _{\theta}(a_t| s_t)\right) \right] \\&= \mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \sum_{t=0}^{\infty}\nabla \log \pi _{\theta}(a_t| s_t)\gamma^{t}G_t \right]

\end{align}

$$

Intuitively, this sums up every single gradient for all steps in all episodes, and each gradient is scaled by how good its step is. It assigns good scores to advantageous state-actions. Since $G_t$ is the Monte Carlo estimate of a single rollout,  we have the following relationship: 
$$\mathbb{E}_{s_t,a_t \sim \pi_{\theta}}\left[G_t \mid(s_t, a_t)\right] = Q^{\pi}(s_t, a_t)$$
