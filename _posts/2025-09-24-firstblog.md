---
title: Form REINFORCE to PPO - Classic Policy Gradient Methods Revisited
date: 2025-09-24
---
# Form REINFORCE to PPO: Classic Policy Gradient Methods revisited

## **Stitching the narrative arc**
When I started studying reinforcement learning, I found blogs to be a very good medium for gaining intuition. There are as many ways of presenting ideas as there are bloggers online. Yet I sometimes find myself struggling to see how the various algorithm boxes fit together into a coherent picture. Inspired by other blogs in the area (@Seita's place and @lil'log), and the recognition that a coherent narrative is conductive to deeper understanding, I decided to write down my own learning diary here. My aim is threefold: to fully understand the logic by writing it down, to create a scaffold I can revisit in the future, and to share the process with anyone who might find value in it. I try to keep the flow didactic, the language casual, and the math tight. There are bound to be typøs, despite my best intentions. All notations apart from the typøs follow Andrew Barto and Richard Sutton's Introduction to Reinforcement Learning.

Policy gradients and value-based are two families of methods. Suppose we want to arrive at a place, I tend to think of value-based as drawing a map, whereas policy gradients is like building intuition. Value-based RL learns action values $Q(s,a)$ and induces a policy by acting near-greedily (i.e. $\epsilon$-greedy). Some algorithms for value-based methods: SARSA, Q-Learning, DQN, etc. 

Contrarily, the **policy-based** methods parameterizes a policy function $\pi(s,a)$ (which represents the probability of taking a given action) and directly pushes parameter by estimating $J_\theta$. For discrete actions, logits from a network go through softmax; for continuous actions, policies can be Gaussian:

$$\pi_\theta(a|s) = \frac{e^{h_\theta(s,a)}}{\sum_{a’} e^{h_\theta(s,a’)}} \quad \quad \pi_\theta(a|s)=\mathcal N(\mu_\theta(s), \Sigma_\theta(s))$$

A subtle question: what's the difference between softmax-weighing-Q-values and a discrete policy? On the surface, both measures a (state, action) preference; but Q values must satisfy Bellman equation whereas policy do not. Quintessentially, that is why we can later bootstrap value functions in Actor-Critic, and why policy methods suffer from high variance.

Here we focus on the policy gradients family, specifically how the vanilla REINFORCE logically develops into PPO.

The fundamental goal of Policy Gradients is to select a $\theta = \arg \max J_{\pi}(\theta)$ , namely, find a parameter $\theta$ for policy $\pi_\theta$ that maximizes $J_\pi(\theta)$ , which is (on average) how much rewards an episode collects under a policy $\pi_{\theta}$.

$$

J_{\pi}(\theta) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^{t}r(s_t, a_t)\right] = \mathbb{E}_{\tau \sim \pi_\theta}\left[r(\tau)\right]

$$
## **How do we maximize this objective?**

We can use $\theta$ to take gradient ascent steps w.r.t $J_\pi$ : this step can be derived neatly from definition, or per [ Berkeley CS285 Notes.](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/slides/lec-5.pdf)

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

This is a trick using the fact: $\mathbb{E}_{s_{0:\infty}, a_{0:\infty}}[r(t)] = \mathbb{E}_{s_{0:t}, a_{0:t}}[r(t)]$, namely: the expectation of a certain reward step does not depend on the transition steps beyond it. So for each $r_t$ we may only keep the trajectory transition leading up to it. This helps us rewrite the sum of $r_t+$ as an estimated $t$-th step Monte Carlo estimate $G_t$:

$$

\begin{align}

\nabla_{\theta} J_{\pi}(\theta) &=

\mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \left( \sum_{t=0}^{\infty} r_{t+1} \right) \left( \sum_{t=0}^{\infty} \nabla \log \pi _{\theta}(a_t| s_t)\right) \right] \\&= \mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \sum_{t=0}^{\infty}\nabla \log \pi _{\theta}(a_t| s_t)\gamma^{t}G_t \right]

\end{align}

$$

Intuitively, this sums up every single gradient for all steps in all episodes, and each gradient is scaled by how good its step is. It assigns good scores to advantageous state-actions. Since $G_t$ is the Monte Carlo estimate of a single rollout,  we have the following relationship: 
$$\mathbb{E}_{s_t,a_t \sim \pi_{\theta}}\left[G_t \mid(s_t, a_t)\right] = Q^{\pi}(s_t, a_t)$$
## **How does this connect to Actor-Critic?**

For each timestamp, suppose we have an estimate of state values, we can use this as a baseline to reduce variance. Interestingly, this state-based baseline does not affect the bias, so it's always preferable to use it.

$$

\begin{align}

\mathbb{E}\left[ \sum_{t=0}^{\infty}\nabla \log \pi _{\theta}(a_t| s_t)G_t \right] &= \mathbb{E}\left[ \sum_{t^{\prime}=0}^{\infty}\nabla \log \pi _{\theta}(a_t| s_t)(G_t - V^{\pi}(s_t)\right] \\ &= \mathbb{E}\left[ \sum_{t=0}^{\infty}\nabla \log \pi _{\theta}(a_t| s_t)A^{\pi}(s_t, a_t)\right] \\ &= \mathbb{E}\left[ \sum_{t=0}^{\infty}\nabla \log \pi _{\theta}(a_t| s_t)\left(r(s_t, a_t) + \gamma \hat{V}^{\pi}(s_{t+1}) - \hat{V}^{\pi}(s_t)\right)\right]

\end{align}

$$

Introducing state-value baseline has multiple implications: this allows algorithm to reduce variance while keeping unbiased, leads to actor-critic style implementation, and introduces advantage function. When writing G-V down as TD error, we can directly bootstrap V function. In actor-critic implementation, this TD error can be used to improve the value function estimation by taking a gradient descent step. I use $\hat {V}^{\pi}_{\phi}$ to emphasize that it's a neural network parameterized by $\phi$:
$$

\phi \leftarrow \phi + \alpha^{\phi} \left[r(s_t, a_t) + \gamma \hat{V_{\phi}}^{\pi}(s_{t+1}) - \hat{V}^{\pi}_{\phi}(s_t) \right]\nabla \hat{V}^{\pi}_{\phi}

$$
## **Why is any value function unbiased as a baseline?**
This is one of the results that initially come across as very surprising for me, but also **undergirds the whole notion of advantage function and actor-critic**. To think about advantage function in the broader sense, it is a measure of how good a state-action is compared with other actions from the state. Given a state-wise baseline $b(s_t)$, we can disentangle the state and action distribution:
$$\begin{align} \mathbb{E}_{s_t, a_t \sim\pi(\cdot|\cdot)}\left[ \sum_{t=0}^{\infty}\nabla \log \pi_\theta(a_t | s_t) b(s_t)\right] &= \sum_{t=0}^{\infty} \mathbb{E}_{s_t\sim p_\theta(s)} \left[ b(s_t)\left(\mathbb{E}_{a \sim \pi(\cdot | s_t)}\left[  \nabla\log\pi_{\theta}(a|s_t)\right] \right)\right] \\ 
&= \sum_{t=0}^{\infty} \mathbb{E}_{s_t\sim p_\theta(s)}\left[b(s_t) \sum_a \nabla \log\pi_{\theta}(a|s_t)\pi(a|s_t)\right] \\ 
&= \sum_{t=0}^{\infty} \mathbb{E}_{s_t\sim p_\theta(s)}\left[b(s_t) \nabla \sum_a  \pi(a|s_t)\right] = 0
\end{align}
$$
so whichever value function we use, it will be an unbiased baseline. However, there is a caveat: When we use TD difference to estimate $G_t$ and **bootstrap** a future $V(s_{t+T})$, this is will no longer be unbiased, though the variance can be further reduced as a tradeoff. 

## **Now we're doing SGD, but can we rewrite an optimization problem?**
(now going into TRPO)
Looking back, we wanted to maximize $J_{\pi}(\theta)$ but that's intractable, so we resorted to using stochastic gradient descent along the steps. But if we fix an old $\theta$, finding new $\theta'$ becomes another optimization problem. At $\theta$, the gradient for $J(\theta)$ is:

$$

\begin{align}

  

\nabla_\theta J(\theta) 

  

&= \mathbb{E}_{s,a \sim \pi_\theta}\!\left[ \nabla_\theta \log \pi_\theta(a|s)\, A^{\pi_\theta}(s,a) \right] \\

  

\end{align}

$$

Notice how we are not assessing $J(\theta)$ directly. Now mentally we may construct another optimization problem: suppose we stand firmly on $\theta$, and look around for a $\theta'$ that's "better". Instead of rolling out an $J(\theta')$ estimate for each $\theta'$, We may construct a surrogate objective $L_{\pi_\theta}(\theta')$ to maximize:
$$L_{\pi_\theta}(\theta') =\mathbb{E}_{s,a \sim \pi_{\theta}}\!\left[ \frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)} \, A^{\pi_{\theta}}(s,a) \right] $$
Note that this is linear to $\pi_{\theta'}(a|s)$, and it just re-weights the advantage estimates according to how much probability the new policy puts on each action. (How to interpret this geometrically I am still pondering). It does not touch $J$ in value (easy to prove $L_{\pi_\theta}(\theta) = 0$) but it does so in slope: when we take gradient at $\theta' \rightarrow \theta$, $\nabla_{\theta'}L_{\pi_\theta}(\theta')$ coincides with $\nabla_\theta J(\theta)$.
$$\nabla_{\theta'}L_{\pi_\theta}(\theta')\big|_{\theta'=\theta} = \mathbb{E}_{s,a \sim \pi_{\theta}}\!\left[ \frac{\nabla_{\theta'}\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)} \, A^{\pi_{\theta}}(s,a) \right] =\mathbb{E}_{s,a \sim \pi_\theta}\!\left[ \nabla_\theta \log \pi_\theta(a|s)\, A^{\pi_\theta}(s,a) \right] = \nabla_\theta J

  

$$
The intuition is that, locally this new-policy function approximate $J$ function gradient-wise. Which is what surrogate is for: a acting substitute, an ersatz. **Instead of taking a vanilla policy gradient or natural gradient step, which could be unstable, this new approach allows us to find a preferable *in the vicinity (Constrained by KL divergence, see below)* of $\theta$, while still measuring in the trajectories and advantages of the old policy.**  After an aside on off-policy I'll go into details on this vicinity constraint. 

(P.S.: Vanilla PG is unstable because the gradient is only valid locally; if the update is too large, the state distribution shifts, variance explodes, and the step may actually decrease true performance)

## **Wait.. so this surrogate is evaluating one policy based off of another, looks a bit like off-policy learning, no?**

(aka: where the $L_{\pi_\theta}(\theta')$ form comes from)

Both uses **importance sampling** ratio. My understanding is, off-policy pg seeks to improve an existing policy by sampling another (perhaps more exploratory) policy, whereas TPRO takes a fixed policy and estimate another surrounding policy from it. 

*A brief recap on off-policy policy gradients:* we have a target $\pi_\theta$, its advantage function $A^{\pi_\theta}$, and sampling from another policy $\mu$.  Intuitively, If a (state, action) is much rarer in sampling policy $\mu$ than target $\theta$, and since we are taking an expectation of advantage-weighted grad-logs, then we must magnify the effect of such samples. Thus, the policy gradient is:
$$
\nabla_\theta J(\theta)
= \mathbb{E}_{s,a\sim \mu} \left[ \frac{\pi_\theta(a|s)}{\mu(a|s)} \nabla_\theta \log \pi_\theta(a|s)\, A^{\pi_\theta}(s,a) \right].
$$
Aside of an aside: why didn't I write discount term here? It is implicit in the -- state distribution subscript. 

*Back to the TRPO objective: given one policy $\theta$,* how to measure the performance of another policy $\theta'$ ? Starting from an identity: [proof](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/slides/lec-9.pdf) of this utilizes the fact that starting state distribution is the same under different policies. 
$$
J(\theta ') = J(\theta) + \mathbb{E}_{s,a \sim \pi_{\theta'}}\left[ \sum_{t=0}^{\infty}\gamma^{t}A^{\pi_\theta}(s_t,a_t)\right]
$$
And directly following this, 

$$
\begin{aligned}
J(\theta') - J(\theta) &= 
\sum_{t=0}^\infty \mathbb{E}_{s_t \sim p_{\theta'}(s_t)}\left[ \mathbb{E}_{a_t\sim\pi_{\theta'}(\cdot \mid s_t)}\left[ \gamma^t A^{\pi_\theta}(s_t,a_t)\right]\right] \\
&= \sum_{t=0}^\infty \mathbb{E}_{s_t \sim p_{\theta'}(s_t)}\left[ \mathbb{E}_{a_t\sim\pi_{\theta}(\cdot \mid s_t)}\left[ \frac{\pi_{\theta'}(s_t,a_t)}{\pi_\theta(s_t,a_t)} \gamma^t A^{\pi_\theta}(s_t,a_t)\right]\right] \\
&\approx \sum_{t=0}^\infty \mathbb{E}_{s_t \sim p_{\theta}(s_t)}\left[ \mathbb{E}_{a_t\sim\pi_{\theta}(\cdot \mid s_t)}\left[ \frac{\pi_{\theta'}(s_t,a_t)}{\pi_\theta(s_t,a_t)} \gamma^t A^{\pi_\theta}(s_t,a_t)\right]\right] \\
&= \mathbb{E}_{s,a \sim \pi_\theta}\left[\frac{\pi_{\theta'} (a|s)}{\pi_{\theta}(a|s)} A^{\pi_{\theta}}(s,a)\right] \\
&= L_{\pi_{\theta}}(\theta') 
\end{aligned}
$$
Nice! From this little transformation we can see that importance sampling factor comes from re-weighing the action when changing policy. And we approximate by assuming state-distribution isn't straying too far (In **chapter 3** there's a [proof](https://arxiv.org/abs/1502.05477) that if we keep KL divergence constrained, finding $\theta^{\star}=\arg \max L_{\pi_\theta}(\theta')$ monotonically improves our $J$ function :)

## **Going deeper in the vicinity...**

What prevents us from arbitrarily defining a new policy that exploits the old advantage function is the fact that this formulation only makes sense locally,. Thus, we need to add a constraint to bound how far $\theta$ can go. In TRPO they use an **averaged** Kullbeck-Leibler Divergence, which measures how different two distributions are. One can bound the KL divergence between old and new policy by $\delta$, and the constrained optimization becomes: 
$$\begin{align}
\max
L_{\pi_\theta}(\theta') &= \mathbb{E}_{s,a \sim \pi_\theta}\left[\frac{\pi_{\theta'} (a|s)}{\pi_\theta(a|s)} A^{\pi_\theta}(s,a)\right]\\ \\
s.t. \quad \mathbb{E}&_{s\sim \eta(\theta)} [D_{KL}(\pi_{\theta}(\cdot\mid s), \pi_{\theta'}(\cdot\mid s)) ] \leq \delta

\\\\ &\text{(Trust Region Policy Optimization)}
\end{align}
$$
Some extra thoughts here, not necessarily correct. If we construct an Lagrangian for this, since $\theta'$ is close to $\theta$, we can simply perform Taylor expansion around the old policy $\theta$!  KL has a quadratic form locally, and the Fisher matrix acts as its curvature (recall natural gradient), in other words, $F$ is second order approximation for KL divergence.
$$\mathbb{E}_{s\sim \eta(\theta)} [D_{KL}(\pi_{\theta}(\cdot\mid s), \pi_{\theta'}(\cdot\mid s)) ] \;\approx\; \tfrac{1}{2} (\theta’ - \theta)^\top F(\theta)(\theta’ - \theta),$$(Oh, where does the 1/2 come from? Fisher information matrix is Hessian for KL divergence at $\theta' = \theta$, and 1/2 comes from second order Taylor expansion.)

The first term becomes zero, which is intuitive: the trajectories following the same policy doesn't update the advantage expectation. To express more precisely, 

$\mathbb{E}_{a\sim \pi\theta(\cdot|s)}[A^{\pi_\theta}(s,a)] = \sum_a \pi_\theta(a|s)\,(Q^{\pi_\theta}(s,a)-V^{\pi_\theta}(s)) = V^{\pi_\theta}(s)-V^{\pi_\theta}(s) = 0.$

So the expanded version: 

$$

L_{\pi_\theta}(\theta',\lambda)
\approx \underbrace{L_{\pi_\theta}(\theta)}_{=0}
+ \nabla_{\theta'}L_{\pi_\theta}(\theta')(\theta'-\theta)
-\tfrac{\lambda}{2}(\theta'-\theta)^\top F(\theta)(\theta'-\theta)
+\lambda\,\delta.
$$
Intuitively, this means the KL constraint defines an ellipsoid around the old policy, and TRPO finds the best direction inside that trust region. Does the RHS return $\theta^{\star}$ in closed form? That connects surprisingly to convex optimization 🤔 

(P.S. Actually, upon closer reading, the TRPO did mention giving a closed form $\theta'$ using natural policy gradients, but it does not achieve )
## **KL Divergence is expensive, are there easier ways to constrain policy change?**

Finally, we're at the last piece of puzzle. KL divergence requires us to perform second order approximation around the trust region, which can be quite computationally expensive. PPO simplifies this process while retaining similar performance. Instead of using a lagrangian to regularize, PPO uses a clipped objective. $r_t(\theta')$ is short for:
$$r_t(\theta') = \frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_{\theta}(a_t \mid s_t)}$$
$$\max L^{\text{CLIP}}(\theta') 

  

= {\mathbb{E}}_t \left[\min \big(r_t(\theta')\,{A}_t, \;\text{clip}(r_t(\theta'), 1-\epsilon, 1+\epsilon)\,{A}_t

    \big) \right]$$
Which clips the extent to which the new policy can exploit juicy steps. For a step with reward or penalty, we want to nudge the parameter to exploit it; but this is capped by $[1-\epsilon, 1+\epsilon ]$. Another more intuitive way to look at it: We want to slide our $\theta$ in parameter space to maximize $L^{CLIP}(\theta')$. When moving towards one direction, benefits from a certain step hit a ceiling (or a bottom ceiling for negative advantage, which some people call a floor), that gives us leeway (because of zero gradient) to move around the $\theta$ to see if we can gain improvements from other steps - perhaps we'll hit ceiling on them as well, and so on. In this case the large policy change is not penalized - but also not preferred, when there are other options to shift $\theta$ that leads to gains in other steps.

When value function and policy share the same network of parameters, the actor-critic update can be combined into maximizing a single objective:
$$\begin{align}
L^{CLIP+VF+S}(\theta) &= 
\mathbb{E}_t \Big[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \Big] \\\\
&(\text{Proximal Policy Optimiation})\end{align}$$
Where S is the entropy term that functions as a regularizer to encourage sufficiently exploratory policies, and is independent of the reward signals. From the actor-critic perspective, we may use the same $\lambda$ for GAE(Generalized Advantage Estimator) (policy) and TD($\lambda$) (value function), which controls how the reward signals flow into objective function. Those are all tricks about finding balance between Monte Carlo and TD(0). 
## **Closing note: Bias-Stability-Variance tradeoff**
From vanilla REINFORCE, we have finally arrived at PPO — a method that openly embraces **controlled bias** in exchange for stability and variance reduction.
- **Surrogate objective (TRPO/PPO)**: The surrogate L is already an approximation to the true objective J. That approximation introduces bias, but in return gives a far more stable update rule than naïve gradient ascent.
- **Clipping**: By truncating large probability ratio changes, PPO deliberately biases the gradient estimate. The gain is stability: the policy is disinclined to make large, destructive shifts in action probabilities.
- **GAE**: Generalized Advantage Estimation interpolates between high-variance Monte Carlo returns and high-bias one-step TD. It trades a bit of bias for substantially lower variance in practice.
- **Entropy bonus**: The entropy acts as a regularizer. It adds a bit of bias to the objective function, but in return it increases stability by making the policy more exploratory. 


  
  
  

#blog
