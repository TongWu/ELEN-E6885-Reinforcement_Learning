# ELEN6885 Final Exam Cheat Sheet

<div align = "center"><font size = 5> Tong Wu, tw2906 <div>

**Function Approximation:** **Motivation: **curse of dimensionality (how to leverage RL to achieve optimal control with the exponential growth of states and actions). **Solution (for large MDP): **Estimate value function with function approximation. 

**Gradient Descent:** Let $J(w)$ be a differentiable function of parameter vector $w$, a column vector, the gradient of $J(w)$ to be: $\nabla_wJ(w)=(\frac{\delta J(w)}{\delta w_1},...,\frac{\delta J(w)}{\delta w_n})^T$, $\Delta w=-\frac12\alpha\nabla_wJ(w)$

**Stochastic Gradient Descent 随机梯度下降法:** Find parameter vector $w$ minimising mean-squared error between approximate value and true value. $J(w)=\mathbb E_{\pi}[(v_{\pi}(S)-\hat v(S,w)^2)]$. 

**Gradient descent** finds a local minimum $\Delta w=-\frac12\alpha\nabla_wJ(w)=\alpha\mathbb E_{\pi}[(v_{\pi}(S)-\hat v(S,w))\nabla_w\hat v(S,w)]$.

**Stochastic gradient descent samples the gradient:** $\Delta w=\alpha(v_{\pi}(S)-\hat v(S,w))\nabla_w\hat v(S,w)$

**RL prediction with value approximation**

**MC: target is the return $G_t$** -> $\Delta w=\alpha(G_t-\hat v(S_t,w))\nabla_w\hat v(S_t,w)$

**TD(0): target is the TD target** -> $\Delta w=\alpha(R_{t+1}+\gamma\hat v(S_{t+1},w)-\hat v(S_t,w))\nabla_w\hat v(S_t,w)$

**TD($\lambda$): target is the $\lambda$-return $G_t^{\lambda}$** -> $\Delta w=\alpha(G_t^{\lambda}-\hat v(S_t,w))\nabla_w\hat v(S_t,w)$

**T/F**

**Finite MDP / MDP**

Reinforcement learning uses the **formal framework** of Markov decision processes (MDP) to define the interaction between a learning agent and its environment in terms of **states**. (F19F)(F21M)

MDP instances with **small **discount factors tend to emphasize (prefer) **near-term** rewards. (F19F)(F21M)

If the only difference between two MDPs is the value of discount factor $\gamma$, then they **may have the same or different** optimal policy. (F19F)(F21M)

For an infinite horizon MDP with a finite number of states and actions and with a discount factor $\gamma\in(0,1)$, value iteration **can guaranteed to converge**. (F19F)(F21M-S)

MDP is **not a mathematical formalisation** of an agent. (F19M)

Assume MRP model is known, then the bellman equation for MRP **can be solved** in a closed matrix form. However, direct solution is **only possible** for small MRPs. (F19M-S)

**$\epsilon$-greedy, SoftMax, UCB**

Incremental implementation is **efficient and memory-saving**.

Compared to $\epsilon$-greedy and SoftMax, UCB does not explore by sampling but rather inflates 夸大 the estimated expected **payout **according to its **uncertainty**. (F19F)

**Policy Iterations / Value Iterations**

Policy and value iterations are **not clear** of which is better. (F21M)

**GPI**

Generalized policy iteration (GPI) is a term used to refer to the general idea of letting **policy-evaluation and policy improvement processes interact**, independent of the granularity and other details of the two processes. (F21M)(S22M)

**Monte Carlo**

Typically, each iteration of a basic version of Monte Carlo Tree Search (MCTS) consists of four steps: **selection, expansion, simulation and backup**. (F19F)

In the case each return is an independent, identically distributed estimate of the value function v(s) with finite variance, the convergence of first-visit Monte Carlo method to the true value v(s) as the number of visits of each s goes to infinity is guaranteed by following the law of large numbers. (F19M-S)

In general, Monte Carlo method is **better on the existing data** (e.g. rewards collected from the episodes) while TD methods produce lower error on future data. (F19M-S)

**$\lambda$ return, TD($\lambda$)**

The on-line $\lambda$ return algorithm is **not equivalent** to the on-line TD($\lambda$) in terms of the total update to a value function at the end of an episode. (F19F)

**SARSA / Q-Learning**

Expected SARSA **may not has higher variance** in its updates than SARSA. (F19M-S)

**Deep Q-Network (QDN)**

In deep Q-Network, experience replay **randomises** over the data, thereby **removing correlations** in the observation sequence and **smoothing **over changes in the data distribution. (F19F)

**SA**

**Discount Factor**

Q: For a gird world, using reward formulation to receive reward +1 on reaching the goal state and 0 for others. If two variants of this reward formulation: (P1) use discounted returns with $\gamma\in(0,1)$. (P2) no discounting is used. As a conclusion, a good policy can be learnt from (P1) but not from (P2), why? (F19F)

A: In (P2), since no discounting, the return for each episode regardless of the number of steps is +1. This **prevents the agent from learning a policy which tries to minimise the number of steps to reach the goal state.** In (P1), **the discount factor ensures that the longer the agent takes to reach the goal**, the lesser reward it gets. This motivates the agent to find the shortest path to the goal state.

Q: *Purpose of discount factor in infinite horizon problem:* (F19M)

A: Give a higher value to plans that **reach a reward sooner**; bounds the utility of a state. 

Q: *Small discount factor have effect:* (F19M)

A: Discount future rewards more heavily, which means it **prefer immediate rewards**. 

**Q-Value Function**

Q: A robot we cannot access in the real world, but can access the simulation software. We know that the software is built using the transition model $P_{sim}(s,a,s')$ that is different than the transition model $P_{real}(s,a,s')$ used in the robot. Select the new update rule for the Q-Value functions for the real world robot: (F19F)

(a) $Q(s,a)\leftarrow Q(s,a) + \alpha(P_{sim}(s,a,s')[r+\gamma max_{\alpha'}Q(s',a')]-Q(s,a))$

(b) $Q(s,a)\leftarrow Q(s,a) + \alpha(\frac {P_{real}(s,a,s')} {P_{sim}(s,a,s')}[r+\gamma max_{\alpha'}Q(s',a')]-Q(s,a))$

(c) $Q(s,a)\leftarrow Q(s,a) + \alpha(\frac {P_{sim}(s,a,s')} {P_{real}(s,a,s')}[r+\gamma max_{\alpha'}Q(s',a')]-Q(s,a))$

A: (b) is correct. This is the importance sampling idea in off-policy learning. The Q-value function should be weighted by $\frac {P_{real}(s,a,s')} {P_{sim}(s,a,s')}$, so that they are correct in expectation instead of sampling from the correct distribution directly.

**Watkins’s Q($\lambda$) algorithm**

Q: What is the disadvantage of Watkins’s Q($\lambda$) algorithm? (F19F)

A: The disadvantage is the **early in learning**, the eligibility trace will be “cut” (zero out) frequently in little advantage to traces.

**Policy Gradient**

Q: In solving a multi-arm bandit problem using the policy gradient method, are we assured of converging to the optimal solution? (F19F)

A: No. Depending upon the properties of the function whose gradient is being ascended, the policy gradient approach **may converge to a local optimum.**

Q: Compare with the stochastic policy gradient, the deterministic policy gradient (DPG) can be estimated much more efficiency and perform better in a high-dimensional task, i.e., much lower computation cost. Why? (F19F)

A: DPG is the expected gradient of action-value function which **integrates over the state space**, while the stochastic policy gradient **integrates over the state and action space**. 

**A3C**

Q: Benefits of using multiple agents in an asynchronous manner in A3C. (F19F)

A: 

(1) It helps to stabilise the training. Since each agent has its own copy of the environment, agents are allowed to explore different parts of the environment as well as to use different policies at the same time. In other words, different agent will likely experience different states and transitions. Therefore, when agents update the global parameters with their local parameters in an asynchronous manner, the global parameters update will be less correlated than using a single agent.

(2) The nature of multi-threads in A3C indicates that A3C needs much less memory to store experience, which means that no need to store the samples for experience replay as that used in DQN.

(3) The practical advantages of A3C is that it allows training on a multi-core CPU rather than GPU. When applied to a variety of Atari games, for instance, agents achieve a better result with asynchronous methods, while using far less resource than these needed on GPU.

**$\epsilon$-greedy:** 

当一系列动作中有reward是小的，则此次行动是exploration，第一次不算。 (F22M-S)

**Policy Evaluation / Value Iteration (IMPORTANT):**
*For iterative policy evaluation: The value function $v(s)=R(s)+\gamma\sum(P_{ss'}V(s))$, the value $v_{k+1}=R_{\pi}+\gamma P_{\pi}v_k$*.(F19F)
*For iterative value evaluation: **By using Bellman optimality backup**, the value function $v(s)=max(R(s)+\gamma V(s))$, the value $v_{k+1}=max(R_{\pi}+\gamma v_{k(des)})$. $v_{k(des)}$即目的地的上一次value*.(F19F)
Bellman optimality equation: $v_1^*=\gamma max(v_1*,v_2^*),v_2^*=\gamma max(v_1*,v_2^*)$, max中的v_1和v_2取迭代极限。(F19F)

**MC / TD:** 
*One-step TD error:* $\delta_t=R_{t+1+\gamma v(S_t+1)-v(S_t)}$. If true state-value function is used, $\mathbb E[\delta_t|S_t=s]=\mathbb E[R_{t+1}+\gamma v.... |S_t=s]\to=v_{\pi}(s)-v_{\pi}(s)=0$  (F22M-S)(F19F)
*Why TD is bootstrapping method:* TD method bases its update in part on existing estimates. (F19F)

**Rewards:** 

If a constant ‘c’ is added to the rewards with maze running task, the effect to optimal policy: $G_t$ will become $G_t+c*(\frac{1-\gamma^T}{1-\gamma})$. (F22M-S)

**SARSA / Q-Learning**

If greedy is used in SARSA rather than nearly greedy: The agent will get stuck assuming that some actions are worse than the current taking, and will not retry other actions, so it cannot learn. (F19F)

**Application**

**2-state MDP** (F19F)

Consider a 2-state MDP, the row and column represents from-state and to-state. For example, the probability of transition from $s_1 $to $s_2 $by taking action $a_1 $is $0.8$. The reward function is $R_{s_1}^{a_1}=R_{s_1}^{a_2}=1$ and $R_{s_2}^{a_1}=R_{s_2}^{a_2}=0$, discount factor $\gamma=0.5$

| action $a_1$ | $s_1$ | $s_2$ |
| :----------- | ----- | ----- |
| $s_1$        | 0.2   | 0.8   |
| $s_2$        | 0.6   | 0.4   |

| action $a_2$ | $s_1$ | $s_2$ |
| :----------- | ----- | ----- |
| $s_1$        | 0.6   | 0.4   |
| $s_2$        | 0.2   | 0.8   |

(1) Let $v^*_1, \ v^*_2$ denote the optimal state value function of state $s_1,\ s_2$. Write the Bellman optimality equations.

$v^*_1=max(R_{s_1}^{a_1}+\gamma(0.2v^*_1+0.8v^*_2), R_{s_1}^{a_2}+\gamma(0.6v^*_1+0.4v^*_2))=max(1+0.1v^*_1+0.4v^*_2,1+0.3v^*_1+0.2v^*_2)$

$v^*_2=max(R_{s_2}^{a_1}+\gamma(0.6v^*_1+0.4v^*_2), R_{s_2}^{a_2}+\gamma(0.2v^*_1+0.8v^*_2))=max(0.3v^*_1+0.2v^*_2,1+0.1v^*_1+0.4v^*_2)$

(2) Prove $v_1^*\gt v^*_2$

Based on the BOE for $v_1^*, v^*_2$, we have $v^*_1=1+v^*_2\gt v^*_2$

(3) Find optimal value $v_1^*, v^*_2$

Find max value from (1), where $v^*_1=1+0.3v^*_1+0.2v^*_2,\ v^*_2=0.3v^*_1+0.2v^*_2$, then solve the function, get $v^*_1=1.6,\ v^*_2=0.6$

**Policy gradient** (F19F)

Q: Consider $\pi(\alpha;\mu,\sigma)=\frac1{\sqrt{2\pi\sigma^2}}e^{-\frac{(\alpha-\mu)^2}{2\alpha^2}},\ \sigma\gt 0$, compute the following equations.

A: $\frac{\delta ln\pi(\alpha;\mu,\sigma)}{\delta\mu}=\frac{\delta ln\frac1{\sqrt{2\pi\sigma^2}}}{\delta\mu}+\frac{\delta-\frac{(\alpha-\mu)^2}{2\sigma^2}}{\delta\mu}=\frac{\alpha-\mu}{\sigma^2}$

$\frac{\delta ln\pi(\alpha;\mu,\sigma)}{\delta\sigma}=\frac{\delta ln\frac1{\sqrt{2\pi\sigma^2}}}{\sigma\mu}+\frac{\delta-\frac{(\alpha-\mu)^2}{2\sigma^2}}{\delta\sigma}=\frac{(\alpha-\mu)^2}{\sigma^3}$

**Deep Q-Network** (F19F)

(1) Consider the loss function $\mathfrak L_i(\theta_i)=E_{s,a,r,s'p}[(r+\gamma max_{a'}Q(s',a';\theta_i)-Q(s,a;\theta_i))^2]$. Rewrite the loss function $\mathfrak L_i$ with the fixed Q learning target.

$\mathfrak L_i(\theta_i)=E_{s,a,r,s'p}[(r+\gamma max_{a'}\hat Q(s',a';\theta^-)-Q(s,a;\theta_i))^2]$

(2) Rewrite $\nabla_{\theta_i}\mathfrak L_i(\theta_i)$ in terms of $\nabla_{\theta_i}Q(s,a;\theta_i)$

 $\nabla_{\theta_i}\mathfrak L_i(\theta_i)=2E_{s,a,r,s'p}[(r+\gamma max_{a'}\hat Q(s',a';\theta^-)-Q(s,a;\theta_i))\nabla_{\theta_i}Q(s,a;\theta_i)]$

(3) What method is widely used to iteratively find the optimal $\theta$ without computing the expectation?

Stochastic gradient descent (SGD)

**TD(0) / TD($\lambda$)**

![image-20221215181818595](https://images.wu.engineer/images/2022/12/15/image-20221215181818595.png)

Consider a grid-world and the following three episodes from runs of the agent through this grid world:

$(1,1),N,0,(2,1),E,0,(2,2),S,-60,(1,2)$; $(1,1),N,0,(2,1),E,0,(2,2),E,0,(2,3),N,+60,(3,3)$; $(1,1),N,0,(2,1),E,0,(2,2),E,0,(2,3),S,+80,(1,3)$

(1) Using a learning rate of $\alpha=0.1$, and assuming initial Q value is 0, what updates to Q((2,2),E) does on-line TD(0) method make after the above three episodes?

There is no update to this point after three episodes, Q((2,2), E)=0

(2) What updates to Q((2,2),E) does on-line backward-view TD($\lambda$) method make after the above three episodes? $\lambda=0.2$

There is no update to Q((2,2),E) after the first episode. The accumulating eligibility trace of the state-action pair ((2,2),E) is 0,0,1, $\gamma\lambda=0.1$ during the second and the third episode. Therefore, the update to Q((2,2),E) after the second episode is:

$Q((2,2),E)=Q((2,2),E)+0.1\times(60-Q((2,3),N)\times0.1)=0.6$

And the update to Q((2,2),E) after the third episode is:

$Q((2,2),E)=Q((2,2),E)+0.1\times(0+Q((2,3),S)-Q((2,2),E))\times1=0.6-0.1\times0.6\times1=0.54$

$Q((2,2),E)=Q((2,2),E)+0.1\times(80-Q((2,3),S))\times0.1=0.54+0.1\times80\times0.1=1.34$

(3) Consider a feature based representation of the Q-value function: $\hat Q(s,a,w)=x_1(s,a)w_1+x_2(s,a)w_2+x_3(s,a)w_3$, where $x_1(s,a)$ is the row number of the state s, $x_2(s,a)$ is the column number of the state s, and x_3(s,a) = 1,2,3,4 if a is N,E,S,W, respectively. For example, if s=(1,1) and a=N, then $x(s,a)=[x_1(s,a),x_2(s,a),x_3(s,a)]^T=[1,1,1]^T$. Given that all $w_i$ are initially 0, what are their values using on-line backward-view TD($\lambda$) after the first episode? Use $\lambda=0.2$ and learning rate $\alpha=0.1$.

By the definition of features, we have $x((1,1),N)=[1,1,1]^T,\ x((2,1),E)=[2,1,2]^T,\ x((2,2),S)=[2,2,3]^T$. The accumulating eligibility trace is $e_t=\gamma\lambda e_{t-1}+x(s,a)$. So the sequence of eligibility traces in the first episode is $[1,1,1]^T $, $[2.1,1.1,2.1]^T$ and $[2.21,2.11,3.21]^T$. Therefore, the update to weights after the first episode is:

$w_1=0+\alpha\times(-60)\times2.21=-13.26$

$w_2=0+\alpha\times(-60)\times2.11=-12.66$

$w_3=0+\alpha\times(-60)\times3.21=-19.26$

**on-line vs off-line update** (HW4)

(1) Difference between on and off line:

In on-line updating, the update are done during the episode, as soon as the increment is computed. In off-line updating, on the other hand, the increments are accumulated “on the side” and are not used to change value estimates until the end of the episode.

(2) Consider an episode: A,+1,B,+2,A,+1,T from an undiscounted MDP, learning rate $\alpha=0.1$. What is total update to V(A) on-line and off-line every-visit constant-$\alpha$ MC method makes after the episode finishes?

On-line: $V(A)=V(A)+\alpha(4-V(A))=0+0.1\times4=0.4$, $V(A)=V(A)+\alpha(1-V(A))=0.4+0.1\times0.6=0.46$

Off-line: $\Delta V(A)=V(A)+\alpha(4-V(A))=0.4$, $\Delta V(A)=\alpha(1-V(A))=0.1\times1=0.1$, Total update is $0.5$

(3) Total update of online and off-line TD(0)?

On-line: $V(A)=V(A)+\alpha(1+V(B)-V(A))=0+0.1\times1=0.1$, $V(A)=V(A)+\alpha(1+V(T)-V(A))=0.1+0.1\times0.9=0.19$

Off-line: $\Delta V(A)=\alpha(1+V(B)-V(A))=0.1\times1=0.1$, $\Delta V(A)=\alpha(1+V(T)-V(A))=0.1\times1=0.1$, total update is $0.2$

(4) On-line/Off-line forward/backward view of TD($\lambda$)

$G_0^{\lambda}=(1-\lambda)G_0^{(1)}+(1-\lambda)\lambda G_0^{(2)}+\lambda^2G_0^{(3)}=0.5\times1+0.25\times3+0.25\times4=2.25$, $G_2^{\lambda}=G_2^{(1)}=1$

On-line forward: $V(A)=V(A)+\alpha(G_0^{\lambda}-V(A))=0+2.25\times0.1=0.225$, $V(A)=V(A)+\alpha(G_2^{\lambda}-V(A))=0.225+0.1\times(1-0.225)=0.3025$

Off-line forward: $\Delta V(A)=\alpha(G_0^{\lambda}-V(A))=0.1\times2.25$, $\Delta V(A)=\alpha(G_2^{\lambda}-V(A))=0.1\times1$, total update is $0.325$

$E_0(A)=1$ $E_2(A)=1.25$

On-line backward: $V(A)=V(A)+\alpha(1+V(B)-V(A))E_0(A)=0+0.1\times1\times=0.1$, $V(A)=V(A)+\alpha(1+V(T)-V(A))E_2(A)=0.1+0.1\times0.9\times1.25=0.2125$

Off-line backward: $\Delta V(A)=\alpha(1+V(B)-V(A))E_0(A)=0.1\times1\times1=0.1$, $\Delta V(A)=\alpha(1+V(T)-V(A))E_2(A)=0.1+0.1\times1\times1.25=0.225$, total update is 0.325

**Linear function approximation** (HW4)

![image-20221215192451328](https://images.wu.engineer/images/2022/12/15/image-20221215192451328.png)

(1) Consider a discounted experiment with actions right, right, right, left. Reward is -1. For two parameters $\hat q(s,a,w)=x_1(s,a)w_1+x_2(s,a)w_2$, if $w_1=w_2=1$, calculate the $\lambda$-return $q_t^{\lambda}$ corresponding to this episode for $\lambda=0.5$:

$q_1=0.5((-1+1)+0.5\times(-2+1)+0.5^2\times(-3+1))+0.5^3\times(-4)=-1$

$q_2=0.5((-1+1)+0.5\times(-2+1))+0.5^2\times(-3)=-1$

$q_3=0.5(-1+1)+0.5\times(-2)=-1$

$q_4=-1$

(2) Using forward-view TD($\lambda$), right the sequence corresponding to the right action, $\lambda=0.5, \gamma=1$:

$\Delta w_1^1=\alpha(q_t^{\lambda}-\hat q(S_t,A_t,w))\nabla_w\hat q(S_t,A_t,w)=0.5\times(-1-1)\times1=-1$

$\Delta w_1^2=\alpha(q_t^{\lambda}-\hat q(S_t,A_t,w))\nabla_w\hat q(S_t,A_t,w)=-1$

$\Delta w_1^3=\alpha(q_t^{\lambda}-\hat q(S_t,A_t,w))\nabla_w\hat q(S_t,A_t,w)=-1$

$\Delta w_1^4=\alpha(q_t^{\lambda}-\hat q(S_t,A_t,w))\nabla_w\hat q(S_t,A_t,w)=0.5\times(-1-1)\times0=0$

(3) Define TD($\lambda$) trace $e_t$, for the right action, $\lambda=0.5, \gamma=1$: 

Where the linear value function approximation in trace $e_t$ is $e_t=\gamma\lambda e_{t-1}+x(s,a)$

The sequence of eligibility traces corresponding to right action should be:$1$, $\frac32$, $\frac 74$, $\frac78$

(4) Backward-view TD($\lambda$) sequence of updates to weight $w_1$, what is the total update to weight $w_1$? $\lambda=0.5,\gamma=1,alpha=0.5,w_1=w_2=1$

$\Delta w_1^1=\alpha\delta_1e_1=0.5\times(-1)\times1=-0.5$

$\Delta w_1^2=\alpha\delta_2e_2=(-0.5)\times\frac32=-\frac34$

$\Delta w_1^3=\alpha\delta_3e_3=(-0.5)\times\frac74=-\frac78$

$\Delta w_1^4=\alpha\delta_4e_4=(-1)\times\frac78=-\frac78$

(5) When using off-line updates and linear function approximation, are forward is equivalent to backward?

Yes. Forward-view and backward-view TD(λ) is equivalent to each other.

**DQN** (HW5)

Why using “experience delay” and “fixed Q-targets” in DQN? Explain why using these two can help stabilise DQN algorithm when the correlations present in the sequence of observations?

Because the experience replay can step out of the correlation that comes with data, which is called independent and identically distributed random variables environment. The fixed Q-targets then, can also step out the correlation between action values and the target, and stabilise the algorithm of DQN.

In sequence of observations like Atari games, the DQN is prefer to forgot the previous situation, so the experience replay can be useful to avoid this case and maintain the previous situation. Fixed Q-targets can replace the attributes which are not good and replace by the new network, which can make the training easier.

**DPG / DDPG** (HW5)

Why discretise the continuous action space may not help in practice in DQN? How DDPG solve this problem?

Because the discretion of the continuous action space needs to maximise the Q value for each step, which is performance-costing and give processor too much pressure. Also, the maximising the Q value for each step needs a lot of storage space, to make the computing more complicated, which makes the continuous space may not be able to explore all states.

DDPG is the fusion of the DPG and DQN, which presenting as a off-policy algorithm, achieved by deep network, so it can handle with this problem.
