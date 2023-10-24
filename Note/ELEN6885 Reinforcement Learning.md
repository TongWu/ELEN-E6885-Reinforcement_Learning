# ELEN6885 Reinforcement Learning

# Week 1 - Introduction

## What is Reinforcement Learning?

- Agent-oriented learning 代理人导向型学习, learning by **interacting with an environment** to achieve a goal
- Learning by trial and error 实践和错误, with only delayed evaluative feedback (reward) 延迟的评价性反馈
    - The kind of machine learning that most like natural learning

## Characteristics of RL

- No supervisor, only a reward signal form the environment
- Feedback is delayed
- Time matters, i.e. sequential decision making 顺序决策
- Agent actions will affect subsequent data/feedback

![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled.png)

## Basic RL Interface

- History and State
    - History is the sequence of **observations**, **actions** and **rewards**
    - State is the information used to determine what happens next. It is a function of the history
- At each time step, an agent:
    - executes action
    - receives observations of the environment (state)
    - receives reward from the environment
- At each time step, the environment:
    - receives action from the agent
    - emits observations/changes its own states
    - emits rewards to the agent
- With a complete story:
    - The agent gives actions, response or control to the environment (world),
    - Then the environment will feedback the reward to the agent, it could be gain, payoff or cost
    - After that, the environment may change, then generate a new state, which will also give to the agent.
    - Finally, the agent will use state and reward that is feedbacked from the environment to decided what next action should it do.
    - It is a sequential process 连续性的处理动作

![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%201.png)

## Elements of RL

- Policy
    - A policy, $\Pi$, is a mapping from perceived states of the environment to the probability of actions to be taken when in those states 是一个从环境的预想状态，到行动概率的映射
    - Policy is the core of RL to determine behaviour
    - Policy can be stochastic or deterministic 随机性的或决定性的
        - For example a robot in a maze, there are walls around the robot and the next step of the robot can only be move forward. This is called deterministic.
        - If the robot have a way in its right and left, then in this state, the policy will be stochastic.
- Reward and return
    - Reward is a mapping from each perceived state of the environment to a single number, indicating the intrinsic desirability of that state 奖励是从每个感知到的环境状态到一个单一数字的映射，表明该状态的内在可取性
    - Reward is immediate 立即的
    - Return is a cumulative sequence of received rewards after a given time step 返回是在给定时间步长后收到的奖励的累积序列
    - Finite step return 有限步长的return:
        - $G_t=R_{t+1}+R_{t+2}+R_{t+3}+...+R_r$
        - $G_t$ is the sum of rewards within $r$ steps, as known as return
        - r is the terminal step
    - Discounted return with $0{<=}γ<=1$
        - $G_t = R_{t+1}+γR_{t+2}+γ^2R_{t+3}+...$
        
        ![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%202.png)
    
- Value function
    - Functions of states that estimate how good it is for the agent to be in a given state
        - How good is refer to the expected return
    - For Markov Decision Process (MDP), the value of a state is defined formally as
    
    ![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%203.png)
    
    - For MDP, the value of an action-state pair is defined formally as
    
    ![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%204.png)
    
- Model
    - Mimic the behaviour of the environment 模仿环境的行为
    - Used for planning (decide on a course of  actions by considering future situations before experienced) 通过考虑未来的情况，在经历之前便做出决定
    - MDP model:
    
    ![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%205.png)
    

## What is RL problem?

- RL problem is a considerable abstraction of the problem of goal-directed learning from interaction with the environment
- RL methods or solutions specify how the agent changes its policy as a result of its experience
- The agent’s goal is to **maximize** the total amount of reward it receives over the long run

## Markov decision process (MDP)

- A MDP is an environment in which all states are Markov
    - Markov analysis is a method used to forecast the value of a variable whose predicted value is influenced only by its current state, and not by any prior activity. In essence, it predicts a random variable based solely upon the current circumstances surrounding the variable.
- RL essentially solves a MDP problem
- A Markov Decision Process is a tuple $<S, A , P, R, \gamma>$
    - $S$ is a finite set of **states**
    - $A$ is a finite set of **actions**
    - $P$ is a state transition probability matrix,
        $$P_{ss'}^a = \mathbb P [S_{t+1}=s' | S_t=s, A_t=a]$$
    - $R$ is a reward function, $R_s^a=\mathbb E[R_{t+1}|S_t=s, A_t=a]$
    - $\gamma$ is a discount factor $\gamma \in [0,1]$


---

# Week 2 - Bandit Problem and MDP

Chapter 3 & 4.3 in RL-CPS book

## n-armed Bandit Problem 多臂匪问题

- You need to repeatedly take a choice among N different options/actions. 你需要在N个选项或动作中做出多次的选择
- You receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. 你会从一个静止的概率分布中获取的奖励，这取决于你所采取的行动
- Your objective is to maximize the expected total reward over a time period. 你的目标是最大化一段时间内的收益
  
    ### Questions
    
    - If there’s two or more machine with different possiability to win (get reward), which machine should you to choose
    - If the expected reward from each machine is known, which will be quite simple, just choose the highest expected reward
    - However, the expected reward is unkown
    - The basic idea is to estimate the expectation from the average reward received so far.   根据现有的平均期望回报来估算（未来的）期望回报
    
    ## Policies
    
    - Greedy policy 贪婪策略
        - always choose the machine with current best expected reward $Q_t(a)$
    - **ϵ-greedy policy ϵ-贪婪策略**
        - Choose machine with current best expected reward with probability 1-ϵ
        - Choose machines in random with probability ϵ, which means, choose random machine with probability ϵ/N
        
        > 当有许多老虎机时，由于我们无法知道哪个机器拥有最佳的回报，所以我们必须采取必要的策略来探索每个机器的平均回报率，并通过已有的数据选择可能的最佳方案。在这个过程中，产生了探索-开发困境(exploration-exploitation dilemma)。即，当我们选择了一个随机的机器时，这被称之为探索（exploration），因为此行为的选择是随机的，但是探索可以增加知识（数据）以提供给策略选择，这在长期中有好处。反之，如果我们选择当前状态下拥有最佳回报率的机器，这被称作开发（exploitation ），我们可以在当前状态下获得最佳的选择，但是总体来说这不一定是最好的选择。ϵ-greedy policy ϵ-贪婪策略，为探索-开发之间的选择有了一个较好的平衡，大概率选择开发利用，而保留了小概率的探索，而这个概率取决于ϵ，大概的平均回报如下图。当ϵ=0.1时，能够取得相对较高的回报。
        > 
        
        ![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%207.png)
        
    
    ## SoftMax
    
    SoftMax is another policy to balance exploration and exploitation
    
    - Use Gibbs (or Boltzmann) distribution to choose action “a” at “t-th” play with probability
      
        ![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%208.png)
        
    - $Q_t(a)$  is the current average rewards for the machine a
    - τ is a positive parameter called the temperature
        - Large τ → nearly equiprobable selection 接近平均的概率，即随机选择一个机器
            - Each machine’s probability will be $1/N$
        - Small τ → greedy action selection 即只选择当前最佳回报率

## Algorithm Implementation

- We don’t need to memory all the rewards to compute $Q_k$
    - Recall that $Q_k$ is the average reward for all previous actions
- Incremental inplementation: let $Q_k$ to be the average of its first k-1 rewards
  
    ![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%209.png)
    
    > $R_k$ 为k时间的reward，即当我们需要计算k+1步长的平均reward时，程序只需要记住：$Q_k$，$R_i$（k步长的reward），k（走过多少步）
    > 
    
    > 此处的公式只用于讨论静止环境（stationary）。如果老虎机的回报随时间而变化，则不会适用
    > 

## Discussion on Step Size (Nonstationary Problem)

> 大部分的RL问题为非平稳的(Unstationary)。在这种情况下，合理的做法是将最近的回报加上较高的权重。这就需要改变步长参数(Step size parameter)。我们将上图的$1/k$改为$a$
> 

$a_t(a)$: the step size parameter of action “a” at time step “t”

The following conditions are required to assure convergence of the incremental implementation with probability 1:

![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%2010.png)

> 在改变步长参数的同时，我们需要确保收敛的概率为1。这里用到两个公式。第一个公式确保了步长足够大，最终能够克服任何初始条件或随机波动。第二个条件保证最终步长变得足够小以确保收敛。
> 

## Markov Decision Process (MDP)

MDP formally describe an environment for reinforcement learning, where the environment is fullt observable.

Almost all RL problem can be formalized as MDPs

### Markov Property (Definition of Markov)

- The future is independent of the past given the present

![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%2011.png)

> 鉴于现在的情况，未来是独立于过去的
> 

## State Transition Matrix

For a Markov state $S$ and successor state $S'$, the state transition probaility is defined by 

![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%2012.png)

State transition matrix $P$ defines transition probailities from all states $S$ to all successor states $S'$

![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%2013.png)

where each row of the matrix sums to 1.

---

- A Markov Process is a memoryless random process, i.e. a sequence of random states, $S_1, S_2...$ with the Markov Property

## Markov Reward Process

The return $G_t$ is the total discounted reward from time step t

![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%2014.png)

### State-Value Function

![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%2015.png)

## Bellman Equation for MRP

- The value function can be decomposed into two parts
    - Immediate reward $R_{t+1}$
    - Discounted value of successor state $γv(S_{t+1})$
    
    ![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%2016.png)
    

![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%2017.png)

### Bellman Equation in Matrix Form

The Bellman Equation can be expressed concisely using matrices,

$v=R+γPv$

where v is a colum vector with one entry per state

![Untitled](ELEN6885%20Reinforcement%20Learning%205fc78de2c64f4312a418ef530e4f284d/Untitled%2018.png)

## Markov Decision  Process

- A Markov Decision Process is a Markov reward process with decisions. It is an environment in which all states are Markov
  
    ### Policies
    

# Week 3 - Model-based RL

## Introduction to dynamic programming

- Dynamic programming (DP) is a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a MDP

> DP 是指一组算法的集合，这些算法可以用来计算给定完美环境模型的最优策略