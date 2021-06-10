---
layout: page
title:  "Reinforcement Learning"
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


{:toc}

* 
{:toc}


<style>
table {
  border-collapse: collapse;
  border: 1px solid black;
  margin: 0 auto;
} 

th,td {
  border: 1px solid black;
  text-align: center;
  padding: 20px;
}

table.a {
  table-layout: auto;
  width: 180px;  
}

table.b {
  table-layout: fixed;
  width: 600px;  
}

table.c {
  table-layout: auto;
  width: 100%;  
}

table.d {
  table-layout: fixed;
  width: 100%;  
}
</style>


<p align="justify">
<a href="https://www.coursera.org/account/accomplishments/specialization/certificate/VRCC6WSKL8Y8"> My certificate.</a><br>
</p>


## 1. Fundamentals of Reinforcement Learning
### 1.1 The K-Armed Bandit Problem
<p align="justify">
Weekly reading: Chapter 2<br>
</p>

#### 1.1.1 Sequential Decision Making with Evaluative Feedback
<p align="justify">
$\bigstar$ Formalize the concept of decision making under uncertainty using k-armed bandits<br>
$\bigstar$ Describe foundamental concepts: rewards, time steps and values<br><br>

<b>Clinical Trials</b><br>
Consider a doctor has 3 differents treatments<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_1_1.png"/></center>
</p>
<p align="justify">
The doctor describes a treatment at random and observe the change after one treatment. After a while, the doctor will notice which treatment is better tan others. Then the doctor must decide between sticking with the best performing treatment or continuing with the randomized study. In fact, it's possible that the other two treatments are better and they work worse this time due to some accidents. This is an example of decision-making under uncertainty.<br><br>

<b>k-armed bandit</b><br>
The medical treatment above is one application of k-armed bandit.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_1_2.png"/></center>
</p>
<p align="justify">
<b>Actions-Values</b><br>
$\bigstar$ The value is the expected reward when we take one action<br>
$$q_{*}(a) \doteq E[R_{t} \mid A_{t} = a] = \sum_{r} p(r \mid a) r, \quad \forall a \in \{1, ..., k\}$$

$\bigstar$ The goal if to maximize the expected reward<br>
$$\arg\max_{a} q_{*}(a)$$

Return to the clinical trial. Suppose the 3 treatment has different distributions: Bernoulli, Binomial and Uniform. For example, the first treatment follows Bernoulli diatribution<br>
$$
p(r \mid a) =
\begin{cases}
  0.5, \quad r = 9 \\
  0.5, \quad r = -11
\end{cases}
$$

<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_1_3.png"/></center>
</p>
<p align="justify">
We can calculate its own action values.<br><br>

<b>Summary</b><br>
$\bigstar$ Decision making under uncertainty can be formalized by the k-armed bandit problem<br>
$\bigstar$ Fundamental ideas: actions, rewards, value functions<br>
</p>

#### 1.1.2 Learning Action Values
<p align="justify">
$\bigstar$ Estimate action value using the sample-average method<br>
$\bigstar$ Describe greedy action selection<br>
$\bigstar$ Introduce the exploration-exploitation dilemma<br><br>

<b>Value of an Action</b><br>
$\bigstar$ The value of an action is the expected reward when that action is taken<br>
$$q_{*}(a) \doteq E[R_{t} \mid A_{t} = a]$$
$\bigstar$ $q_{*}(a)$ is unknown, so we estimate it<br><br>

<b>Sample-Average Method</b><br>
We take sample-average method to estimate $q_{*}(a)$<br>
$$Q_{t}(a) \doteq \frac{\text{sum of rewards when a taken prior to t}}{\text{number of times a taken prior to t}} = \frac{\sum_{i=1}^{t-1} R}{t-1}$$

For example, a doctor chooses one of 3 treatments and he will get a reward 1 if the treatment works. After a while, he gets three value of actions.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_2_1.png"/></center>
</p>
<p align="justify">
In the next step, he will decide which atcion to take. If he always picks up the biggest $Q_{a}$, we call this greedy action selection due to exploitation. In contrast, we have non-greedy action selection based on exploration. In fact, an agent cannot hold exploitation and exploration simultaneously. This is a fundamental problem in Reinforcement Learning.<br><br>

<b>Summary</b><br>
$\bigstar$ Sample-average method can be used to estimate action values<br>
$\bigstar$ The greedy action is the action with the highest value estimate<br>
</p>

#### 1.1.3 Estimating Action Values Incrementally
<p align="justify">
$\bigstar$ Describe how action values can be estimated incrementally using the sample-average method<br>
$\bigstar$ Identify how the incremental update rule is an instance of a more general update rule<br>
$\bigstar$ Descirbe how general update rule can be used to solve non-stationary bandit problem<br><br>

<b>Incremental Update Rule</b>
$$
\begin{aligned}
Q_{n+1} &= \frac{1}{n}\sum_{i=1}^{n} R_{i} \\
&= \frac{1}{n} (R_{n} + \sum_{i=1}^{n-1} R_{i}) \\
&= \frac{1}{n}(R_{n} + (n-1)\frac{1}{n-1} \sum_{i=1}^{n-1} R_{i}) \\
&= \frac{1}{n} (R_{n} + (n-1)Q_{n}) \\
&= Q_{n} + \frac{1}{n}(R_{n} - Q_{n}) \\
\text{NewEstimate} &\leftarrow \text{OldEstimate} + \text{StepSize} \cdot (\text{Target} - \text{OldEstimate})
\end{aligned}
$$

The step size can be set a small number between 0 and 1. In sample average, the step size is $\frac{1}{n}$.<br><br>

<b>Non-Stationary Bandit Problem</b><br>
A non-stationary problem is same as stationary bandit problem except that our distribution is no logner constant. In such cases it makes sense to give more weight to recent rewards than to long-past rewards. <b>One of the most popular ways of doing this is to use a constant step-size parameter.</b> 
$$Q_{n+1} = Q_{n} + \alpha(R_{n} - Q_{n})$$

The reward weight decreases exponentially as time increases<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_3_1.png"/></center>
</p>
<p align="justify">
<b>Decay Past Rewards</b><br>
$$
\begin{aligned}
Q_{n+1} &= Q_{n} + \alpha(R_{n} - Q_{n}) \\
&= \alpha R_{n} + Q_{n} - \alpha Q_{n} \\
&= \alpha R_{n} + (1 - \alpha)Q_{n} \\
&= \alpha R_{n} + (1 - \alpha)[Q_{n-1} + \alpha(R_{n-1} - Q_{n-1})] \\
&= \alpha R_{n} + (1 - \alpha)\alpha R_{n-1} + (1-\alpha)^{2} Q_{n-1} \\
&= (1-\alpha)^{n} Q_{1} + \sum_{i=1}^{n} \alpha (1 - \alpha)^{n-i} R_{i}
\end{aligned}
$$

$Q_{1}$ is our initial action-value. The first term tells the contribution of $Q_{1}$ to $Q_{n+1}$ decreases exponentially over time. The second term tells the reward further back in time contribute exponentially less to the sum. Take them all together, the influence of $Q_{1}$ goes to 0 with more and more data. The most recent rewards contribute most to our current estimate.<br><br>

A constant $\alpha \in (0, 1]$ results in $Q_{n+1}$ being a weight average of past rewards and the initial estimate $Q_{1}$. We can check
$$(1-\alpha)^{n} + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} = 1$$

<b>In fact, the weight decays exponentially according to the exponent on $1 - \alpha$.</b> If $\alpha = 1$, all past rewards' weights become 1 because of $0^{0} = 1$.<br><br>

<b>Summary</b><br>
$\bigstar$ Derive incremental sample-average method<br>
$\bigstar$ Generalize the incremental update rule into a more general update rule<br>
$\bigstar$ A constant step size parameter can be used to solve a non-stationary bandit problem<br>
</p>

#### 1.1.4 What is the trade-off
<p align="justify">
$\bigstar$ Define the exploration and exploitation tradeoff<br>
$\bigstar$ Define epsilon greedy, a simple method to balance exploration and exploitation<br><br>

<b>Exploration and Exploitation</b><br>
$\bigstar$ Exploration: improve knowledge for long-term benefit<br>
$\bigstar$ Exploitation: exploit knowledge for short-term benefit<br><br>

<b>Epsilon-Greedy Action Selection</b>
$$
A_{t} \leftarrow
\begin{cases}
  \arg\max_{a} Q_{t}(a), \quad & \text{with probability } 1-\epsilon \\
  a \sim \text{Uniform}(\{a_{1}, a_{2}, ..., a_{k}\}), \quad & \text{with probability } \epsilon
\end{cases}
$$

Here is an experiment for 10-armed Testbed with different $\epsilon$<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_4_1.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ We have discussed the tradeoff between exploration and exploitation<br>
$\bigstar$ We introduce epsilon-greedy which is a simple method for balancing exploration and exploitation<br>
</p>

#### 1.1.5 Optimistic Initial Values
<p align="justify">
<b>Example: Clinical Trials</b><br>
A reward of 1 if the treatment works otherwise 0. We give an initial value $Q_{1} = 2$ for all actions.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_5_1.png"/></center>
</p>
<p align="justify">
After a while, our estimated $Q_{n}(a)$ is close to $q_{*}(n)$. Recall $q_{*}$ is the true optimal action value. We always want to estimate it with data.<br><br>

All the methods we have discussed so far are dependent to some extent on the initial action-value estimates $Q_{1}(a)$. In the language of statistics, these methods are biased by their initial estimates. For the sample-average methods, the bias disappears once all actions have been selected at least once, but for methods with constant $\alpha$, the bias is permanent.<br><br>

In practice, this kind of bias is usually not a problem and can sometimes be very helpful. The downside is that the initial estimates become a set of parameters that must be picked by the user, if only to set them all to zero. The upside is that they provide an easy way to supply some prior knowledge about what level of rewards can be expected.<br><br>

Initial action values can also be used as a simple way to encourage exploration.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_5_2.png"/></center>
</p>
<p align="justify">
An initial estimate of +5 is thus wildly optimistic because <b>$q_{*}(a)$ in this problem are selected from a normal distribution with mean 0 and variance 1</b>. But this optimism encourages action-value methods to explore. Whichever actions are initially selected, the reward is less than the starting estimates; the learner switches to other actions, being “disappointed” with the rewards it is receiving. The result is that all actions are tried several times before the value estimates converge. The system does a fair amount of exploration even if greedy actions are selected all the time.<br><br>

We call this technique for encouraging exploration optimistic initial values. We regard it as a simple trick that can be quite effective on stationary problems, but it is far from being a generally useful approach to encouraging exploration. For example, it is not well suited to nonstationary problems because its drive for exploration is inherently temporary. Indeed, any method that focuses on the initial conditions in any special way is unlikely to help with the general nonstationary case.<br><br>

<b>Limitations of Optimistic Initial Values</b><br>
$\bigstar$ Optimistic initial values only drive early exploration<br>
$\bigstar$ They are not well-suited for non-stationary problems<br>
$\bigstar$ We may not know what the optimistic initial value should be<br><br>

<b>Summary</b><br>
$\bigstar$ Optimistic initial values encourage early exploration<br>
$\bigstar$ Described limitations of optimistic initial value<br>
</p>

#### 1.1.6 Upper-Confidence Bound (UCB) Action Selection
<p align="justify">
<b>Uncertainty in Estimates</b><br>
When we use epsilon-greedy action selection, we have a small probability that pick up a random action. So, there will be uncertainty in estimating $q_{*}$.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_6_1.png"/></center>
</p>
<p align="justify">
If the confidence interval is small, we are pretty sure $Q_{n}(a)$ is close to $q_{*}(a)$.<br><br>

In Upper-Confidence Bound (UCB), we follow the principle of optimism in face of uncertainty. For example, we have 3 action-values with uncertainty<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_6_2.png"/></center>
</p>
<p align="justify">
Our agent has no idea which is best, so it optimistically picks the action that has the highest upper bound.<br><br>

<b>Upper-Confidence Bound (UCB) Action Selection</b><br>
$$A_{t} = \arg\max_{a} [Q_{t}(a) + c \sqrt{\frac{\ln t}{N_{t}(a)}}]$$

$N_{t}(a)$ denotes the number of times that action a has been selected prior to time t. c > 0 controls the degree of exploration. If $N_{t}(a) \rightarrow 0$ , then a is considered to be a maximizing action because this results a huge confidence interval.<br><br>

For example, we have 10000 trials and $N_{t}(a)$ is 5000, we have<br>
$$c \sqrt{\frac{\ln t}{N_{t}(a)}} = c \sqrt{\frac{\ln10000}{5000}} = 0.043c$$

While $N_{t}(a)$ is 100<br>
$$c \sqrt{\frac{\ln t}{N_{t}(a)}} = c \sqrt{\frac{\ln10000}{100}} = 0.303c$$

Therefore, we have a much larger confidence interval with $N_{t}(a) = 100$.<br><br>

Performance of optimistic initial values on the 10-armed Testbed<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_6_3.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ Upper-Confidence Bound action-selection uses uncertainty in value estimates for balancing exploration and exploitation<br>
</p>

#### 1.1.7 Practice Quiz: Exploration/Exploitation
<p align="justify">
<b>1. What is the incremental rule (sample average) for action values?</b><br>
A. $Q_{n+1} = Q_{n} + \frac{1}{n}Q_{n}$<br>
B. $Q_{n+1} = Q_{n} - \frac{1}{n}(R_{n} - Q_{n})$<br>
C. $Q_{n+1} = Q_{n} + \frac{1}{n}(R_{n} - Q_{n})$<br>
D. $Q_{n+1} = Q_{n} + \frac{1}{n}(R_{n} + Q_{n})$<br><br>

<b>Answer:</b> C.<br><br>

<b>2.</b><br>
This exercise will give you a better hands-on feel for how it works. The blue line is the target that we might estimate with equation 2.5. The red line is our estimate plotted over time.<br>
$$q_{n+1} = q_{n} + \alpha_{n}(R_{n} - q_{n})$$

<b>2.1</b><br>
Given the estimate update in red, <b>what do you think was the value of the step size parameter we used to update the estimate on each time step?</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_7_1.png"/></center>
</p>
<p align="justify">
A. 1<br>
B. 1/(t-1)<br>
C. 1/8<br>
D. 1/2<br><br>

<b>Answer:</b> C.<br><br>

<b>2.2</b><br>
Given the estimate update in red, <b>what do you think was the value of the step size parameter we used to update the estimate on each time step?</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_7_2.png"/></center>
</p>
<p align="justify">
A. 1<br>
B. 1/(t-1)<br>
C. 1/8<br>
D. 1/2<br><br>

<b>Answer:</b> A.<br><br>

<b>2.3</b><br>
Given the estimate update in red, <b>what do you think was the value of the step size parameter we used to update the estimate on each time step?</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_7_3.png"/></center>
</p>
<p align="justify">
A. 1<br>
B. 1/(t-1)<br>
C. 1/8<br>
D. 1/2<br><br>

<b>Answer:</b> B.<br><br>

<b>3.</b><br>
<b>What is the exploration/exploitation tradeoff?</b><br>
A. The agent wants to explore to get more accurate estimates of its values. The agent also wants to exploit to get more reward. The agent cannot, however, choose to do both simultaneously.<br>
B. The agent wants to explore the environment to learn as much about it as possible about the various actions. That way once it knows every arm’s true value it can choose the best one for the rest of the time.<br>
C. The agent wants to maximize the amount of reward it receives over its lifetime. To do so it needs to avoid the action it believes is worst to exploit what it knows about the environment. However to discover which arm is truly worst it needs to explore different actions which potentially will lead it to take the worst action at times.<br><br>

<b>Answer:</b> A.<br><br>

<b>4.</b><br>
Here is a diagram about different epsilon<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_1_7_4.png"/></center>
</p>
<p align="justify">
<b>4.1</b><br>
<b>Why did epsilon of 0.1 perform better over 1000 steps than epsilon of 0.01?</b><br>
A. The 0.01 agent did not explore enough. Thus it ended up selecting a suboptimal arm for longer.<br>
B. The 0.01 agent explored too much causing the arm to choose a bad action too often.<br>
C. Epsilon of 0.1 is the optimal value for epsilon in general.<br><br>

<b>Answer:</b> A.<br><br>

<b>4.2</b><br>
<b>If exploration is so great why did epsilon of 0.0 (a greedy agent) perform better than epsilon of 0.4?</b><br>
A. Epsilon of 0.4 doesn’t explore often enough to find the optimal action.<br>
B. Epsilon of 0.4 explores too often that it takes many sub-optimal actions causing it to do worse over the long term.<br>
C. Epsilon of 0.0 is greedy, thus it will always choose the optimal arm.<br><br>

<b>Answer:</b> B.<br>
</p>

#### 1.1.8 Programming Assignment: Bandits and Exploration/Exploitation
<p align="justify">

</p>

### 1.2 Markov Decision Processes
<p align="justify">
Weekly reading: Chapter 3.3 (pages 47-56)<br>
</p>

#### 1.2.1 Markov Decision Processes
<p align="justify">
Markov decision process (MDP) is a framework in Reinforcement Learning to describe how an agent interacts with its environment.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_2_1_1.png"/></center>
</p>
<p align="justify">
<b>The dynamics of an MDP</b><br>
At state S, an agent takes an action a then it goes into a next state S' (S' could be S, but here we use different notations to distinguish them) with a reward (positive or negative). We take use of a join distribution to describe this process.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_2_1_2.png"/></center>
</p>
<p align="justify">
The present state contains all the information necessary to predict the future.<br><br>

<b>Summary</b><br>
$\bigstar$ MDPs provide a general framework for sequential decision-making<br>
$\bigstar$ The dynamics of an MDP are defined by a probability distribution<br>
</p>

#### 1.2.2 The Goal of Reinforcement Learning
<p align="justify">
<b>Return</b><br>
$$G_{t} = R_{t+1} + R_{t+2} + ... + R_{T}$$

$G_{t}$ is a random variable and it represents a reward in long term.<br><br>

<b>Goal of an Agent: maximize the expected return</b><br>
$$E[G_{t}] = E[R_{t+1} + R_{t+2} + ... + R_{T}]$$

Specifically, we call $R_{T}$ as final time step where the agent environment interation ends. <b>At termination, the agent is reset to a start state.</b><br><br>

Every episode has a final state called terminal state.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_2_2_1.png"/></center>
</p>
<p align="justify">
We call these tasks episodic tasks.<br><br>

<b>Summary</b><br>
$\bigstar$ The goal of an agent is to maximize the expected return.<br>
$\bigstar$ In episodic tasks, the agent-environment interatcion breaks up into episode.<br>
</p>

#### 1.2.3 Continuing Tasks
<p align="justify">
<b>A comparison between episodic tasks and continuing tasks</b><br>
</p>
<p align="justify">
<table class="c">
  <tr><th>Episodic Tasks</th><th>Continuing Tasks</th></tr>
  <tr><td>Interation breaks naturally into episodes</td><td>Interation goes on continually</td></tr>
  <tr><td>Each episode ends in a terminal state</td><td>No terminal state</td></tr>
  <tr><td>Episodes are independent</td><td></td></tr>
  <tr><td>$G_{t} = R_{t+1} + R_{t+2} + ... + R_{T}$</td><td>$G_{t} = R_{t+1} + R_{t+2} + ...$</td></tr>
</table><br>
</p>
<p align="justify">
For example, a thermostat can be formulated as a continuing task since the thermostat never stops interacting with the environment.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_2_3_1.png"/></center>
</p>
<p align="justify">
We hope our thermostat is intelligent to adjust the temperature. So, the reward is -1 when temperature is adjusted manually.<br><br>

<b>Discounting</b><br>
We mignht notice that the return for continuing task could be infinite. How to make sure $G_{t}$ is finite?<br>
Discount the rewards in the future by $\gamma$ where $0 \leq \gamma < 1$
$$G_{t} = R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + ... + \gamma^{k-1} R_{t+k} + ... = \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}$$

$G_{t}$ is finite as long as $0 \leq \gamma < 1$. Why?<br>
Assume $R_{max}$ is the maximum reward the agent can receive<br>
$$G_{t} = \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \leq \sum_{k=0}^{\infty} \gamma^{k} R_{max} = R_{max} \sum_{k=0}^{\infty} \gamma^{k} = R_{max} \frac{1}{1 - \gamma}$$

<b>Effect of $\gamma$ on agent behavior</b><br>
If $\gamma = 0$, $G_{t} = R_{t+1}$. Agent only cares about the immediate reward. $\Rightarrow$ Short-sighted agent.<br>
If $\gamma \rightarrow 1$, Agent takes future rewards into account more strongly<br><br>

<b>Recursive nature of returns</b><br>
$$G_{t} = R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + ... = R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + ... )$$
$$G_{t} = R_{t+1} + \gamma G_{t+1}$$

<b>Summary</b><br>
$\bigstar$ In continuing tasks, the agent-environment interatcion goes on indefinitely.<br>
$\bigstar$ Discounting is used to ensure return are finite.<br>
$\bigstar$ Return can be defined recusively.<br>
</p>

#### 1.2.4 Practice Quiz: MDPs
<p align="justify">
<b>1.</b><br>
The learner and decision maker is the _______.<br>
A. Environment<br>
B. Agent<br>
C. State<br>
D. Reward<br><br>

<b>Answer:</b> B.<br><br>

<b>2.</b><br>
At each time step the agent takes an _______.
A. Environment<br>
B. Action<br>
C. State<br>
D. Reward<br><br>

<b>Answer:</b> B.<br><br>

<b>3.</b><br>
What equation(s) define $q_{\pi}(S_{t}, A_{t})$ in terms of subsequent reward?<br>
A. $q_{\pi}(s, a) = [G_{t} \mid S_{t} = s, A_{t} = a] , \text{where } G_{t} = R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + ...$<br>
B. $q_{\pi}(s, a) = E_{\pi} [R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + ...  \mid S_{t} = s, A_{t} = a]$<br>
C. $q_{\pi}(s, a) = E_{\pi}  [G_{t}] , \text{where } G_{t} = R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + ...$<br>
D. $q_{\pi}(s, a) =E_{\pi} [G_{t} \mid S_{t} = s, A_{t} = a] , \text{where } G_{t} = R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + ...$<br>
E. $q_{\pi}(s, a) =E_{\pi} [R_{t+1} \mid S_{t} = s, A_{t} = a]$<br><br>

<b>Answer:</b> B, D.<br><br>

<b>4.</b><br>
Imagine the agent is learning in an episodic problem. Which of the following is true?<br>
A. The number of steps in an episode is always the same.<br>
B. The number of steps in an episode is stochastic: each episode can have a different number of steps.<br>
C. The agent takes the same action at each step during an episode.<br><br>

<b>Answer:</b> B.<br><br>

<b>5.</b><br>
If the reward is always +1 what is the sum of the discounted infinite return when $\gamma$ < 1?<br>
$$G_{t} = \sum_{k = 0}^{\infty} \gamma^{k} R_{t+k+1}$$

A. $G_t = \frac{1}{1-\gamma}$<br>
B. Infinity<br>
C. $G_{t} = 1 * \gamma^{k}$<br>
D. $G_t = \frac{\gamma}{1-\gamma}$<br><br>

<b>Answer:</b> A.<br><br>

<b>6.</b><br>
What is the difference between a small gamma (discount factor) and a large gamma?<br>
A. The size of the discount factor has no effect on the agent.<br>
B. With a smaller discount factor the agent is more far-sighted and considers rewards farther into the future.<br>
C. With a larger discount factor the agent is more far-sighted and considers rewards farther into the future.<br><br>

<b>Answer:</b> C.<br><br>

<b>7.</b><br>
Suppose $\gamma$ = 0.8 and we observe the following sequence of rewards: $R_{1}$ = -3, $R_{2}$ = 5, $R_{3}$ = 2, $R_{4}$ = 7 and $R_{5}$ = 1 with T = 5. What is $G_{0}$? Hint: work backward and recall $G_{t} = R_{t+1} + \gamma G_{t+1}$<br>
A. 11.592<br>
B. 8.24<br>
C. 6.2736<br>
D. -3<br>
E. 12<br><br>

<b>Answer:</b> C.<br><br>

<b>8.</b><br>
Suppose $\gamma$ = 0.8 and reward sequence is $R_{1}$ = 5 followed by an infinite sequence of 10s. What is $G_{0}$<br>
A. 45<br>
B. 55<br>
C. 15<br><br>

<b>Answer:</b> A.<br>
$G_{2}$ = 10 / (1 - 0.8) = 50<br>
$G_{1}$ = 10 + 0.8 * 50 = 50<br>
$G_{0}$ = 5 + 0.8 * 50 = 45<br><br>

<b>9.</b><br>
Suppose reinforcement learning is being applied to determine moment-by-moment temperatures and stirring rates for a bioreactor (a large vat of nutrients and bacteria used to produce useful chemicals). The actions in such an application might be target temperatures and target stirring rates that are passed to lower-level control systems that, in turn, directly activate heating elements and motors to attain the targets. The states are likely to be thermocouple and other sensory readings, perhaps filtered and delayed, plus symbolic inputs representing the ingredients in the vat and the target chemical. The rewards might be moment-by-moment measures of the rate at which the useful chemical is produced by the bioreactor. Notice that here each state is a list, or vector, of sensor readings and symbolic inputs, and each action is a vector consisting of a target temperature and a stirring rate. Is this a valid MDP?<br>
A. Yes<br>
B. No<br><br>

<b>Answer:</b> A.<br><br>

<b>10.</b><br>
Consider using reinforcement learning to control the motion of a robot arm in a repetitive pick-and-place task. If we want to learn movements that are fast and smooth, the learning agent will have to control the motors directly and have low-latency information about the current positions and velocities of the mechanical linkages. The actions in this case might be the voltages applied to each motor at each joint, and the states might be the latest readings of joint angles and velocities. The reward might be +1 for each object successfully picked up and placed. To encourage smooth movements, on each time step a small, negative reward can be given as a function of the moment-to-moment “jerkiness” of the motion. Is this a valid MDP?<br>
A. Yes<br>
B. No<br><br>

<b>Answer:</b> A.<br><br>

<b>11.</b><br>
Imagine that you are a vision system. When you are first turned on for the day, an image floods into your camera. You can see lots of things, but not all things. You can't see objects that are occluded, and of course you can't see objects that are behind you. After seeing that first scene, do you have access to the Markov state of the environment? Suppose your camera was broken that day and you received no images at all, all day. Would you have access to the Markov state then?<br>
A. You have access to the Markov state before and after damage.<br>
B. You have access to the Markov state before damage, but you don’t have access to the Markov state after damage.<br>
C. You don’t have access to the Markov state before damage, but you do have access to the Markov state after damage.<br>
D. You don’t have access to the Markov state before or after damage.<br><br>

<b>Answer:</b> A.<br>
Because there is no history before the first image, the first state has the Markov property. The Markov property does not mean that the state representation tells all that would be useful to know, only that it has not forgotten anything that would be useful to know. The case when the camera is broken is different, but again we have the Markov property. The key in this case is that the future is impoverished. All the possible futures are the same (all blank), so nothing need be remembered in order to predict them.<br><br>

<b>12.</b><br>
What does MDP stand for?<br>
A. Meaningful Decision Process<br>
B. Markov Decision Process<br>
C. Markov Deterministic Policy<br>
D. Markov Decision Protocol<br><br>

<b>Answer:</b> B.<br><br>

<b>13.</b><br>
What is the reward hypothesis?<br>
A. Always take the action that gives you the best reward at that point.<br>
B. Goals and purposes can be thought of as the maximization of the expected value of the cumulative sum of rewards received.<br>
C. Ignore rewards and find other signals.<br>
D. Goals and purposes can be thought of as the minimization of the expected value of the cumulative sum of rewards received.<br><br>

<b>Answer:</b> B.<br><br>

<b>14.</b><br>
Imagine, an agent is in a maze-like gridworld. You would like the agent to find the goal, as quickly as possible. You give the agent a reward of +1 when it reaches the goal and the discount rate is 1.0, because this is an episodic task. When you run the agent its finds the goal, but does not seem to care how long it takes to complete each episode. How could you fix this? (Select all that apply)<br>
A. Give the agent -1 at each time step.<br>
B. Give the agent a reward of +1 at every time step.<br>
C. Set a discount rate less than 1 and greater than 0, like 0.9.<br>
D. Give the agent a reward of 0 at every time step so it wants to leave.<br><br>

<b>Answer:</b> A, C.<br><br>

<b>15.</b><br>
When may you want to formulate a problem as episodic?<br>
A. When the agent-environment interaction naturally breaks into sequences. Each sequence begins independently of how the episode ended.<br>
B. When the agent-environment interaction does not naturally break into sequences. Each new episode begins independently of how the previous episode ended.<br><br>

<b>Answer:</b> A.<br><br>

<b>16.</b><br>
When may you want to formulate a problem as continuing?<br>
A. When the agent-environment interaction does not naturally break into sequences. Each new episode begins independently of how the previous episode ended.<br>
B. When the agent-environment interaction naturally breaks into sequences and each sequence begins independently of how the previous sequence ended.<br><br>

<b>Answer:</b> A.<br>
</p>

### 1.3 Value Functions & Bellman Equations
<p align="justify">
Weekly reading: Chapter 3.5 - 3.8 (pages 58-67)<br>
</p>

#### 1.3.1 Specifying Policies
<p align="justify">
<b>Deterministic policy notation</b><br>
If the agent is at state s, it will take an specific action a.<br>
$$\pi(s) = a$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_3_1_1.png"/></center>
</p>
<p align="justify">
<b>Stochastic policy notation</b><br>
$$\pi(a \mid s) \geq 0$$
$$\sum_{a \in \mathbf{A}(s)} \pi(a \mid s) = 1$$

<b>Summary</b><br>
$\bigstar$ A policy maps the current state onto a set of probabilities.<br>
$\bigstar$ Policies can only depend on the current state.<br>
</p>

#### 1.3.2 Value Functions
<p align="justify">
<b>State-value functions</b><br>
$$v_{\pi}(s) = E_{\pi}[G_{t} \mid S_{t} = s], \quad G_{t} = \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}$$

<b>Action-value functions</b><br>
$$q_{\pi}(s, a) = E_{\pi} [G_{t} \mid S_{t} = s, A_{t} = a]$$

<b>Value functions predict rewards into the future</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_3_2_1.png"/></center>
</p>
<p align="justify">
Value functions are able to judge the quality of policies.<br><br>

<b>Summary</b><br>
$\bigstar$ State-value function represent the expected return form a given state under a specific policy.<br>
$\bigstar$ Action-value functions represent the expected return form a given state after taking a specific action, later following a specific policy.<br>
</p>

#### 1.3.3 Bellman Equation Derivation
<p align="justify">
<b>State-value Bellman Equation</b><br>
$$v_{\pi}(s) = E_{\pi}[G_{t} \mid S_{t} = s], \quad G_{t} = \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}$$
$$v_{\pi}(s) = E_{\pi} [R_{t+1} + \gamma G_{t+1} \mid S_{t} = s]$$
$$= \sum_{a} \pi(a \mid s) \sum_{s'} \sum_{r} p(s', r \mid s, a) \left \{ r + \gamma E_{\pi} [G_{t+1} \mid S_{t+1} = s'] \right \}$$

We can continue to unroll the term $E_{\pi} [G_{t+1} \mid S_{t+1} = s']$ if we'd like to<br>
$$E_{\pi} [G_{t+1} \mid S_{t+1} = s'] = \sum_{a'} \pi(a' \mid s') \sum_{s''} \sum_{r'} p(s'', r' \mid s', a') \left \{ r' + \gamma E_{\pi} [G_{t+2} \mid S_{t+2} = s''] \right \}$$

But we can notice that<br>
$$E_{\pi} [G_{t+1} \mid S_{t+1} = s'] = v_{\pi} (s')$$

So, we find a relationship of state-value function between two steps.<br>
$$v_{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} \sum_{r} p(s', r \mid s, a)(r + \gamma v_{\pi}(s')), \forall s \in S$$

We can use a diagram to represent<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_3_3_1.png"/></center>
</p>
<p align="justify">
<b>Action-value Bellman Equation</b><br>
$$q_{\pi} (s, a) = E_{\pi}[G_{t} \mid S_{t} = s, A_{t} = a]$$
$$= \sum_{s'} \sum_{r} p(s' , r \mid s, a) \left \{ r + \gamma E_{\pi} [G_{t+1} \mid S_{t+1} = s'] \right \}$$
$$E_{\pi} [G_{t+1} \mid S_{t+1} = s'] = \sum_{a'} \pi(a' \mid s) E_{\pi} [G_{t+1} \mid S_{t+1} = s', A_{t+1} = a']$$
$$q_{\pi} (s, a) = \sum_{s'} \sum_{r} p(s', r \mid s, a) [r + \gamma \sum_{a'} \pi(a' \mid s') q_{\pi} (s', a')]$$

We can notice a relationship between $v_{\pi}$ and $q_{\pi}$<br>
$$v_{\pi} (s) = \sum_{a} \pi(a \mid s) q_{\pi} (s, a)$$
$$q_{\pi} (s, a) = \sum_{s'} \sum_{r} p(s' , r \mid s, a) [r + \gamma v_{\pi}(s')]$$

<b>Summary</b><br>
$\bigstar$ The current time step's state-action values can be written recusively in terms of future state-action values.<br>
</p>

#### 1.3.4 Optimal Policies
<p align="justify">
An optimal policy $\pi*$ is as good as or better than all the other policies<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_3_4_1.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ An optimal policy is defined as the policy with the highest possible value functions in all states.<br>
$\bigstar$ At least one optimal policy always exists, but they may be more than one.<br>
$\bigstar$ The exponential number of possible policies making searching for the optimal policy by brute-force intractable.<br>
</p>

#### 1.3.5 Optimal Value Functions
<p align="justify">
$\pi_{1} \geq \pi_{2}$ if and only if $v_{\pi_{1}}(s) \geq v_{\pi_{2}}(s)$ for all s $\in$ S.<br><br>

Shared state value function $v*$<br>
$$v_{\pi_{*}}(s) = E_{\pi}[G_{t} \mid S_{t} = t] = \max_{\pi} v_{\pi}(s), \quad  \forall \in S$$

Shared action avlue function $q*$<br>
$$q_{\pi_{*}}(s, a) = \max_{\pi} q_{\pi}(s, a), \quad \forall s \in S \text{ and } a \in A$$

<b>Bellman Equation for $v_{*}$</b><br>
$$v_{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} \sum_{r} p(s', r \mid s, a)[r + \gamma v_{\pi}(s')]$$
$$v_{*}(s) = \sum_{a} \pi_{*}(a \mid s) \sum_{s'} \sum_{r} p(s', r \mid s, a)[r + \gamma v_{*}(s')]$$
$$v_{*}(s) = \max_{a} \sum_{s'} \sum_{r} p(s', r \mid s, a)[r + \gamma v_{*}(s')]$$

<b>Bellman Equation for $q_{*}$</b><br>
$$q_{\pi} (s, a) = \sum_{s'} \sum_{r} p(s', r \mid s, a) [r + \gamma \sum_{a'} \pi(a' \mid s') q_{\pi} (s', a')]$$
$$q_{*} (s, a) = \sum_{s'} \sum_{r} p(s', r \mid s, a) [r + \gamma \sum_{a'} \pi_{*} (a' \mid s') q_{*} (s', a')]$$
$$q_{*} (s, a) = \sum_{s'} \sum_{r} p(s', r \mid s, a) [r + \gamma \max_{a'} q_{*} (s', a')]$$

We cannot solve for a $v_{*}$ with linear system solver because max is not a linear operation.<br><br>

<b>Summary</b><br>
$\bigstar$ The Bellman optimality equation relates the value of a state, or state-action pair, to its possible successors under any optimal policy.<br>
</p>

#### 1.3.6 Using Optimal Value Functions to Get Optimal Policies
<p align="justify">
<b>Determining an Optimal Policy</b><br>
$$v_{*}(s) = \max_{a} \sum_{s'} \sum_{r} p(s', r \mid s, a)[r + \gamma v_{*}(s')]$$
$$\pi_{*}(s) = \arg\max_{a} \sum_{s'} \sum_{r} p(s', r \mid s, a)[r + \gamma v_{*}(s')]$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_3_6_1.png"/></center>
</p>
<p align="justify">
For example<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_3_6_2.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ Once we have the optimal state value function or action value function, its relatively easy to work out the optimal policy.<br>
</p>

#### 1.3.7 Practice Quiz: Value Functions and Bellman Equations
<p align="justify">
<b>1.</b><br>
A policy is a function which maps ____ to ____.<br>
A. States to values.<br>
B. States to actions.<br>
C. States to probability distributions over actions.<br>
D. Actions to probabilities.<br>
E. Actions to probability distributions over values.<br><br>

<b>Answer:</b> C.<br><br>

<b>2.</b><br>
The term “backup” most closely resembles the term ___ in meaning.<br>
A. Value<br>
B. Update<br>
C. Diagram<br><br>

<b>Answer:</b> B.<br><br>

<b>3.</b><br>
At least one deterministic optimal policy exists in every Markov decision process.<br>
A. False<br>
B. True<br><br>

<b>Answer:</b> B.<br><br>

<b>4.</b><br>
The optimal state-value function:<br>
A. Is not guaranteed to be unique, even in finite Markov decision processes.<br>
B. Is unique in every finite Markov decision process.<br><br>

<b>Answer:</b> B.<br><br>

<b>5.</b><br>
Does adding a constant to all rewards change the set of optimal policies in episodic tasks?<br>
A. Yes, adding a constant to all rewards changes the set of optimal policies.<br>
B. No, as long as the relative differences between rewards remain the same, the set of optimal policies is the same.<br><br>

<b>Answer:</b> A.<br>
Adding a constant to the reward signal can make longer episodes more or less advantageous (depending on whether the constant is positive or negative).<br><br>

<b>6.</b><br>
Does adding a constant to all rewards change the set of optimal policies in continuing tasks?<br>
A. Yes, adding a constant to all rewards changes the set of optimal policies.<br>
B. No, as long as the relative differences between rewards remain the same, the set of optimal policies is the same.<br>

<b>Answer:</b> B.<br>
Since the task is continuing, the agent will accumulate the same amount of extra reward independent of its behavior.<br><br>

<b>7.</b><br>
Select the equation that correctly relates $v_{*}$ to $q_{*}$.  Assume $\pi$ is the uniform random policy.<br>
A. $v_{*}(s) = \sum_{a, r, s'} \pi(a \mid s) p(s', r \mid s, a) [r + q_{*}(s')]$<br>
B. $v_{*}(s) = \sum_{a, r, s'} \pi(a \mid s) p(s', r \mid s, a) [r + \gamma q_{*}(s')]$<br>
C. $v_{*}(s) = \sum_{a, r, s'} \pi(a \mid s) p(s', r \mid s, a) q_{*}(s')$<br>
D. $v_{*}(s) = \max_{a} q_{*}(s, a)$<br><br>

<b>Answer:</b> D.<br><br>

<b>8.</b><br>
Select the equation that correctly relates $q_{*}$ to $v_{*}$ using four-argument function p<br>
A. $q_{*}(s, a) = \sum_{s', r} p(s', r \mid a, s) [r + v_{*}(s')]$<br>
B. $q_{*}(s, a) = \sum_{s', r} p(s', r \mid a, s) \gamma [r + v_{*}(s')]$<br>
C. $q_{*}(s, a) = \sum_{s', r} p(s', r \mid a, s) [r + \gamma v_{*}(s')]$<br><br>

<b>Answer:</b> C.<br><br>

<b>9.</b><br>
Write a policy $\pi_{*}$ in terms of $q_{*}$<br>
A. $\pi_{*}(a \mid s) = q_{*}(s, a)$<br>
B. $\pi_{*}(a \mid s) = \max_{a'} q_{*}(s, a')$<br>
C. $\pi_{\ast}(a|s) = 1 \mbox{ if } a = \mbox{argmax}_{a'} q_{\ast}(s, a'), \mbox{ else } 0$<br><br>

<b>Answer:</b> C.<br><br>

<b>10.</b><br>
Give an equation for some $\pi_{*}$ in terms of $v_{*}$ and the four-argument p.<br>
A. $\pi_{\ast}(a|s) = 1 \mbox{ if } v_{\ast}(s) = \max_{a'} \sum_{s', r} p(s', r | s, a') [ r + \gamma v_{\ast}(s')], \mbox{ else } 0$<br>
B. $\pi_{*} (a \mid s) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_{*}(s')]$<br>
C. $\pi_{*} (a \mid s) = \max_{a'} \sum_{s', r} p(s', r \mid s, a') [r + \gamma v_{*}(s')]$<br>
D. $\pi_{\ast}(a|s) = 1 \mbox{ if } v_{\ast}(s) = \sum_{s', r} p(s', r | s, a) [ r + \gamma v_{\ast}(s')], \mbox{ else } 0$<br><br>

<b>Answer:</b> D.<br>
</p>

#### 1.3.8 Quiz: Value Functions and Bellman Equations
<p align="justify">
<b>1.</b><br>
A function which maps ___ to ___ is a value function. [Select all that apply]<br>
A. State-action pairs to expected returns.<br>
B. Values to actions.<br>
C. Values to states.<br>
D. States to expected returns.<br><br>

<b>Answer:</b> A, D.<br><br>

<b>2.</b><br>
Consider the continuing Markov decision process shown below. The only decision to be made is in the top state, where two actions are available, left and right. The numbers show the rewards that are received deterministically after each action. There are exactly two deterministic policies, $\pi_{\mbox{left}}$ and $\pi_{\mbox{right}}$. Indicate the optimal policies if $\gamma$ = 0? If $\gamma$ = 0.9? If $\gamma$ = 0.5? [Select all that apply]<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_3_8_1.png"/></center>
</p>
<p align="justify">
A. For $\gamma$ = 0.9, $\pi_{\mbox{left}}$<br>
B. For $\gamma$ = 0.9, $\pi_{\mbox{right}}$<br>
C. For $\gamma$ = 0, $\pi_{\mbox{right}}$<br>
D. For $\gamma$ = 0.5, $\pi_{\mbox{right}}$<br>
E. For $\gamma$ = 0.5, $\pi_{\mbox{left}}$<br>
F. For $\gamma$ = 0, $\pi_{\mbox{left}}$<br><br>

<b>Answer:</b> B, D, E, F.<br><br>

<b>3.</b><br>
Every finite Markov decision process has __. [Select all that apply]<br>
A. A unique optimal value function<br>
B. A dertiministic optimal policy<br>
C. A stochastic optimal policy<br>
D. A unique optimal policy<br><br>

<b>Answer:</b> .<br>
A<br><br>

<b>4.</b><br>
The ___ of the reward for each state-action pair, the dynamics function p, and the policy π is _____ to characterize the value function $v_{\pi}$. (Remember that the value of a policy π at state s is<br>
$$v_{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} \sum_{r} p(s', r \mid s, a)(r + \gamma v_{\pi}(s'))$$

A. Mean; sufficient<br>
B. Distribution; necessary<br><br>

<b>Answer:</b> A.<br><br>

<b>5.</b><br>
The Bellman equation for a given a policy π: [Select all that apply]<br>
A. Expresses state values $v(s)$ in terms of state values of successor states.<br>
B. Expresses the improved policy in terms of the existing policy.<br>
C. Holds when the policy is greedy with respect to the value function.<br><br>

<b>Answer:</b> A.<br><br>

<b>6.</b><br>
An optimal policy:<br>
A. Is unique in every finite Markov decision process.<br>
B. Is unique in every Markov decision process.<br>
C. Is not guaranteed to be unique, even in finite Markov decision processes.<br><br>

<b>Answer:</b> C.<br>
For example, imagine a Markov decision process with one state and two actions. If both actions receive the same reward, then any policy is an optimal policy.<br><br>

<b>7.</b><br>
The Bellman optimality equation for $v_{*}$ [Select all that apply]<br>
A. Holds when the policy is greedy with respect to the value function.<br>
B. Expresses state values $v_{*}(s)$ in terms of state values of successor states.<br>
C. Holds when $v_{*} = v_{\pi}$ for a given policy π.<br>
D. Expresses the improved policy in terms of the existing policy.<br>
E. Holds for the optimal state value function.<br><br> 

<b>Answer:</b> B, E.<br><br>

<b>8.</b><br>
Give an equation for $v_{\pi}$ in terms of $q_{\pi}$ and $\pi$<br>
A. $v_{\pi}(s) = \max_{a} \gamma \pi(a \mid s) q_{\pi}(s, a)$<br>
B. $v_{\pi}(s) = \sum_{a} \gamma \pi(a \mid s) q_{\pi}(s, a)$<br>
C. $v_{\pi}(s) = \max_{a} \pi(a \mid s) q_{\pi}(s, a)$<br>
D. $v_{\pi}(s) = \sum_{a} \pi(a \mid s) q_{\pi}(s, a)$<br><br>

<b>Answer:</b> D.<br><br>

<b>9.</b><br>
Give an equation for $q_{\pi}$ in terms of $v_{\pi}$ and the four-argument p.<br>
A. $q_{\pi}(s, a) = \max_{s', r} p(s', r \mid s, a) [r + v_{\pi}(s')]$<br>
B. $q_{\pi}(s, a) = \max_{s', r} p(s', r \mid s, a)\gamma [r + v_{\pi}(s')]$<br>
C. $q_{\pi}(s, a) = \sum_{s', r} p(s', r \mid s, a)\gamma [r + v_{\pi}(s')]$<br>
D. $q_{\pi}(s, a) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_{\pi}(s')]$<br>
E. $q_{\pi}(s, a) = \sum_{s', r} p(s', r \mid s, a) [r + v_{\pi}(s')]$<br>
F. $q_{\pi}(s, a) = \max_{s', r} p(s', r \mid s, a)[r + \gamma v_{\pi}(s')]$<br><br>

<b>Answer:</b> D.<br><br>

<b>10.</b><br>
Let r(s,a) be the expected reward for taking action a in state s, as defined in equation 3.5 of the textbook. Which of the following are valid ways to re-express the Bellman equations, using this expected reward function? [Select all that apply]<br>
A. $v_{\pi}(s) = \sum_{a} \pi(a \mid s) [r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) v_{\pi}(s')]$<br>
B. $v_{*}(s) = \max_{a} [r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) v_{*}(s')]$<br>
C. $q_{\pi}(s, a) = r(s, a) + \gamma \sum_{s', a} p(s' \mid s, a) \pi(a' \mid s') q_{\pi}(s', a')$<br>
D. $q_{*}(s, a) = r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) \max_{a'} q_{*}(s', a')$<br><br>

<b>Answer:</b> A, B, C, D.<br><br>

<b>11.</b><br>
Consider an episodic MDP with one state and two actions (left and right). The left action has stochastic reward 1 with probability p and 3 with probability 1−p. The right action has stochastic reward 0 with probability q and 10 with probability 1−q. What relationship between p and q makes the actions equally optimal?<br>
A. 7 + 2p = 10q<br>
B. 7 + 2p = -10q<br>
C. 13 + 2p = 10q<br>
D. 7 + 3p = 10q<br>
E. 13 + 3p = -10q<br>
F. 13 + 3p = 10q<br>
G. 7 + 3p = -10q<br>
H. 13 + 2p = -10q<br><br>

<b>Answer:</b> A.<br>
</p>

### 1.4 Dynamic Programming
<p align="justify">
Weekly reading: Chapter 4.1, 4.2, 4.3, 4.4, 4.6, 4.7 (pages 73-88)<br>
</p>

#### 1.4.1 Policy Evaluation vs. Control
<p align="justify">
Policy evaluation is the task of determining the value function for a specific policy. Control is the task of finding a policy to obtain as much reward as possible.<br><br>

<b>Policy Evaluation</b><br>
$$\pi \rightarrow v_{\pi}$$

<b>Control is the task of improving a policy</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_4_1_1.png"/></center>
</p>
<p align="justify">
We use dynamic programming for policy evaluation and control.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_4_1_2.png"/></center>
</p>
<p align="justify">
$$
\\
v_{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} \sum_{r} p(s', r \mid s, a)(r + \gamma v_{\pi}(s'))\\

q_{\pi} (s, a) = \sum_{s'} \sum_{r} p(s' , r \mid s, a) [r + \gamma \sum_{a'} \pi(a' \mid s') q_{\pi} (s', a')]\\

v_{*}(s) = \max_{a} \sum_{s'} \sum_{r} p(s', r \mid s, a)[r + \gamma v_{*}(s')]\\

q_{*} (s, a) = \sum_{s'} \sum_{r} p(s', r \mid s, a) [r + \gamma \max_{a'} q_{*} (s', a')]
$$

<b>Summary</b><br>
$\bigstar$ Policy evaluation is the task of determining the state-value function $v_{\pi}$ for a particular policy $\pi$.<br>
$\bigstar$ Control is the task of improving an existing policy.<br>
$\bigstar$ Dynamic programming techniques can be used to solve these tasks, if we have access to the dynamic function p.<br>
</p>

#### 1.4.2 Iterative Policy Evaluation
<p align="justify">
<b>Iterative Policy Evaluation in a Nutshell</b><br>
$$v_{k+1}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} \sum_{r} p(s', r \mid s, a)(r + \gamma v_{k}(s'))$$

s and s' are states. k+1 is next iteration.<br><br>

When $v_{k+1} \approx v_{k}$, we stop.<br><br>

Algorithm<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_4_2_1.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ We can turn the Bellman equation into an update rule, to iteratively compute value function.<br>
</p>

#### 1.4.3 Policy Improvement
<p align="justify">
$$\pi_{*}(s) = \arg\max_{a} \sum_{s'} \sum_{r} p(s', r \mid s, a) [r + \gamma v_{*}(s')]$$

The new policy is a trict improvement over $\pi$ unless $\pi$ is already optimal.<br><br>

<b>Policy Improvement Theorem:</b><br>
$$q_{\pi}(s, \pi'(s)) \geq q_{\pi}(s, \pi(s)), \quad \text{for all }s \in S \rightarrow \pi' \geq \pi$$
$$q_{\pi}(s, \pi'(s)) > q_{\pi}(s, \pi(s)), \quad \text{for at leastl one }s \in S \rightarrow \pi' > \pi$$

<b>Summary</b><br>
$\bigstar$ The policy Improvement theorem tells us that a greedified policy is a strict improvement.<br>
</p>

#### 1.4.4 Policy Iteration
<p align="justify">
Policy Iteration contains <b>Policy Evaluation</b> and <b>Policy Improvement</b>.<br><br>

$$\pi_{1} \rightarrow v_{\pi_{1}} \rightarrow \pi_{2} \rightarrow v_{\pi_{2}} \rightarrow \pi_{3} \rightarrow ...$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_4_4_1.png"/></center>
</p>
<p align="justify">
Finally, we have an optimal policy and shared value functions.<br>
$$\pi_{*} \leftrightarrow v_{*}$$

<b>Summary</b><br>
$\bigstar$ Policy Iteration works by alternating policy evaluation and policy improvement.<br>
$\bigstar$ Policy Iteration follows a sequence of better and better policies and value functions until it reaches the optimal policy and associated optimal value functions<br>
</p>

#### 1.4.5 Flexibility of the Policy Iteration Framework
<p align="justify">
<b>Policy Iteration</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_4_5_1.png"/></center>
</p>
<p align="justify">
<b>Generalized Policy Iteration</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_4_5_2.png"/></center>
</p>
<p align="justify">
Synchronous methods repeatedly sweep over the entire state space. Asynchronous methods are more flexible, and they can update states in any order. Asynchronous methods can more efficiently propagate value information. This can be especially helpful when the state space is very large.<br><br>

<b>Summary</b><br>
$\bigstar$ Value Iteration allows us to combine policy evaluation and improvement into a single update.<br>
$\bigstar$ Asynchronous dynamic programming methods give us the freedom to update states in any order.<br>
$\bigstar$ Generalized Policy iteration unifies classical DP methods, value iteration and synchronous DP.<br>
</p>

#### 1.4.6 Efficiency of Dynamic Programming
<p align="justify">
<b>A Sampling Alternative for Policy Evaluation</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_4_6_1.png"/></center>
</p>
<p align="justify">
Bootstrapping<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_4_6_2.png"/></center>
</p>
<p align="justify">
<b>DP is much faster in practice than brutal force.</b><br><br>

<b>The Curse of Dimensionality</b><br>
The size of the state space grows exponentially as the number of relevant features increases.<br>
This is not an issue with DP, but an inherent complexity of the problem.<br><br>

<b>Summary</b><br>
$\bigstar$ Bootstrapping can save us from performing a huge amount of unnecessary work.<br>
</p>

#### 1.4.7 Practice Quiz: Dynamic Programming
<p align="justify">
<b>1.</b><br>
The value of any state under an optimal policy is ___ the value of that state under a non-optimal policy. [Select all that apply]<br>
A. Strictly greater than<br>
B. Greater than or equal to<br>
C. Strictly less than<br>
D. Less than or equal to<br><br>

<b>Answer:</b> B.<br><br>

<b>2.</b><br>
If a policy is greedy with respect to the value function for the equiprobable random policy, then it is guaranteed to be an optimal policy.<br>
A. True<br>
B. False<br><br>

<b>Answer:</b> B.<br>
Only policies greedy with respect to the optimal value function are guaranteed to be optimal.<br><br>

<b>3.</b><br>
If a policy $\pi$ is greedy with respect to its own value function $v_{\pi}$, then it is an optimal policy.<br>
A. False<br>
B. True<br><br>

<b>Answer:</b> B.<br>
If a policy is greedy with respect to its own value function, it follows from the policy improvement theorem and the Bellman optimality equation that it must be an optimal policy.<br><br>

<b>4.</b><br>
Let $v_{\pi}$ be the state-value function for the policy $\pi$. Let $\pi'$ be greedy with respect to $v_{\pi}$, then $v_{\pi'} \geq v_{\pi}$.<br>
A. True<br>
B. False<br><br>

<b>Answer:</b> A.<br>
This is a consequence of the policy improvement theorem.<br><br>

<b>5.</b><br>
let $v_{\pi}$ be the state-value function for the policy $\pi$. Let $v_{\pi'}$ be the state-value function for the policy $\pi'$. Assume $v_{\pi} = v_{\pi'}$, then this means $\pi = \pi'$.<br>
A. True<br>
B. False<br><br>

<b>Answer:</b> B.<br>
Two policies might share the same value function, but differ due to random tie breaking.<br><br>

<b>6.</b><br>
What is the relationship between value iteration and policy iteration? [Select all that apply]<br>
A. Policy iteration is a special case of value iteration.<br>
B. Value iteration and policy iteration are both special cases of generalized policy iteration.<br>
C. Value iteration is a special case of policy iteration.<br><br>

<b>Answer:</b> B.<br><br>

<b>7.</b><br>
The word synchronous means "at the same time". The word asynchronous means "not at the same time". A dynamic programming algorithm is: [Select all that apply]<br>
A. Asynchronous, if it does not update all states at each iteration.<br>
B. Asynchronous, if it updates some states more than others.<br>
C. Synchronous, if it systematically sweeps the entire state space at each iteration.<br><br>

<b>Answer:</b> A, B, C.<br><br>

<b>8.</b><br>
All Generalized Policy Iteration algorithms are synchronous.<br>
A. False<br>
B. True<br><br>

<b>Answer:</b> A.<br>
A Generalized Policy Iteration algorithm can update states in a non-systematic fashion.<br><br>

<b>9.</b><br>
Policy iteration and value iteration, as described in chapter four, are synchronous.<br>
A. True<br>
B. False<br><br>

<b>Answer:</b> A.<br><br>

<b>10.</b><br>
Which of the following is true?<br>
A. Synchronous methods generally scale to large state spaces better than asynchronous methods.<br>
B. Asynchronous methods generally scale to large state spaces better than synchronous methods.<br><br>

<b>Answer:</b> B.<br><br>

<b>11.</b><br>
Why are dynamic programming algorithms considered planning methods? [Select all that apply]<br>
A. They use a model to improve the policy.<br>
B. They compute optimal value functions.<br>
C. They learn from trial and error interaction.<br><br>

<b>Answer:</b> A.<br><br>

<b>12.</b><br>
Consider the undiscounted, episodic MDP below. There are four actions possible in each state, A = {up, down, right, left}, which deterministically cause the corresponding state transitions, except that actions that would take the agent off the grid in fact leave the state unchanged. The right half of the figure shows the value of each state under the equiprobable random policy.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/1_4_7_1.png"/></center>
</p>
<p align="justify">
<b>12.1</b><br>
If $\pi$ is the equiprobable random policy, what is $q(11, \mbox{down})$?<br>
A. -15<br>
B. -1<br>
C. 0<br>
D. -14<br><br>

<b>Answer:</b> B.<br>
Moving down incurs a reward of -1 before reaching the terminal state, after which the episode is over.<br><br>

<b>12.2</b><br>
If $\pi$ is the equiprobable random policy, what is $q(7, \mbox{down})$?<br>
A. -20<br>
B. -21<br>
C. -15<br>
D. -14<br><br>

<b>Answer:</b> C.<br>
Moving down incurs a reward of -1 before reaching state 11, from which the expected future return is -14.<br><br>

<b>12.3</b><br>
If $\pi$ is the equiprobable random policy, what is v(15)?<br>
A. -22<br>
B. -23<br>
C. -24<br>
D. -25<br>
E. -21<br><br>

<b>Answer:</b> C.<br>
right, down, left: v(15); up: v(13)<br>
</p>

#### 1.4.8 Programming Assignment: Optimal Policies with Dynamic Programming
<p align="justify">

</p>


## 2. Sample-based Learning Methods
### 2.1 Monte Carlo Methods for Prediction & Control
<p align="justify">
Weekly reading: Chapter 5.0-5.5 (pp. 91-104).<br>
</p>

#### 2.1.1 What is Monte Carlo?
<p align="justify">
To use pure DP approach, the agent needs to know the environment's transition probabilities. In some situations, we have no idea about transition probabilities.<br><br>

<b>Monte Carlo methods estimate values by averaging over a large number of random samples.</b><br><br>

<b>Monte Carlo methods for Policy Evaluation</b><br>
$$v_{\pi} = E_{\pi}[G_{t} \mid S_{t} = s]$$

Monte Carlo method for learning a value function would first observe multiple returns from the same state. Then, it average those observed returns to estimate the expected return from that state. As the number of samples increases, the average tends to get closer and closer to the expected return. The more returns the agent observes from a state, the more likely it is that the sample average is close to the state value. These returns can only be observed at the end of an episode.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_1_1.png"/></center>
</p>
<p align="justify">
<b>MC Algorithm</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_1_2.png"/></center>
</p>
<p align="justify">
For example, we have episodic tasks with 5 rewards and $\gamma = 0.5$. This is one sample.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_1_3.png"/></center>
</p>
<p align="justify">
In order to compute efficiently, we should calculate return form terminal state then backward.<br>
$$G_{5} = 0$$
$$G_{4} = R_{5} + \gamma G_{5} = 2 + 0.5 \cdot 0 = 2$$
$$G_{3} = R_{4} + \gamma G_{4} = 1 + 0.5 \cdot 2 = 2$$
$$G_{2} = R_{3} + \gamma G_{3} = 7 + 0.5 \cdot 2 = 8$$
$$G_{1} = R_{2} + \gamma G_{2} = 4 + 0.5 \cdot 8 = 8$$
$$G_{0} = R_{1} + \gamma G_{1} = 3 + 0.5 \cdot 8 = 7$$

Then, we use expected return to update $v_{\pi}$.<br><br>

<b>Summary</b><br>
$\bigstar$ We talked about how Monte Carlo methods learn directly from interaction.<br>
$\bigstar$ We showed a Monte Carlo Algorithm for learning state valeus in episodic problem.<br>
</p>

#### 2.1.2 Using Monte Carlo for Prediction
<p align="justify">
<b>Blackjack</b><br>
Collect cards so that their sum is as large as possible without exceeding 21.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_2_1.png"/></center>
</p>
<p align="justify">
The game begins with 2 cards dealt to both the player and the dealer. The player can see one of the dealer's cards, but the other is face down. If the player immediately has 21, they win unless the dealer also has 21 in which case they draw. If the player doesn't have 21 immediately, they can request more cards one at a time or a hit. If the sum of the player's cards ever exceeds 21, we say the player goes bust and loses. Otherwise when the player decides to stop requesting cards or sticks, it becomes the dealer's turn. The dealer only hits if the sum of their cards is less than 17, if the dealer goes bust the player wins. Otherwise, the winner of the game is a player who's sum is closer to 21.<br><br>

<b>Problem Formulation</b><br>
$\bigstar$ Undiscounted MDP where each game of blackjacks corresponds to an episode.<br>
$\bigstar$ Reward -1 for a loss, 0 for a draw and 1 for a win.<br>
$\bigstar$ Actions: Hit ir stick<br>
$\bigstar$ State (200 state in total)<br>
-- Whether the player has a uable ace (yes or no)<br>
-- The sum of the player's card (12-21)<br>
-- The card the dealer shows (Ace-10)<br>
We assume the cards are dealt from the deck with replacement. This means that there's no point in keeping track of the cards that had been dealt and that the state respects the Markov property.<br>
$\bigstar$ Policy: Stop requesting cards when the player's sum is 20 or 21.<br><br>

<b>Result</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_2_2.png"/></center>
</p>
<p align="justify">
<b>Implication of Monte Carlo learning</b><br>
$\bigstar$ We do not need to keep a large model of the environment.<br>
$\bigstar$ We are estimating the value of an individual state independently of the value of other states.<br>
$\bigstar$ The computation need to update the value of each state does not depend on the size of the MDP.<br><br>

<b>Summary</b><br>
$\bigstar$ We showed how to use Mothe Carlo Predictions to learn the value function of a policy.<br>
$\bigstar$ Monte Carlo learning is computationally efficient.<br>
</p>

#### 2.1.3 Using Monte Carlo for Action Values
<p align="justify">
$$q_{\pi}(s, a) = E_{\pi}[G_{t} \mid S_{t} = s, A_{t} = a]$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_3_1.png"/></center>
</p>
<p align="justify">
<b>Action-values are useful for learning a policy</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_3_2.png"/></center>
</p>
<p align="justify">
<b>Exploration Starts</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_3_3.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ Monte Carlo algorithm for estimating action-value function<br>
$\bigstar$ Importance of maintaining explorations<br>
</p>

#### 2.1.4 Using Monte Carlo methods for Generalized Policy Iteration
<p align="justify">
<b>Monte Carlo Generalized Policy Iteration</b><br>
GPI includes a <b>policy evaluation</b> and a <b>policy improvement</b> step. GPI algorithms produce sequences of policies that are at least as good as the policies before them.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_4_1.png"/></center>
</p>
<p align="justify">
In the GPI framework, the value estimates need only improve a little, not all the way to the correct action values.<br><br>

Monte Carlo Algorithm for Exploration Start<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_4_2.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ We described how to use Monte Carlo methods to create GPI algorithm.<br>
$\bigstar$ We showed an example algorithm, Monte Carlo with Exploration Starts.<br>
</p>

#### 2.1.5 Epsilon-soft policies
<p align="justify">
We can't always use Exploration Stars. For example, how could we randomly sample the initial state action pair for a self-driving car? How could we ensure the agent can start in all possible states? We could need to put the car in many different configurations in the middle of a busy freeway. This is dangerous.<br><br>

<b>$\epsilon$-Greedy Exploration</b><br>
Epsilon soft policies take each action with probability at least Epsilon over the number of actions $\frac{\epsilon}{\left | A \right |}$. The uniform random policy is another notable Epsilon South policy.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_5_1.png"/></center>
</p>
<p align="justify">
$\epsilon$ soft policies continue to explore and they are always stochastic.<br><br>

<b>$\epsilon$-greedy policies and deterministic policies</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_5_2.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ We talked about how sampling an initial state-action pair for exploring stats is not always feasible.<br>
$\bigstar$ We discussed Monte Carlo contol with $\epsilon$-soft policies.<br>
</p>

#### 2.1.6 Why does off-policy learning matter?
<p align="justify">
<b>On-Policy and Off-Policy</b><br>
$\bigstar$ On-Policy: improve and evaluate the policy being used to select actions<br>
$\bigstar$ Off-Policy: improve and evaluate a different policy from the one used to select action.<br><br>

<b>Summary</b><br>
$\bigstar$ Off-Policy learning allows learning an optimal policy form suboptimal behavior.<br>
$\bigstar$ The policy that we are learning is the target policy.<br>
$\bigstar$ The policy that we are choosing actions from is the behavior policy.<br>
</p>

#### 2.1.7 Importance Sampling
<p align="justify">
Importance sampling allows us to do off-policy learning, learning with one policy while following another.<br><br>

<b>Derivation of Importance Sampling</b><br>
We have a random variable X which is sampled from a distribution b. We want to estimate the espected value of x but with respect to the target distibution $\pi$.<br>
$$\text{Sample: } X \sim b$$
$$\text{Estimate: } E_{\pi}[X]$$

We cannot use sample average to compute the expectation under $\pi$, because X is sampled from b. The sample average will give us an expectation under b instead.<br><br>

We start by the definition of expected value<br>
$$E_{\pi}[X] = \sum_{x \in X} x \pi(x)$$
$$= \sum_{x \in X} x \pi(x) \frac{b(x)}{b(x)}$$

Note: $b(x)$ is the probability of observed outcome x under b.<br>
$$= \sum_{x \in X} x  \frac{\pi(x)}{b(x)} b(x)$$
$\frac{\pi(x)}{b(x)}$ is called <b>Importance Sampling Ratio</b> $\rho(x)$<br>
$$= \sum_{x \in X} x  \rho(x) b(x)$$

If we treat $x  \rho(x)$ as a new random variable, mutiplied by $b(x)$, we can rewrite this sum as an expectation under b.<br>
$$E_{\pi}[X] = \sum_{x \in X} x \rho(x) b(x) = E_{b} [X \rho(x)]$$

Recall expectation can be calculate as sample average for all data<br>
$$E[X] \approx \frac{1}{n} \sum_{i=1}^{n} x_{i}$$

So,
$$E_{b} [X \rho(X)] = \sum_{x \in X} x \rho(x) b(x) \approx \frac{1}{n} \sum_{i=1}^{n} x_{i} \rho(x_{i}), \quad x_{i} \sim b$$

Note: $x_{i}$ is drawed from b instead of $\pi$.<br><br>

Finally, to estimate an expectation of X under $\pi$<br>
$$E_{\pi}[X] \approx \frac{1}{n} \sum_{i=1}^{n} x_{i} \rho(x_{i}), \quad x_{i} \sim b$$
$$\rho(x) = \frac{\pi(x)}{b(x)}$$

<b>Look at an example of Estimation of via Sampling</b><br>
We have two distirbutions $b(x)$ and $\pi(x)$.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_7_1.png"/></center>
</p>
<p align="justify">
We draw samples from $b$ then estimate an expectated value under $\pi$. For reference, 2.2 is a true expected value uner $\pi$.<br>
Firstly, we draw a sample, say x = 1 with a probability $b(x = 1) = 0.85$. With this sample, we calcuate $E_{\pi}[X]$<br>
$$E_{\pi}[X] = \frac{1}{n} \sum_{i=1}^{n} x_{i} \rho(x_{i}) = 1 \times \frac{0.3}{0.85} = 0.35$$

We get an estimated value 0.35.<br><br>

We continue to sample, say x= 3 with $b(x = 3) = 0.05$, $\pi(x = 3) = 0.1$<br>
$$E_{\pi}[X] = \frac{1}{2} (1 \times \frac{0.3}{0.85} + 3 \times \frac{0.1}{0.05}) = 3.18$$

Next sample, say x = 1 with $b(x = 1) = 0.85$, $\pi(x = 1) = 0.3$<br>
$$E_{\pi}[X] = \frac{1}{3} (1 \times \frac{0.3}{0.85} + 3 \times \frac{0.1}{0.05} + 1 \times \frac{0.3}{0.85}) = 2.24$$

We get an estimated value which is close to the true value.<br><br>

By contrast, if we take a sample average directly with the data x = 1, 3, 1, then we will get an expected value of 0.85 + 0.15 + 0.85 = 1.85. This is under $b(x)$ instead of $\pi(x)$.<br><br>

<b>Summary</b><br>
$\bigstar$ Importance Sampling uses samples from one probability distribution to estimate the expectation of a different distribution.<br>
$\bigstar$ We can apply Importance Sampling for off-policy.<br>
</p>

#### 2.1.8 Quiz: Graded Quiz
<p align="justify">
<b>1.</b><br>
Which approach ensures continual exploration? (Select all that apply)<br>
A. Exploring starts<br>
B. On-policy learning with a deterministic policy<br>
C. On-policy learning with an ϵ-soft policy<br>
D. Off-Policy learning with an ϵ-soft behavior policy and a deterministic target policy<br>
E. Off-Policy learning with an ϵ-soft target policy and a deterministic behavior policy<br><br>

<b>Answer:</b> A, C, D.<br><br>

<b>2.</b><br>
When can Monte Carlo methods, as defined in the course, be applied? (Select all that apply)<br>
A. When the problem is continuing and there are sequences of states, actions, and rewards<br>
B. When the problem is continuing and there is a model that produces samples of the next state and reward<br>
C. When the problem is episodic and there are sequences of states, actions, and rewards<br>
D. When the problem is episodic and there is a model that produces samples of the next state and reward<br><br>

<b>Answer:</b> C, D.<br>
Well-defined returns are available in episodic tasks.<br><br>

<b>3.</b><br>
Which of the following learning settings are examples of off-policy learning? (Select all that apply)<br>
A. Learning about multiple policies simultaneously while following a single behavior policy<br>
B. Learning the optimal policy while continuing to explore<br>
C. Learning from data generated by a human expert<br><br>

<b>Answer:</b> A, B, C.<br>
Off-policy learning enables learning about multiple target policies simultaneously using a single behavior policy.<br>
An off-policy method with an exploratory behavior policy can assure continual exploration.<br>
Applications of off-policy learning include learning from data generated by a non-learning agent or human expert. The policy that is being learned (the target policy) can be different from the human expert’s policy (the behavior policy).<br><br>

<b>4.</b><br>
If a trajectory starts at time t and ends at time T, what is its relative probability under the target policy π and the behavior policy b?<br>
A. $\prod_{k=t}^{k=T} \frac{\pi(A_{t} \mid S_{k})}{b(A_{k} \mid S_{k})}$<br>
B. $\sum_{k=t}^{k=T} \frac{\pi(A_{t} \mid S_{k})}{b(A_{k} \mid S_{k})}$<br>
C. $\frac{\pi(A_{T-1} \mid S_{T-1})}{b(A_{T-1} \mid S_{T-1})}$<br>
D. $\frac{\pi(A_{t} \mid S_{t})}{b(A_{t} \mid S_{t})}$<br><br>

<b>Answer:</b> A.<br><br>

<b>5.</b><br>
When is it possible to determine a policy that is greedy with respect to the value functions $v_{\pi}$, $q_{\pi}$, for the policy $\pi$? (Select all that apply)<br>
When state values $v_{\pi}$ and a model are available<br>
When state values $v_{\pi}$ are available but no model is available.<br>
When action values $q_{\pi}$ and a model are available<br>
When action values $q_{\pi}$ are available but no model is available.<br><br>

<b>Answer:</b> .<br>
A, C, D<br><br>

<b>6.</b><br>
Monte Carlo methods in Reinforcement Learning work by...<br>
A. Averaging sample rewards<br>
B. Averaging sample returns<br>
C. Performing sweeps through the state set<br>
D. Planning with a model of the environment<br><br>

<b>Answer:</b> B.<br><br>

<b>7.</b><br>
Which of the following is a requirement for using Monte Carlo policy evaluation with a behavior policy b for a target policy $\pi$?<br>
A. For each state s and action a, if $b(a \mid s) > 0$ then $\pi(a \mid s) > 0$<br>
B. For each state s and action a, if $\pi(a \mid a) > 0$ then $b(a \mid s) > 0$<br>
C. All actions have non-zero probabilities under $\pi$<br><br>

<b>Answer:</b> B.<br>
Every action taken under $\pi$ must have a non-zero probability under b: $\rho(x) = \frac{\pi(x)}{b(x)}$<br><br>

<b>8.</b><br>
Suppose the state s has been visited three times, with corresponding returns 8, 4, and 3. What is the current Monte Carlo estimate for the value of s?<br>
A. 3<br>
B. 15<br>
C. 5<br>
D. 3.5<br><br>

<b>Answer:</b> C.<br><br>

<b>9.</b><br>
When does Monte Carlo prediction perform its first update?<br>
A. After the first time step<br>
B. When every state is visited at least once<br>
C. At the end of the first episode<br><br>

<b>Answer:</b> C.<br>
Prediction means updating value function. It must be done after one sample, namely after one episode.<br><br>

<b>10.</b><br>
In Monte Carlo prediction of state-values, memory requirements depend on (select all that apply)<br>
A. The number of states<br>
B. The number of possible actions in each state<br>
C. The length of episodes<br><br>

<b>Answer:</b> A, C.<br><br>

<b>11.</b><br>
For Monte Carlo Prediction of state-values, the number of updates at the end of an episode depends on<br>
A. The number of states<br>
B. The number of possible actions in each state<br>
C. The length of the episode<br><br>

<b>Answer:</b> C.<br>
One episode contains at least 1 state. MC prediction has to update all states in this episode according samples.<br><br>

<b>12.</b><br>
Which approach can find an optimal deterministic policy? (select all that apply)<br>
A. Exploring Starts<br>
B. $\epsilon$-greedy exploration<br>
C. Off-policy learning with an $\epsilon$-soft behavior policy and a deterministic target policy<br><br>

<b>Answer:</b> A, C.<br>
Exploring starts ensure that every state-action pair is visited even if the policy is deterministic.<br><br>

<b>13.</b><br>
In an $\epsilon$-greedy policy over A actions, what is the probability of the highest valued action if there are no other actions with the same value?<br>
A. $1 - \epsilon$<br>
B. $\epsilon$<br>
C. $1 - \epsilon + \frac{\epsilon}{A}$<br>
D. $\frac{\epsilon}{A}$<br><br>

<b>Answer:</b> C.<br><br>
</p>

#### 2.1.9 Off-Policy Monte Carlo Prediction
<p align="justify">
MC prediction for off-policy according to Importance Sampling<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_9_1.png"/></center>
</p>
<p align="justify">
<b>Off-Policy Monte Carlo</b><br>
$$\rho = \frac{P(\text{trajectory under } \pi)}{P(\text{trajectory under }b)}$$
$$v_{\pi}(s) = E_{b}[\rho G_{t} \mid S_{t} = s]$$

In order to calculate $\rho$, we have to calculate $P(\text{trajectory})$.<br><br>

<b>Off-Policy Trajectories</b><br>
$$P(A_{t}, S_{t+1}, A_{t+1}..., S_{T} \mid S_{t}, A_{t: T})$$
$$ = b(A_{t} \mid S_{t}) p(S_{t+1} \mid S_{t}, A_{t}) b(A_{t+1} \mid S_{t+1}) p(S_{t+2} \mid S_{t+1}, A_{t+1})...p(S_{T} \mid S_{T-1}, A_{A_{T-1}})$$
$$= \prod_{k=t}^{T-1} b(A_{k} \mid S_{k}) p(S_{k+1} \mid S_{k}, A_{k})$$

Probability of an trajectory under b<br>
$$P(\text{trajectory under }b) = \prod_{k=t}^{T-1} b(A_{k} \mid S_{k}) p(S_{k+1} \mid S_{k}, A_{k})$$

We define<br>
$$\rho_{t:T-1} = \frac{P(\text{trajectory under } \pi)}{P(\text{trajectory under }b)} = \prod_{k=t}^{T-1} \frac{\pi(A_{k} \mid S_{k}) p(S_{k+1} \mid S_{k}, A_{k})}{b(A_{k} \mid S_{k}) p(S_{k+1} \mid S_{k}, A_{k})}$$

The transition dynamics of the environment cancel out on each time step<br>
$$= \prod_{k=t}^{T-1} \frac{\pi(A_{k} \mid S_{k})}{b(A_{k} \mid S_{k})}$$

<b>Off-Policy Values</b><br>
$$E_{b}[\rho_{t:T-1} G_{t} \mid S_{t} = s] = v_{\pi}(s)$$

<b>Every-visit MC prediction algorithm for on-policy and off-policy</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_9_2.png"/></center>
</p>
<p align="justify">
<b>Computing $\rho_{t:T-1}$ incrementally</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_1_9_3.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ Used importance sampling ratios to correct the returns<br>
$\bigstar$ Modified the on-policy Monte Carlo prediction algorithm for off-policy learning<br>
</p>

### 2.2 Temporal Difference Learning Methods for Prediction
<p align="justify">
Weekly reading: Chapter 6.3 (pp. 116-128).<br>
</p>

#### 2.2.1 What is Temporal Difference (TD) learning?
<p align="justify">
Both TD and Monte Carlo methods use experience to solve the prediction problem. Given some experience following a policy $\pi$, both methods update their estimate V of $v_{\pi}$ for the nonterminal states $S_{t}$ occurring in that experience.<br><br>

<b>Review: Estimating Values from Returns</b><br>
$$v_{\pi}(s) = E_{\pi} [G_{t} \mid S_{t} = s]$$

<b>Bootstrapping</b><br>
$$G_{t} = R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + ... = R_{t+1} + \gamma G_{t+1}$$
$$v_{\pi}(s) = E_{\pi}[G_{t} \mid S_{t} = s] = E_{\pi} [R_{t+1} + \gamma G_{t+1} \mid S_{t} = s] = R_{t+1} + \gamma v_{\pi}(S_{t+1})$$

<b>Monte Carlo method suitable for nonstationary environments</b><br>
$$V(S_{t}) \leftarrow V(S_{t}) + \alpha [G_{t} - V(S_{t})]$$

<b>Temporal Difference</b><br>
$$V(S_{t}) \leftarrow V(S_{t}) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_{t})]$$

$R_{t+1} + \gamma V(S_{t+1})$ is called target and $R_{t+1} + \gamma V(S_{t+1}) - V(S_{t})$ is called TD error $\delta_{t}$.<br><br>

TD updates the value of one state towards its own estimate of the value in the next state.<br><br>

<b>Whereas Monte Carlo methods must wait until the end of the episode to determine the increment to $V(S_{t})$ (when $G_{t}$ is known), TD methods need to wait only until the next time step.</b><br><br>

In DP, we update all possible next states, so we need a model p of the environment to compute this expectation.<br>
$$v_{\pi}(s) \leftarrow \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_{\pi}(s')]$$

But TD needs only the next state. We can get directly from the environment without a model.<br><br>

<b>1-step TD or TD(0)</b><br>
Consider t+1 is current time step and t is previous time step. We use the state in previous time step $s_{t}$ to compute the TD error.<br>
$$V(S_{t}) \leftarrow V(S_{t}) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_{t})]$$
$$s_{t} \leftarrow s_{t+1}$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_2_1_1.png"/></center>
</p>
<p align="justify">
TD(0) algorithm<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_2_1_2.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ Temporal Difference Learning is a way to incrementally estimate the return through bootstrapping.<br>
$\bigstar$ TD error $\delta_{t} = R_{t+1} + \gamma V(S_{t+1}) - V(S_{t})$<br>
</p>

#### 2.2.2 The advantages of temporal difference learning
<p align="justify">
TD methods don't require a model of the environment. They can learn directly from experience. Unlike Monte Carlo, TD can update the values on every state. Bootstrapping allows us to update the episodes based on other estimates. TD asymptotically converges to the correct predictions. In addition, TD methods usually converge faster than Monte Carlo methods.<br>
</p>

#### 2.2.3 Comparing TD and Monte Carlo
<p align="justify">
<b>Random Walk</b><br>
In this experiment we have 7 states: A, B, C, D, E are non-terminal states. Our agent has two available actions to take: left and right with a equiprob policy. Remember that once the agent arrives at a terminal state, this episode ends and V(terminal state) is 0.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_2_3_1.png"/></center>
</p>
<p align="justify">
The value of the start state is 0.5, that means the probability of terminating from the center is a half.<br><br>

We have a reference state value functions.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_2_3_2.png"/></center>
</p>
<p align="justify">
We run one episode for TD method and MC method.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_2_3_3.png"/></center>
</p>
<p align="justify">
We run more episodes to calculate the RMS error for TD and MC.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_2_3_4.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ We ran a careful experiment comparing TD and Monte Carlo.<br>
$\bigstar$ The result shows that TD converges faster to a low final error in this problem.<br>
</p>

#### 2.2.4 Quiz: Practice Quiz
<p align="justify">

</p>

#### 2.2.5 Programming Assignment: Policy Evaluation with Temporal Difference Learning
<p align="justify">
All codes are available <a href="https://github.com/chaopan95/PROJECTS/tree/master/Reinforcement-Learning-Course-PA/Policy-Evaluation-with-Temporal-Difference%20Learning">here.</a><br>
</p>

### 2.3 Temporal Difference Learning Methods for Control
<p align="justify">
Weekly reading: Chapter 6.4-6.6 (pp. 129-134).<br>
</p>

#### 2.3.1 Sarsa
<p align="justify">
<b>Recall Generalized Policy Iteration</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_1_1.png"/></center>
</p>
<p align="justify">
<b>Sarsa</b><br>
Sarsa is a short of state, action, reward, next state, next action.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_1_2.png"/></center>
</p>
<p align="justify">
Sarsa makes predictions about the values of state action pair. It takes an action $A_{t}$ in current state $S_{t}$ and observes reward $R_{t+1}$ as well next state $S_{t+1}$. In Sarsa, the agent needs to know its next state action pair before updating its value estimate. Since the agent is learning action values for a specific policy, it uses that policy to sample next action.<br>
$$Q(S_{t}, A_{t}) \leftarrow Q(S_{t}, A_{t}) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_{t}, A_{t})]$$

This update is done after every transition from a nonterminal state $S_{t}$. If $S_{t+1}$ is terminal state, $Q(S_{t+1}, A_{t+1})$ is 0.<br><br>

The backup diagram for Sarsa<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_1_3.png"/></center>
</p>
<p align="justify">
This equation is for policy evaluation. Tnaks for the GPI framework, we can turn it into a control algorithm.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_1_4.png"/></center>
</p>
<p align="justify">
We can improve the policy every time step rather than after an episode or after convergence.<br><br>

$\bigstar$ We can combine generalized policy iteration with TD learning to find improved policies.<br>
$\bigstar$ Sarsa is an action value form of TD which combines these ideas.<br><br>

<b>The Windy Gridworld</b><br>
We have a start state S and a goal state G. The agent can move four directions and each move gets a reward of -1.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_1_5.png"/></center>
</p>
<p align="justify">
Besides, we impose wind from down to up. Due to the wind, each horizontal move will shift one grid upward. This is an undiscounted episodic task. We apply a $\epsilon$-greedy Sarsa with $\epsilon$ = 0.1, $\alpha$ = 0.5 and initial value $Q(s, a)$ = 0.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_1_6.png"/></center>
</p>
<p align="justify">
The increasing slope shows the goal was reached more quickly over time. By 8000 steps, the policy is optimal with an average step length of 17.<br><br>

Monte Carlo is hard to work here because termination is not guaranteed for all policies. Without termination, MC cannot update value function.<br>

<b>Sarsa Algorithm</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_1_7.png"/></center>
</p>
<p align="justify">
Sarsa is on-policy TD control.<br>
</p>

#### 2.3.2 Q-Learning
<p align="justify">
<b>Q-Learning Algorithm</b><br>
Q-learning was developed in 1989 and is one of the first major online Reinforcement Learning algorithm.<br>
$$Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_{a} Q(S', a) - Q(S, A)]$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_2_1.png"/></center>
</p>
<p align="justify">
<b>Revisiting Bellman equations</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_2_2.png"/></center>
</p>
<p align="justify">
In fact, Sarsa is a sample-based algorithm to solve the Bellman equation for action values. Q-learning also solves the Bellman equation using samples from the environment. But instead of using the standard Bellman equation, Q-learning uses the Bellman's Optimality Equation for action values. The optimality equations enable Q-learning to directly learn Q-star instead of switching between policy improvement and policy evaluation steps.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_2_3.png"/></center>
</p>
<p align="justify">
Even though Sarsa and Q-learning are both based on Bellman equations, they're based on very different Bellman equations. Sarsa is sample-based version of <b>policy iteration</b> which uses Bellman equations for action values, that each depend on <b>a fixed policy</b>. Q-learning is a sample-based version of <b>value iteration</b> which iteratively applies the Bellman optimality equation. Applying the Bellman's Optimality Equation strictly improves the value function, unless it is already optimal. So value iteration continually improves as value function estimate, which eventually converges to the optimal solution. For the same reason, Q-learning also converges to the optimal value function as long as the aging continues to explore and samples all areas of the state action space.<br><br>

We take s same example like before, but with Q-Learning and Sarsa.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_2_4.png"/></center>
</p>
<p align="justify">
Towards the end, Q-Learning seems to find a better policy.<br><br>

Why Q-learning does better than Sarsa here?<br>
Q-Learning takes the max over next action values. So it only changes when the agent learns that one action is better than another. In contrast, SARSA uses the estimate of the next action value in its target.<br><br>

If we set $\alpha$ = 0.1<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_2_5.png"/></center>
</p>
<p align="justify">
Sarsa learns a same policy but more slowly. This experiment highlights the impact of parameters in Reinforcement Learning, including $\alpha$, $\epsilon$, initial value function etc.<br><br>

<b>Comparison between Sarsa and Q-Learning</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_2_6.png"/></center>
</p>
<p align="justify">
<b>Target and Behavior Policies for Q-Learning</b><br>
Target Policy is always greedy with respect to its current values, while Behavior Policy can be anything that continues to visit all pairs during the learning, e.g. $\epsilon$-greedy.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_2_7.png"/></center>
</p>
<p align="justify">
Here is a question, <b>since Q-learning learns off-policy, why don't we see any important sampling ratios?</b><br>
It is because the agent is estimating action values with unknown policy. It does not need important sampling ratios to correct for the difference in action selection. The action value function represents the returns following each action in a given state. The agent's target policy represents the probability of taking each action in a given state. Putting these two elements together, the agent can calculate the expected return under its target policy from any given state, in particular, the next state, $S_{t+1}$. Q-learning uses exactly this technique to learn off-policy. Since the agents target policies greedy, with respect to its action values, all non-maximum actions have probability 0. As a result, the expected return from that state is equal to a maximal action value from that state.<br><br>

<b>Cliff Walking</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_2_8.png"/></center>
</p>
<p align="justify">
Since Q-learning learns the optimal value function, it quickly learns that an optimal policy travels right alongside the cliff. However, since his actions or epsilon greedy, traveling alongside the cliff occasionally results and falling off of the cliff. Sarsa learns about his current policy, taking into account the effect of epsilon greedy action selection. Accounting for occasional exploratory actions, it learns to take the longer but more reliable path. They usually avoids randomly falling into the cliff. Because of it's safer path, Sarsa is able to reach the goal more reliably.<br>
</p>

#### 2.3.3 Expected Sarsa
<p align="justify">
<b>The Bellman Equation for action-values</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_3_1.png"/></center>
</p>
<p align="justify">
The agent has already known the policy, it can compute the expected value function directly.<br><br>

<b>Expected Sarsa Algorithm</b><br>
The expected Sarsa algorithm is similar to Sarsa expcet TD error uses the expected estimate of next action value instead of a sample of next action value. In other word, the agent has to average the next state's action value under its policy on every step.<br>
$$Q(S_{t}, A_{t}) \leftarrow Q(S_{t}, A_{t}) +\alpha [R_{t+1} + \gamma \sum_{a'} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a') - Q(S_{t}. A_{t})]$$

For example,<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_3_2.png"/></center>
</p>
<p align="justify">
Expected Sarsa is more table in update target than Sarsa.<br><br>

<b>Satbility in the Update Target</b><br>
Consider a reward is 1. Both Sarsa and Expected Sarsa start with true value function for next state.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_3_3.png"/></center>
</p>
<p align="justify">
The next action sampling that Sarsa does can cause it uo update its value in a wrong direction. It relies on the fact that in expectation across mutilple updates, the direction is correct. But, Expected Sarsa update targets are exactly correct and do not change the estimated values away from the true values.<br><br>

<b>In fact, Expected Sarsa update targets are much lower variance than Sarsa.</b> But Expected Sarsa need much computation ressource. If there are many actions, computing the average might take a long time.<br><br>

The backup diagrams for Q-learning and Expected Sarsa.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_3_4.png"/></center>
</p>
<p align="justify">
<b>Off-Policy Expected Sarsa</b><br>
Consider the Expected Sarsa update. The next action is sampled from $\pi$ in this case. However, notice that the expectation over actions is computed independently of the action actually selected in the next state.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_3_5.png"/></center>
</p>
<p align="justify">
In other word, $\pi$ need not to be equal to behavior policy. So, Expected Sarsa is off-policy like Q-Learning.<br><br>

<b>Greedy Expected Sarsa</b><br>
If we take a greedy policy as target policy with respect to action value estimates.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_3_6.png"/></center>
</p>
<p align="justify">
In other word, Q-Learning is a special case of Expected Sarsa with greedy target policy.<br><br>

<b>TD control and Bellman equations</b><br>
Sarsa uses a sample based version of the Bellman euqation to learn $q_{\pi}$, Q-Learning uses the Bellman optimality equation to learn $q_{*}$. Expected Sarsa uses a same Bellman equation like Sarsa, but Expected Sarsa takes an average over the next action values.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_3_3_7.png"/></center>
</p>
<p align="justify">
Sarsa can do better than Q-learning when performance is measured online. This is because on-policy control methods account for their own exploration.<br>
</p>

#### 2.3.4 Programming Assignment: Q-learning and Expected Sarsa
<p align="justify">

</p>

#### 2.3.5 Quiz: Practice Quiz
<p align="justify">

</p>

### 2.4 Planning, Learning & Acting
<p align="justify">
Weekly reading: Chapter 8.1-8.3 (pp. 159-166).<br>
</p>

#### 2.4.1 What is a Model?
<p align="justify">
<b>Model store knowledge</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_1_1.png"/></center>
</p>
<p align="justify">
In this course, model store knowledge about the transition and reward dynamics.<br><br>

A model allows for planning. Planning refers to the process of using a model to improve a policy.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_1_2.png"/></center>
</p>
<p align="justify">
<b>Types of Models</b><br>
We have two types: sample model and distribution model.<br>
Sample model produces an actual outcome drawn from some underlying probabilities; distribution model completely specifies the likelihood or probability of every outcome.<br><br>

<b>Advantages</b><br>
Sample models require less memory; distribution models can be used to compute the exact expected outcome.<br>
</p>

#### 2.4.2 Random Tabular Q-planning
<p align="justify">
<b>Planning improves policies</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_2_1.png"/></center>
</p>
<p align="justify">
One possible approach to planning is to first sample experience from the model.<br><br>

<b>Connection with Q-Learning</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_2_2.png"/></center>
</p>
<p align="justify">
<b>Random-sample one-step tabular Q-Planning</b><br>
Suppose we have a sample model of the transition dynamics and a strategy for sampling relevant state action pairs. One possible option is sample states and actions uniformly.<br><br>

his algorithm first chooses a state action pair at random from the set of all states and actions. It then queries the sample model with this state action pair to produce a sample of the next state and reward. It then performs a Q-Learning Update on this model transition. Finally, it improves the policy by beautifying with respect to the updated action values.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_2_3.png"/></center>
</p>
<p align="justify">
A key point is that this planning method only uses imagined or simulated experience.<br>
</p>

#### 2.4.3 Dyna
<p align="justify">
<b>The Dyna Architecture</b><br>
First, we have environment and policy, which generate a stream of experience. Then we use this experience to perform direct RL updates. To do planning, we need a model coming from somewhere. Well, the environment experience can be used to learn the model. This model will generate model experience. Besides, we want to control how the model generates this simulated experience. We call this process <b>search control</b>. Planning update are performed using the experience generated by the model.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_3_1.png"/></center>
</p>
<p align="justify">
Look at an example of a simple maze where a robot try to find the exit E. This is a discounted problem. The robot receieves 1 at goal, 0 otherwise.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_3_2.png"/></center>
</p>
<p align="justify">
One direct RL update using Q-Learning<br>
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \cdot \max_{a} Q(s', a') - Q(s, a)]$$

As usually, the robot knows nothing. It randomly traverses the maze until it reaches the goal for the first time. At this time, the robot only updates one action value showed in a up-arrow because this is only one transition with non-zero reward This update is done by direct RL. <b>Dyna makes use of all the experience correspond to states visited during the first episode.</b> Dyna performs planning on every time step. However, planning has no impact on the policy during the first episode. After the first episode completes, planning can really shine.<br><br>

$\bigstar$ Direct RL updates use environment experience to improve a policy or value function.<br>
$\bigstar$ Planning updates use model experience to improve a policy or value function.<br><br>

<b>The Dyna Algorithm</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_3_3.png"/></center>
</p>
<p align="justify">
<b>More planning $\rightarrow$ faster learning</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_3_4.png"/></center>
</p>
<p align="justify">
<b>Bonus rewards for exploration</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_3_5.png"/></center>
</p>
<p align="justify">
<b>The Dyna-Q+ Algorithm</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_3_6.png"/></center>
</p>
<p align="justify">
Adding this reward bonus to Dyna-Q's planning updates results in the Dyna-Q+ algorithm.<br><br>

<b>Dyna-Q vs Dyna-Q+ in a chaning environment</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_3_7.png"/></center>
</p>
<p align="justify">
Dyna-Q+ can find the short caut while Dyna-Q cannot.<br><br>

Backup diagrams for all the one-step updates considered in this book<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_3_8.png"/></center>
</p>
<p align="justify">
If there is enough time to complete an expected update, then the resulting estimate is generally better than that of b sample updates because of the absence of sampling error. But if there is insu cient time to complete an expected update, then sample updates are always preferable because they at least make some improvement in the value estimate with fewer than b updates. In a large problem with many state–action pairs, we are often in the latter situation.<br>
</p>
<p align="justify">
We summarize all algorithms in a map<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/2_4_3_9.png"/></center>
</p>

#### 2.4.4 Quiz: Practice Assessment
<p align="justify">

</p>

#### 2.4.5 Programming Assignment: Dyna-Q and Dyna-Q+
<p align="justify">

</p>

## 3. Prediction and Control with Function Approximation
### 3.1 On-policy Prediction with Approximation
<p align="justify">
Ch. 9.1 - 9.4 (pp. 197- 209)<br>
</p>

#### 3.1.1 Parameterized Functions
<p align="justify">
<b>Parameterizing the value function</b><br>
$$\hat{v}(s, w) \approx v_{\pi}(s)$$

w is weight. For example<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_1_1.png"/></center>
</p>
<p align="justify">
<b>Linear Vlaue Function Approximation</b><br>
$$\hat{v}(s, w) = \sum w_{i}x_{i}(s) = < w, x(s) >$$

Tabular value functions are in fact linear functions.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_1_2.png"/></center>
</p>
<p align="justify">
<b>Nonlinear Function Approximation</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_1_3.png"/></center>
</p>
<p align="justify">
<b>Generalization: Update to One State Affect the Value of Other State.</b><br>
Generalization can speed learning by making better use of the experience we have.<br><br>

<b>Discrimination: The ability to make the value of two states different.</b><br><br>

Tabular methods have high discrimination but low generalization; Aggregat all states methods have high generalization but low discrimination.<br><br>

<b>Framing Value Estimation as Supervised Learning</b><br><br>
Monte Carlo methods estimate the value function using samples of the return. We can think of this as an example of a supervised learning problem, where input is the state and targets/labels are the returns.<br>
TD can also be framed as supervised learning, where the targets are the one-step bootstrap return. In principle, any function approxiamtion technique from supervised learning can be applied to be policy evaluation task.<br><br>

The function approximator should be compatible with bootstrapping.<br><br>

<b>The Mean Squared Value Error Objective</b><br>
The mean squared value error is a metric to measure the difference between value function and estimated value function.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_1_4.png"/></center>
</p>
<p align="justify">
$$\text{Mean Squared Value Error} = \sum_{s} \mu(s) [v_{\pi}(s) - \hat{v}(s, w)]^{2}$$

$\mu(s)$ means how much we care about each state. A natural measure is the fraction of time we spend in S when following policy $\pi$.<br><br>

<b>Adapting the weights to minimize the mean squared value error objective</b><br>
$$\overline{VE} = \sum_{s} \mu(s) [v_{\pi}(s) - \hat{v}(s, w)]^{2}$$
</p>

#### 3.1.2 Gradient Descent
<p align="justify">
<b>Gradient: Derivatives in multiple domensions</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_2_1.png"/></center>
</p>
<p align="justify">
<b>Gradient Descent</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_2_2.png"/></center>
</p>
<p align="justify">
<b>Gradient of the Mean squared Value Error Objective</b><br>
$$\nabla \sum_{s} \mu(s) [v_{\pi}(s) - \hat{v}(s, w)]^{2} = - \sum_{s \in S} \mu(s) 2 [v_{\pi} - \hat{v}(s, w)] \nabla \hat{v}(s, w)$$
$$\nabla \hat{v}(s, w) = < w, x(s) > = x(s)$$
$$\nabla w \propto \sum_{s \in S} \mu(s) 2 [v_{\pi} - \hat{v}(s, w)] \nabla \hat{v}(s, w)$$

Stochastic Gradient Descent is much less computational.<br><br>

<b>Gradient Monte Carlo for Policy Evaluation</b><br>
$$w_{t+1} = w_{t} + \alpha [v_{\pi}(S_{t}) - \hat{v}(S_{t}, w_{t})] \nabla \hat{v}(S_{t}, w_{t})$$
$$w_{t+1} = w_{t} + \alpha [G_{t} - \hat{v}(S_{t}, w_{t})] \nabla \hat{v}(S_{t}, w_{t})$$
$$E_{\pi} [2[v_{\pi}(S_{t}) - \hat{v}(S_{t}, w)]\nabla \hat{v}(S_{t}, w)] = E_{\pi} [2[G_{t} - \hat{v}(S_{t}, w)] \nabla \hat{v}(S_{t}, w)]$$

<b>Gradient Monte Carlo Algorithm</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_2_3.png"/></center>
</p>
<p align="justify">
<b>State Aggregation</b><br>
We group some states with a same feature. We update a group of state at a same time.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_2_4.png"/></center>
</p>
<p align="justify">
<b>State Aggregation with Monte Carlo</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_2_5.png"/></center>
</p>
<p align="justify">
<b>Gradient Monte Carlo</b><br>
$$w \leftarrow w + \alpha [G_{t} - \hat{v}(S_{t}, w)] \nabla \hat{v}(S_{t}, w)$$

<b>The TD update for Function Approximation</b><br>
$$w \leftarrow w + \alpha [U_{t} - \hat{v}(S_{t}, w)]\nabla \hat{v}(S_{t}, w)$$
$$U_{t} = R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$$

$U_{t}$ is biased $\rightarrow$ w may bot converge to a local optimum.<br><br>

<b>Semi-Gradient TD for Policy Evaluation</b><br>
TD is a semi-gradient method.<br>
$$\nabla \frac{1}{2} [U_{t} - \hat{v}(S_{t}, w)]^{2} = (U_{t} - \hat{v}(S_{t}, w))(\nabla U_{t} - \nabla \hat{v}(S_{t}, w)) \neq -(U_{t} - \hat{v}(S_{t}, w))\nabla \hat{v}(S_{t}, w)$$

Why $\nabla U_{t} \neq 0$?<br>
$$\nabla U_{t} = \nabla (R_{t+1} + \gamma \hat{v}(S_{t+1}, w)) = \gamma \nabla \hat{v}(S_{t+1}, w) \neq 0$$

Semi-gradient TD(0) algorithm<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_2_6.png"/></center>
</p>
<p align="justify">
<b>Comparing TD and Monte Carlo with State Aggregation</b><br>
$\bigstar$ Gradient Monte Carlo will converge to a local minimum of the Mean Squared Value Error, because of unbiased $G_{t}$<br>
$\bigstar$ Semi-gradient TD will not necessarily converge to a local minimum of the Mean Squared Value Error.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_2_7.png"/></center>
</p>
<p align="justify">
<b>TD update with Linear Function Approximation</b><br>
$$w \leftarrow w +\alpha \delta_{t} \nabla \hat{v}(S_{t}, w)$$
$$\delta_{t} = R_{t+1} + \gamma \hat{v} (S_{t+1}, w) - \hat{S_{t}, w}$$
$$\hat{v}(S_{t}, w) = w^{T} x(S_{t})$$
$$\nabla \hat{v}(S_{t}, w) = x(S_{t})$$
$$w \leftarrow w +\alpha \delta_{t} x(S_{t})$$

<b>Tabular TD is a special case of linear TD</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/RL/3_1_2_8.png"/></center>
</p>
<p align="justify">
<b>The utility of Linear Function Approximation</b><br>
$\bigstar$ Linear methods are simple to understand and analyse mathematically.<br>
$\bigstar$ With good features, linear methods can learn quickly and achieve good prediction accuracy.<br><br>

<b>Expected TD Update</b><br>
$$\hat{v}(s, w) = w^{t}x(s)$$
$$w_{t+1} = w_{t} + \alpha [R_{t+1} + \gamma \hat{v}(S_{t+1}, w_{t}) - \hat{v}(S_{t}, w_{t})]x_{t}$$
$$= w_{t} + \alpha [R_{t+1} + \gamma w_{t}^{T}x_{t+1} - w_{t}^{T}x_{t}]x_{t}$$
$$= w_{t} + \alpha [R_{t+1} x_{t} - x_{t} (x_{t} - \gamma x_{t+1})^{T} w_{t}]$$

$$E[\Delta w_{t}] = \alpha (\mathbf{b} - \mathbf{A} w_{t})$$
$$\mathbf{b} = E[R_{t+1} x_{t}]$$
$$\mathbf{A} = E[x_{t} (x_{t} - \gamma x_{t+1})^{T}]$$

<b>TD Fixed Point</b><br>
$$E[\Delta w_{TD}] = \alpha (\mathbf{b} - \mathbf{A} w_{TD}) = 0$$
$$\Rightarrow w_{TD} = \mathbf{A}^{-1} \mathbf{b}$$

Linear TD converges to fixed point.
$$w_{TD} \text{ minimizes } (b - Aw)^{T}(b - Aw)$$

<b>Relating the TD Fixed Point and the Minimum of the Value Error</b><br>
$$\overline{VE}(w_{TD}) \leq \frac{1}{1 - \gamma} \min_{w} \overline{VE}(w)$$

If $\gamma \rightarrow 1$, the TD Fixed Point will very close to the Minimum Squared Value Error solution.<br>
</p>

#### 3.1.3 Practice Quiz: On-policy Prediction with Approximation
<p align="justify">
<b>1.</b><br>
Which of the following statements is true about function approximation in reinforcement learning? (Select all that apply)<br>
A. We only use function approximation because we have to for large or continuous state spaces. We would use tabular methods if we could, and learn an independent value per state.<br>
B. It can be more memory efficient.<br>
C. It can help the agent achieve good generalization with good discrimination, so that it learns faster and represent the values quite accurately.<br>
D. It allows faster training by generalizing between states.<br><br>

<b>Answer:</b> B, C, D.<br><br>

<b>2.</b><br>
We learned how value function estimation can be framed as supervised learning. But not all supervised learning methods are suitable. What are some key differences in reinforcement learning that can make it hard to apply standard supervised learning methods?<br>
A. When using bootstrapping methods like TD, the target labels change.<br>
B. Reinforcement learning is usually done in an online setting where the full dataset is not fixed and unavailable from the beginning.<br>
C. Data is available as a fixed batch.<br>
D. Data is temporally correlated in reinforcement learning.<br><br>

<b>Answer:</b> A, B, D.<br><br>

<b>3.</b><br>
Value Prediction (or Policy Evaluation) with Function Approximation can be viewed as supervised learning mainly because _________. [choose the most appropriate completion of the proceeding statement]<br>
A. We use stochastic gradient descent to learn the value function.<br>
B. Each state and its target estimate (used in the Monte Carlo update, TD(0) update, and DP update) can be seen as input-output training examples to estimate a continuous function.<br>
C. We can learn the value function by training with batches of data obtained from the agent’s interaction with the world.<br><br>

<b>Answer:</b> B.<br><br>

<b>4.</b><br>
Which of the following is true about using Mean Squared Value Error<br>
$$\overline{VE} = \sum_{s} \mu(s) [v_{\pi}(s) - \hat{v}(s, w)]^{2}$$
as the prediction objective? $\mu(s)$ represents the weighted distribution of visited states (Select all that apply).<br>
A. Gradient Monte Carlo with linear function approximation converges to the global optimum of this objective, if the step size is reduced over time.<br>
B. The agent can get zero MSVE when using a linear representation that cannot represent the true values.<br>
C. The agent can get zero MSVE when using a tabular representation that can represent the true values.<br>
D. This objective makes it explicit how we should trade-off accuracy of the value estimates across states, using the weighting $\mu$.<br><br>

<b>Answer:</b> A, C, D.<br><br>

<b>5.</b><br>
Which of the following is true about $\mu(S)$ in Mean Squared Value Error? (Select all that apply)<br>
A. It serves as a weighting to minimize the error more in states that we care about.<br>
B. If the policy is uniformly random, $\mu(S)$ would have the same value for all states.<br>
C. It has higher values for states that are visited more often.<br>
D. It is a probability distribution.<br><br>

<b>Answer:</b> A, C, D.<br><br>

<b>6.</b><br>
If we are given the true value function $v_{\pi}(S_{t})$, the stochastic gradient descent update would be as follows. Fill in the blanks (A), (B), (C ) and (D) with correct terms. (Select all correct answers)<br>
$$w_{t+1} = w_{t} (A) \frac{1}{2} \alpha \nabla [(C) - (D)]^{2} = w_{t} (B) \alpha [(C) - (D)] \nabla \hat{v}(S_{t}, w_{t}) \quad \alpha > 0$$
A. -, -, $\hat{v}(S_{t}, w_{t})$, $v_{\pi}(S_{t})$<br>
B. -, +, $v_{\pi}(S_{t})$, $\hat{v}(S_{t}, w_{t})$<br>
C. +, +, $\hat{v}(S_{t}, w_{t})$, $v_{\pi}(S_{t})$<br>
D. +, -, $v_{\pi}(S_{t})$, $\hat{v}(S_{t}, w_{t})$<br><br>

<b>Answer:</b> A, B.<br><br>

<b>7.</b><br>
In a Monte Carlo Update with function approximation, we do stochastic gradient descent using the following gradient:<br>
$$\nabla [G_{t} - \hat{v}(s, w)]^{2} = 2 [G_{t} - \hat{v}(s, w)] \nabla (-\hat{v}(S_{t}, w_{t})) = -1 * 2 [G_{t} - \hat{v}(s, w)] \nabla \hat{v}(S_{t}, w_{t})$$

But the actual Monte Carlo Update rule is the following:
$$w_{t+1} = w_{t} + \alpha [G_{t} - \hat{v}(S_{t}, w_{t})] \nabla \hat{v}(S_{t}, w_{t}), \quad \alpha > 0$$

Where did the constant -1 and 2 go when $\alpha$ is positive? (Choose all that apply)<br>
A. We assume that the 2 is included in the step-size.<br>
B. We assume that the 2 is included in $\nabla \hat{v}(S_{t}, w_{t})$<br>
C. We are performing gradient descent, so we subtract the gradient from the weights, negating -1.<br>
D. We are performing gradient ascent, so we subtract the gradient from the weights, negating -1.<br><br>

<b>Answer:</b> A, C.<br><br>

<b>8.</b><br>
When using stochastic gradient descent for learning the value function, why do we only make a small update towards minimizing the error instead of fully minimizing the error at each encountered state?<br>
A. Because small updates guarantee we can slowly reduce approximation error to zero for all states.<br>
B. Because we want to minimize approximation error for all states, proportionally to $\mu$.<br>
C. Because the target value may not be accurate initially for both TD(0) and Monte Carlo method.<br><br>

<b>Answer:</b> B.<br><br>

<b>9.</b><br>
The general stochastic gradient descent update rule for state-value prediction is as follows:<br>
$$w_{t+1} = w_{t} + \alpha [U_{t} - \hat{v}(S_{t}, w_{t})] \nabla \hat{v}(S_{t}, w_{t})$$

For what values of $U_{t}$ would this be a semi-gradient method?<br>
A. $R_{t+1} +\hat{v}(S_{t+1}, w_{t})$<br>
B. $G_{t}$<br>
C. $v_{\pi}(S_{t})$<br>
D. $R_{t+1} + R_{t+2} + ... + R_{T}$<br><br>

<b>Answer:</b> A.<br><br>

<b>10.</b><br>
Which of the following statements is true about state-value prediction using stochastic gradient descent? (Select all that apply)<br>
$$w_{t+1} = w_{t} + \alpha [U_{t} - \hat{v}(S_{t}, w_{t})] \nabla \hat{v}(S_{t}, w_{t})$$

A. Stochastic gradient descent updates with Monte Carlo targets always reduce the Mean Squared Value Error at each step.<br>
B. Using the Monte Carlo return as target, and under appropriate stochastic approximation conditions,  the value function will  converge to a local optimum of the Mean Squared Value Error.<br>
C. When using $U_t = R_{t+1} +\hat{v}(S_{t+1},\mathbf{w_t})$, the weight update is not using the true gradient of the TD error.<br>
D. Using the Monte Carlo return or true value function as target results in an unbiased update.<br>
E. Semi-gradient TD(0) methods typically learns faster than gradient Monte Carlo methods.<br><br>

<b>Answer:</b> B, C, D, E.<br><br>

<b>11.</b><br>
Which of the following is true about the TD fixed point? (Select all correct answers)<br>
A. At the TD fixed point, the mean squared value error is not larger than $\frac{1}{1-\gamma}$ times the mean squared value error of the global optimum, assuming the same linear function approximation.<br>
B. The weight vector corresponding to the TD fixed point is a local minimum of the Mean Squared Value Error.<br>
C. The weight vector corresponding to the TD fixed point is the global minimum of the Mean Squared Value Error.<br>
D. Semi-gradient TD(0) with linear function approximation converges to the TD fixed point.<br><br>

<b>Answer:</b> A, D.<br>
The weight vector corresponding to the TD fixed point may not even be a stationary point of the Mean Squared Value Error.<br><br>

<b>12.</b><br>
Which of the following is true about Linear Function Approximation, for estimating state-values? (Select all that apply)<br>
A. The size of the feature vector is not necessarily equal to the size of the weight vector.<br>
B. The gradient of the approximate value function $\hat{v}(s,w)$ with respect to w is just the feature vector.<br>
C. Features are often called basis functions because every approximate value function we consider can be written as a linear combination of these features.<br>
D. State aggregation is one way to generate features for linear function approximation.<br><br>

<b>Answer:</b> B, C, D.<br>
</p>

#### 3.1.4 Programming Assignment: Semi-gradient TD(0) with State Aggregation
<p align="justify">

</p>

### 3.2 Constructing Features for Prediction
<p align="justify">
Ch. 9.4 - 9.5.0 (pp. 204 - 210), 9.5.3 - 9.5.4 (pp 215 - 222) and 9.7 (pg 223 - 228)<br>
</p>

#### 3.2.1 Coarse Coding
<p align="justify">

</p>

#### 3.2.2 Generalization Properties of Coarse Coding
<p align="justify">

</p>

#### 3.2.3 Tile Coding
<p align="justify">

</p>

#### 3.2.4 Using Tile Coding in TD
<p align="justify">

</p>

#### 3.2.5 What is a Neural Network?
<p align="justify">

</p>

#### 3.2.6 Non-linear Approximation with Neural Networks
<p align="justify">

</p>

#### 3.2.7 Deep Neural Networks
<p align="justify">

</p>

#### 3.2.8 Gradient Descent for Training Neural Networks
<p align="justify">

</p>

#### 3.2.9 Optimization Strategies for NNs
<p align="justify">

</p>

#### 3.2.10 David Silver on Deep Learning + RL = AI?
<p align="justify">

</p>

#### 3.2.11 Practice Quiz: Constructing Features for Prediction
<p align="justify">

</p>

#### 3.2.12 Programming Assignment: Semi-gradient TD with a Neural Network
<p align="justify">

</p>

### 3.3 Control with Approximation
<p align="justify">
Chapter 10 (pp. 243-246) & 10.3 (pp. 249-252)<br>
</p>

#### 3.3.1 Episodic Sarsa with Function Approximation
<p align="justify">

</p>

#### 3.3.2 Episodic Sarsa in Mountain Car
<p align="justify">

</p>

#### 3.3.3 Expected Sarsa with Function Approximation
<p align="justify">

</p>

#### 3.3.4 Exploration under Function Approximation
<p align="justify">

</p>

#### 3.3.5 Average Reward: A New Way of Formulating Control Problems
<p align="justify">

</p>

#### 3.3.6 Satinder Singh on Intrinsic Rewards
<p align="justify">

</p>

#### 3.3.7 Practice Quiz: Control with Approximation
<p align="justify">

</p>

#### 3.3.8 Programming Assignment: Function Approximation and Control
<p align="justify">

</p>

### 3.4 Policy Gradient
<p align="justify">
Chapter 13 (pp. 321 - 336)<br>
</p>

#### 3.4.1 Learning Policies Directly
<p align="justify">

</p>

#### 3.4.2 Advantages of Policy Parameterization
<p align="justify">

</p>

#### 3.4.3 The Objective for Learning Policies
<p align="justify">

</p>

#### 3.4.4 The Policy Gradient Theorem
<p align="justify">

</p>

#### 3.4.5 Estimating the Policy Gradient
<p align="justify">

</p>

#### 3.4.6 Actor-Critic Algorithm
<p align="justify">

</p>

#### 3.4.7 Actor-Critic with Softmax Policies
<p align="justify">

</p>

#### 3.4.8 Demonstration with Actor-Critic
<p align="justify">

</p>

#### 3.4.9 Gaussian Policies for Continuous Actions
<p align="justify">

</p>

#### 3.4.10 Practice Quiz: Policy Gradient Methods
<p align="justify">

</p>

#### 3.4.11 Programming Assignment: Average Reward Softmax Actor-Critic with Tile-coding
<p align="justify">

</p>


## 4. A Complete Reinforcement Learning System (Capstone)
### 4.1 Quiz
#### 4.1.1 Choosing the Right Algorithm
<p align="justify">

</p>

#### 4.1.2 Impact of Parameter Choices in RL
<p align="justify">

</p>

### 4.2 Notebook
#### 4.2.1 MoonShot Technologies
<p align="justify">

</p>

#### 4.2.2 Implement Your Agent
<p align="justify">

</p>

#### 4.2.3 Completing the parameter study
<p align="justify">

</p>


## 5. Reference:
<p align="justify">
[1] <a href="https://books.google.fr/books?id=sWV0DwAAQBAJ&printsec=frontcover&dq=Sutton,+Richard+S.,+and+Andrew+G.+Barto.+Reinforcement+learning:+An+introduction.+MIT+press,+2018.&hl=en&sa=X&ved=0ahUKEwiI_ZO4zJ7oAhXTAWMBHX8ZDwcQ6AEIKDAA#v=onepage&q=Sutton%2C%20Richard%20S.%2C%20and%20Andrew%20G.%20Barto.%20Reinforcement%20learning%3A%20An%20introduction.%20MIT%20press%2C%202018.&f=false"> Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.</a><br>
</p>
