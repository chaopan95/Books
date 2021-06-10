---
layout: post
title:  "Immortal Flappy Bird"
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


<p align="justify">
The complete code is available <a href="https://github.com/chaopan1995/PROJECTS/tree/master/ImmortalFlappyBird">here</a>.
</p>


## 1. Introduction
<p align="justify">
Flappy Bird is a simple but interesting game, where one player controls the "bird" up or down in order to succeed in passing some obstacles. Two actions can be taken during the game: press the "up" button to make the bird jump, in contrast it will fall without pressing any button. In this paper, we want to realize that a computer can play this game automatically and accurately Reinforcement Learning.
</p>


## 2. Reinforcement Learning
<p align="justify">
Reinforcement learning is a branch of machine learning. It highlights how to act in accordance with the environment to maximize the expected benefits.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/ImmortalFlappyBird/1.png"/></center>
</p>
<p align="justify">
One agent is in a specific environment and the agent determine a next action according to the current state. Once an action is carried out, the agent will go to the next state and it will receive a reward from the environment.
</p>

### 2.1 Markov Decision Processes
<p align="justify">
Markov decision processes (MDP) involves evaluative feedback, but also an associative aspect—choosing different actions in different situations. MDPs are a classical formalization of sequential decision making, where actions influence not just immediate rewards, but also subsequent situations, or states, and through those future rewards.
$$S_{0}, A_{0}, R_{1}, S_{1}, A_{1}, R_{2}, S_{2}, A_{2}, R_{3}, S_{3}, ...$$
</p>

### 2.2 Value functions
<p align="justify">
We denote the action selected on time step t as , and the corresponding reward as . The value then of an arbitrary action a, denoted , is the expected reward given that a is selected
$$q_{*}(a) = E[R_{t} \mid A_{t} = a]$$

In general, we don’t know the exact value of $q_{*}(a)$, so we usually estimate this value using $Q_{t}(a)$
$$Q_{t}(a) = \frac{\sum_{i=1}^{t-1} R_{i} \mathbf{1}_{A_{i}=a}}{\sum_{i=1}^{t-1} \mathbf{1}_{A_{i}=a}} = \frac{\sum_{i=1}^{t-1} R_{i}}{t-1}$$

As the denominator goes to infinity, by the law of large numbers, $Q_{t}(a)$ converges to $q_{*}(a)$. The simplest action selection rule is to select one of the actions with the highest estimated value, which is called greedy action
$$A_{t} = \arg \max_{a} Q_{t}(a)$$

One of the challenges that arise in reinforcement learning is the tradeoff between exploration and exploitation. To obtain a lot of reward, an agent must prefer actions that it has tried in the past and found to be effective in producing reward. But to discover such actions, it has to try actions that it has not selected before. The agent has to exploit what it has already experienced in order to obtain reward, but it also has to explore in order to make better action selections in the future. This is a dilemma between exploitation and exploration.<br><br>

we focus on a next state
$$Q_{n+1} = \frac{\sum_{i=1}^{n}R_{i}}{n} = \frac{1}{n}(R_{n} + \sum_{i=1}^{n-1}R_{i}) = \frac{1}{n}(R_{n} + (n-1)\frac{1}{n-1}\sum_{i=1}^{n-1}R_{i})$$
$$= \frac{1}{n}(R_{n} + (n-1)Q_{n}) = Q_{n} + \frac{1}{n}(R_{n} - Q_{n})$$

we get an iteration equation for Q and this equation is fundamental in reinforcement learning
$$\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \cdot (\text{Target} - \text{OldEstimate})$$
</p>

### 2.3 Return
<p align="justify">
In reinforcement learning, the agent’s goal is to maximize the total amount of reward it receives. This means maximizing not immediate reward, but cumulative reward in the long run. For a sequence of states with a discount rate, the return is defined as follows
$$G_{t} = R_{t+1} + \gamma R_{t+2} + \gamma^{2}R_{t+3}+ \cdots = \sum_{k=0}^{\infty} \gamma^{k}R_{t+k+1}$$
</p>

### 2.4 Deep QLearning Network
<p align="justify">
QLearning is an off-policy algorithm of Temporal-Difference (TD) Learning without a transition of probabilities.
$$Q(S_{t}, A_{t}) \leftarrow Q(S_{t}, A_{t}) + \alpha[R_{t+1} + \gamma \max_{a}Q(S_{t+1}, a) - Q(S_{t}, A_{t})]$$

This update is done after each transition from a nonterminal state $S_{t}$, if $S_{t+1}$ is a terminal, then $Q(S_{t+1}, a)$ is define as zeros. Besides, QLearning contains two main parts, choose an action from Q and update Q. The first part takes a -greedy method while the second takes greedy method. The policy evaluation and policy improvement use different method, which is called off-policy.<br><br>

The traditional QLearning approximate the value functions in a form of table, while Deep QLearning Network estimate the value functions with a series of parameters or weight
$$\hat{q_{}}(s, a, w) \approx q_{*}(s, a)$$

Gradient descent update for state action value functions
$$w_{t+1} = w_{t} + \alpha[R_{t+1} + \gamma \max_{a}Q(S_{t+1}, a) - \hat{q_{}}(S_{t}, A_{t}, w_{t})]\nabla \hat{q_{}}(S_{t}, A_{t}, w_{t})$$

The loss function is defined as a deference between a current Q et an updated Q
$$L_{t}(w_{t}) = \left \| \hat{q_{}}_{update}(S_{t}, A_{t}) - \hat{q_{}}(S_{t}, A_{t}) \right \|^{2}$$

To crack this problem, we take neural network to implement our algorithm. In details, there is a specific container for our training data. Because our bird learns from its past experience, our training data is a set of ancient images or experiences. In order to facilitate neural network, we resize our input image from (400, 700) to (80, 80) and convert a color image into a binary image. Since our game is dynamic, one image is not enough to detect a velocity of moving for our bird, we put 4 consecutive images together, like an image with 4 channels. Then we construct our forward network with 3 convolutional layers (convolutional kernel 8×8×4×32, 4×4×32×64, 3×3×64×64) and 2 fully-connected layers for getting 2 output nodes. Besides, in order to avoid overfitting, we introduce a regularizer (0.001). then in the backpropagation step, we choose the Adam optimizer with a learning rate 1e-6.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/ImmortalFlappyBird/2.png"/></center>
</p>

### 2.5 Algorithm 
<p align="justify">
<b>01</b>. Input: Well-programmed game Flappy Bird<br>
<b>02</b>. Output: Well-trained DQN<br>
<b>03</b>. Initialize a game. Give an initial action (generally 0) for bird and get an initial state value function $S_{t}$. Step counter t = 0, episode = 0. Construct forward propagation network. Set a container D for ancient experience with a maximum size $D_{max}$. Set a mini batch size B. Set error tolerance e<br>
<b>04</b>. While True do:<br>
<b>05</b>. &emsp;Calculate $\hat{q_{}}(S_{t}, A_{t}, w_{t})$ via neural network<br>
<b>06</b>. &emsp;Choose $A_{t}$ with ε-greedy method then get $S_{t+1}$, $R_{t}$, $IsTerminal_{t+1}$<br>
<b>07</b>. &emsp;Put a tuple ($S_{t}$, $A_{t}$, $R_{t}$, $S_{t+1}$, $IsTerminal_{t+1}$) into D<br>
<b>08</b>. &emsp;if $\left \| D \right \| \geq D_{max}$ do:<br>
<b>09</b>. &emsp;&emsp;Pop left D<br>
<b>10</b>. &emsp;end if<br>
<b>11</b>. &emsp;Pick up randomly B data from D and calculate $\hat{q_{}}_{update}(S_{t}, A_{t})$<br>
<b>12</b>. &emsp;Get a loss function $L_{t}(w_{t})$ and put it in backward propagation<br>
<b>13</b>. &emsp;if t % N == 1 do:<br>
<b>14</b>. &emsp;&emsp;Save compute graph and the parameters as checkpoint file<br>
<b>15</b>. &emsp;end if<br>
<b>16</b>. &emsp;if $IsTerminal_{t+1}$ is true do:<br>
<b>17</b>. &emsp;&emsp;Restart a game and episode += 1<br>
<b>18</b>. &emsp;end if<br>
<b>19</b>. &emsp;if loss value if less than e do:<br>
<b>20</b>. &emsp;&emsp;Break loop<br>
<b>21</b>. &emsp;end if<br>
<b>22</b>. &emsp;Step t += 1<br>
<b>23</b>. return<br><br>

We have trained our bird with TensorFlow. After 12000 episodes, it can acquire 350 scores. It proves that our algorithm works well, and our bird have the intelligence to some extent.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/ImmortalFlappyBird/3.png"/></center>
</p>


## 3. How to run it?
<p align="justify">
We use tensorflow 1.x<br><br>

Run DQN.py to see a well-trained bird
</p>


## 4. Reference:
<p align="justify">
[1] Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.<br>
[2] Kevin Chen. Deep Reinforcement Learning for Flappy Bird<br>
[3] https://github.com/yenchenlin/DeepLearningFlappyBird<br>
[4] https://blog.csdn.net/songrotek/article/details/50951537<br>
[5] https://blog.csdn.net/qq_32892383/article/details/89646221<br>
</p>
