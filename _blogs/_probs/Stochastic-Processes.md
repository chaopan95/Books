---
layout: page
title:  "Stochastic Processes"
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


## 1. Introduction & Renewal processes
### 1.1 Introduction
#### 1.1.1 Difference between deterministic and stochastic world
<p align="justify">
<table class="c">
  <tr>
    <th></th>
    <th>Deterministic world</th>
    <th>Stochastic world</th>
  </tr>
  <tr>
    <td>Single variable e.g. temperature T of a sick man</td>
    <td>$T=39^{o} \in \mathbb{R}$</td>
    <td>Random variable, E, Var, ...</td>
  </tr>
  <tr>
    <td>Variable changing over time<br>T in the first 3 days</td>
    <td>$R_{+}\rightarrow R$<br>$T(1)=39^{o}$<br>$T(2)=38^{o}$<br>$T(3)=37^{o}$</td>
    <td>Stochastic process</td>
  </tr>
</table><br>
</p>

#### 1.1.2 Difference between various fields of stochastics
<p align="justify">
$\bigstar$ Stochastics is composed of<br>
-- Probability theory<br>
-- Mathematical statistics<br>
-- Stochastic process.<br>
For example, we have a pond with M fishes. If we know a distribution of the N fishes, we can estimate E, v and others characteristics with Probability theory. Mathematical statistics solves problems which are upper side to the probability theory. Namely, one can ask how to estimate the amount of fishes N by providing some statistical experiments. For instance, we can catch M fishes and mark them, and then we put them back in the pond, so the moment we have M marked fishes and N-M unlabeled fishes. Next, we recatch n fishes, in which m fishes are labeled. Here is a question, what is the probability that the number of labled fished are m? The answer is (x denotes a number of labled fishes):
$$\mathbb{P}\left [x=m \right ] = \frac{C_{M}^{m}C_{N-M}^{n-m}}{C_{N}^{M}}$$

If we repeat such an experiment $m_{0}, m_{1},..., m_{q}$, then we take a log likelihood function.
$$\log \prod_{k=0}^{q} P[x = m_{k}] = \sum_{k=0}^{q} \log(P[ x=m_{k} ])$$

In this formula, we notice that M, n, m are known in each experiment while N is unknown. So we can maximize the log likelihood function to get N, which is called maximum likelihood estimator. Okay, this was a practical solution for a very simple task, and in this context, we can also introduce the notion of stochastic process because the m fishes in the pond, definitely depends on T events on time, and therefore, we can ask the same questions, but taking into account that the amount of fishes changes and this is exactly the objective of the theory of stochastic processes.
</p>

#### 1.1.3 Probability space
<P align="justify">
We have 3 fundamental notions $(\Omega, \mathbb{F}, \mathbb{P})$.
<table class="c">
  <tr>
    <th>General theory</th>
    <th>Bernouli {0, 1}</th>
    <th>[0, 1]</th>
  </tr>
  <tr>
    <td>$\Omega$: sample sapce. It is basically a space of any nature which includes all possible outcomes of an experiment</td>
    <td>$(a_{1}, a_{2}, ..., a_{n})$<br>$\left \| \Omega \right \| = 2^{n}$</td>
    <td>A set of possible outcomes are [0, 1]</td>
  </tr>
  <tr>
    <td>$\mathbb{F}$ is $\sigma$-algebra<br>It is a set of subsets $\Omega$ with the following properties<br>(1) $\Omega \in\mathbb{F}$<br>(2) $\mathbb{F}$ should be close on the complement $A\in F\Rightarrow \Omega \setminus A \in F$<br>(3) $\mathbb{F}$ should be close under countable units $A_{1}, ...A_{n},... \in F \Rightarrow \bigcup_{i}^{\infty }A_{I}\in F$</td>
    <td>$\mathbb{F}$ is a power set.<br>$\left \| \mathbb{F} \right \| = 2^{\left \| \Omega \right \|}=2^{2^{n}}$</td>
    <td>Borel $\sigma$-algebra<br>$[\alpha, \beta]$<br>$(\alpha, \beta]$<br>$[\alpha, \beta)$<br>$(\alpha, \beta)$<br>$(\beta)$</td>
  </tr>
  <tr>
    <td>$\mathbb{P}$ is called Probability measure.<br>(1) $\mathbb{P}[\Omega]=1$<br>(2) if $A_{1}, A_{2}, ..., A_{n}\in F$ and all elements are not intersectable with others, then $\mathbb{P}[\bigcup_{i}^{n}A_{i}]=\sum_{i}^{n}\mathbb{P}[A_{i}]$</td>
    <td>$\mathbb{P}[1]=p$ and $\mathbb{P}[0]=1-p$</td>
    <td>$\mathbb{P}[[\alpha, \beta]]=\beta-\alpha$</td>
  </tr>
</table><br>
</p>

#### 1.1.4 Definition of a stochastic function
<p align="justify">
Let us consider the following example. An agent flips a coin 2 times. In this model, Ω = {(h,h); (h,t); (t,h); (t,t)}, where t means tails and h means heads.σ-algebra F contains all possible combinations of those 4 elements in Ω (by the way, the number of elements in F is exactly $2^{4}$).
$$
\begin{aligned}
\text{Let } & \xi(h, h) = 1 \\
& \xi(h, t) = 2 \\
& \xi(t, h) = 3 \\
& \xi(t, t) = 4 \\
\text{Note } & P(\xi = k) = \frac{1}{4}, \quad k = \{1, 2, 3, 4\} \\
\text{So, } & \xi^{-1}(1) = (h, h) \\
& \xi^{-1}(2) = (h, t) \\
& \xi^{-1}(3) = (t, h) \\
& \xi^{-1}(4) = (t, t) \\
& \forall B \in \mathbb{B}: \xi^{-1}(B) \subset \mathbb{F}
\end{aligned}
$$

Let ($\Omega$, $\mathbb{F}$, $\mathbb{P}$), random variable is a function
$$\xi : \Omega \rightarrow \mathbb{R} \text{ such that } \forall B \in \mathbb{B}(\mathbb{R}) : \xi^{-1}(B) \in \mathbb{F}$$

We take time T into account. If $\forall t \in T, X_{t}=X(t)$ is a random variable on $(\Omega, \mathbb{F}, \mathbb{P})$, Random function is
$$X:T\times \Omega\rightarrow \mathbb{R}$$
$$
\text{Random function }
\begin{cases}
T = \mathbb{R}_{+}, & \text{random process} \quad
\begin{cases}
T = \mathbb{N}(\mathbb{Z}), & \text{discrete function} \\
T = \mathbb{R}_{+}(R), & \text{continuous function}
\end{cases} \\
T = \mathbb{R}_{+}^{n}, & \text{random field}
\end{cases}
$$
</p>

#### 1.1.5 Trajectories and finite-dimensional distributions
<p align="justify">
$\bigstar$ Trajectory (path) is a mapping from T to R when we fix an element from omage.<br>
$\bigstar$ Any stochastic process at any fixed time point is a random variable.<br>
$\bigstar$ There is a notion of finite dimensional distribution, this is a distribution of random vector
$$(X_{t1}, X_{t2}, ...,, X_{tn}), \quad t_{1}, t_{2}, \cdots, t_{n} \in R$$

For example, a stochastic process $X_{t}$ is defined as follows:
$$X_{t} = \xi_{1} \sin(t) + \xi_{2} \cos(t), \quad t \geq 0$$
$\xi_{1}$ and $\xi_{2}$ are two i.i.d. random variables with standard normal distribution. Assume {A, B} $\in \Omega$, i.e. A is a fixed value for $\xi_{1}$ and B is a fixed value for $\xi_{2}$, explicitly shows the trajectories of the process $X_{t}$
$$
\begin{aligned}
& \text{Substitute random variables with constant} \\
& X_{t} = \xi_{1} \sin(t) + \xi_{2} \cos(t) = A \sin(t) + B \cos(t) \\
& \text{If } A^{2} + B^{2} = 0, A = B = 0, \\
& \text{trajectory is a horizontal line } X_{t} = 0, \quad t \geq 0 \\
& \text{If } A^{2} + B^{2} \geq 0, \\
&
\begin{aligned}
X_{t} &= \sqrt{A^{2} + B^{2}} (\frac{A}{\sqrt{A^{2} + B^{2}}} \sin(t) + \frac{B}{\sqrt{A^{2} + B^{2}}} \cos(t)) \\
&= \sqrt{A^{2} + B^{2}} \sin(t + \theta), \quad \text{where }
\begin{cases}
\sin(\theta) = \frac{B}{\sqrt{A^{2} + B^{2}}}, \\
\cos(\theta) = \frac{A}{\sqrt{A^{2} + B^{2}}}
\end{cases}
\end{aligned}
\end{aligned}
$$
</p>

### 1.2 Renewal processes
#### 1.2.1 Counting process
<p align="justify">
Renewal process Sn is a discrete tie process that it is equal to 0 at 0.<br>
A renewal process at any time point n can be represented as a sum of i.i.d. non-negative random variables
$$
\begin{aligned}
& S_{0} = 0 \\
& S_{n} = S_{n-1} + \xi_{n} \\
& \xi_{1}, \xi_{2}, \cdots,\xi_{n} \text{ a sequence of i.i.d positive random variables} \\
& P(\xi_{1}>0) = 1 \Leftrightarrow F(0) = 0 \\
& N_{t} = \arg\max_{k} (S_{k} \leq t) \text{ counting process} \\
& \{S_{n} > t\} \Leftrightarrow \{N_{t} < n\} \\
& S_{n} = S_{n-1} + \xi_{n} = \sum_{i=1}^{n} \xi_{i} \\
& F^{n*} = F * \cdots * F \quad \text{where F is a distribution function}
\end{aligned}
$$

We have 2 properties
$$
\begin{aligned}
\mathbf{(1)} \quad & F^{n*}(x) \leq F^{n}(x), \quad \text{if } F(0) = 0 \\
& \because \xi_{1}, \xi_{2}, \cdots, \xi_{n} \text{ are i.i.d postive random variables} \sim F \\
& \therefore P(\xi_{i} \geq 0) = 1 \quad i = 1, 2, \cdots n \\
& \text{If } \xi_{1} + \xi_{2} + \cdots + \xi_{n} \leq x \\
& \text{Then, } \xi_{1} \leq x, \xi_{2} \leq x, \cdots, \xi_{n} \leq x \\
& \because \{\xi_{1} + \xi_{2} + \cdots + \xi_{n} \leq x \} \subset \{\xi_{1} \leq x, \xi_{2} \leq x, \cdots \xi_{n} \leq x \} \\
& \therefore P(\xi_{1} + \xi_{2} + \cdots + \xi_{n} \leq x) \leq P(\xi_{1} \leq x, \xi_{2} \leq x, \cdots \xi_{n} \leq x) = \prod_{i} P(\xi_{i} \leq x) \\
& \therefore F^{n*}(x) \leq F^{n}(x) \\
\\
\mathbf{(2)} \quad & F^{(n+1)*}(x) \leq F^{n*}(x) \\
& \because \{\xi_{1} + \xi_{2} + \cdots + \xi_{n+1} \leq x \} \subset \{\xi_{1} + \xi_{2} + \cdots + \xi_{n} \leq x \} \\
& \therefore P(\sum_{i}^{n} \xi_{i}) \geq P(\sum_{i=1}^{n+1} \xi_{i})
\end{aligned}
$$

Two important theories
$$
\begin{aligned}
\boldsymbol{(1)} \quad & U(t) = \sum_{n=1}^{\infty} F^{n*}(t) < \infty \text{ converge} \\
\\
\boldsymbol{(2)} \quad & E[N_{t} ] = U(t) \\
&
\begin{aligned}
E[N_{t} ] &= E[\text{No. of } \{n: S_{n} \leq t \}] \\
&= E[\sum_{n=1}^{\infty} \boldsymbol{1}_{S_{n} \leq t}] \\
&= \sum_{n=1}^{\infty} P(S_{n} \leq t) \\
&= \sum_{n=1}^{\infty} F^{n*}(t)
\end{aligned}
\end{aligned}
$$

We draw a trajectory of Nt
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROB/SP/1_2_1_1.png"/></center>
</p>

#### 1.2.2 Convolution
<p align="justify">
All of the statements are about independent variables
$$X \perp Y$$

Distribution function F
$$
\begin{aligned}
& X \sim F_{X}, Y \sim F_{Y} \\
&
\begin{aligned}
F_{X+Y}(x) &= \int_{-\infty}^{\infty} F_{X}(x-y)dF_{Y}(y) \\
&= F_{X} * F_{Y} \quad \text{convolution in terms of distribution function}
\end{aligned}
\end{aligned}
$$

X has a density $P_{X}$, Y has a density $P_{Y}$
$$
\begin{aligned}
& X \sim P_{X}, Y \sim P_{Y} \\
&
\begin{aligned}
P_{X+Y}(x) &= \int_{-\infty}^{\infty} P_{X}(x-y)P_{Y}(y)dy \\
&= P_{X} * P_{Y} \quad \text{convolution in terms of density}
\end{aligned}
\end{aligned}
$$

For example, $X\sim e^{\lambda},Y\sim e^{\mu} \quad \text{where } \lambda \neq \mu$, Find the probability distribution function of X+Y
$$
\begin{aligned}
& P_{X}(x) = \lambda e^{-\lambda x}, \quad x \geq 0 \\
& P_{Y}(y) = \mu e^{-\mu y}, \quad y \geq 0 \\
&
\begin{aligned}
P_{X+Y}(x) &= \int_{-\infty}^{\infty} P_{X}(x-y)P_{Y}(y)dy \\
&=
\begin{cases}
\int_{0}^{x} \lambda e^{-\lambda (x-y)} \mu e^{-\mu y} dy, & \quad x \geq y \geq 0 \\
0, & \quad \text{else}
\end{cases} \\
&= \lambda \mu e^{-\lambda x} \int_{0}^{x} e^{(\lambda - \mu) y} dy \\
&= \frac{\lambda \mu}{\lambda - \mu} (e^{-\mu x} - e^{-\lambda x})
\end{aligned}
\end{aligned}
$$
</p>

#### 1.2.3 Laplace transform
<p align="justify">
Laplace transform $\mathfrak{L}$
$$
\begin{aligned}
& \mathfrak{L}_{f}(s) = \int_{0}^{\infty} e^{-s x} f(x) dx \quad \text{where } f: R_{+} \rightarrow R \\
&
\begin{aligned}
\boldsymbol{(1)} \quad & f \text{ is density of } \xi, \quad \text{then } \mathfrak{L}_{f}(s) = E[e^{-s \xi} ] \\
\boldsymbol{(2)} \quad & f_{1}, f_{2}: \mathfrak{L}_{f_{1} * f_{2}} = \mathfrak{L}_{f_{1}}(\xi) \cdot \mathfrak{L}_{f_{2}}(\xi) \\
\boldsymbol{(3)} \quad & F \text{ is density function for postive random variable }, F(0) = 0, P = F' \\
& \mathfrak{L}_{F}(s) = \frac{\mathfrak{L}_{P}(s)}{s} \\
& \because
\begin{aligned}
\mathfrak{L}_{F}(s) &= \int_{0}^{\infty} e^{-sx} F(x) dx \\
&= -\int_{0}^{\infty} F(x) \frac{d(e^{-sx})}{s} \\
&= -F(x) \frac{e^{-sx}}{s} \mid_{0}^{\infty} + \int_{0}^{\infty} \frac{e^{-sx}}{s} d(F(x)) \\
&= \int_{0}^{\infty} P(x) \frac{e^{-sx}}{s} dx \\
&= \frac{\mathfrak{L}_{P}(s)}{s}
\end{aligned} \\
& \therefore \mathfrak{L}_{F}(s) = \frac{\mathfrak{L}_{P}(s)}{s}
\end{aligned} 
\end{aligned}
$$

For example
$$
\begin{aligned}
\boldsymbol{(1)} \quad & f(x) = x^{n}, \quad \text{where } n = 1, 2, \cdots \\
&
\begin{aligned}
\mathfrak{L}_{x^{n}}(s) &= \int_{R_{+}} x^{n} e^{-sx} dx \\
&= \int_{R_{+}} x^{n} \frac{d(e^{-sx})}{-s} \\
&= \frac{n}{s} \int_{R_{+}} x^{n-1} e^{-sx} dx \\
&= \frac{n}{s} \cdot \frac{n-1}{s} \cdots \frac{1}{s} \int_{R_{+}} e^{-sx} dx \\
&= \frac{n!}{s^{n+1}}
\end{aligned} \\
\boldsymbol{(2)} \quad & f(x) = e^{ax}, \quad \text{where } a < s \\
&
\begin{aligned}
\mathfrak{L}_{x^{n}}(s) &= \int_{R_{+}} e^{ax} e^{-sx} dx \\
&= \frac{e^{(a-s)x}}{a-s} \mid_{0}^{\infty} \\
&= \frac{1}{s-a}
\end{aligned}
\end{aligned}
$$

To calculate $E[ N_{t} ]$
$$
\begin{aligned}
& F \rightarrow E[N_{t}] \\
& E[N_{t}] = U(t) = \sum_{n=1}^{\infty} F^{n*}(t) = F(t) + (\sum_{n=1}^{\infty} F^{n*}) *F(t) \\
& U(t) = F(t) + U(t) * F(t) = F + U*P, \quad P = F' \\
& \int_{R} U(x - y) d(F(y)) = \int_{R} U(x - y) P(y) dy \\
& \mathfrak{L}_{U} (s) = \mathfrak{L}_{F} (s) + \mathfrak{L}_{U} (s) \cdot \mathfrak{L}_{P}(s) = \frac{\mathfrak{L}_{P} (s)}{s} + \mathfrak{L}_{U} (s) \cdot \mathfrak{L}_{P}(s) \\
& \mathfrak{L}_{U} (s) = \frac{\mathfrak{L}_{P} (s)}{s(1 - \mathfrak{L}_{P} (s))} \\
& F \rightarrow \mathfrak{L}_{P} \rightarrow \mathfrak{L}_{U} \rightarrow U
\end{aligned}
$$

For example
$$
\begin{aligned}
& S_{n} = S_{n-1} + \xi_{n} \\
& \xi_{1}, \xi_{2}, \cdots, \xi_{n} \sim P(x) = \frac{e^{-x}}{2} + e^{-2x}, \quad x > 0 \\
& E[N_{t}] = ? \\
& \mathbf{(1)} \quad P \rightarrow \mathfrak{L}_{P} \\
& \quad \quad
\begin{aligned}
\mathfrak{L}_{P} &= \frac{1}{2} \mathfrak{L}_{e^{-x}} (s) + \mathfrak{L}_{e^{-2x}}(s) \\
&= \frac{1}{2(s+1)} + \frac{1}{s+2} \\
&= \frac{3s+4}{2(s+1)(s+2)}
\end{aligned} \\
& \mathbf{(2)} \quad \mathfrak{L}_{P} \rightarrow \mathfrak{L}_{U} \\
& \quad \quad \mathfrak{L}_{U}(s) = \frac{\mathfrak{L}_{P}(s)}{s(1-\mathfrak{L}_{P}(s))} = \frac{3s+4}{s^{2}(2s+3)} \\
& \mathbf{(3)} \quad \mathfrak{L}_{U} \rightarrow U \\
& \quad \quad
\begin{aligned}
& \mathfrak{L}_{U}(s) = \frac{3s+4}{s^{2}(2s+3)} = \frac{A}{s^{2}} + \frac{B}{s} + \frac{C}{2s+3} \\
&
\begin{cases}
A = \frac{4}{3} \\
B = \frac{1}{9} \\
C = -\frac{2}{9}
\end{cases} \\
& U_{t} = \frac{4}{3} t + \frac{1}{9} - \frac{2}{9} e^{-\frac{3}{2}t}
\end{aligned}
\end{aligned}
$$
</p>

#### 1.2.4 Limit theorems for renewal processes
<p align="justify">
$$S_{n} = S_{n-1} + \xi_{n}, \quad \xi_{1}, \xi_{2}, \cdots, \xi_{n} \text{ -i.i.d} > 0$$

$\bigstar$ Thoery 1
$$
\begin{aligned}
& \text{If } \mu = E[ \xi_{1} ] < \infty \\
& \text{Then } \lim_{x \to \infty} \frac{N_{t}}{t} = \frac{1}{\mu} \\
& \text{which is analogue to law of large number (LLN) } \\
& \lim_{n \to \infty} \frac{\xi_{1} + \cdots + \xi_{n}}{n} = \mu
\end{aligned}
$$

-- prove
$$
\begin{aligned}
& \because S_{N_{t}} \leq t \leq S_{N_{t} + 1} \\
& \therefore \frac{N_{t}}{S_{N_{t} + 1}} \leq \frac{N_{t}}{t} \leq \frac{N_{t}}{S_{N_{t}}} \\
& \because \lim_{t \to \infty} \frac{N_{t}}{S_{N_{t}+1}} = \lim_{n \to \infty} \frac{n}{S_{n}} = \lim_{n \to \infty} \frac{n}{\sum_{i=1}^{n} \xi_{i}} = \frac{1}{\mu} \\
& \quad \lim_{t \to \infty} \frac{N_{t}}{S_{N_{t} + 1}} = \lim_{t \to \infty} \frac{N_{t}}{N_{t}+1} \frac{N_{t}+1}{S_{N_{t}+1}} = \lim_{t \to \infty} \frac{N_{t}}{N_{t}+1} \cdot \lim_{t \to \infty} \frac{N_{t}+1}{S_{N_{t}+1}} = \frac{1}{\mu} \\
& \therefore \lim_{t \to \infty} \frac{N_{t}}{t} = \frac{1}{\mu}
\end{aligned}
$$

$\bigstar$ Thoery 2
$$
\begin{aligned}
& E[ \xi_{1} ] = \mu \\
& \text{If } \sigma^{2} = \text{Var}(\xi_{1}) < \infty \\
& \text{Then } \lim_{t \to \infty} \frac{N_{t} - \frac{t}{\mu}}{\sigma \sqrt{t} \mu^{-\frac{3}{2}}} \sim N(0, 1) \\
& Z_{t} = \frac{N_{t} - \frac{t}{\mu}}{\sigma \sqrt{t} \mu^{-\frac{3}{2}}} \\
& P(Z_{t} \leq x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-\frac{u^{2}}{2}} du \\
& \text{Which is analogue to Central Limit Theorem (CLT) } \\
& \lim_{n \to \infty} \frac{\xi_{1} + \cdots + \xi_{n} - n \mu}{\sigma \sqrt{n}} \sim N(0, 1)
\end{aligned}
$$

-- prove
$$
\begin{aligned}
& \because \lim_{n \to \infty} \frac{\xi_{1} + \cdots + \xi_{n} - n \mu}{\sigma \sqrt{n}} \sim N(0, 1) \quad (\text{CLT}) \\
& \therefore \lim_{n \to \infty} P(\frac{S_{n} - n \mu}{\sigma \sqrt{n}} \leq x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-\frac{u^{2}}{2}} du = \Phi(x) \\
& \therefore \lim_{n \to \infty} P(S_{n} \leq n\mu + \sigma \sqrt{n} x) = \Phi(x) \\
& \because \{S_{n} \leq t \} \Leftrightarrow \{N_{t} \geq n\} \\
& \therefore t = n\mu + \sigma \sqrt{n} x \\
& \therefore n = \frac{t}{\mu} - \frac{\sigma \sqrt{n} x}{t} \\
& \because n\mu \approx t \quad \text{LLN} \\
& \therefore n = \frac{t}{n} - \frac{\sigma \sqrt{t}}{u^{\frac{3}{2}}}x \\
& \therefore \lim_{t \to \infty} P(N_{t} \geq \frac{t}{n} - \frac{\sigma \sqrt{t}}{u^{\frac{3}{2}}}x) = \Phi(x) \\
& \therefore \lim_{t \to \infty} P(Z_{t} \geq -x) = \lim_{t \to \infty} P(Z_{t} \leq x) = \Phi(x)
\end{aligned}
$$
</p>

### 1.3 Exercises
#### 1.3.1 Quiz: Introduction & Renewal processes
<p align="justify">
<b>1.</b><br>
Let $\eta$ be a random variable with distribution function $F_{\eta}$. Define a stochastic process $X_{t} = \eta + t$. Compute the distribution function of a finite-dimensional distribution ($X_{t_{1}}$, ..., $X_{t_{n}}$), where $t_{1}, \cdots, t_{n} \in R_{+}$<br>
A. $F_{\eta} \{\min(x_{1}-t_{1}, \cdots, x_{n}-t_{n})\}$<br>
B. $F_{\eta}\{\min(x_{1}, \cdots, x_{n})\}$<br>
C. $F_{\eta}\{\min(t_{1}, \cdots, t_{n})\}$<br>
D. None<br>
<b>Answer</b>: A.
$$
\begin{aligned}
F_{X}(\vec{x}) &= P(X_{t_{1}} \leq x_{1}, \cdots, X_{t_{n}} \leq x_{n}) \\
&= P(\eta+t_{1} \leq x_{1}, \cdots, \eta+t_{n} \leq x_{n}) \\
&= P(\eta \leq x_{1} - t_{1}, \cdots, \eta \leq x_{n} - t_{n}) \\
&= F_{\eta} \{\min(x_{1}-t_{1}, \cdots, x_{n}-t_{n})\}
\end{aligned}
$$

<b>2.</b><br>
Let $S_{n}$ be a renewal process such that $\xi_{n} = S_{n} - S_{n-1}$ takes the values 1 or 2 with equal probabilities p=1/2. Find the mathematical expectation of the counting process $N_{t}$ at t=3<br>
A. 3<br>
B. 7/8<br>
C. 1/8<br>
D. 15/8<br>
E. None<br>
<b>Answer</b>: D.
$$
\begin{aligned}
E(N_{3}) &= \sum_{k=0}^{\infty} k \cdot P(N_{3} = k) \\
&= 0 + 1 \cdot P(N_{3} = 1) + 2 \cdot P(N_{3} = 2) + 3 \cdot P(N_{3} = 3) + 4 \cdot P(N_{4} = 4) \\
&= 1 \cdot P(\xi_{1} = 2, \xi_{2} = 2) + \\
& \quad \quad 2 \cdot (P(\xi_{1}=1, \xi_{2}=2) + P(\xi_{1}=1, \xi_{2}=2) + P(\xi_{1}=1, \xi_{2}=1, \xi_{3}=2)) + \\
& \quad \quad 3 \cdot P(\xi_{1} = 1, \xi_{2} = 1, \xi_{3} = 1) + 4 \cdot 0 \\
&= 1 \cdot \frac{1}{4} + 2 \cdot (\frac{1}{4} + \frac{1}{4} + \frac{1}{8}) + 3 \cdot \frac{1}{8} \\
&= \frac{15}{8}
\end{aligned}
$$

<b>3.</b><br>
Let $S_{n} = S_{n-1} + \xi_{n}$ be a renewal process and $P_{\xi}(x) = \lambda e^{-\lambda x}$ Find the mathematical expectation of the corresponding counting process $N_{t}$<br>
A. $\lambda$<br>
B. $\lambda^{2}$<br>
C. $\frac{1}{\lambda^{2}}$<br>
D. $\frac{1}{\lambda}$<br>
E. None<br>
<b>Answer</b>: E.
$$
\begin{aligned}
\mathbf{(1)} \quad & P \rightarrow \mathfrak{L}_{P}(s) \\
& \mathfrak{L}_{P}(s) = \int_{0}^{\infty} e^{-sx} \lambda e^{-\lambda x} dx = \frac{\lambda}{\lambda + s} \\
\mathbf{(2)} \quad & \mathfrak{L}_{P}(s) \rightarrow \mathfrak{L}_{U}(s) \\
& \mathfrak{L}_{U}(s) = \frac{\mathfrak{L}_{P}(s)}{s(1-\mathfrak{L}_{P}(s))} = \frac{\lambda}{s^{2}} \\
\mathbf{(2)} \quad & \mathfrak{L}_{U}(s) \rightarrow U \\
& U(t) = \lambda t
\end{aligned}
$$

<b>4.</b><br>
Let η be a random variable with distribution function $F_{\eta}$. Define a stochastic process $X_{t} = e^{\eta} t^{2}$. What is the distribution function of ($X_{t_{1}}$, ..., $X_{t_{n}}$)for positive $t_{1}$, ..., $t_{n}$?<br>
A. $F_{\eta} \{\min(\ln \frac{x_{1}}{t_{1}^{2}}, \cdots, \ln \frac{x_{n}}{t_{n}^{2}})\}$<br>
B. $F_{\eta} \{\min(\ln \frac{x_{1}}{t_{1}}, \cdots, \ln \frac{x_{n}}{t_{n}})\}$<br>
C. 0<br>
D. None<br>
<b>Answer</b>: A.
$$
\begin{aligned}
F_{X}(\vec{x}) &= P(X_{t_{1}} \leq x_{1}, \cdots, X_{t_{n}} \leq x_{n}) \\
&= P(e^{\eta}t_{1}^{2} \leq x_{1}, \cdots, e^{\eta}t_{n}^{2} \leq x_{n}) \\
&= P(\eta \leq \ln \frac{x_{1}}{t_{1}^{2}}, \cdots, \eta \leq \ln \frac{x_{n}}{t_{n}^{2}}) \\
&= F_{\eta} \{\min(\ln \frac{x_{1}}{t_{1}^{2}}, \cdots, \ln \frac{x_{n}}{t_{n}^{2}})\}
\end{aligned}
$$

<b>5.</b><br>
Let $N_{t}$ be a counting process of a renewal process $S_{n} = S_{n-1} + \xi_{n}$ such that the i.i.d. random variables $\xi_{1}$, $\xi_{2}$, ... have a probability density function
$$
P_{\xi}(x) =
\begin{cases}
\frac{1}{2} e^{-x} (x+1), \quad & x \geq 0 \\
0, \quad & x < 0
\end{cases}
$$
Find the mean of $N_{t}$<br>
A. $-\frac{1}{9} + \frac{4}{3} t + \frac{1}{9} e^{-\frac{3}{2}t}$<br>
B. $-\frac{1}{9} + \frac{2}{3} t + \frac{1}{9} e^{-\frac{3}{2}t}$<br>
C. $-\frac{1}{9} + \frac{2}{3} t + \frac{1}{9} e^{\frac{3}{2}t}$<br>
D. None<br>
<b>Answer</b>: B.
$$
\begin{aligned}
\mathbf{(1)} \quad & P \rightarrow \mathfrak{L}_{P} \\
& \mathfrak{L}_{P}(s) = \int_{0}^{\infty} \frac{1}{2} x e^{-x} e^{-sx} dx + \int_{0}^{\infty} \frac{1}{2} e^{-x} e^{-sx} dx = \frac{s+2}{2(s+1)^{2}} \\
\mathbf{(2)} \quad & \mathfrak{L}_{P} \rightarrow \mathfrak{L}_{U} \\
& \mathfrak{L}_{U}(s) = \frac{\mathfrak{L}_{P}}{s(1-\mathfrak{L}_{P})} = \frac{s+2}{s^{2}(2s+3)} = \frac{A}{s} + \frac{B}{s^{2}} + \frac{C}{2s+3} \\
&
\begin{cases}
A = -\frac{1}{9} \\
B = \frac{2}{3} \\
C = \frac{2}{9}
\end{cases} \\
& \mathfrak{L}_{U}(s) = -\frac{\frac{1}{9}}{s} + \frac{\frac{2}{3}}{s^{2}} + \frac{\frac{2}{9}}{2s+3} \\
\mathbf{(3)} \quad & \mathfrak{L}_{U} \rightarrow U \\
& U(t) = -\frac{2}{9} + \frac{2}{3} t + \frac{2}{9} e^{-\frac{3}{2}t}
\end{aligned}
$$

<b>6.</b><br>
Let ξ and η be 2 random variables. It is known that the distribution of η is symmetric, that is, $P(\eta > x) = P(\eta < -x)$ for any x > 0, and moreover $P(\eta = 0) = 0$. Find the probability of the event that the trajectories of stochastic process $X_{t} = \xi^{2} + t(\eta + t), t \geq 0$ increase<br>
A. 1/4<br>
B. 0<br>
C. 1<br>
D. 1/2<br>
E. None<br>
<b>Answer</b>: D.
$$
\begin{aligned}
P(\frac{d}{dt} X_{t} > 0, \forall t \geq 0) &= P(2t+\eta, \forall t \geq 0) \\
&= P(\eta \geq 0) \\
&= \frac{1}{2}
\end{aligned}
$$
</p>

## 2. Poisson Processes
### 2.1 Poisson processes
#### 2.1.1 Definition of a Poisson process as a special example of renewal process
<p align="justify">

</p>

### 2.2 Models, related to Poisson processes
<p align="justify">

</p>

## 9. Final Exam
### 9.1 Final exam
#### 9.1.1 Quiz: Final Exam
<p align="justify">

</p>
