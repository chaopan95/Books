---
layout: page
title:  "Probabilistic Graphical Models"
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
<a href="https://www.coursera.org/account/accomplishments/specialization/certificate/XGWQ9URSZW3G"> My certificate.</a><br>
</p>


## 1. Representation
### 1.1 Introduction and Overview
#### 1.1.1 Overview and Motivation
<p align="justify">
Model: a declarative representation of our understanding of the world.<br>
Probalistic: the modeling limitations of complicated systems are such that one might as well view the world as inherently stochastic.<br>
Graph: a perspective of computer science to allow us to represent systems that are very complicated that involved large numbers of variables.<br><br>

Graphical representation:<br>
(1) Intuitive and compact data structure<br>
(2) Efficient reasoning using general-purpose algorithms<br>
(3) Sparse parameterization: feasible elicitation, learning from data<br><br>
</p>

#### 1.1.2 Distributions
<p align="justify">
Consider 3 random variables and each of them has some possible values:<br>
Intelligence(I) -- $i^{0}$ (low), $i^{1}$ (high)<br>
Difficulty(D) -- $d^{0}$ (easy), $d^{1}$ (hard)<br>
Grade(G) -- $g^{1}$ (A), $g^{2}$ (B), $g^{3}$ (C)<br><br>

We have a joint distribution $P(I, D, G)$<br>
<table class="a">
  <tr><th>I</th><th>D</th><th>G</th><th>Prob</th></tr>
  <tr><td>$i^{0}$</td><td>$d^{0}$</td><td>$g^{1}$</td><td>0.126</td></tr>
  <tr><td>$i^{0}$</td><td>$d^{0}$</td><td>$g^{2}$</td><td>0.168</td></tr>
  <tr><td>$i^{0}$</td><td>$d^{0}$</td><td>$g^{3}$</td><td>0.126</td></tr>
  <tr><td>$i^{0}$</td><td>$d^{1}$</td><td>$g^{1}$</td><td>0.009</td></tr>
  <tr><td>$i^{0}$</td><td>$d^{1}$</td><td>$g^{2}$</td><td>0.045</td></tr>
  <tr><td>$i^{0}$</td><td>$d^{1}$</td><td>$g^{3}$</td><td>0.126</td></tr>
  <tr><td>$i^{1}$</td><td>$d^{0}$</td><td>$g^{1}$</td><td>0.252</td></tr>
  <tr><td>$i^{1}$</td><td>$d^{0}$</td><td>$g^{2}$</td><td>0.0224</td></tr>
  <tr><td>$i^{1}$</td><td>$d^{0}$</td><td>$g^{3}$</td><td>0.0056</td></tr>
  <tr><td>$i^{1}$</td><td>$d^{1}$</td><td>$g^{1}$</td><td>0.06</td></tr>
  <tr><td>$i^{1}$</td><td>$d^{1}$</td><td>$g^{2}$</td><td>0.036</td></tr>
  <tr><td>$i^{1}$</td><td>$d^{1}$</td><td>$g^{3}$</td><td>0.024</td></tr>
</table><br>
</p>
<p align="justify">
Conditional probability $P(I, D \mid g^{1})$<br>
<table class="a">
  <tr><th>I</th><th>D</th><th>Prob</th></tr>
  <tr><td>$i^{0}$</td><td>$d^{0}$</td><td>0.282</td></tr>
  <tr><td>$i^{0}$</td><td>$d^{1}$</td><td>0.02</td></tr>
  <tr><td>$i^{1}$</td><td>$d^{0}$</td><td>0.564</td></tr>
  <tr><td>$i^{1}$</td><td>$d^{1}$</td><td>0.134</td></tr>
</table><br>
</p>
<p align="justify">
Marginalization $\sum_{I} P(I, D) = P(D)$<br>
<table class="a">
  <tr><th>D</th><th>Prob</th></tr>
  <tr><td>$d^{0}$</td><td>0.846</td></tr>
  <tr><td>$d^{1}$</td><td>0.154</td></tr>
</table><br>
</p>

#### 1.1.3 Factors
<p align="justify">
A factor $\phi(X_{1}, \cdots, X_{k})$<br>
$\phi$: $Val(X_{1}, \cdots, X_{k}) \rightarrow R$<br>
Scope = $\{ X_{1}, \cdots, X_{k} \}$<br><br>

Scope of the factor product $\phi(A, B, C) \cdot \phi(C, D)$: {A, B, C, D}<br><br>
P(I, D, G) and P(I, D, $g^{1}$) are factors.<br><br>

Why factors?<br>
Fundamental building block for defining distributions in high-dimensional spaces.<br>
Set of basic operations for manipulating these probability distributions.<br><br>
</p>

#### 1.1.4 Quiz
<p align="justify">
<b>1. Factor product</b><br>
Let X,Y and Z be binary variables.<br>
If $\phi_1(X,Y)$ and $\phi_2(Y, Z)$ are the factors shown below, compute the selected entries (marked by a '?') in the factor $\psi(X, Y, Z) = \phi_1(X,Y) \cdot \phi_2(Y, Z)$, giving your answer according to the ordering of assignments to variables as shown below.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_1_4_1.png"/></center>
</p>
<p align="justify">
<b>Answer:</b> 0.16 0.45 0.6<br><br>

<b>2. Factor reduction</b><br>
Let X, Z be binary variables, and let Y be a variable that takes on values 1, 2, or 3.<br>
Now say we observe Y = 2. If $\phi(X,Y,Z)$ is the factor shown below, compute the missing entries of the reduced factor $\psi(X, Z)$ given that Y = 2, giving your answer according to the ordering of assignments to variables as shown below.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_1_4_2.png"/></center>
</p>
<p align="justify">
<b>Answer:</b> 40 27 54 3<br><br>

<b>3. Factor marginalization</b><br>
Let X, Z be binary variables, and let Y be a variable that takes on values 1, 2, or 3. If $\phi(X,Y,Z)$ is the factor shown below, compute the entries of the factor
$$\psi(Y, Z) = \sum_X \phi(X,Y,Z)$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_1_4_3.png"/></center>
</p>
<p align="justify">
<b>Answer:</b> 108 135 79 141<br><br>
</p>

### 1.2 Bayesian Network
#### 1.2.1 Semantics & Factorization
<p align="justify">
Consider we have 5 random variables<br>
G: grade<br>
D: course diffucilty<br>
I: student intelligence<br>
S: student SAT<br>
L:reference letter<br><br>

Here is a possible model
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_1_1.png"/></center>
</p>
<p align="justify">
Factorization of joint distribution P(D, I, G, S, L) is $P(D) P(I) P(G \mid I,D) P(S \mid I) P(L \mid G)$ with chain rule
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_1_2.png"/></center>
</p>
<p align="justify">
For example, we want to know a joint of probability for $d^{0}$, $i^{1}$, $g^{3}$, $s^{1}$, $l^{1}$
$$P(d^{0}, i^{1}, g^{3}, s^{1}, l^{1}) = 0.6 \cdot 0.3 \cdot 0.02 \cdot 0.01 \cdot 0.8$$

<b>Bayesian Network</b> is:<br>
$\bigstar$ A <b>dircted acyclic</b> graph (DAG) G whose nodes represent the random variables $X_{1}$, $\cdots$, $X_{n}$<br>
$\bigstar$ For each node $X_{i}$ has a CPD (conditional probability distribution) $P(X_{i} \mid Par_{G}(X_{i}))$ (no parent is possible)<br><br>

The BN represents a joint distribution via the chain rule for Bayesian netwroks
$$P(X_{1}, \cdots, X_{n}) = \prod_{i}P(X_{i} \mid Par_{G}(X_{i}))$$

BN is a legal distribution: $P \geq 0$ and $\sum P = 1$
$$\sum_{L}P(L \mid G) = 1$$
$$
\begin{aligned}
\sum_{D, I, G, S, L} P(D, I, G, S, L) & = \sum_{D, I, G, S, L} P(D)P(I)P(G \mid I,D)P(S \mid I)P(L \mid G) \\
& = \sum_{D, I, G, S} P(D)P(I)P(G \mid I,D)P(S \mid I)\sum_{L}P(L \mid G) \\
& = \sum_{D, I, G, S} P(D)P(I)P(G \mid I,D)P(S \mid I) \\
& = \sum_{D, I, G} P(D)P(I)P(G \mid I,D)\sum_{S}P(S \mid I) \\
& = \sum_{D, I, G} P(D)P(I)P(G \mid I,D) \\
& = \sum_{D, I} P(D)P(I)\sum_{G}P(G \mid I,D) = \sum_{D, I} P(D)P(I) \\
& = 1
\end{aligned}
$$
Let G be a graph over $X_{1}$, $\cdots$, $X_{n}$<br>
<b>P factorizes over G</b> if $P(X_{1}, \cdots, X_{n}) = \prod_{i}P(X_{i} \mid Par_{G}(X_{i}))$
</p>
{% highlight Python %}
D = {0: 0.6, 1: 0.4}
I = {0: 0.7, 1: 0.3}
G = {1: {(0, 0): 0.3, (0, 1): 0.05, (1, 0): 0.9, (1, 1): 0.5},
        2: {(0, 0): 0.4, (0, 1): 0.25, (1, 0): 0.08, (1, 1): 0.3},
        3: {(0, 0): 0.3, (0, 1): 0.7, (1, 0): 0.02, (1, 1): 0.2}}
S = {0: {0: 0.95, 1: 0.2},
        1: {0: 0.05, 1: 0.8}}
L = {0: {1: 0.1, 2: 0.4, 3: 0.99},
        1: {1: 0.9, 2: 0.6, 3: 0.01}}
P = 0
def Prob(d=None, i=None, g=None, s=None, l=None):
    if d is None:
        D_keys = D.keys()
    else:
        D_keys = [d]
    if i is None:
        I_keys = I.keys()
    else:
        I_keys = [i]
    if g is None:
        G_keys = G.keys()
    else:
        G_keys = [g]
    if s is None:
        S_keys = S.keys()
    else:
        S_keys = [s]
    if l is None:
        L_keys = L.keys()
    else:
        L_keys = [l]
    P = 0
    for d in D_keys:
        for i in I_keys:
            for g in G_keys:
                for s in S_keys:
                    for l in L_keys:
                        P += D[d]*I[i]*G[g][(i, d)]*S[s][i]*L[l][g]
    return P
Prob()
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 1.2.2 Reasoning Patterns
<p align="justify">
<b>Causal Reasoning</b><br>
The reasoning goes, in the causal direction from top to bottom.<br><br>

For example, in a natural condition, the probability that one student can obtain a reference letter without other factors
$$P(i^{1}) = 0.5$$
If we add a factor of intelligence, say low intelligence $i^{0}$, then
$$P(l^{1} \mid i^{0}) = \frac{P(l^{1}, i^{0})}{P(i^{0})} = \frac{0.27}{0.7} = 0.39$$

If we add another factor of difficulty, say easy $d^{0}$, then
$$P(l^{1} \mid i^{0}, d^{0}) = 0.51$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_2_1.png"/></center>
</p>
<p align="justify">
<b>Evidential Reasoning</b><br>
The reasoning goes, in the evidential direction from bottom to top.<br><br>

For example, we know $P(d^{1}) = 0.4$ for a hard course. If a student have a a grade C for this course. What is the probability that this course is hard?
$$P(d^{1} \mid g^{3}) = \frac{P(d^{1}, g^{3})}{P(g^{3})} = \frac{0.22}{0.35} = 0.63$$

We know $P(i^{1}) = 0.3$ for a high intelligence and what is the probability that a student with C has high intelligence
$$P(i^{1} \mid g^{3}) = 0.08$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_2_2.png"/></center>
</p>
<p align="justify">
<b>Intercausal Reasoning</b><br>
It's flow of information between two causes of a single effect.<br><br>

For example, a student with C has a high intelligence with a probability $P(i^{1} \mid g^{3}) = 0.08$. If we have another information this course is hard $d^{1}$, the probability that this student with C has high intelligence goes up
$$P(i^{1} \mid g^{3}, d^{1}) = 0.11$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_2_3.png"/></center>
</p>
<p align="justify">
If a student aces the SAT, say $s^{1}$ but has a C in one course, the probability that he has high intelligence goes up
$$P(i^{1} \mid g^{3}, s^{1}) = 0.58$$

The probability that this course is really hard goes up too
$$P(d^{1} \mid g^{3}, s^{1}) = 0.76$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_2_4.png"/></center>
<br>
</p>

#### 1.2.3 Flow of Probabilistic Influence
<p align="justify">
<table class="c">
  <tr><th>Structure</th><th>Instance</th><th>X can influence Y?</th></tr>
  <tr><td>$X \rightarrow Y$</td><td>$D \rightarrow G$</td><td>Yes</td></tr>
  <tr><td>$X \leftarrow Y$</td><td>$G \leftarrow I$</td><td>Yes</td></tr>
  <tr><td>$X \rightarrow W \rightarrow Y$</td><td>$D \rightarrow G \rightarrow L$</td><td>Yes</td></tr>
  <tr><td>$X \leftarrow W \leftarrow Y$</td><td>$L \leftarrow G \leftarrow I$</td><td>Yes</td></tr>
  <tr><td>$X \leftarrow W \rightarrow Y$</td><td>$G \leftarrow I \rightarrow S$</td><td>Yes</td></tr>
  <tr><td>$X \rightarrow W \leftarrow Y$</td><td>$D \rightarrow G \leftarrow I$</td><td>No</td></tr>
</table><br>
</p>
<p align="justify">
$X \rightarrow W \leftarrow Y$ is also called v-strcuture.<br>
<table class="c">
  <tr><th>Structure</th><th>Instance</th><th>X can influence Y given evidence Z?</th></tr>
  <tr><td>$X \rightarrow Y$</td><td>$D \rightarrow G$</td><td>Yes</td></tr>
  <tr><td>$X \leftarrow Y$</td><td>$G \leftarrow I$</td><td>Yes</td></tr>
  <tr><td>$X \rightarrow W \rightarrow Y$</td><td>$D \rightarrow G \rightarrow L$</td><td>W $\notin$ Z, Yes<br> W $\in$ Z, No</td></tr>
  <tr><td>$X \leftarrow W \leftarrow Y$</td><td>$L \leftarrow G \leftarrow I$</td><td>W $\notin$ Z, Yes<br> W $\in$ Z, No</td></tr>
  <tr><td>$X \leftarrow W \rightarrow Y$</td><td>$G \leftarrow I \rightarrow S$</td><td>W $\notin$ Z, Yes<br> W $\in$ Z, No</td></tr>
  <tr><td>$X \rightarrow W \leftarrow Y$</td><td>$D \rightarrow G \leftarrow I$</td><td>W and all W's descendants $\notin$ Z, No<br> W $\in$ Z, Yes</td></tr>
</table><br>
</p>
<p align="justify">
For example, $S \leftarrow I \rightarrow G \leftarrow D$ allows influence to flow between S and D when:<br>
$\bigstar$ I is not observed and G or one of G's descendants is observed<br><br>

<b>Active Trails</b><br>
A trail $X_{1}$ -- ... -- $X_{k}$ is active if it has no v-structures $X_{i-1} \rightarrow X_{i} \leftarrow X_{i+1}$<br><br>

A trail $X_{1}$ -- ... -- $X_{k}$ is active given Z if:<br>
$\bigstar$ for any v-structure $X_{i-1} \rightarrow X_{i} \leftarrow X_{i+1}$, we have that $X_{i}$ or one of its descendants $\in$ Z<br>
$\bigstar$ no other $X_{i}$ is in Z<br><br>

For example, we have a Bayesian Network like this
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_3_1.png"/></center>
</p>
<p align="justify">
Given evidence G:<br>
$I \rightarrow S \rightarrow J \rightarrow H$ is active because G is not in the flow<br>
$C \rightarrow D \rightarrow G \leftarrow I \rightarrow S$ is active, because observing G active v-structrue<br>
$C \rightarrow D \rightarrow G \leftarrow I \rightarrow S \rightarrow J \leftarrow L$ is not active because of v-structure  $S \rightarrow J \leftarrow L$<br>
$I \rightarrow G \rightarrow L \rightarrow J \rightarrow H$ is not active because observing G blocks a flow I to J.<br><br>
</p>

#### 1.2.4 Independencies
<p align="justify">
<b>Independence</b>:<br>
For events $\alpha$, $\beta$, $P \models \alpha \perp \beta$, if:<br>
$\bigstar$ $P(\alpha, \beta) = P(\alpha)P(\beta)$<br>
$\bigstar$ $P(\alpha \mid \beta) = P(\alpha)$<br>
$\bigstar$ $P(\beta \mid \alpha) = P(\beta)$<br><br>

For random variables X, Y, $P \models X \perp Y$, if:<br>
$\bigstar$ $P(X, Y) = P(X)P(Y)$<br>
$\bigstar$ $P(X \mid Y) = P(X)$<br>
$\bigstar$ $P(Y \mid X) = P(Y)$<br><br>

<b>Conditional Independence</b>:<br>
For (sets of) random variables X, Y, Z, $P \models (X \perp Y \mid Z)$ if:<br>
$\bigstar$ $P(X, Y \mid Z) = P(X \mid Z)P(Y \mid Z)$<br>
$\bigstar$ $P(X \mid Y, Z) = P(X \mid Z)$<br>
$\bigstar$ $P(Y \mid X, Z) = P(Y \mid Z)$<br>
$\bigstar$ $P(X, Y, Z) \propto \phi_{1}(X, Z) \cdot \phi_{2}(Y, Z)$<br><br>

For example, Grade, Intelligence, SAT bayesian network
$$G \leftarrow I \rightarrow S$$

G and S are not independent but G and S are conditionally independent given condition I.<br><br>

<b>Conditioning can lose independence</b>:<br>
For example, Difficulty, Grade, Intelligence network
$$D \rightarrow G \leftarrow I$$

D and I are independent but D and I are not independent given condition G<br><br>
</p>

#### 1.2.5 D-separation
<p align="justify">
<b>Definition:</b> X and Y are d-separated in G given Z if there is no active trail in G between X and Y given Z.<br>
Notation: $d-sep_{G}(X, Y \mid Z)$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_3_1.png"/></center>
</p>
<p align="justify">
For example, $d-sep_{G}(D, J \mid L, I)$ is a correct d-separation, while $d-sep_{G}(D, J \mid L, H, I)$, $d-sep_{G}(D, I \mid L)$ and $d-sep_{G}(D, J \mid L)$ are not correct.<br><br>
</p>

#### 1.2.6 I-map
<p align="justify">
d-separation in G $\rightarrow$ P satisfies corresponding independence statement
$$I(G) = \{(X \perp Y \mid Z): d-sep_{G}(X, Y \mid Z)\}$$

$\bigstar$ <b>Definition:</b> If P satisfies I(G), G is an I-map (independency map) of P<br><br>

For example, we have 2 joint distributions $P_{1}$ and $P_{2}$ and 2 graphs $G_{1}$ and $G_{2}$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_6_1.png"/></center>
</p>
<p align="justify">
For $G_{1}$, $I(G_{1}) = {D \perp I}$; while for $G_{2}$, $I(G_{2}) = \varnothing$ because D and I are not independent.<br><br>

$P_{1} \models D \perp I$, so $G_{1}$ is an I-map of $P_{1}$. But $P_{2}$ doesn't satisfy $D \perp I$, so $G_{1}$ is not an I-map of $P_{2}$.<br><br>

What about $G_{2}$? Well, $G_{2}$ has no independent assumptions, and so both $P_{1}$ and $P_{2}$ satisfy the empty set of independence assumptions. As it happens $P_{1}$ also satisfies additional independence assumptions but the definitions of I-map didn't require that that G, that a graph exactly represent the independents in in a distribution, only that any independence that's implied by the graph holds for the distribution. So, the $G_{2}$ is actually an I-map for both $P_{1}$ and $P_{2}$.<br><br>
</p>

#### 1.2.7 Independence & Factorization
<p align="justify">
$P(X, Y) = P(X)P(Y)$ -- X, Y independent<br>
$P(X, Y, Z) \propto \phi_{1}(X, Z) \cdot \phi_{2}(Y, Z)$ -- $X \perp Y \mid Z$<br>
Factorization of a distribution P implies independences that hold in P<br>
If P factorizes ove G, can we read these independences from the structure of G?<br><br>

<b>Factorization $\rightarrow$ Independence: BNs</b>:<br>
<b>Theorem</b>: If P factorizes over G and $d-sep_{G}(X, Y \mid Z)$, then P satisfies $X \perp Y \mid Z$<br><br>

For example, without any condition, $D \perp S$ because there is no active trail between D and S
$$
\begin{aligned}
P(D, S) & = \sum_{G, I, L} P(D)P(I)P(G \mid D, I)P(S \mid I)P(L \mid G) \\
& = \sum_{I}P(D)P(I)P(S \mid I) \sum_{G}(P(G \mid D,I) \sum_{L}P(L \mid G)) \\
& = P(D) \sum_{I}P(I)P(S \mid I) = P(D)P(S)
\end{aligned}
$$

Any node is d-separated from its non-descendants given its parents.<br>
If P factorizes over G, then in P, any variable is independent of its non-descendants given its parents.<br><br>

$\bigstar$ <b>Theorem</b>: If P factorizes over G, then G is an I-map for P<br><br>

For example, we applies chain rule for probability to get a joint distribution P(D, I, G, S, L). Here, we have no bayesian network.
$$P(D, I, G, S, L) = P(D) P(I \mid D) P(G \mid D, I) P(S \mid D, I, G) P(L \mid D, I, G, S)$$

Because $D \perp I$, $P(I \mid D) = P(I)$. Because I is S's parent and D and G are non-descendants of S, so we can eliminate D, G from $P(S \mid D, I, G)$.
$$P(D, I, G, S, L) = P(D) P(I) P(G \mid D, I) P(S \mid I) P(L \mid G)$$

This is exactly a form of joint distribution based on bayesian network.<br><br>

Two equivalent views of graph structures:<br>
$\bigstar$ Factorization: G allows P to be represented<br>
$\bigstar$ I-map: Independencies encoded by G hold in P<br><br>

If P factorizes over a graph G, we can read from the graph independencies that must hold in P (an independency map)<br><br>
</p>

#### 1.2.8 Naive Bayes
<p align="justify">
<b>Naive Bayes model</b><br>
We have a class variable ans some feature variables. Features are observed and class is not observed. Our goal is to infer the class.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_8_1.png"/></center>
</p>
<p align="justify">
Naive Bayes model make independence assumption: Given the class variable, each observed variable is independent of the other observed variables.<br>
If we observe the class variable, influence cannot flow between any of the other variables, because the only path between two other variables is through the class variable. If we don't observe the class variable, however, all the variables are dependent.<br><br>

($X_{i} \perp X_{j} \mid C$) for all $X_{i}, X_{j}$ means $X_{i}$ and $X_{j}$ are conditionally independent given class C.
$$P(C, X_{1}, \cdots, X_{n}) = P(C) \prod_{i=1}^{n} P(X_{i} \mid C)$$

<b>Naive Bayes classifier</b><br>
We have more than 1 class
$$\frac{P(C=c^{1} \mid x_{1}, \cdots, x_{n})}{P(C=c^{2} \mid x_{1}, \cdots, x_{n})} = \frac{P(C=c^{1})}{P(C=c^{2})} \prod_{i=1}^{n} \frac{P(x_{i} \mid C=c^{1})}{P(x_{i} \mid C=c^{2})}$$

Bernoulli Naive Bayes for text
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_8_2.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ Simple approach for classification: computationally efficient; easy to construct<br>
$\bigstar$ Surprisingly effective in domains with many weakly relevant features<br>
$\bigstar$ Strong independence assumption reduce performance when many features are strongly correlated.<br><br>
</p>

#### 1.2.9 Quiz
<p align="justify">
<b>1.</b><br>
In general, if we have an edge $X \rightarrow Y$ in some Bayesian network, is there any set of other variables that we can condition on to make X and Y independent ?<br>
<b>Answer</b>: No, because the direct edge from X to Y means that in general, influence can flow from X to Y regardless of whether other variables are observed.<br><br>

<b>2.</b><br>
Say we observe Intelligence. <b>Are Grade and SAT conditionally independent?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_9_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: Yes.<br>
If we don't observe Intelligence, then Grade and SAT are dependent, because observing Grade gives us some information about Intelligence and therefore about SAT, and vice versa. However, if we have already observed Intelligence, then observing Grade can't affect SAT and vice versa, so they are conditionally independent.<br><br>

<b>3. Independencies in a graph</b><br>
<b>3.1</b><br>
A multinomial distribution over m possibilities $x_{1},\cdots,x_{m}$ has m parameters, but m−1 independent parameters, because we have the constraint that all parameters must sum to 1, so that if you specify m−1 of the parameters, the final one is fixed. In a CPD P(X∣Y), if X has m values and Y has k values, then we have k distinct multinomial distributions, one for each value of Y, and we have m−1 independent parameters in each of them, for a total of k(m−1). More generally, in a CPD $P(X \mid Y_{1}, \cdots, Y_{r})$, if each $Y_{i}$ has $k_{i}$ values, we have a total of $\prod_{i=1}^{r}k_{i}(m-1)$ independent parameters. <b>How many independent parameters are required to uniquely define the CPD of E (the conditional probability distribution associated with the variable E) in the same graphical model as above, if A, B, and D are binary, and C and E have three values each?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_9_2.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: $2 \cdot 3 \cdot (3-1) = 12$<br><br>

<b>3.2</b><br>
<b>Which pairs of variables are independent in the graphical model below, given that none of them have been observed?</b><br>
A. (A, B)<br>
B. (A, E)<br>
C. (D, E)<br>
D. (A, C)<br>
E. (A, D)<br>
F. None, there is no pairs of independent variables<br>
<b>Answer</b>: A, E.<br>
(D, E) are not independent because of $X \leftarrow W \rightarrow Y$.<br><br>

<b>3.3</b><br>
Now assume that the value of E is known. (E is observed. A, B, C, and D are not observed.) <b>Which pairs of variables (not including E) are independent in the same graphical model, given E?</b><br>
A. None, given E, there are no pairs of variables that are independent<br>
B. (A, B)<br>
C. (A, C)<br>
D. (A, D)<br>
E. (B. D)<br>
F. (D, C)<br>
G. (B, C)<br>
<b>Answer</b>: A. <br>
Observing E activates v-structure $A \rightarrow C \leftarrow B$.<br><br>

<b>4. Conditional independence</b><br>
We have shown that $P(S, G \mid i^{0}) = P(S \mid i^{0})P(G \mid i^{0})$. In this case, to check if S and G are conditionally independent given I, <b>do we still need to check if $P(S, G \mid i^{1}) = P(S \mid i^{1})P(G \mid i^{1})$</b><br>
<b>Answer:</b> Yes, we need to check if independence holds for all values of I, because the statement is that S and G are conditionally independent given I, not that they are conditionally independent given $i^{0}$.<br><br>

<b>5. I-maps</b><br>
<b>5.1</b><br>
I-maps can also be defined directly on graphs as follows. Let I(G) be the set of independences encoded by a graph G. Then $G_{1}$ is an I-map for $G_{2}$ if $I(G_1) \subseteq I(G_2)$. <b>Which of the following statements about I-maps are true? You may select 1 or more options</b><br>
A. A graph K is an I-map for a graph G if and only if all of the independences encoded by K are also encoded by G<br>
B. A graph K is an I-map for a graph G if and only if K encodes all of the independences that G has and more<br>
C. An I-map is a function f that maps a graph G to itself, i.e., f(G) = G<br>
D. The graph K that is the same as the graph G, except thta all of the edges are oriented in the opposite direction as the corresponding edges in G, is always an I-maps for G regardless of the structure G.<br>
<b>Answer:</b> A.<br>
K is an I-map for G if K does not make independence assumptions that are not true in G. An easy way to remember this is that the complete graph, which has no independencies, is an I-map of all distributions.<br><br>

<b>5.2</b><br>
Suppose $(A \perp B) \in \mathcal{I}(P)$, and G is an I-map of P, where G is a Bayesian network and P is a probability distribution. <b>Is it necessarily true that $(A \perp B) \in \mathcal{I}(G)$?</b><br>
<b>Answer:</b> No.<br>
Since G is an I-map of P, all independencies in G are also in P. However, this doesn't mean that all independencies in P are also in G. An easy way to remember this is that the complete graph, which has no independencies, is an I-map of all distributions.<br><br>

<b>6. Naive Bayes</b><br>
Consider the following Naive Bayes model for flu diagnosis:
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_2_9_3.png"/></center>
</p>
<p align="justify">
Assume a population size of 10,000. <b>Which of the following statements are true in this model? You may select 1 or more options.</b><br>
A. Say we observe that 1000 people have the flu, out of which 500 people have a headache (and possibly other syn=mptoms) and 500 have a fever (and possibly other symptoms). We would expect that approximately 250 people with the flu also have both a headache and fever.<br>
B. Say we observe that 1000 people have the flu, out of which 500 people have a headache (and possibly other syn=mptoms) and 500 have a fever (and possibly other symptoms). Without more information, we cannot estimate how many people with the flu also have both a headache and fever.<br>
C. Say we observe that 1000 people have a headache (and possibly other syn=mptoms), out of which 500 people have the flu (and possibly other syn=mptoms) and 500 have a fever (and possibly other symptoms). Without more information, we cannot estimate how many people with a headache also have both the flu and fever.<br>
D. Say we observe 500 people have a headache (and possibly other syn=mptoms) and 500 people have a fever (and possibly other syn=mptoms). Without more information, we cannot estimate how many people have both a headache and fever<br>
<b>Answer:</b> A, C, D.<br>
For A and B, headache and fever are conditionally independent given flu.
$$
\begin{aligned}
P(headache = 1, fever = 1 \mid flu = 1) & = P(headache = 1 \mid flu = 1) P(fever = 1 \mid flu = 1) \\
& = 0.5 \cdot 0.5 \\
& = 0.25
\end{aligned}
$$

So, We would expect that approximately 250 people with the flu also have both a headache and fever.<br><br>

For C, even after observing the Headache variable, there is still an active trail from Flu to Fever. Thus, the probability of someone with a headache also having a flu is dependent on the probability of his having a fever as well. For example, if someone has a flu, he could be more likely to have a fever, irrespective of whether he has a headache or not. We cannot estimate
$$P(flu = 1, fever = 1 \mid headache = 1) = P(flu = 1 \mid headache = 1) P(fever = 1 \mid headache = 1)$$

For D, without observing flu variable, headache and fever are not independent. We cannot infer
$$P(headache = 1 \mid fever = 1) = P(headache = 1) P(fever = 1)$$<br>
</p>

### 1.3 Template Models for Bayesian Networks
#### 1.3.1 Overview of Template Models
<p align="justify">
Template variables $X(U_{1}, \cdots, U_{k})$ is instantiated (duplicated) multiple times<br>
$\bigstar$ Location, Sonar<br>
$\bigstar$ Genotype (person), Phenotype (person)<br>
$\bigstar$ Lable (pixel)<br>
$\bigstar$ Difficulty (course), Intelligence (student), Grade (course, student)<br><br>

Template models<br>
$\bigstar$ Languages that specify how variables inherit dependency model from template<br>
$\bigstar$ Dynamic Bayesian network<br>
$\bigstar$ Object-relational models: Directed (plate model) and undirected<br><br>

Advantages of using template models:<br>
$\bigstar$ Template models can often capture events that occur in a time series.<br>
$\bigstar$ Template models can capture parameter sharing within a model.<br>
$\bigstar$ CPDs in template models can often be copied many times.<br>
Template models are a convenient way of representing Bayesian networks that have a high amount of parameter sharing and structure. At the end of the day, however, they are merely compact representations of a fully unrolled Bayesian network, and thus have no additional representative powers.<br><br>
</p>

#### 1.3.2 Temporal Model: DBNs
<p align="justify">
<b>Distributions over Trajectoires</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_2_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ Pick time granularity $\Delta$<br>
$\bigstar$ $X^{(t)}$ - variable X at time $t\Delta$<br>
$\bigstar$ $X^{(t:t')} = \{ X^{(t)}, \cdots, X^{(t')} \}$ ($t \leq t'$)<br><br>

Our goal is to represent $P(X^{(t:t')})$ for any t, t'.<br><br>

<b>Markov Assumption</b><br>
The probability of the set of variables expanding the time from zero all the way to capital T (Time flow forward) by chain rule
$$P(X^{(0:T)}) = P(X^{(0)}) \prod_{t=0}^{T-1} P(X^{(t+1)} \mid X^{(0:t)})$$

Markov assumption is effectively a type of conditional independence assumption
$$(X^{(t+1)} \perp X^{(0:t-1)} \mid X^{(t)})$$

Apply this assumption
$$P(X^{(0:T)}) = P(X^{(0)}) \prod_{t=0}^{T-1} P(X^{(t+1)} \mid X^{(t)})$$

Is this true? For example, X represent the location of a robot.
$$L^{t+1} \perp L^{t-1} \mid L^{t}$$

In most cases, this assumption is proably not true. The reason is that it completely ignores the velocity, which direction are you moving and how fast? So, this is a classical example of where the Markov assumption for this particular model is probably too strong of an assumption. So, what do we do to fix it?<br><br>

The one way to fix it is to enrich the stay description. So, estimate the Markov assumption a better approximation. Just like any independent assumption, the Markov assumption is always going to be an approximation but the question is how good of that approximation and if we add, for example, $V^{t}$, which is the velocity at times t, maybe the exploration of time t, maybe the robot's intent, where its goal is. I mean, all sorts of additional stuff into the state. Then, at that point, the Markov assumption becomes much more warranted. And so that's one way of making the Markhov assumption true. An alternative strategy is to move away from the Markov assumption by adding dependencies that go further back in time, back in time. And that's called a <b>semi-Markov model</b>.<br><br>

<b>Time Invariance</b><br>
In order to simplify the model, we've reduced the model to encoding a probability of $X^{t+1}$, given $X^{t}$, but it's still an unbounded number of conditional probabilities. Now, at least each of them is compact, but there's still a probabilistic model for every P. And this is where we're going to end up with a with a template based model. We're going to stipulate that there is a template probability model $P(X' \mid X)$, X' denotes the next time point and X denotes the current tiem point. For all t
$$P(X^{(t+1)} \mid X^{(t)}) = P(X' \mid X)$$

The model is replicated for every single time point. That is, when you're moving from time zero to time one, you use this model. When you're moving time one to time two, you also use this model. And and that assumption, for obvious reasons, is called time invariance. Because it assumes that the dynamics of the system, not the actual position of the robot, but rather the dynamics that move it <b>from state to state</b>, or the dynamics of the system, don't depend on the current time point t. And once again, it's an assumption that's warranted in certain cases and not in others.<br><br>

For example, let's imagine that this represents now the traffic on some road. Well, does the dynamics of that traffic depend on the current time point of the system? On most roads, the answer is probably yes. It might depend on the time of day, on the day of the week, on whether there is a big football match, on all sorts of things that might affect the dynamics of traffic. The point being that just like in the previous example, we can correct inaccuracies in our assumption by enriching the model. So, once again, we can enrich the model by including these variables in it. And once we have that, the model becomes a much better reflection of reality.<br><br>

<b>Template Transition Model</b><br>
let's assume that our stayed description is composed of a set of random variables. And we have little baby traffic system where we have the weather at the current time point, the location of a vehicle, the velocity of the vehicle. We also have a sensor, who's observation we get at each of those time points. And the sensor may or may not be failing at the current time point. And what we've done here is we've encoded the the probabilistic model of this next state. So, W', V', L', F' and O', given the previous states, W, V, L, and F.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_2_2.png"/></center>
</p>
<p align="justify">
$$P(W', V', L', F', O' \mid W, V, L, F) = P(W' \mid W) P(V' \mid W, V) P(L' \mid V, L) P(F' \mid F, W) P(O' \mid L', F')$$

Why is O not here on the right-hand side? Because it doesn't affect any of the next state variables. So, it would be kind of hanging down here if we included it. But that doesn't affect anything, we don't choose to to represent it. So, this model represents a conditional distribution.<br><br>

We have dependencies both within and across time. For example, $W \rightarrow W'$ and $L' \rightarrow O'$. This is logic because observation is relatively instantaneous compared to our time granularity and it's a better reflection for which variable actually influences the observation.<br><br>

Inter-time slice edge are edges that go from a variable at one time point to the value of that variable at the next time point, these are often called persistence edges because they indicate the the tendency of a variable to persist in state from one time point to another.<br><br>

We have CPDs for the variables on the right-hand side, the prime variables. But there's no CPDs for the variables that are unprimed, the variables on the left. And this is because the model doesn't actually try and represent the distribution, O over W, V, L, and F. It doesn't try and do that. It tries to represent the probability of the next time slice, given the previous one. So, as we can see, this graphical model only has CPD's for a subset of the variables in it. The ones that represent the next time point. So, that represents the transition dynamics.<br><br>

<b>Initial State Distribution</b><br>
Time slice 0
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_2_3.png"/></center>
</p>
<p align="justify">
$$P(W^{(0)}, V^{(0)}, L^{(0)}, F^{(0)}, O^{(0)}) = P(W^{(0)}) P(V^{(0)} \mid L^{(0)}) P(L^{(0)}) P(F^{(0)}) P(O^{(0)} \mid F^{(0)}, L^{(0)})$$

<b>Ground Bayesian Network</b><br>
With those two pieces, we can now represent probability distributions over arbitrarily long trajectories, by taking for time slice zero and copying the times zero Bayesian network, which represent the probability distribution over the time zero variables. And now, we have a bunch of copies that represent the probability distribution at time one, given time zero. And here, we have another copy of exactly the same set of parameters that represents time two given time one. And we can continue copying this indefinitely and each copy gives us the probability distribution of the next time slice given the one that we just had and so we can construct arbitrarily along Bayesian network.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_2_4.png"/></center>
</p>
<p align="justify">
<b>2-time-slice Bayesian Network</b><br>
$\bigstar$ A transition model (2TBN) over $X_{1}, \cdots, X_{n}$ is specified as a BN fragment such that:<br>
-- The nodes include $X'_{1}, \cdots, X'_{n}$ and a subset of $X_{1}, \cdots, X_{n}$<br>
-- Only the nodes $X'_{1}, \cdots, X'_{n}$ have parents and a CPD<br>
$\bigstar$ The 2TBN defines a conditional distribution by chain rule
$$P(X' \mid X) = \prod_{i=1}^{n}P(X'_{i} \mid Parent_{X'_{i}})$$

<b>Dynamic Bayesian Network</b><br>
$\bigstar$ Adynamic Bayesian network (DBN) over $X_{1}, \cdots, X_{n}$ is defined by<br>
-- 2TBN $BN_{\rightarrow}$ over $X_{1}, \cdots, X_{n}$<br>
-- a Bayesian network $BN^{(0)}$ over $X_{1}^{(0)}, \cdots, X_{n}^{(0)}$<br><br>

<b>Ground Network</b><br>
$\bigstar$ For a trajectory over $O, \cdots, T$ we define a ground (unrolled network) such that<br>
-- The dependency model for $X_{1}^{(0)}, \cdots, X_{n}^{(0)}$ is copied from $BN^{(0)}$<br>
-- The dependency model for $X_{1}^{(t)}, \cdots, X_{n}^{(t)}$ for all t > 0 is copied from previous network $BN_{\rightarrow}$<br><br>

<b>Summary</b><br>
$\bigstar$ DBNs are a compact representation for encoding structured distributions over <b>arbitrarily long temporal trajectories</b><br>
$\bigstar$ They make assumptions that may require appropriate model (re)design: Markov assumption and Time invariance<br><br>
</p>

#### 1.3.3 Temporal Model: HMMs
<p align="justify">
<b>Hidden Markov Models</b><br>
One simple yet extraordinarily class of probabilistic temporal models is the hidden Markov models. Although these are models can be viewed as a subclass of dynamic Bayesian networks. We'll see that they have their own type of structure that makes them particularly useful for a broad range of applications.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_3_1.png"/></center>
</p>
<p align="justify">
A hidden Markov model in its simplest form can be viewed as a probabilistic model that has a state variable S and a single observation variable O. And so the model really has only two publistic pieces, there is the transition model that tells us the transition from one state to the next over time and then there is the observation model, that tells us in a given state how likely we are to see different observations.<br><br>

We can unroll 2TBN to produce an unrolled network, which has the same repeated structure state at time zero move to the state at time one, and so on, and at each state, we make an appropriate observation. But what's interesting about hidden Markov models is that they often have a lot of internal structure that manifests most notably here in the transition model, but sometimes that was on the observation model.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_3_2.png"/></center>
</p>
<p align="justify">
Here is an example of what the of what a structured transition model would look like and each of these nodes in here is not a random variable, rather it is a particular assignment to the state variable, sort of state that the model might been. And what you see here is the structure of the transition matrix that tells us that from S1 for example, the the model is likely to transition to S2 with probability of 0.7 or stay in S1 with a probability of 0.3. And these two outgoing probabilities have to sum to one because it's a probability distribution over the next state, given that in the current time point the model is in state S1. And we similarly have that for, all other states, so here from S4, for example, there is the probability of 0.9 of going to S2 and 0.1 of staying at S4.<br><br>

<b>Numerous Applications</b><br>
$\bigstar$ Robot localization<br>
$\bigstar$ Speech recognition<br>
$\bigstar$ Biological sequence analysis<br>
$\bigstar$ Text annotation<br><br>

We take robot localization for an example
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_3_3.png"/></center>
</p>
<p align="justify">
$S^{(i)}$ is a state variable that represents the robot pose that is the position, and potentially orientation of the robot within a map at each point in time.<br><br>

$u^{(i)}$ is an external control signal, which is the guidance that the robot is told of move left, move right, turn around, since these variables are observed and externally imposed, they're not really stochastic random variables, they are just inputs to the system.<br><br>

In addition, the observation variables $o^{(i)}$ is what the robot observes at each point in the process, which depends on both their position, and of course on, the map position. So that the robot's observations depend on the overall architecture of the space that they're in and the state that they're building.<br><br>

we're going to have sparsity in the transition model, because the robot had jump around from one state to the other so within a single step. and so there is only going to be a limited set of positions at time T plus one given where the robot is at time T.<br><br>

<b>Summary</b><br>
$\bigstar$ HMMs can be viewed as a subset of DBNs<br>
$\bigstar$ HMMs seem unstructured at the level of random variables<br>
$\bigstar$ HMMs structure typically manifests in sparity and repeated elements within the transition matrix<br>
$\bigstar$ HMMs are used in a wide variety of applications for modeling sequences<br><br>
</p>

#### 1.3.4 Plate Models
<p align="justify">
<b>Modeling Repetition</b><br>
Imagine that we're repeatedly tossing the same coin again and again. So we have an <b>outcome</b> variable, and what we'd like to model is the repetition of multiple tosses. And so we're going to put a little box around that outcome variable, and this box which is called a <b>plate</b>, which is a way of denoting that the outcome variable is indexed. Usually we don't denote explicitly by different tosses of the coin t. And the reason for calling it a plate is because the intuition of the this is a stack of identical plates. That's kind of where the idea comes from for plate model.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_4_1.png"/></center>
</p>
<p align="justify">
And looking at what that model denotes is if we have a set of coins. The coin tosses t1 up to tk.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_4_2.png"/></center>
</p>
<p align="justify">
$$\{ t_{1}, \cdots, t_{k} \}$$

It basically says that we have a set of random variables, outcome t1 up to outcome tk. So we've just reproduced the outcome variable in its multiple copies.<br><br>

A random variable $\theta$ who is the actual CPD parameterization is explicitly into the model, so that we can show how different variable depend on that. And we can see that $\theta$ is outside of the plate, which means that it's not indexed by t and it's the same for all values of t. In other word, we have all of these outcomes depend on the exact same parameterization. And the CPD of the outcome of t1 is copied from this parameterization $\theta$.<br><br>

Going back to our university with multiple students, we now have a two variable model where we have intelligence and grade. And we now index that by different students s, which again indicates that we have a repetition, a copying of this template model. In this case, I only made two copies for one for student 1 and the other one for student 2.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_4_3.png"/></center>
</p>
<p align="justify">
And once again, if we wanted to encode dependence on the parameters. So we might have $\theta_{I}$, which represents the CPD for Intelligence. And we might have $\theta_{G}$, which represents the CPD for Grade. $\theta_{I}$ and $\theta_{G}$ are out of the plate. And we would have exactly the same idea that $\theta_{I}$ enforces the two I variables and $\theta_{G}$ enforces the two g variables. Sometimes in many models, we will include those parameters explicitly within the model. But often when we have a parameter that's outside of all plates. We won't denote it explicitly.<br><br>

<b>Nested Plates</b><br>
We want to know how different types of objects in the model overlap with each other. In this case, we have two kinds of objects: courses and students. So the difficulty variable belongs in the course plate because it's a property of course and the students plate is inside the course plate.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_4_4.png"/></center>
</p>
<p align="justify">
Because the intelligence variable is in both the s plate and c plate, it's going to be indexed by both.<br><br>

We build our model by unraveling course ans student plate: two courses and two students. Both I and G are parametered by s and c. The implications are that the intelligence is now a property of both the student and course. Besides, the intelligence of the student in a particular course can influence the grade of the student in that course.<br><br>

<b>Overlapping Plates</b><br>
Plate s and plate c have a common property grade
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_4_5.png"/></center>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_4_6.png"/></center>
</p>
<p align="justify">
<b>Explicit Parameters Sharing</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_4_7.png"/></center>
</p>
<p align="justify">
<b>Collective Inference</b><br>
Consider a student takes two courses Geo101 and CS101. Our priority believe is 80% student have high intelligence.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_4_8.png"/></center>
</p>
<p align="justify">
According to CPDs, the probability that this student have high intelligence goes up if we know he has a A for Geo101, while the probability will go down if we know he has C for CS101.<br><br>

What if a bunch of students takes a bunch of course?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_4_9.png"/></center>
</p>
<p align="justify">
So, we are pretty sure CS101 is an easy course and we can determine si some student have high intelligence.<br><br>

<b>Plate Dependency Model</b><br>
$\bigstar$ For a template variable $A(U_{1}, \cdots, U_{k})$:<br>
-- It has template parents $B_{1}(\overline{U_{1}}), \cdots, B_{m}(\overline{U_{m}})$ and for each i, we must have 
$$\overline{U_{i}} \subseteq \{ U_{1}, \cdots, U_{k} \}$$

which means we cannot have an index in the parent that doesn't appear in the child<br>
-- CPD $P(A \mid B_{1}, \cdots, B_{m})$<br><br>

For example
$$G(s, c) \sim A(U_{1}, U_{2})$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_4_4.png"/></center>
</p>
<p align="justify">
We have 2 template parents I(s) for s and D(c) for c. This exactly says that the grade of a particular student in a particular course depends on the difficulty of that course and on the intelligence of that student.<br><br>

<b>Summary</b><br>
$\bigstar$ Template for an infinite set of BNs, each induced by a different set of domain objects<br>
$\bigstar$ Parameters and structure are reused within a BN and across different BNs<br>
$\bigstar$ Models encode correlations across multiple objects, allowing collective inferences<br>
$\bigstar$ Multiple 'languages', each with different tradeoffs in expressive power<br><br>
</p>

#### 1.3.5 Quiz
<p align="justify">
<b>1.</b><br>
<b>Which of the following plate models could have induced the ground network shown?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_5_1.png"/></center>
</p>
<p align="justify">
<b>Answer:</b> C.<br>
From the ground network, we can see that A and B belong only to plate x, C belongs to x and y, D belongs to x and z and E belongs to all 3. Moreover, there needs to be a direct edge from A to E.<br><br>

<b>2. Independencies in DBNs</b><br>
<b>In the following DBN, which of the following independence assumptions are true? You may select 1 or more options.</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_5_2.png"/></center>
</p>
<p align="justify">
A. $(X^{(t+1)} \perp X^{(t)} \mid X^{(t-1)})$<br>
B. $(X^{(t)} \perp X^{(t-1)})$<br>
C. $(O^{(t)} \perp O^{(t-1)})$<br>
D. $(O^{(t)} \perp O^{(t-1)} \mid X^{(t)})$<br>
<b>Answer:</b> D.<br>
$O^{(t)}$ and $O^{(t-1)}$ are dependent because $O^{(t-1)} \leftarrow X^{(t-1)} \rightarrow X^{(t)} \rightarrow O^{(t)}$<br><br>

<b>3. Applications of DBNs</b><br>
<b>For which of the following applications might one use a DBN (i.e. the Markov assumption is satisfied)? You may select 1 or more options.</b><br>
A. Modeling the behavior of people, where a person's behavior is influenced by only the behavior of people in the same generation and the people in his/her parents' generation.<br>
B. Modeling time-series data, where the events at each time-point are influenced by only the events at the one time-point directly before it<br>
C. Modeling data taken at different locations along a road, where the data at each location is influenced by only the data at the same location and at the location directly to the East.<br>
D. Predicting the probability that today will be a snow day (school will be closed because of the snow), when this probability depends only on whether yesterday was a snow day.<br>
<b>Answer:</b> B, D.<br><br>

<b>4. Plate Semantics</b><br>
Let A and B be random variables inside a common plate indexed by i. <b>Which of the following statements must be true? You may select 1 or more options.</b><br>
A. For each i, A(i) and B(i) are not independent.<br>
B. For each i, A(i) and B(i) have edges connecting them to the same variables outside of the plate.<br>
C. For each i, A(i) and B(i) are independent.<br>
D. There is an instance of A and an instance of B for every i.<br>
<b>Answer:</b> D.<br><br>

<b>5.</b><br>
<b>5.1 *Plate Interpretation</b><br>
Consider the plate model below (with edges removed).
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_5_3.png"/></center>
</p>
<p align="justify">
<b>Which of the following might a given instance of $X$ possibly represent in the grounded model? (You may select 1 or more options.</b> Keep in mind that this question addresses the variable's semantics, not its CPD.)<br>
A. Whether a specific teacher T is a tough grader<br>
B. None of these options can represent $X$ in the grounded model
C. Whether someone with expertise E taught something of difficulty D at a place in location L<br>
D. Whether a specific teacher T taught a specific course C at school S<br>
E. Whether someone with expertise E taught something of difficulty D at school S<br>
<b>Answer:</b> D.<br><br>

<b>5.2 Grounded Plates</b><br>
Using the same plate model, now assume that there are s schools, t teachers in each school, and c courses taught by each teacher. <b>How many instances of the Expertise variable are there?</b><br>
A. t<br>
B. stc<br>
C. s<br>
D. st<br>
<b>Answer:</b> D.<br><br>

<b>6. Template Models</b><br>
Consider the plate model shown below. Assume we are given K Markets, L Products, M Consumers and N Locations.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_5_4.png"/></center>
</p>
<p align="justify">
<b>6.1 What is the total number of instances of the variable P in the grounded BN?</b><br>
A. $K \cdot L \cdot M$<br>
B. $K \cdot (L + M)$<br>
C. $L \cdot M$<br>
D. $(L \cdot M)^{K}$<br>
<b>Answer:</b> A.<br><br>

<b>6.2 What might P represent?</b><br>
A. Whether a specific product PROD was consumed by consumer C in market M<br>
B. Whether a specific product PROD was consumed by consumer C in all markets<br>
C. Whether a specific product of brand q was consumed by a consumer with age t in a market of type m that is in location a<br>
D. Whether a specific product PROD was consumed by consumer C in market M in location L<br>
<b>Answer:</b> A.<br><br>

<b>7. Time-Series Graphs</b><br>
<b>Which of the time-series graphs satisfies the Markov assumption? You may select 1 or more options.</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_5_5.png"/></center>
</p>
<p align="justify">
A. (a)<br>
B. (b)<br>
C. (c)<br>
<b>Answer:</b> B.<br>
(c) fails because of backward edges, which cause time-slices to depend on both the previous and the following time-slice.<br><br>

<b>8. *Unrolling DBNs</b><br>
Below are 2-TBNs that could be unrolled into DBNs. Consider these unrolled DBNs (note that there are no edges within the first time-point).
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_3_5_6.png"/></center>
</p>
<p align="justify">
<b>In which of them will $(X^{(t)} \perp Z^{(t)} \mid Y^{(t)})$ hold for all t, assuming $Obs^{(t)}$ is observed for all t and $X^{(t)}$ and $Z^{(t)}$ are never observed? You may select 1 or more options.</b><br>
Hint: Unroll these 2-TBNs into DBNs that are at least 3 time steps long (i.e., involving variables from t-1 to t+1)<br>
A. (a)<br>
B. (b)<br>
C. (c)<br>
<b>Answer:</b> B.<br>
(a) is not correct because of an active trail $X^{(t)} \leftarrow X^{(t-1)} \rightarrow Y^{(t-1)} \rightarrow Z^{(t-1)} \rightarrow Z^{(t)}$<br>
(c) is not correct because of $X^{(t)} \rightarrow X^{(t+1)} \rightarrow Y^{(t+1)} \rightarrow Obs^{(t+1)} \leftarrow Z^{(t+1)} \leftarrow Z^{(t)}$<br><br>
</p>

### 1.4 Structured CPDs for Bayesian Networks
#### 1.4.1 Structured CPDs
<p align="justify">
<b>Tabular Representations</b><br>
Usually, a table is used to represent the CPDs, but if we have enormous variables, tabular representation is not suitable.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_1_1.png"/></center>
</p>
<p align="justify">
<b>General CPDs</b><br>
$\bigstar$ $P(X \mid Y_{1}, \cdots, Y_{k})$ specifies distribution over X for each assignment $Y_{1}, \cdots, Y_{k}$.<br>
$\bigstar$ Can use any function to specify a factor $\phi(X, Y_{1}, \cdots, Y_{k})$ such that $\sum_{X} \phi(X, Y_{1}, \cdots, Y_{k}) = 1$ for all $Y_{1}, \cdots, Y_{k}$<br><br>

<b>Many Models</b><br>
$\bigstar$ Deterministic CPDs<br>
$\bigstar$ Tree-structured CPDs<br>
$\bigstar$ Logistic CPDs & generalizations<br>
$\bigstar$ Noisy OR / AND<br>
$\bigstar$ Linear Gaussiana & generalizations<br><br>

<b>Context-Specific Independence</b><br>
$$P \models (X \perp_{c} Y \mid Z, c)$$
$$P(X, Y \mid Z, c) = P(X \mid Z, c)P(Y \mid Z, c)$$
$$P(X \mid Y, Z, c) = P(X \mid Z, c)$$
$$P(Y \mid X, Z, c) = P(Y \mid Z, c)$$<br>
</p>

#### 1.4.2 Tree-Structured CPDs
<p align="justify">
<b>Tree CPD</b><br>
Imagine a student is applying a job and the prospect of the student to get the job depends on 3 variables: apply, letter and SAT. Besides, we construct a possible CPDs for this model.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_2_1.png"/></center>
</p>
<p align="justify">
This is a branching process. For example, if the student didn't apply this job, the probability of getting the job is 0.2. If the student did apply and he has a SAT score, the probability of getting the job will be 0.9.<br><br>

<b>Which context-specific independencies are implied by the structure of this CPD?</b><br>
A. $(J \perp_{c} L \mid a^{1}, s^{1})$<br>
B. $(J \perp_{c} L \mid a^{1})$<br>
C. $(J \perp_{c} L, S \mid a^{0})$<br>
D. $(J \perp_{c} L \mid s^{1}, A)$<br>
<b>Answer:</b> A, C, D.<br><br>

Another example
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_2_2.png"/></center>
</p>
<p align="justify">
<b>Which non-context-specific independency is implied by the structure of this CPD?</b><br>
A. $(J \perp_{c} C \mid L_{1}, L_{2})$<br>
B. $(L_{1} \perp_{c} C \mid J, L_{2})$<br>
C. $(J \perp_{c} L_{1} \mid C, L_{2})$<br>
D. $(L_{1} \perp_{c} L_{2} \mid J, C)$<br>
<b>Answer:</b> D.<br>
Given J activates v-structure $L_{1} \rightarrow J \leftarrow L_{2}$.<br><br>

<b>Multiplexer CPDs</b><br>
Multiplexer CPD in this case actually has the following structure.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_2_3.png"/></center>
</p>
<p align="justify">
We have a set of random variables $Z_{1}$ up to $Z_{k}$ all of which take up some value in some particular space and the variable Y is a copy of one of the $Z_{i}$. Variable A is multiplexer, selector variable and it takes some value in space $\{ 1, \cdots, k \}$<br><br>

Y is deterministic, and its CPD given A and $Z_{1}, \cdots, Z_{k}$
$$
P(Y \mid A, Z_{1}, \cdots, Z_{k})
\begin{cases}
  1, \quad Y = Z_{A} \\
  0, \quad \text{otherwise}
\end{cases}
$$

Which means $A = a \Rightarrow Y = Z_{a}$. A tells us which Z need to be copied for Y.<br><br>

<b>Summary</b><br>
$\bigstar$ Compact CPD representation that captures context-specific dependencies<br>
$\bigstar$ Relevant in multiple applications<br>
-- Hardware configuration variables<br>
-- Medical settings<br>
-- Dependence on agent's action<br>
-- Perceptual ambiguity<br><br>
</p>

#### 1.4.3 Independence of Causal Influence
<p align="justify">
<b>Noisy OR CPD</b><br>
Consider a situation that we have a variable Cough that depends on multiple different factors, such as pneumonia, flu, tuberculosis, bronchitis, and so on.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_3_1.png"/></center>
</p>
<p align="justify">
Tree-CPD doesn't work because this is not the case where we depend on one only certain context and not on others. In other word, Cough depends on all of them. In order to represent this model, we use Noisy OR CPD.<br><br>

We break down the dependencies of y on its parents $X_{1}, \cdots, X_{k}$ by introducing a bunch of interventing variables Z. Imagine y is Cough variable, $X_{i}$ is disease variable. The intermediate variable e.g. $Z_{1}$, captures the event at disease $X_{1}$ and it is $X_{1}$ by itself causing y. We can regard $Z_{i}$ as a noisy transmitter. If $X_{1}$ is true, y is true. Finally, if y is true, some X is succeed in making it true, which means y is a deterministic OR of its parents $Z_{0}, \cdots, Z_{k}$.
$$P(Z_{0} = 1) = \lambda_{0}$$
$$
P(Z_{i} = 1 \mid X_{i}) = 
\begin{cases}
  0, \quad X_{i} = 0 \\
  \lambda_{i}, \quad X_{i} = 1
\end{cases}
$$

$Z_{0}$ is the leak probability that Y gets turned on just by itself. $\lambda_{i} \in [0, 1]$ means how good $X_{i}$ is at turning y on.<br><br>

We want to know all of X fail to turing y on
$$P(Y = 0 \mid X_{1}, \cdots, X_{k}) = (1 - \lambda_{0}) \prod_{i: X_{i} = 1} (1 - \lambda_{i})$$

In contrast, y is turned on
$$P(Y = 1 \mid X_{1}, \cdots, X_{k}) = 1 - P(Y = 0 \mid X_{1}, \cdots, X_{k})$$

<b>What context-specific independencies are induced by a noisy OR CPD?</b><br>
A. $(Y \perp_{c} X_{2} \mid x_{1}^{1})$<br>
B. $(X_{1} \perp_{c} X_{2} \mid y^{1})$<br>
C. $(X_{1} \perp_{c} X_{2} \mid y^{0})$<br>
D. A noisy OR CPD induces no context-specific independencies.<br>
<b>Answer:</b> C.<br>
Given that y = 0, all $Z_{i}$ = 0, so that the trail of influence from $X_{1}$ to $X_{2}$ is blocked.<br><br>

<b>Independence of Causal Influence</b><br>
We develop the Noisy OR CPD a little
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_3_2.png"/></center>
</p>
<p align="justify">
In fact, we can generalize this model by remplacing OR by AND, MAX, sigmoid etc.<br><br>

<b>Sigmoid CPD</b><br>
Based on the graph above, we have a Sigmoid CPD
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_3_3.png"/></center>
</p>
<p align="justify">
$$P(y^{1} \mid X_{1}, \cdots, X_{k}) = sigmoid(Z)$$
$$sigmoid(z) = \frac{e^{z}}{1 + e^{z}}$$<br>
</p>

#### 1.4.4 Continuous Variables
<p align="justify">
Imagine we want to measure temperature
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_4_1.png"/></center>
</p>
<p align="justify">
We have a sensor for temperature now with a gaussian distribution $S \sim N(T; \sigma_{S}^{2})$. Similarly, temperature soon also follow a gaussian distribution
$$T' \sim N(\alpha T + (1 - \alpha)O; \sigma_{T}^{2})$$

O is outside temperature. This model is called linear gaussian.<br><br>

What about introduing a variable Door.
$$
T' \sim 
\begin{cases}
  N(\alpha_{0}T + (1 - \alpha_{0})O; \sigma_{0T}^{2}), \quad D = 0 \\
  N(\alpha_{1}T + (1 - \alpha_{1})O; \sigma_{1T}^{2}), \quad D = 1
\end{cases}
$$

This is called conditional linear gaussian<br><br>

<b>Linear Gaussian</b><br>
We have a general variable y and its parents $X_{1}, \cdots, X_{k}$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_4_2.png"/></center>
</p>
<p align="justify">
$$Y \sim N(w_{0} + \sum_{i=1}^{k} w_{i}X_{i}; \sigma^{2})$$

The mean of Y is a linear function of its parents but the variance don't depend anyone at all.<br><br>

<b>Conditional Linear Gaussian</b><br>
Based on linear gaussian, we introduce a discrete parent of y, say A, so linear gaussian's parameters depend on A
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_4_3.png"/></center>
</p>
<p align="justify">
$$Y \sim N(w_{a0} + \sum_{i=1}^{k} w_{ai} X_{i}; \sigma_{a}^{2})$$

In this case, the variance depends on A.<br><br>
</p>

#### 1.4.5 Quiz
<p align="justify">
<b>1.</b><br>
<b>Which of the following context-specific dependencies hold when X is a deterministic OR of $Y_{1}$ and $Y_{2}$? (Mark all that apply)</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_5_1.png"/></center>
</p>
<p align="justify">
A. $(X \perp Y_{1} \mid y_{2}^{1})$<br>
B. $(Y_{1} \perp Y_{2} \mid x^{0})$<br>
C. $X \perp Y_{1} \mid y_{2}^{0}$<br>
D. $(Y_{1} \perp Y_{2} \mid x^{1})$<br>
<b>Answer:</b> A, B.<br>
$y_{2}^{1}$ means $Y_{2}$ is true, so that we don't care $Y^{1}$, because X is true. But $Y_{2}$ is false, X is the same as $Y_{1}$, because $X = OR(Y_{1}, Y_{2})$<br>
If X is false, both $Y_{1}$ and $Y_{2}$ are false. If they're both false, then they're independent of each other. Because if you tell me that one of them is false, I already know the other one's false, so they're independent. Then they are independent of each other because if one is false, the other must be false. What if X is true? We cannot determine which Y is true to make X true.<br><br>

<b>2.</b><br>
$$P(y^{1} \mid X_{1}, \cdots, X_{k}) = sigmoid(w_{0} + \sum_{i=1}^{k} w_{i}X_{i})$$
The odds ratio of Y is
$$O(\textbf{X}) = \frac{P(y^{1}|\mathbf{X})}{P(y^{0}|\mathbf{X})}$$
It captures the relative likelihood of the two values of Y. <b>By what factor does $O(\textbf{X})$ change if the value of $X_{i}$ goes from 0 to 1?</b><br>
A. $\frac{e^{w_{i}}}{1 + e^{w_{i}}}$<br>
B. $w_{i}$<br>
C. $e^{w_{i}}$<br>
D. It depends on the values of the other $X_{i}$<br>
<b>Answer:</b> C.<br>
$$O(P) = \frac{P}{1 - P} = \frac{\frac{e^{Z}}{1 + e^{Z}}}{1 - \frac{e^{Z}}{1 + e^{Z}}} = e^{Z}$$
$$\frac{O(X_{i} = 0)}{O(X_{i} = 1)} = e^{w_{i}}$$

<b>3.</b><br>
Let L and V be the location and velocity of a car. Assume that the CPD on the right is a linear Gaussian.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_5_2.png"/></center>
</p>
<p align="justify">
<b>Which of the following statements could possibly be consistent with that CPD?</b><br>
A. The distance moved, $\left | L^{(t+1)} - L^{(t)} \right |$, will never be more than $2 V^{(t)} \Delta t$<br>
B. $L^{(t+1)}$ might possibly end up far from its expected position.<br>
C. Cars that move faster skid more and have greater variance in position.<br>
D. Due to friction, the single most likely value for $L^{(t+1)}$ is $L^{(t)} + 0.9 * V^{(t)} \Delta t$<br>
<b>Answer:</b> B, D.<br>
A would require the variance in position to be dependent on the current value of L, which cannot be coded for a linear Gaussian model (in which variables have fixed variances).<br>
C would not be possible, as Gaussian distributions are unbounded, and there will always be a non-zero (though small) probability of moving a large distance.<br><br>

<b>4.</b><br>
Consider the CPD below.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_5_3.png"/></center>
</p>
<p align="justify">
<b>4.1 Causal Influence</b><br>
<b>What is the probability that $E = e_{0}$ in the following graph, given an observation $A = a_{0}$, $B = b_{1}$, $C = c_{1}$, $D = d_{1}$?</b> Note that, for the pairs of probabilities that make up the leaves, the probability on the left is the probability of $e_{0}$, and the probability on the right is the probability of $e_{1}$<br>
<b>Answer:</b> 0.5.<br><br>

<b>4.2 Context-Specific Independencies in Bayesian Networks</b><br>
<b>Which of the following are context-specific independences that do exist in the tree CPD below?</b> (Note: Only consider independencies in this CPD, ignoring other possible paths in the network that are not shown here. You may select 1 or more options.)<br>
A. $(E \perp_{c} C \mid a^{0}, b^{0})$<br>
B. $(E \perp_{c} D \mid a^{0})$<br>
C. $(E \perp_{c} C \mid b^{0}, d^{0})$<br>
D. $(E \perp_{c} D, B \mid a^{1})$<br>
<b>Answer:</b> A, D.<br><br>

<b>5.</b><br>
We have a Bayesian network.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_4_5_4.png"/></center>
</p>
<p align="justify">
<b>5.1 Independencies with Deterministic Functions</b><br>
If the node B is a deterministic function of its parent A, <b>which of the following is an independence statement that holds in the network? You may select 1 or more options.</b><br>
A. $(A \perp B \mid C, D)$<br>
B. $(A \perp D \mid B)$<br>
C. $(B \perp D \mid C)$<br>
D. $(C \perp D \mid B)$<br>
<b>Answer:</b> B, D.<br><br>

<b>5.2 Independencies in Bayesian Networks</b><br>
Let B no longer be a deterministic function of its parent A. <b>Which of the following is an independence statement that holds in the modified Bayesian network? You may select 1 or more options.</b><br>
A. $(B \perp D \mid C)$<br>
B. $(A \perp D \mid C)$<br>
C. $(A \perp D \mid B)$<br>
D. $(C \perp D \mid A)$<br>
<b>Answer:</b> C.<br>
Since C is not on the active trail from A to D, observing C does not make A and D independent.<br><br>
</p>

### 1.5 Markov Networks
<p align="justify">
Undirected graph<br><br>
</p>

#### 1.5.1 Pairwise Markov Networks
<p align="justify">
Imagine 4 students are in a group. Well, Alice and Charles don't get along and Bob and Debbie had a bad breakup so they don't talk to each other. So, we have the study pairs that are marked by the edges on this diagram
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_1_1.png"/></center>
</p>
<p align="justify">
If two students study together, they influence each other. The influence flows both the directions. For example, Alice and Bob study together, if one of them has the misconception, then the other is likely to have too. So how to parametrize such an undirected graph?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_1_2.png"/></center>
</p>
<p align="justify">
The numbers in each table mean the local happiness of variables with taking a particular joint assignement. For example, A and B take a joint assignement, the happiest assignement for A and B in isolation of everything else is $a^{0}b^{0}$, which means neither has the misconception, so that that is the happiest assignement. $a^{1}b^{1}$ means both of them have the misconception.<br><br>

We want to define a joint probability distribution by putting these pieces together
$$\widetilde{P}(A, B, C, D) = \phi_{1}(A, B) \times \phi_{2}(B, C) \times \phi_{3}(C, D) \times \phi_{4}(A, D)$$

Why is this not a proper probability distribution?<br>
-- It does not sum to 1<br>
-- It is not necessarily between 0 and 1.<br><br>

We normalize it
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_1_3.png"/></center>
</p>
<p align="justify">
$$P(A, B, C, D) = \frac{1}{Z} \widetilde{P}(A, B, C, D)$$

Z is called partition function.<br><br>

Consider the pairwise factor $\phi_{1}(A, B)$. That potential is proportional to:<br>
A. The marginal probability P(A, B)<br>
B. The conditional probability P(A | B)<br>
C. The conditional probability P(A, B | C, D)<br>
D. None of the above<br>
<b>Answer:</b> D.<br><br>

Here, we have a set of factors that we used to construct this distribution
<table class="c">
  <tr><th>A</th><th>B</th><th>Prob</th></tr>
  <tr><td>$a^{0}$</td><td>$b^{0}$</td><td>0.13</td></tr>
  <tr><td>$a^{0}$</td><td>$b^{1}$</td><td>0.69</td></tr>
  <tr><td>$a^{1}$</td><td>$b^{0}$</td><td>0.14</td></tr>
  <tr><td>$a^{1}$</td><td>$b^{1}$</td><td>0.04</td></tr>
</table>
</p>
<p align="justify">
The probability of A and B is defines by a factor $\Phi = \{ \phi_{1}, \phi_{2}, \phi_{3}, \phi_{4} \}$. We note it $P_{\Phi}(A, B)$<br><br>

We can see $\Phi$ respect that A and B like to agree with each other.
$$
\begin{aligned}
& \phi_{1}(A = a^{0}, B = b^{0}) = 30 \\
& \phi_{1}(A = a^{1}, B = b^{1}) = 10 \\
& P(A = a^{0}, B = b^{0}) = 0.13 \\
& P(A = a^{1}, B = b^{1}) = 0.04
\end{aligned}
$$

Why do we have the highest propability when $a^{0}b^{1}$? B really likes to aggree with C and A and D are closely tied, while C and D strongly disagree. Now all three of these factors are actually stronger. So, it doesn't work that D agrees with A, A agrees with B, B agrees with C and C disagrees with D. In other word, somewhere in this loop has to be broken. The place where it gets broken is A and B and it's a week factor.<br><br>

<b>So the A and B probability is actually some kind of complicated aggregate of these different factors that are used to compose the Markov network.</b> There isn't a natural mapping between the probability distribution and factors that are used to compose it. This is in direct contrast to Bayesian network, where factors were all conditional propabilities and you could just look at the distribution and compute them. Here we can't do that and that often turns out to affect things like how we can learn, from data, because we can't just extract them directly from the propability distribution.<br><br>

<b>A pairwise Markov network is an undirected graph whose nodes are $X_{1}, \cdots, X_{n}$ and each edge $X_{i} - X_{j}$ is associated with a factor (potential) $\phi_{ij}(X_{i}, X_{j})$</b><br><br>
</p>

#### 1.5.2 General Gibbs Distribution
<p align="justify">
Imagine we have 4 random variables A, B, C, D and pairwise edges between each two of them.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_2_1.png"/></center>
</p>
<p align="justify">
Consider a fully connected pariwise Markov network over $X_{1}, \cdots, X_{n}$ where each $X_{i}$ has d values. How many parameters does the network have?<br>
$$\binom{n}{2} \times d^{2} = O(n^{2}d^{2})$$

What if a general probability distribution? $O(d^{n})$, which is much bigger than $O(n^{2}d^{2})$. So, without getting into formal arguement, Pairwise Markov networks are not sufficiently expressive to capture all probability distributions.<br><br>

<b>Not every distribution can be represented as a pairwise Markov network</b><br><br>

<b>Gibbs Distribution</b><br>
$\bigstar$ Parameters:<br>
-- General factros $\phi_{i}(D_{i})$<br>
-- $\Phi = \{ \phi_{i}(D_{i}) \}$
$$\Phi = \{ \phi_{1}(D_{1}), \cdots, \phi_{k}(D_{k}) \}$$
$$\widetilde{P}_{\Phi}(X_{1}, \cdots, X_{n}) = \prod_{i=1}^{k} \phi_{i}(D_{i})$$
$$Z_{\Phi} = \sum_{X_{1}, \cdots, X_{n}} \widetilde{P}_{\Phi}(X_{1}, \cdots, X_{n})$$
$$P_{\Phi}(X_{1}, \cdots, X_{n}) = \frac{1}{Z_{\Phi}} \widetilde{P}_{\Phi}(X_{1}, \cdots, X_{n})$$

<b>Induced Markov Network</b>
$$\Phi = \{ \phi_{1}(D_{1}), \cdots, \phi_{k}(D_{k}) \}$$

Induced markov network $H_{\Phi}$ has an edge $X_{i} - X_{j}$ whenever there exists $\phi_{m} \in \Phi$ and $X_{i}, X_{j} \in \phi_{m}$.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_2_2.png"/></center>
</p>
<p align="justify">
For example, we have two factors $\phi_{1}(A, B, C)$ and $\phi_{2}(B, C, D)$. Then we represent them in our diagram with two colors.<br><br>

<b>Factorization</b><br>
P factorizes over H if there exists $\Phi = \{ \phi_{1}(D_{1}), \cdots, \phi_{k}(D_{k}) \}$, such that<br>
-- $P = P_{\Phi}$ (normalized product of factors)<br>
-- H is the induced graph for $\Phi$.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_2_3.png"/></center>
</p>
<p align="justify">
For example, <b>which Gibbs distribution would be induce the graph H</b><br>
A. $\phi_{1}(A, B, D)$, $\phi_{2}(B, C, D)$<br>
B. $\phi_{1}(A, B)$, $\phi_{2}(B, C)$, $\phi_{3}(C, D)$, $\phi_{4}(A, D)$, $\phi_{5}(B, D)$<br>
C. $\phi_{1}(A, B, D)$, $\phi_{2}(B, C)$, $\phi_{3}(C, D)$<br>
D. All of the above<br>
<b>Answer:</b> D.<br><br>

This tells us that we cannot read factorizations form the graphs because of more than one factorization.<br><br>

<b>Flow of Influence</b><br>
$\bigstar$ Influence can flow along any trail, regardless of the form of the factors.<br><br>

<b>Active Trail</b><br>
$\bigstar$ A trail $X_{1} - \cdots, - X_{n}$ is active given Z if no $X_{i}$ is in Z.<br><br>

<b>What is the difference between an active trail in a Markov Network and an active trail in a Bayesian Network?</b><br><br>

<b>Answer:</b> They are different in the case where Z is the descendant of a v-structure..<br>
Let's look at a v-structure and Z is the descendent of the v-structure. If Z is observed in Bayesian network, it will create an active trail allowing influence to flow. If Z is observed in a Markov Network, then influence stops at Z and the trail is inactive.<br><br>

<b>Summary</b><br>
$\bigstar$ Gibbs distribution represents distribution as a product of factors<br>
$\bigstar$ Induced Markov network connects every pair of nodes that are in the same factor<br>
$\bigstar$ Markov network structure doesn't fully specify the factorization of P<br>
$\bigstar$ But active trails depend only on graph structure<br><br>
</p>

#### 1.5.3 Conditional Random Fields
<p align="justify">
<b>Correlated Features</b><br>
Consider the case of image segmentation
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_3_1.png"/></center>
</p>
<p align="justify">
Imagine we are try to predict $C_{i}$ (target), a label for a particular superpixel and $X_{i1}, \cdots, X_{ik}$ represent some features, such as color, texture histogram etc.<br><br>

The problem is that these features are much informative about the classes of pixel of which are very correlated to each other. If we have features that are very correlated to each other, we will get redundant information.<br><br>

<b>CRF Representation</b><br>
$$\Phi = \{ \phi_{1}(D_{1}), \cdots, \phi_{k}(D_{k}) \}$$

Unnormalized measure
$$\widetilde{P}_{\Phi}(X, Y) = \prod_{i=1}^{k} \phi_{i}(D_{i})$$

Partition function of X
$$Z_{\Phi}(X) = \sum_{Y} \widetilde{P}_{\Phi}(X, Y)$$

Normalized conditional distribution
$$P_{\Phi}(Y \mid X) = \frac{1}{Z_{\Phi}(X)} \widetilde{P}_{\Phi}(X, Y)$$

Consider a new factor $\phi(\mathbf{D})$ such that $\mathbf{D} \subseteq X$.<b>What is the effect of adding this factor $\phi$ on the distribution P(Y∣X)?</b><br>
A. Introducing $\phi$ can change the distribution P(Y∣X).<br>
B. $\phi$ multiplies both the unnormalized measure and the partition function and therefore cancels out, so P(Y∣X) remains unchanged.<br>
C. $\phi$ affects only the partition function and therefore P(Y∣X) remains unchanged.<br>
D. Introducing $\phi$ doesn't change P(Y∣X) only if D is a single variable.<br>
<b>Answer:</b> B.<br>
$$P_{\Phi}(Y \mid X) = \frac{\widetilde{P}_{\Phi}(X, Y)}{Z_{\Phi}(X)}$$
$$Z_{\Phi}(X) = \sum_{Y} P_{\Phi}(X, Y)$$
$$\tilde{P}_{\Phi}(X, Y) = \prod_{i} \phi_{i} (D_{i})$$

Adding a new factor means it will multiplied in the numerator and the denominator and therefore will cancel out.<br><br>

<b>CRF and Logistic Model</b><br>
Imagine we have binary variable X and Y
$$\phi_{i}(X_{i}, Y) = e^{w_{i} I_{X_{i}=1, Y = 1}}$$

Where $I_{X_{i}=1, Y = 1}$ is an indicator function.
$$
\phi_{i}(X_{i}, Y=1) = 
\begin{cases}
  1, \quad X_{i} = 0 \\
  e^{w_{i}}, \quad X_{i} = 1
\end{cases}
= e^{w_{i} X_{i}}
$$
$$\phi_{i}(X_{i}, Y=0) = 1$$

Compute our unnormalized density
$$\tilde{P_\Phi}(X,Y=1) = e^{\sum_{i} w_{i}X_{i}}$$
$$\tilde{P_\Phi}(X,Y=0) = 1$$

Normalized density
$$P_{\Phi}(Y = 1 \mid X) = \frac{e^{\sum_{i} w_{i}X_{i}}}{1 + e^{\sum_{i} w_{i}X_{i}}}$$

In fact, this is the sigmoid function. Logistic model is a simple CRF.<br><br>

<b>CRF for Image Segmentation</b><br>
$\bigstar$ Node factors can use any features of the image<br>
-- Color histograms<br>
-- Texture features<br>
-- Discriminative patches<br>
$\bigstar$ Features can be in or out of the superpixel<br>
$\bigstar$ Correlations don't matter<br>
$\bigstar$ Can train a discriminative classifier (SVM, boosting) to improve performance<br><br>

<b>CRF for Language</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_3_2.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ A CRF is parameterized the same as a Gibbs distribution, but normalized differently.<br>
-- generalizes logistic regression model<br>
$\bigstar$ Don't need to model distribution over variables we don't care about<br>
$\bigstar$ Allows models with highly expressive features, without worrying about wrong independencies<br><br>
</p>

#### 1.5.4 Independencies in Markov Networks
<p align="justify">
<b>Seperation in MNs</b><br>
$\bigstar$ <b>Definition:</b> X and Y are seperated in H given Z if there is no active trail in H between X and Y given Z. In other word, no node along trail is observed.<br><br>

For example, here is a diagram H
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_4_1.png"/></center>
</p>
<p align="justify">
A and E are seperated in the diagram above given B, D, or given B, C.<br><br>

<b>Factorization $\rightarrow$ Independence: MNs</b><br>
$$I(H) = \{ (X \perp Y \mid Z): sep_{H}(X, Y \mid Z) \}$$

If P satisfies I(H), H is an I-map (independency map) of P.<br><br>

<b>Theorem:</b> If P factorizes over H, then H is an I-map of P.<br><br>

For the graph H, <b>which independence assertions are in I(H)?</b><br>
A. $(A \perp F \mid B, D)$<br>
B. $(A \perp C \mid B, D)$<br>
C. $(A \perp E \mid B)$<br>
D. $(C \perp F \mid B)$<br>

<b>Answer:</b> A, B, D.<br><br>

<b>Independence $\rightarrow$ Factorization</b><br>
$\bigstar$ <b>Theorem (Hammersley Clifford):</b> For a positive distribution P, if H is an I-map for P, then P factorizes over H.<br><br>

Here, probability is strictly greater than 0 for all assignments X. If we have a distribution that involves a terministic relationship, this property doesn't hold.<br><br>

<b>Summary</b><br>
$\bigstar$ Two equivalent views of graph structure (for positive distribution):<br>
-- Factorization: H allows P to be represented<br>
-- I-map: Independencies encoded by H hold in P<br>
$\bigstar$ If P factorizes over a graph H, we can read from the graph independencies that must hold in P (an independency map)<br><br>
</p>

#### 1.5.5 I-maps and perfect maps
<p align="justify">
How to take a distribution that has a certain set of independencies that it satisfies and encoded within a graph's structure? How well can we take that distribution and capture its independencies in the context of a graphical model?<br><br>

<b>Capturing Independencies in P</b><br>
We define this notion of I for a distribution P, as the set of all independent statements (X is independent of Y given Z), that hold for the distribution P
$$I(P) = \{ (X \perp Y \mid Z): P \models (X \perp Y \mid Z) \}$$

$\bigstar$ P factorizes over G $\rightarrow$ G is an I-map for P:
$$I(G) \subseteq I(P)$$

$\bigstar$ But not always vice versa: there can be independencies in I(P) that are not in I(G)<br><br>

If G is an I-map for P, <b>which of the following must be true?</b><br>
A. P and G have the same set of independence variables.<br>
B. Every independence in G is also in P.<br>
C. Every independence in P is also in G.<br>
D. P factorizes over G.<br>
<b>Answer:</b> B, D.<br>
A probability distribution is an I-map for a graph if every independence in the graph also is satisfied in the probability distribution. When this happens, the probability distribution can factorize over the graph.<br><br>

<b>Sparse Graph</b><br>
$\bigstar$ If a graph encodes more independencies<br>
-- it is sparser (has fewer parameters) and more informative about P<br>
$\bigstar$ Want a graph that captures as much of the structure in P as possible<br><br>

<b>Minimal I-map</b><br>
$\bigstar$ Minimal I-map: I-map without redundant edges<br>
For example, imagine a graph with two binary variables
$$X \rightarrow Y$$

If we have $P(Y \mid x^{1}) = P(Y \mid x^{0})$, then the edges can be removed because it is redundant.<br><br>

$\bigstar$ Minimal I-map may still not capture I(P)<br><br>

For example, we have a student netwok
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_5_1.png"/></center>
</p>
<p align="justify">
(a) is our original network and it is a minimal I-map, but (b) is also a minimal I-map because if no edge is redundant. If we remove any edge we will get a graph like one of (c), (d), (e), which doesn't represent the same I-map as (b).<br><br>

So, minimal I-maps are not necessarily the best tools for capturing structure in a distribution.<br><br>

<b>Perfect Map</b><br>
$\bigstar$ Perfect Map: I(G) = I(P)<br>
-- G perfectly captures independencies in P<br><br>

Unfortunately, perfect map is hard to come by. For example, one scenario doesn't have a perfect map. This is a distribution P that is actually represented by the pairwise Markov networks
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_5_2.png"/></center>
</p>
<p align="justify">
(a) is our original network, P gives a set of independency statements. None of (b), (c), (d) is an I-map of P.<br><br>

<b>MN as a Perfect Map</b><br>
$\bigstar$ Perfect Map: I(H) = I(P)<br>
-- H perfectly captures independencies in P<br><br>

For example, here is a Bayesian network (a) and MN graphs (b), (c), (d)
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_5_3.png"/></center>
</p>
<p align="justify">
We can see that no MN graph can exactly represent I(P), so there is no prefect map I(H) in this case<br><br>

<b>Uniqueness of a Perfect Map</b><br>
Imagine two graphs
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_5_4.png"/></center>
</p>
<p align="justify">
We can see these two graphs can represent the same distribution.<br><br>

<b>I-equivalence</b><br>
$\bigstar$ <b>Definition:</b> Two graphs $G_{1}$ and $G_{2}$ over $X_{1}, \cdots, X_{n}$ are I-equivalent if $I(G_{1}) = I(G_{2})$<br><br>

For example, $X \rightarrow Y \rightarrow Z$, $X \leftarrow Y \leftarrow Z$ and $X \leftarrow Y \rightarrow Z$ are I-equivalent.<br><br>

$\bigstar$ Most graphs have many I-equivalent variants.<br><br>

<b>Summary</b><br>
$\bigstar$ Graphs that captures more of I(P) are <b>more compact</b> and provide more insight.<br>
$\bigstar$ A minimal I-map may fail to capture a lot of structure even if present<br>
$\bigstar$ A perfect map is ideal, but may not exist<br>
$\bigstar$ Converting BNs $\leftrightarrow$ MNs loses independencies<br>
-- BNs $\rightarrow$ MNs: loses independencies in v-structure<br>
-- MNs $\rightarrow$ BNs: must add triangulating edges to loops<br><br>
</p>

#### 1.5.6 Log-Linear Models
<p align="justify">
<b>Log-Linear Representation</b><br>
$$\widetilde{P} = \prod_{i} \phi_{i}(D_{i}) \rightarrow \widetilde{P} = e^{-\sum_{j}w_{j}f_{j}(D_{j})}$$

Where $w_{j}$ is coefficients, $f_{j}$ is features like factors, $D_{i}$ is scopes
$$\widetilde{P} = \prod_{j} e^{-w_{j}f_{j}(D_{j})}$$

$\bigstar$ Each feature $f_{j}$ has a scope $D_{j}$<br>
$\bigstar$ Different features can have same scope<br><br>

<b>Representing Table Factors</b><br>
$$
\phi(X_{1}, X_{2}) =
\begin{pmatrix}
a_{00} & a_{01}\\
a_{10} & a_{11}
\end{pmatrix}
$$
$$
\begin{matrix}
f_{12}^{00} = \mathbf{I}_{X_{1}=0, X_{2}=0}\\
\\
f_{12}^{01} = \mathbf{I}_{X_{1}=0, X_{2}=1}\\
\\
f_{12}^{10} = \mathbf{I}_{X_{1}=1, X_{2}=0}\\
\\
f_{12}^{11} = \mathbf{I}_{X_{1}=1, X_{2}=1}
\end{matrix}
$$
$$\phi(X_{1}, X_{2}) = e^{-\sum_{kl}w_{kl}f_{ij}^{kl}(X_{1}, X_{2})}$$
$$w_{kl} = -log a_{kl}$$

<b>Ising Model</b><br>
Ising model have pairs of variables, so it's a pair-wise Markov network
$$E(x_{1}, \cdots, x_{n}) = -\sum_{i< j}w_{i,j}x_{i}x_{j} - \sum_{i}u_{i}x_{i}$$

Where $x_{i} \in \{ +1, -1 \}$ and $f_{i, j}(X_{i}, X_{j}) = X_{i} \cdot X_{j}$

$$P(\mathbf{X}) \propto e^{-\frac{1}{T}E(\mathbf{X})}$$

<b>Metric MRFs</b><br>
$\bigstar$ All $X_{i}$ take values in label space V
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_6_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ Distance function $\mu: V \times V \rightarrow \mathbb{R}^{+}$<br>
-- Reflexivity: $\mu(v, v) = 0$ for all v<br>
-- Symmetry: $\mu(v_{1}, v_{2}) = \mu(v_{2}, v_{1})$ for all $v_{1}$, $v_{2}$<br>
-- Triangle inequality: $\mu(v_{1}, v_{2}) \leq \mu(v_{1}, v_{3}) + \mu(v_{3}, v_{2})$ for all $v_{1}$, $v_{2}$, $v_{3}$<br><br>

If $\mu$ satisfies reflexivity and symmetry, we call it semi-metric; if it all 3 are satisfied, we call it metric.<br>
$$f_{i, j}(X_{i}, X_{j}) = \mu(X_{i}, X_{j})$$
$$e^{-w_{ij}f_{ij}(X_{i}, X_{j})}, \quad w_{ij} > 0$$

Values of $X_{i}$ and $X_{j}$ far in $\mu$ $\rightarrow$ lower probability<br><br>

For example
$$
\mu(v_{k}, v_{l}) =
\begin{cases}
  0, \quad v_{k} = v_{l}\\
  1, \quad \text{otherwise}
\end{cases}
$$
$$
\begin{pmatrix}
0 & 1 & 1 & 1\\
1 & 0 & 1 & 1\\
1 & 1 & 0 & 1\\
1 & 1 & 1 & 0
\end{pmatrix}
$$<br>
</p>

#### 1.5.7 Shared Features in Log-Linear Models
<p align="justify">
$\bigstar$ In most MRFs, same features and weight are used over many scopes<br><br>

Take Ising Model for example, we have a bunch of binary variables
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_7_1.png"/></center>
</p>
<p align="justify">
$$E(x_{1}, \cdots, x_{n}) = -\sum_{(i, j) \in Edegs} w_{i, j}x_{i}x_{j} - \sum_{i}u_{i}x_{i}$$

Where $x_{i}x_{j}$ is $f(x_{i}, x_{j})$. We can use $w$ to replace $w_{i, j}$<br><br>

For Natural Language Processing
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_7_2.png"/></center>
</p>
<p align="justify">
Some energy terms $w_{k}f_{k}(X_{i}, y_{i})$ repeat for all position i in the sequence. Some energy terms $w_{m}f_{m}(y_{i}, y_{i+1})$ also repeat for all positions i.<br><br>

<b>In the NLP MRF, of the following types of energy terms would it make sense to repeat?</b><br>
A. An energy term regarding whether a word ends in "ing".<br>
B. An energy term regarding whether two consecutive words begin with capital letters.<br>
C. An energy term regarding whether the third word is "sun".<br>
D. An energy term regarding whether there is a period at the end of a word.<br>
<b>Answer:</b> A, B, D.<br>
Energy terms for which you would not want different parameterization for different positions should be repeated. An energy term regarding whether the third word is "sun" is position-specific and it would be incorrect to repeat it.<br><br>

For Image Segmentation
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_7_3.png"/></center>
</p>
<p align="justify">
Some features and weights for all superpixels in the image.<br><br>

<b>Repeated Features</b><br>
$\bigstar$ Need to specify for each feature $f_{k}$ a set of scopes $Scopes[f_{k}]$<br>
$\bigstar$ For each $D_{k} \in Scopes[f_{k}]$ we have a term $w_{k}f_{k}(D_{k})$ in the energy function
$$w_{k} \sum_{D_{k} \in Scopes(f_{k})} f_{k}(D_{k})$$

<b>Summary</b><br>
$\bigstar$ Same feature & weight can be used for multiple subsets of variables<br>
-- Pairs of adjacent pixels/atoms/words<br>
-- Occurrences of same word in document<br>
$\bigstar$ Can provide a single template for multiple MNs<br>
-- Different images<br>
-- Different sentences<br>
$\bigstar$ Parameters and structure are reused within an MN and across different MNs<br>
$\bigstar$ Need to specify set of scopes for each feature<br><br>
</p>

#### 1.5.8 Quiz
<p align="justify">
<b>1. Factor Scope</b><br>
Let $\phi(a,c)$ be a factor in a graphical model, where a is a value of A and c is a value of C. <b>What is the scope of $\phi$?</b><br>
A. {A}<br>
B. {A, C, E}<br>
C. {A, C}<br>
D. {A, B, C, E}<br>
<b>Answer:</b> C.<br><br>

<b>2. Independence in Markov Networks</b><br>
Consider this graphical model and all of the edges are undirected (see modified graph below)
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_8_1.png"/></center>
</p>
<p align="justify">
<b>Which pairs of variables are independent in this network? You may select 1 or more options.</b><br>
A. No pair of variables are independent on each other.<br>
B. B, E<br>
C. A, D<br>
<b>Answer:</b> A.<br><br>

<b>3. Factorization</b><br>
<b>Which of the following is a valid Gibbs distribution over this graph?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_8_2.png"/></center>
</p>
<p align="justify">
A. $\frac{\phi(A) \times \phi(B) \times \phi(C) \times \phi(D) \times \phi(E) \times \phi(F)}{Z}$, where Z is the partition function<br>
B. $\phi(A, B, D) \times \phi(C, E, F)$<br>
C. $\phi(A) \times \phi(B) \times \phi(D) \times \phi(E) \times \phi(F)$<br>
D. $\frac{\phi(A, B, D) \times \phi(C, E, F)}{Z}$, where Z is the partition function<br>
<b>Answer:</b> A.<br><br>

<b>4. Factors in Markov Network</b><br>
<b>4.1 </b><br>
Let $\phi(A,B,C)$ be a factor in a probability distribution that factorizes over a Markov network. <b>Which of the following must be true? You may select 1 or more options.</b><br>
A. $\phi(a,b,c) \geq 0$, where a is a value of A, b is a value of B, and c is a value of C.<br>
B. A, B, and C do not form a clique in the network.<br>
C. A, B, and C form a clique in the network.<br>
D. There is no path connecting A, B, and C in the network.<br>
E. There is a path from A to B, a path from B to C, and a path from A to C in the network.<br>
<b>Answer:</b> A, C, E.<br><br>

<b>4.2 </b><br>
Let $\pi_{1}(A, B)$, $\pi_{2}(B, C)$ and $\pi_{3}(A, C)$ be all of the factors in a particular undirected graphical model. <b>Then what is $\sum_{A, B, C} \pi_{1}(A, B) \times pi_{2}(B, C) \times \pi_{3}(A, C)$? More than one answer could be correct</b><br>
A. Always equal to the partition function Z<br>
B. Always equal to 1<br>
C. Always greater than or equal to $\pi_{1}(a, b) \times pi_{2}(b, c) \times \pi_{3}(a, c)$, where a is a value of A, b is a value of B and c is a value of C<br>
D. Always less than or equal to 1<br>
E. Always less than or equal to $\pi_{1}(a, b) \times pi_{2}(b, c) \times \pi_{3}(a, c)$, where a is a value of A, b is a value of B and c is a value of C<br>
F. Always greater than or equal to 0<br>
<b>Answer:</b> A, C, F.<br><br>

<b>5. I-Map</b><br>
Graph G is a perfect I-map for distribution P, i.e. $\mathcal{I}(G)=\mathcal{I}(P)$.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_8_3.png"/></center>
</p>
<p align="justify">
<b>Which of the other graphs is a perfect I-map for P?</b><br>
<b>Answer:</b> None.<br>
I isn't because it has the extra independence relation $(A \perp C)$. II has $(B \perp C \mid D)$ but G has $(B \perp C)$ and III does not perserve an independence relationship in G.<br><br>

<b>6. I-Equivalence</b><br>
<b>6.1</b><br>
In the figure below, <b>graph G is I-equivalent to which other graph(s)?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_5_8_4.png"/></center>
</p>
<p align="justify">
<b>Answer:</b> I.<br><br>

<b>6.2</b><br>
Let Bayesian network G be a simple directed chain $X_{1} \rightarrow X_{2} \rightarrow \cdots \rightarrow X_{n}$ for some number n. <b>How many Bayesian networks are I-equivalent to G including G itself?</b><br>
<b>Answer:</b> n.<br>
The chain $X_{1} \rightarrow X_{2} \rightarrow \cdots \rightarrow X_{n}$ is I-equivalent, where i can be 2 through n (when i=n, all arrows point left). Thus, there are n−1 I-equivalent networks like this. Including the original network makes n.<br><br>
</p>

### 1.6 Decision Making
#### 1.6.1 Maximum Expected Utility
<p align="justify">
<b>Simple Decision Making</b><br>
A simple decision making situation D:<br>
$\bigstar$ A set of possible actions Val(A) = {$a^{1}, \cdots, a^{k}$}<br>
$\bigstar$ A set of states Val(X) = {$x^{1}, \cdots, x^{N}$}<br>
$\bigstar$ A distribution P(X | A)<br>
$\bigstar$ A utility function U(X, A)<br>
-- defines the agent's preferences, it allows us to evaluate different actions in terms of how much we prefer them<br><br>

<b>Expected Utility</b><br>
$$EU[D(a)] = \sum_{x}P(x \mid a)U(x, a)$$

$\bigstar$ Want to choose action a that maximizes the expected utility
$$a^{*} = arg \max_{a}EU[ D[ a ] ]$$

<b>Simple Influence Diagram</b><br>
We have a state random variable market, an action variable found (the agent peak one with peak probability), our situation is a budding entrepreneur, who just graduated from college and wants to decide whether to found a widget making company or not.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_1_1.png"/></center>
</p>
<p align="justify">
<b>Assuming that Found is a binary decision, what is the size of the CPD for Found?</b><br><br>

<b>Answer:</b> It does not have a CPD.<br>
Found is an action, not a random variable, so it does not have a CPD.<br><br>

We calculate the expected utility
$$EU(f^{0}) = 0$$
$$EU(f^{1}) = 0.5*(-7) + 0.3*5 + 0.2*20 = 2$$

So, the best action (or decision) is to found the market.<br><br>

<b>More Complex Influence Diagram</b><br>
We have a student diagram, where only one action is study and three utility functions $V_{Q}$, $V_{G}$ and $V_{S}$. $V_{Q}$ represents how much happiness for the student life, for example, if the course that he takes is easy and he studies hard, he may be happy. Similarly, $V_{G}$ measures how much his happiness for the grade and $V_{S}$ shows how much happiness for getting a job.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_1_2.png"/></center>
</p>
<p align="justify">
We have a composed utility function
$$V_{G} + V_{Q} + V_{S}$$

<b>Assuming Grade, Job, Study and Difficulty are binary, how many values we have to elicit for a joint utility function over all variables?</b><br><br>

<b>Answer:</b> $2^{4}$ = 16 values since there are 4 binary variables.<br><br>

<b>Information Edges</b><br>
We introduce another random variable Survey for Market-found diagram
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_1_3.png"/></center>
</p>
<p align="justify">
Decision rule $\delta$ at action node A is a CPD P(A | Parents(A))<br><br>

<b>Expected Utility with Information</b><br>
$$EU[D(\delta_{A})] = \sum_{x, a} P_{\delta_{A}}(x, a)U(x, a)$$

$\bigstar$ Want to choose the decision rule $\delta_{A}$ that maximizes the expected utility
$$arg \max_{\delta_{A}} EU[ D(\delta_{A}) ]$$
$$MEU(D) = \max_{\delta_{A}} EU[ D(\delta_{A}) ]$$

<b>Finding MEU DEcision Rules</b><br>
Take an example of market-survey-found as an example to illustrate how we find a MEU under decision rules
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_1_4.png"/></center>
</p>
<p align="justify">
$$EU[D(\delta_{A})] = \sum_{x, a} P_{\delta_{A}}(x, a)U(x, a)$$
$$EU[D(\delta_{F})] = \sum_{M, S, F} P(M)P(S \mid M) \delta_{F}(F \mid S)U(F, M)$$
$$= \sum_{S, F} \delta_{F}(F \mid S) \sum_{M} P(M)P(S \mid M)U(F, m)$$
$$= \sum_{S, F} \delta_{F}(F \mid S) \mu(F, S)$$

Where $\mu(F, S)$ is a marginalization of joint distribution over M. In fact, we can regard $\mu$ as utility function U.<br><br>

Suppose we have an inept cousin, Ben Bumbler, as a co-founder to your company. Ben is so inept that the outcome of founding a company even when the market is great is in doubt. We choose to model the situation by introducing a new random variable, Outcome, denoted by O, that has parents M and F. The Utility is now a function solely of O. <b>How would this change the calculation just presented?</b><br>
A. The term $\delta_F(F \vert S)$ would become $\delta_F(F \vert S, O)$.<br>
B. The term μ(S,F) would become μ(S,F,M,O) because we can no longer sum out the M variable since it is not directly connected to the utility U.<br>
C. The term μ(S,F) would become μ(S,F,O)<br>
D. You would have a different μ(S,F)<br>
<b>Answer:</b> D.<br>
The original summation would be
$$\sum_{M,S,F,O} P(M) P(S \mid M) \delta_{F}(F \mid S) P(O \mid F, M) U(O)$$

We would push the summation over M and O into the terms:
$$\sum_{S, F}\delta_{F}(F \mid S) \sum_{M} P(M) P(S \mid M) \sum_O P(O \mid F, M) U(O)$$

Thus, the μ(S,F) term would be different.<br><br>

<b>More Generally</b><br>
$$EU[D(\delta_{A})] = \sum_{x, a} P_{\delta_{A}}(x, a)U(x, a)$$
$$= \sum_{X_{1}, \cdots, X_{n}, A} ((\prod_{i}P(X_{i} \mid Parent_{X_{i}}))U(Parent_{U})\delta_{A}(A \mid Z))$$

Where $Z = Parent_{A}$<br>
$$= \sum_{Z, A}\delta_{A}(A \mid Z)\sum_{W}((\prod_{i}P(X_{i} \mid Parent_{X_{i}}))U(Parent_{U}))$$

Where $W = {X_{1}, \cdots, X_{n}} - Z$, all variables except Z<br>
$$= \sum_{Z, A}\delta_{A}(A \mid Z)\mu(A, Z)$$

$$
\delta_{A}^{*}(a \mid z) =
\begin{cases}
  1, \quad a = arg\max_{A}\mu(A, z) \\
  0, \quad \text{otherwise}
\end{cases}
$$

<b>MEU Algorithm</b><br>
$\bigstar$ To compute MEU & optimize decision at A:<br>
-- Treat A as random variable with arbitrary CPD<br>
-- Introduce utility factor with scope $Parent_{U}$<br>
-- Eliminate all variables except A, Z (A's parents) to produce factor $\mu(A, Z)$<br>
-- For each z, set
$$
\delta_{A}^{*}(a \mid z) =
\begin{cases}
  1, \quad a = arg\max_{A}\mu(A, z) \\
  0, \quad \text{otherwise}
\end{cases}
$$

<b>Decision Making under Uncertainty</b><br>
$\bigstar$ MEU principle provides rigorous foundation<br>
$\bigstar$ PGMs provide structured representation for probabilities, actions and utilities<br>
$\bigstar$ PGM inference methods (VE) can be used for<br>
-- Finding the optimal strategy<br>
-- Determining overall value of the decision situation<br>
$\bigstar$ Efficient methods also exist for<br>
-- MUltiple utility components<br>
-- Multiple decisions<br><br>
</p>

#### 1.6.2 Utility Functions
<p align="justify">
<b>St. Petersburg Paradox</b><br>
$\bigstar$ Faire coin is tossed repeatedly until it comes up heads , say on the $n^{th}$ toss<br>
$\bigstar$ Payoff = $2^{n}$ dollars<br><br>

<b>What is the expected payout for this game?</b><br>
$$\frac{1}{2} \cdot 2 + \frac{1}{4} \cdot 2^{2} + \frac{1}{8} \cdot 2^{3} + \cdots = +\infty$$

So in principal people might be willing to pay any amount to play this game, because their expected pay-off is bigger than any amount, but for most people, the value of playing this game is approximately $2 which is a strong indication that their preferences are not linear in the amount of money that they earn.<br><br>

<b>Utilitiy Curve</b><br>
In order to quantify this case, we use utility curve whose x axis is the reward and y axis is the utility that an agent describes to that.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_2_1.png"/></center>
</p>
<p align="justify">
If a reward is 500 dollars, we can find its corresponded utility function value in this diagram.<br><br>

Look at a situation that involves some risk. Consider a set of lotteries where I get 0 dollars with 1-p and 1000 dollars with p
$$
D = 
\begin{cases}
  0, \quad 1-p \\
  1000, \quad p
\end{cases}
$$

Because of the linearity of expected utility for all these lotteries, we have a blue line. If p = 0.5, we have a utility function value for 500 in this diagram and through the curve, it points to 400 dollars. So this 400 dollars is called <b>certainty equivalent</b> of the lottery D. That is to say, if we trade for this lottery, we will get 400 dollars for certain, while 500 is our expected reward. So the difference between 400 and 500 is called <b>insurance/risk premium</b>. It is so called because that's where insurance companies make their money. It shows a person's willness to take less money with certainty over a more risky proposition.<br><br>

Say Thelma has a lottery ticket for a game which pays out 1000 dollars with probability p = 0.5 and 0 dollar otherwise. Her friend, Louise, thinks that the probability of paying out 1000 dollars, is actually 0.6. Louise offers Thelma 450 dollars for her ticket. <b>Should Thelma accept her offer?</b> Assume Louise's utility function for money is the same as was shown in lecture.<br>
A. Cannot be determined from the information given.<br>
B. Yes<br>
C. No<br>
D. She should accept with probability $log_{2}1.1$<br>
<b>Answer:</b> A.<br>
We do not have enough information to answer this question because Thelma's utility function is not specified.<br><br>

Besides, this curve's shape is concave, representing a risk profile which is <b>risk averse</b>. If a curve is linear, we call it <b>risk neutral</b>. Conversely, if we have a convex curve, we call it <b>risk seeking</b>. The risk seeking behavior occurs for example in other gambling situations, where one is willing to actually take a loss in terms of the expected reward for the small chance of a getting a really high pay-off.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_2_2.png"/></center>
</p>
<p align="justify">
<b>Multi-Attribute Utilities</b><br>
$\bigstar$ All attributes affecting preferences must be integrated into one utility function<br><br>

Take prenatal diagnosis for an example, we care about whether a baby is going to end up with some kind of genetic disorder and we have five revelent attributes: testing, knowledge, down's syndrome, loss of fetus and future pregnancy. Here is one combination of all attributes.
$$U_{1}(T) + U_{2}(K) + U_{3}(D, L) + U_{4}(L, F)$$

If the variables T, K, D, L, and F are all binary valued variables, and the utility function is decomposed into U(T)+U(K)+U(D,L)+U(L,F), <b>how many different utility values have to be elicited to characterize the utility function?</b><br>
<b>Answer:</b> 12.<br>
2 + 2 + 2*2 + 2*2 = 12 (as opposed to 32 if we used a joint utility function over all variables).<br><br>

<b>Summary</b><br>
$\bigstar$ Our utility function determines our preferences about decisions that involve uncertainty<br>
$\bigstar$ Utility generally depends on multiple factors<br>
-- Money, time, chances of death<br>
$\bigstar$ Relationship is usually non-linear<br>
-- Shape of of utility curve determines attitude to risk<br>
$\bigstar$ Multiple-attribute utilities can help decompose high-dimensional function into tractable pieces<br><br>
</p>

#### 1.6.3 Value of Perfect Information
<p align="justify">
<b>Value of Information</b><br>
$\bigstar$ VPI(A | X) is the value of observing X before choosing an action of A.<br>
$\bigstar$ D = original influence diagram<br>
$\bigstar$ $D_{X \rightarrow A}$ = influence diagram with edge $X \rightarrow A$
$$VPI(A \mid X) := MEU(D_{X \rightarrow A}) - MEU(D)$$

Consider two situations, one is that the agent founds the company without any additional information about the value of the market; the other is that the agent makes an observation reagrding the survey variable prior to making the decision whether to found this company
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_3_1.png"/></center>
</p>
<p align="justify">
So, the value of decision making is
$$VPI(F \mid S) = MEU(D_{S \rightarrow F}) - MEU(D)$$

Assume the agent makes optimal decision of the original decision making situation, we can see how much the agent gain by observing S. For example,
$$VPI(F \mid S) = 3.15 - 2 = 1.25$$

Which means should be willing be pay anything up to 1.25 utility points in order to conduct the survey because doing this will increase his expected utility.<br><br>

<b>Theorem</b><br>
$\bigstar$ VPI(A | X) $\geq$ 0<br>
$\bigstar$ VPI(A | X) = 0 if and only if the optimal decision rule for D is still optimal $D_{X \rightarrow A}$<br><br>

For the first property, MEU is obtained by optimizing over the decision rule. For $MEU(D)$, the decision rule is a CPD $\delta_{A \mid Z}$, where Z is the set parents A in original diagram, while for $MEU(D_{X \rightarrow A})$, the decision rule is $\delta(A \mid Z, X)$, Z is original parents of A and X is additional parent of X.
$\delta(A \mid Z, X) > \delta_{A \mid Z}$

Which means any CPD $\delta(A \mid Z)$ is also a CPD $\delta(A \mid Z, X)$. In other word, any decision rule that we implement in our original diagram can be also implemented in our current diagram. For example, if the agent decides to found thsi company regardless of the value of Survey, then this is also a legitimate decision when it observes the Survey.<br><br>

For the second property, VPI(A | X) is 0 means an agent cannot gain any additional expected utility after observing X.<br><br>

Say we discover a magical lantern and upon rubbing it, a genie tells us that $\delta(A \mid \overline{Z}, X) \neq \delta(A \mid \overline{Z})$. <b>What does this allow us to conclude about the value of knowing X?</b><br>
A. It is equal to 0.<br>
B. We cannot tell anything because it depends on the magnitude of the difference.<br>
C. It is positive.<br>
D. It is negative.<br>
<b>Answer:</b> C.<br>
The value of knowing X > 0 because $\delta(A \mid \overline{Z})$ is not optimal for $D_{X \rightarrow A}$.<br><br>

<b>Value of Information Example</b><br>
Consider two companies, we are considering to join one of them. We have two state variables about management, say $s^{1}$ is bad management. The Funding variable depends on the state with a CPD. Our utility function is V, 1 means we choose to join and 0 means no.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_3_2.png"/></center>
</p>
<p align="justify">
Based on figure (a), For company 1
$$EU(D(c_{1})) = \sum_{S_{1}, F_{1}, c_{1}}P(S_{1})P(F_{1} \mid S_{1})U(F_{1}, c_{1}) = 0.1 \cdot 0.1 + 0.2 \cdot 0.4 + 0.7 \cdot 0.9 = 0.72$$

For company 2
$$EU(D(c_{2})) = \sum_{S_{2}, F_{2}, c_{2}}P(S_{2})P(F_{2} \mid S_{2})U(F_{2}, c_{2}) = 0.4 \cdot 0.1 + 0.5 \cdot 0.4 + 0.1 \cdot 0.9 = 0.33$$

If the agent observe $S_{2}$ before making decision like (b), the optimal decision rule
$$
\delta^{*}(C \mid S_{2}) =
\begin{cases}
  P(c_{2}) = 1, \quad S_{2} = s^{3} \\
  P(c_{1}) = 1, \quad otherwise
\end{cases}
$$

In other word, the expected utility when the agent chooses company 2 and $S_{2} = s^{1}$ is 0.1; EU = 0.4 when the agent chooses company 2 and $S_{2} = s^{2}$. Both the two conditions have a lower expected utility compared to our original diagram, so the agent still prefers to the original choice, namely company 1. But EU = 0.9 when the agent chooses company 2 and $S_{2} = s^{3}$, in this case, the agent prefers company 2.<br><br>

You just saw that the optimal decision only changes our mind if the second company is doing very well ($s^{3}$). Recall that this only happens with prior probability 0.1. <b>Do you think VPI will be high or low in this case?</b><br>
A. Low<br>
B. HIgh<br>
C. Low when $S_{2} = s^{3}$<br>
D. HIgh when $S_{2} = s^{3}$<br>
<b>Answer:</b> A.<br>
The basic idea is that the gain in utility in this case has to be weighted by the probability of the state of the world that brings about that gain (i.e. the second company is doing very well). Because the probability of $s^{3}$ is not very large we should not expect a very large gain in expected utility so the value of the information will likely be small.<br><br>

Under the optimal decision rule, maximum expected utility
$$MEU(D_{S_{2} \rightarrow C}) = 0.738$$
</p>
{% highlight Python %}
S1 = {1: 0.1, 2: 0.2, 3: 0.7}
S2 = {1: 0.4, 2: 0.5, 3: 0.1}
F = {0: {1: 0.9, 2: 0.6, 3: 0.1},
     1: {1: 0.1, 2: 0.4, 3: 0.9}}
C = {2: 1, 3: 2, 1: 1}
V = {(0, 0, 1): 0, (0, 1, 1): 0, (1, 0, 1): 1, (1, 1, 1): 1,
     (0, 0, 2): 0, (0, 1, 2): 1, (1, 0, 2): 0, (1, 1, 2): 1}

def MEU():
    meu = 0
    for s1 in S1.keys():
        for f1 in F.keys():
            for s2 in S2.keys():
                for f2 in F.keys():
                    c = C[s2]
                    meu += S1[s1]*F[f1][s1]*S2[s2]*F[f2][s2]*V[(f1, f2, c)]
    return meu

MEU()
{% endhighlight %}
<p align="justify">
Which means that the agent shouldn't be willing to pay his company too much money in order to get information about the detail.<br><br>

If we change the distribution of $S_{1}$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_3_3.png"/></center>
</p>
<p align="justify">
In the original diagram
$$EU(D(c_{1})) = 0.35$$
$$EU(D(c_{2})) = 0.33$$

In our new diagram
$$MEU(D_{S_{2} \rightarrow C}) = 0.43$$

It goes up to 0.43, which is a much more significant increase in their expected utility relative to what we had before. Because now there is more value to the information.<br><br>

<b>Summary</b><br>
$\bigstar$ Influence diagrams provide clear and coherent semantics for the value of making an observation<br>
-- Difference between values of two IDs<br>
$\bigstar$ Information is valuable if and only if it induces a change in action in at least one context<br><br>
</p>

#### 1.6.4 Quiz
<p align="justify">
<b>1. Utility Curves</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_4_1.png"/></center>
</p>
<p align="justify">
<b>1.1</b><br>
<b>What does the point marked A on the Y axis correspond to? (Mark all that apply)</b><br>
A. U(l) where l is a lottery that pays 0 dollar with probability 0.5 and 1000 dollar with probability 0.5.<br>
B. 500 dollar<br>
C. 0.5 U(0 dollar) + 0.5 U(1000 dollar)<br>
D. U(500 dollar)<br>
<b>Answer:</b> A, C.<br><br>

<b>1.2</b><br>
<b>What does the point marked B on the Y axis correspond to? (Mark all that apply)</b><br>
A. U(l) where l is a lottery that pays 0 dollar with probability 0.5 and 1000 dollar with probability 0.5.<br>
B. 500 dollar<br>
C. 0.5 U(0 dollar) + 0.5 U(1000 dollar)<br>
D. U(500 dollar)<br>
<b>Answer:</b> B.<br><br>

<b>2. *Uninformative Variables</b><br>
Here is an influence diagram
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_6_4_2.png"/></center>
</p>
<p align="justify">
<b>What is an appropriate way to have the model account for the fact that if the Test wasn’t performed ($t^{0}$), then the survey is uninformative?</b><br>
A. Set $P(S \mid M, t^{0})$ to be uniform<br>
B. Set $P(S \mid M, t^{0}) = P(S \mid M, t^{1})$<br>
C. Set $P(S \mid M, t^{0})$ so that S takes some new value "not performed" with probability of 1<br>
D. Set $P(S \mid M, t^{0})$ so that S takes the value $s^{0}$ with probability of 1<br>
<b>Answer:</b> C.<br>
A implies that all survey outcomes are equally likely, but that is not the same as saying that the survey was not carried out.<br>
B means the test is always performed, since the distribution of results when the test is not performed is the same as the distribution when the test is performed.<br>
C s the appropriate action. Assigning S to any other value would not be desirable, as these other values may represent survey results, but we have not actually conducted the survey.<br>
D is not desirable, since may correspond to a value of "Good Market", for example, and not that the survey was not performed.<br><br>
</p>

### 1.7 Summary
#### 1.7.1 Knowledge Engineering
<p align="justify">
<b>Important Distinctions</b><br>
$\bigstar$ Template based versus specific<br>
$\bigstar$ Directed versus undirected<br>
$\bigstar$ Generative versus discriminative<br>
$\bigstar$ Hybrids are also common<br><br>

Medical diagnosis is usually a specific model. We have a particular set of symptoms diseases and so on that we want to encode in our model. On the other side on the template based side we have things like image segmentation. There are all sorts of applications that sit in between those two, for example, fault diagnosis.<br><br>

Template based models usually have a fairly small number of variable types. So, for example, in our image segmentation setting, we have the class label, that is one variable type. Nevertheless, we manage to construct very richly expressive models about this, because of interesting interactions between multiple class labels for adjacent, for different pixels in the image. But it's a very small number of variable types, and most of the effort goes into figuring out things like which features are most predictive. On the spcific model side we have usually a large number because unless we build small models, each variables going to be unique, so a large number, of unique variables.<br><br>

On the discriminatave side we have a particular task in mind. A particular prediction task is often better solved by having richly expressive features, richly discriminative features, and then modeling this as a discriminative model allows me to avoid dealing with correlations, so that it gives me usually a high performance model. When would I use a generative model? One answer is when I don't have a predetermined task. So for example, when I have a medical diagnosis pack, every patient presents differently. In each patients case I have a different subset of things that I happen to know about that patient such as the symptoms that they present with, and the tests that I happened to perform. And so, I don't want to train a discriminative model that uses a predetermined set of variables as inputs and a predetermined set of diseases as outputs. Rather, I want something that gives me flexibility to measure different variables and predict others. The second reason for using a generative model is that generative models are easier to train in certain regimes. for example, in the case where the data is not fully labeled.<br><br>

<b>For which of the following applications might a generative model be a better choice than a discriminative model?</b><br>
A. Making fast predictions of Y given X where the distribution of X is not important.<br>
B. Medical diagnosis where obtaining ground-truth labels for the data is expensive.<br>
C. Image segmentation where the extracted features are high-dimensional.<br>
D. Modeling different music composers for classification of music and creating new music pieces.<br>
<b>Answer:</b> B, D.<br><br>

<b>Variable Types</b><br>
$\bigstar$ Target<br>
$\bigstar$ Observed<br>
-- Including complex, constructed features<br>
$\bigstar$ Latent<br>
-- Hidden<br><br>

Sometimes latent variables can simplify our structure. For example, imagine that I asked all of you in this class, what time does your watch show?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_7_1_1.png"/></center>
</p>
<p align="justify">
So $W_{i}$ is the time on the watch of each of you in the class. Now, these variables are all correlated with each other. But in fact, they're not correlated with each other. Unless we all had a watch setting party just before class. Really, what they're all correlated with is Greenwich mean time. So we have a model, in this case it's a naive base model, where we have Greenwich Mean Time influencing a bunch of random variables that are conditionally independent given that. Now Greenwich Mean Time is latent unless we actually end up calling Greenwich to find out what the current time is right now in Greenwich, which I don't think any of us really care about. But why would we want to include Greenwich Mean Time in our model? Because if we don't include Greenwich Mean Time, so if we basically eliminate Greenwich Mean Time from our model, what happens to the dependency structure, of our model? We end up with a complicated model.<br><br>

<b>Structure</b><br>
$\bigstar$ Causal versus non-causal ordering<br><br>

When we think about Bayesian networks specifically, do the arrows correspond to causality? That is, is an arrow from x to y indicative of having a causal connection from x to y?
$$X \rightarrow Y \quad X \leftarrow Y$$

So, the answer to that is yes and no. so what does no mean in this case? We have X pointing to Y, where X is a parent of Y. In the Bayes Net I can equally invert that edge with Y pointing to X. So, in this example, we can reverse the edges and have a model that's equally expressive. But that model might be very nasty. For example, $X_{1}$ and $X_{2} are both parents of Y.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_7_1_2.png"/></center>
</p>
<p align="justify">
If we want to invert the directionality of the edges and put Y as a parent of say $X_{2}$, this is not in our case. So causal directionality is often simpler.

So let's go back to our Greenwich mean time example.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_7_1_3.png"/></center>
</p>
<p align="justify">
Where we have the Greenwich mean time is in some way the, the cause or the parent of the different watch, times that we see our in different individuals. And now I'm going to force Grenwich mean time to be the child of all these. Is this the correct model? No, because this says that all of the watch times are independent which is not the case. And so, what we're going to end up with as the model is the same horrific model that I showed before where everything is connected to everything else. Although causal ordering is not more correct than a non-causal ordering, it's sparser, more intuitive and easier to parameterize.<br><br>

<b>Parameters: Values</b><br>
$\bigstar$ What matters:<br>
-- Zeros<br>
-- Order of magnitude<br>
-- Relative values<br>
$\bigstar$ Structured CPDs<br><br>

<b>Parameters: Local Structure</b><br>
$\bigstar$ Table CPDs are the exception<br>
<table class="c">
  <tr><th></th><th>Context-specific</th><th>Aggregating</th></tr>
  <tr><td>Discrete</td><td>Tree CPDs</td><td>Sigmoid, Noisy max</td></tr>
  <tr><td>Continuous</td><td>Regression tree (threshold)</td><td>Linear Gaussian</td></tr>
</table><br>
</p>
<p align="justify">
Assume that you are indoors and you can't see the weather outside. <b>When determining the weather based only on observations of the attire of people entering the building from outside, which of the following modeling decisions makes sense?</b><br>
A. Putting a larger weight on features related to the types of clothing that people are wearing (raincoat vs. t-shirt vs. winter coat) than on features related to the color of clothing that people are wearing.<br>
B. Making the naïve Bayes assumption that the attire of different people independent given the weather.<br>
C. Using a linear Gaussian if the weather states are discrete.<br>
D. Making the attire of people observed variables and the weather states target variables.<br>
<b>Answer:</b> A, D.<br><br>

<b>Iterative Refinement</b><br>
$\bigstar$ Model testing<br>
$\bigstar$ Sensitivity analysis for parameters<br>
$\bigstar$ Error analysis<br>
-- Add features<br>
-- Add dependencies<br><br>
</p>

#### 1.7.2 Quiz
<p align="justify">
<b>1. Template Model Representation</b><br>
Consider the following scenario:<br>
On each campus there are several Computer Science students and several Psychology students (each student belongs to one xor the other group). We have a binary variable L for whether the campus is large, a binary variable S for whether the CS student is shy, a binary variable C for whether the Psychology student likes computers, and a binary variable F for whether the Computer Science student is friends with the Psychology student. <b>Which of the following plate models can represent this scenario?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_7_2_1.png"/></center>
</p>
<p align="justify">
A. (A)<br>
B. (B)<br>
C. (C)<br>
D. (D)<br>
E. None of these plate models can represent this scenario<br>
<b>Answer:</b> A.<br><br>

<b>2. Partition Function</b><br>
<b>Which of the following is a use of the partition function?</b><br>
A. The partition function is used only in the context of Bayesian networks, not Markov networks.<br>
B. One can subtract the partition function from factor products in order to convert them into probabilities.<br>
C. One can divide factor products by the partition function in order to convert them into probabilities.<br>
D. The partition function is useless and should be ignored<br>
<b>Answer:</b> C.<br><br>

<b>3. *I-Equivalence</b><br>
Let T be any directed tree (not a polytree) over n nodes, where n≥1. A directed tree is a traditional tree, where each node has at most one parent and there is only one root, i.e., all but one node has exactly one parent. (In a polytree, nodes may have multiple parents.) <b>How many networks (including itself) are I-equivalent to T?</b><br>
A. n<br>
B. 1<br>
C. n+1<br>
D. n!<br>
E. Dependes on the specific structure of T<br>
<b>Answer:</b> A.<br><br>

<b>4. *Markov Network Construction</b><br>
Consider the unrolled network for the plate model shown below, where we have n students and m courses. Assume that we have observed the grade of all students in all courses. <b>In general, what does a pairwise Markov network that is a minimal I-map for the conditional distribution look like?</b> (Hint: the factors in the network are the CPDs reduced by the observed grades. We are interested in modeling the conditional distribution, so we do not need to explicitly include the Grade variables in this new network. Instead, we model their effect by appropriately choosing the factor values in the new network.)
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_7_2_2.png"/></center>
</p>
<p align="justify">
A. A fully connected graph with instantiations of the Difficulty and Intelligence variables.<br>
B. Impossible to tell without more information on the exact grades observed.<br>
C. A fully connected bipartite graph where instantiations of the Difficulty variables are on one side and instantiations of the Intelligence variables are on the other side.<br>
D. A graph over instantiations of the Difficulty variables and instantiations of the Intelligence variables, not necessarily bipartite; there could be edges between different Difficulty variables, and there could also be edges between different Intelligence variables.<br>
E. A bipartite graph where instantiations of the Difficulty variables are on one side and instantiations of the Intelligence variables are on the other side. In general, this graph will not be fully connected.<br>
<b>Answer:</b> C.<br><br>

<b>5. Grounded Plates.</b><br>
Which of the following is a valid grounded model for the plate shown? You may select 1 or more options.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_7_2_3.png"/></center>
</p>
<p align="justify">
A. (a)<br>
B. (b)<br>
C. (c)<br>
<b>Answer:</b> C.<br><br>

<b>6. Independencies in Markov Networks</b><br>
Consider the following set of factors: $\Phi = \{\Phi_1(A, B), \Phi_2(B, C, D), \Phi_3(D), \Phi_4(C, E, F)\}$. Now, consider a Markov Network G such that $P_\Phi$ factorizes over G. <b>Which of the following is an independence statement that holds in the network? You may select 1 or more options.</b><br>
A. $(C \perp D \mid A)$<br>
B. $(B \perp E \mid C)$<br>
C. $(B \perp E \mid A)$<br>
D. $(C \perp E \mid B)$<br>
E. $(A \perp E \mid B)$<br>
F. $(A \perp F \mid C)$<br>
<b>Answer:</b> B, E, F.<br><br>

<b>7. Factorization of Probability Distributions</b><br>
Consider a directed graph G. We construct a new graph G' by removing one edge from G. <b>Which of the following is always true? You may select 1 or more options.</b><br>
A. If G and G' were undirected graphs, the answers to the other options would not change.<br>
B. Any probability distribution P that factorizes over G also factorizes over G'.<br>
C. Any probability distribution P that factorizes over G' also factorizes over G.<br>
D. No probability distribution P that factorizes over G also factorizes over G'<br>
<b>Answer:</b> A, C.<br><br>

<b>8. Template Model in CRF</b><br>
The CRF model for OCR with only singleton and pairwise potentials that you played around with in PA3 and PA7 is an instance of a template model, with variables $C_{1}, \cdots, C_{n}$ over the characters and observed images $I_{1}, \cdots, I_{n}$. The model we used is a template model in that the singleton potentials are replicated across different $C_{i}$ variables, and the pairwise potentials are replicated across character pairs. The structure of the model is shown below:
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/1_7_2_4.png"/></center>
</p>
<p align="justify">
Now consider the advantages of this particular template model for the OCR task, as compared to a non-template model that has the same structure, but where there are distinct singleton potentials for each $C_{i}$ variable, and distinct potentials for each pair of characters. <b>Which of the following about the advantage of using a template model is true? You may select 1 or more options.</b><br>
A. The template model can incorporate position-specific features, e.g. q-u occurs more frequently at the beginning of a word, while a non-template model cannot.<br>
B. Parameter sharing could make the model less susceptible to over-fitting when there is less training data.<br>
C. The same template model can be used for words of different lengths.<br>
D. The inference is significantly faster with the template model.<br>
<b>Answer:</b> B, C.<br><br>
</p>


## 2. Inference
### 2.1 Inference Overview
#### 2.1.1 Conditional Probability Queries
<p align="justify">
<b>Conditional Probability Queries</b><br>
$\bigstar$ Evidence: E = e<br>
$\bigstar$ Query: a subset of variables Y<br>
$\bigstar$ Task: compute P(Y | E = e)<br>
$\bigstar$ Application:<br>
-- Medical/fault diagnosis<br>
-- Pedigree analysis<br><br>

<b>NP-Hardness</b><br>
The following are all NP-Hard:<br>
$\bigstar$ Given a PGM $P_{\phi}$, a variable X and a value $x \in Val(X)$, <b>compute $P_{\phi}(X=x)$</b><br>
-- even decide if $P_{\phi}(X=x) > 0$<br>
$\bigstar$ Let $\epsilon < 0.5$. Given a PGM $P_{\phi}$, a variable X and a value $x \in Val(X)$, and observation $e \in Val(E)$, find a number p that has
$$\left | P_{\phi}(X=x \mid E=e) - p \right | < \epsilon$$

<b>How many entries are in a table describing the full joint distribution, where each of the n variables has k possible values?</b><br><br>

<b>Answer:</b> $k^{n}$.<br><br>

<b>Sum-Product</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_1_1_1.png"/></center>
</p>
<p align="justify">
For Bayesian network
$$\sum_{C, D, I, G, S, L, J, H} \phi_{C}(C) \cdot \phi_{D}(C, D) \cdot \phi_{I}(I) \cdot \phi_{G}(G, I, D) \cdot \phi_{S}(S, I) \cdot \phi_{L}(L, G) \cdot \phi_{J}(J, L, S) \cdot \phi_{H}(H, G, J)$$

For Markov network
$$\sum_{A, B, C, D} \phi_{1}(A, B) \cdot \phi_{2}(B, C) \cdot \phi_{3}(C, D) \cdot \phi_{4}(A, D)$$

<b>Evidence: Reduced Factors</b><br>
$$P(Y \mid E=e) = \frac{P(Y, E=e)}{P(E=e)}$$

Consider <b>W</b> denotes all variabless except the query Y and evidence E
$$W = \{ X_{1}, X_{2}, \cdots, X_{n} \} - Y - E$$

$\bigstar$ For the numerator
$$P(Y, E=e) = \sum_{W} P(Y, W, E=e) = \sum_{W} \frac{1}{Z} \prod_{k} \phi_{k}(D_{k}, E=e) = \sum_{W} \frac{1}{Z} \prod_{k} \phi'_{k}(D'_{k})$$

So, we can reduce the factors by the evidence. For example, if we observe A = $a^{0}$, we can eliminate A = $a^{1}$ while computing the distribution above because A = $a^{1}$ is not consistent with our evidence.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_1_1_2.png"/></center>
</p>
<p align="justify">
$\bigstar$ For the denominator
$$P(E=e) = \sum_{Y}\sum_{W} \frac{1}{Z} \prod_{k}\phi'_{k}(D'_{k})$$

There is something in common for P(Y, E=e) and P(E=e)
$$\sum_{W} \prod_{k} \phi'_{k}(D'_{k})$$

Then we need to normalize it.<br><br>

<b>Algorithms: Conditional Probability</b><br>
$\bigstar$ Push summations into factor product<br>
-- Variable elimination<br>
$\bigstar$ Message passing over a graph<br>
-- Belief propagation<br>
-- Variational approximations<br>
$\bigstar$ Random sampling instantiations<br>
-- Markov chain Monte Carlo (MCMC)<br>
-- Importance sampling<br><br>

<b>Why is the expression $\sum_{\bar{W}}P(\bar{Y},\bar{W},\bar{e})$ hard to compute in general?</b><br>
A. It may be hard to compute $P(\bar{Y},\bar{W},\bar{e})$.<br>
B. It may be intractable to sum over all the different values that $\bar{W}$ can take.<br>
C. $P(\bar{Y},\bar{W},\bar{e})$ may be zero for some evidence $\bar{e}$.<br>
D. It isn't hard. It's always tractable.<br><br>

<b>Answer:</b> B.<br>
The summation over all values of $\bar{W}$ is exponential. If $\bar{W}$ has 100 binary variables, then summing will take $2^{100}$ operations. $P(\bar{Y},\bar{W},\bar{e})$ is always easy to compute because it is just the product of all CPDs.<br><br>
</p>

#### 2.1.2 MAP Inference
<p align="justify">
<b>Maximum a Posterior (MAP)</b><br>
$\bigstar$ Evidence: E = e<br>
$\bigstar$ Query: all other variables except evidence $Y = \{ X_{1}, \cdots, X_{n} \} - E$<br>
$\bigstar$ Task: compute $MAP(Y \mid E = e) = arg\max_{y} P(Y = y \mid E = e)$<br>
-- Note: there may be more than one possible solution<br>
$\bigstar$ Applications:<br>
-- Message decoding: most likely transmitted message<br>
-- Image segmentation: most likely segmentation<br><br>

<b>MAP $\neq$ Max over Marginal</b><br>
For example,
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_1_2_1.png"/></center>
</p>
<p align="justify">
We can calculate its Joint distribution
$$P(a^{0}, b^{0}) = 0.04$$
$$P(a^{0}, b^{1}) = 0.36$$
$$P(a^{1}, b^{0}) = 0.3$$
$$P(a^{1}, b^{1}) = 0.3$$

So, MAP is A = $a^{0}$ and B = $b^{1}$. But, if we focus on the marginal distribution, say, P(A), we will get A = $a^{1}$ makes P(A) biggest.<br><br>

<b>what is $\mathrm{argmax}_{a} P(A=a \mid B=0)$?</b> <b>How about $\mathrm{argmax}_{a} P(A=a \mid B=1)$?</b><br>
<b>Answer:</b> 1 and 0.<br><br>

<b>NP-Hardness</b><br>
The following are NP-hard<br>
$\bigstar$ Given a PGM $P_{\phi}$, find a joint assignment x with highest probability $P_{\phi}(x)$<br>
$\bigstar$ Given a PGM $P_{\phi}$ and a probability p, decide if there is an assignment x such that $P_{\phi}(x) > p$<br><br>

<b>Max Product</b><br>
Similar to Sum Product, we have a Max Product
$$arg\max_{C, D, I, G, S, L, J, H} \phi_{C}(C) \cdot \phi_{D}(C, D) \cdot \phi_{I}(I) \cdot \phi_{G}(G, I, D) \cdot \phi_{S}(S, I) \cdot \phi_{L}(L, G) \cdot \phi_{J}(J, L, S) \cdot \phi_{H}(H, G, J)$$

Recall $Y = \{ X_{1}, \cdots, X_{n} \} - E$
$$P(Y \mid E=e) = \frac{P(Y, E=e)}{P(E=e)} \propto P(Y, E=e)$$

Because P(E = e) is constant with respect to Y. Then,
$$P(Y, E=e) = \frac{1}{Z} \prod_{k} \phi'_{k}(D'_{k}) \propto \prod_{k} \phi'_{k}(D'_{k})$$

Where $D'_{k}$ is a reduced factor relative to evidence and Z is partition function. Finally, we deduce the MAP
$$arg\max_{Y} P(Y \mid E=e) = arg\max_{Y} \prod_{k} \phi'_{k}(D'_{k})$$

<b>Algorithms MAP</b><br>
$\bigstar$ Push maximization into factor product<br>
-- Variable elimination<br>
$\bigstar$ Message passing over a graph<br>
-- Max-product belief propagation<br>
$\bigstar$ Using methods for integer programming<br>
$\bigstar$ For some networks: graph-cut methods<br>
$\bigstar$ Combinatorial search<br><br>

<b>Summary</b><br>
$\bigstar$ MAP: single coherent assignment of highest probability<br>
-- Not the same as maximizing individual marginal probabilities<br>
$\bigstar$ Maxing over factor product<br>
$\bigstar$ Combinatorial optimization problem<br>
$\bigstar$ Many exact and approximate algorithms<br><br>
</p>

### 2.2 Variable Elimination
#### 2.2.1 Variable Elimination Algorithm
<p align="justify">
<b>Elimination in Chains</b><br>
Consider a graph with a joint distribution
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_1_1.png"/></center>
</p>
<p align="justify">
We want to get P(E), so
$$P(E) \propto \sum_{D}\sum_{C}\sum_{B}\sum_{A} \widetilde{P}(A, B, C, D, E)$$
$$= \sum_{D}\sum_{C}\sum_{B}\sum_{A} \phi_{1}(A, B)\phi_{2}(B, C)\phi_{3}(C, D)\phi_{4}(D, E)$$
$$= \sum_{D}\sum_{C}\sum_{B} \phi_{2}(B, C)\phi_{3}(C, D)\phi_{4}(D, E) \sum_{A} \phi_{1}(A, B)$$

We notice that $\sum_{A} \phi_{1}(A, B)$ ia a function of B, so we call that $\tau_{1}(B)$
$$\tau_{1}(B) = \sum_{A} \phi_{1}(A, B)$$
$$P(E) \propto \sum_{D}\sum_{C}\sum_{B} \phi_{2}(B, C)\phi_{3}(C, D)\phi_{4}(D, E) \tau_{1}(B)$$
$$= \sum_{D}\sum_{C} \phi_{3}(C, D)\phi_{4}(D, E) \sum_{B} \phi_{2}(B, C)\tau_{1}(B)$$

Similarly, $\sum_{B} \phi_{2}(B, C)\tau_{1}(B)$ is a function of $\tau_{2}(C)$
$$P(E) \propto \sum_{D}\sum_{C} \phi_{3}(C, D)\phi_{4}(D, E) \tau_{2}(C)$$
$$= \sum_{D} \phi_{4}(D, E) \sum_{C} \phi_{3}(C, D)\tau_{2}(C)$$

We can continue such a process until a solution<br><br>

For another example, we want to compute P(J), so we have to eliminate all other variables C, D, I, H, G, S, L
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_1_2.png"/></center>
</p>
<p align="justify">
$$P(J) = \sum_{C, D, I, G, S, L, H} \phi_{J}(J, L, S)\phi_{L}(L, G)\phi_{S}(S, I)\phi_{G}(G, I, D)\phi_{H}(H, G, J)\phi_{I}(I)\phi_{D}(C, D)\phi_{C}(C)$$
$$= \sum_{D, I, G, S, L, H} \phi_{J}(J, L, S)\phi_{L}(L, G)\phi_{S}(S, I)\phi_{G}(G, I, D)\phi_{H}(H, G, J)\phi_{I}(I) \sum_{C} \phi_{D}(C, D)\phi_{C}(C)$$
$$\tau_{1}(D) = \sum_{C} \phi_{D}(C, D)\phi_{C}(C)$$
$$P(J) = \sum_{D, I, G, S, L, H} \phi_{J}(J, L, S)\phi_{L}(L, G)\phi_{S}(S, I)\phi_{G}(G, I, D)\phi_{H}(H, G, J)\phi_{I}(I) \tau_{1}(D)$$
$$= \sum_{I, G, S, L, H} \phi_{J}(J, L, S)\phi_{L}(L, G)\phi_{S}(S, I)\phi_{H}(H, G, J)\phi_{I}(I) \sum_{D} \phi_{G}(G, I, D)\tau_{1}(D)$$
$$\tau_{2}(G, I) = \sum_{D} \phi_{G}(G, I, D)\tau_{1}(D)$$
$$P(J) = \sum_{I, G, S, L, H} \phi_{J}(J, L, S)\phi_{L}(L, G)\phi_{S}(S, I)\phi_{H}(H, G, J)\phi_{I}(I) \tau_{2}(G, I)$$
$$= \sum_{G, S, L, H} \phi_{J}(J, L, S)\phi_{L}(L, G)\phi_{H}(H, G, J) \sum_{I} \phi_{I}(I)\phi_{S}(S, I)\tau_{2}(G, I)$$
$$\tau_{3}(S, G) = \sum_{I} \phi_{I}(I)\phi_{S}(S, I)\tau_{2}(G, I)$$
$$P(J) = \sum_{G, S, L, H} \phi_{J}(J, L, S)\phi_{L}(L, G)\phi_{H}(H, G, J) \tau_{3}(G, S)$$
$$= \sum_{G, S, L} \phi_{J}(J, L, S)\phi_{L}(L, G)\tau_{3}(G, S) \sum_{H} \phi_{H}(H, G, J)$$
$$\tau_{4}(G, J) = \sum_{H} \phi_{H}(H, G, J) = \sum_{H} P(H \mid G, J) = 1$$
$$P(J) = \sum_{G, S, L} \phi_{J}(J, L, S)\phi_{L}(L, G)\tau_{3}(G, S) \tau_{4}(G, J)$$
$$= \sum_{S, L} \phi_{J}(J, L, S) \sum_{G} \phi_{L}(L, G)\tau_{3}(G, S) \tau_{4}(G, J)$$
$$\tau_{5}(L, S, J) = \sum_{G} \phi_{L}(L, G)\tau_{3}(G, S) \tau_{4}(G, J)$$
$$P(J) = \sum_{S, L} \phi_{J}(J, L, S) \tau_{5}(L, S, J)$$
$$\tau_{6}(J) = \sum_{S, L} \phi_{J}(J, L, S) \tau_{5}(L, S, J)$$

<b>Variable Elimination with evidence</b><br>
Under a same graph, we want to comput P(J, I=i, H=h), so we have to eliminate C, D, G, S, L because variable I and H have a specific value.<br><br>

How to get $P(J \mid I=i, H=h)$?
$$P(J \mid I=i, H=h) = \frac{P(J, I=i, H=h)}{P(I=i, H=h)}$$

<b>Which variables do we sum over when performing Variable Elimination with evidence?</b><br>
<b>Answer:</b> All except the evidence variables.<br>
Imagine we are trying to compute
$P(\bar{Y} \mid \bar{e}) = \frac{P(\bar{Y},\bar{e})}{P(\bar{e})}$

where $\bar{Y}$ are the query variables and $\bar{E}$ are the evidence. Computing the numerator involves summing out everything except for the query and evidence variables, and computing the denominator requires further summing over the query variables so that the entire equation sums to 1.<br><br>

<b>Eliminate Variable Z from $\Phi$</b><br>
$$\Phi' = \{ \phi_{i} \in \Phi: Z \in Scope[\phi_{i}] \}$$
$$\psi = \prod_{\phi_{i} \in \Phi'} \phi_{i}$$
$$\tau = \sum_{Z} \psi$$
$$\Phi := \Phi - \Phi' \cup \{ \tau \}$$

<b>VE Algorithm</b><br>
$\bigstar$ Reduce all factors by evidence<br>
-- Get a set of factors $\Phi$<br>
$\bigstar$ For each non-query variable Z<br>
-- Eliminate Var Z from $\Phi$<br>
$\bigstar$ Multiply all remaining factors<br>
$\bigstar$ Renormalizing to get distribution<br><br>

<b>Summary</b><br>
$\bigstar$ Simple algorithm<br>
$\bigstar$ Works for both BNs and MNs<br>
$\bigstar$ Factors product ans summation steps can be done in any order, subject to:<br>
-- when Z is eliminated, all factors involving Z have been multiplied<br><br>
</p>

#### 2.2.2 Complexity of Variable Elimination
<p align="justify">
<b>Eliminating Z</b><br>
There are two steps for eliminating Z: factor product and marginalization.
$$\psi_{k}(X_{k}) = \prod_{i=1}^{m_{k}} \phi_{i}$$
$$\tau_{k}(X_{k} - \{ Z \}) = \sum_{Z} \psi_{k}(X_{k})$$

$X_{k}$ denotes a set of factors which are relevant to Z<br><br>

How many times does each entry in $\psi_i$ get added to one of the $\tau_{i}$ entries?<br><br>

<b>Answer:</b> At most once.<br>
We are computing a single sum over some entries in $\psi$, so each entry is involved in the sum at most once.<br><br>

For factor product, imagine we have two factors
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_2_1.png"/></center>
</p>
<p align="justify">
We define the number of lines for the new table by
$$N_{k} = \left | Val(X_{k}) \right |$$

In fact, the new table's cardinality = a product of all variables' cardinality. So factor product's cost: $(m_{k}-1)N_{k}$ multiplications<br><br>

For factor marginalization
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_2_2.png"/></center>
</p>
<p align="justify">
Cost: ~ $N_{k}$ additions<br><br>

<b>Complexity of Variable Elimination</b><br>
$\bigstar$ Start with m factors<br>
-- m $\leq$ n for Bayesian networks<br>
-- can be larger for Markov networks<br>
$\bigstar$ At each elimination step generate 1 factor (a new factor)<br>
$\bigstar$ At most n elimination steps<br>
$\bigstar$ Total number of factors: $m^{\star} = m + n$<br>
$\bigstar$ N = max($N_{k}$) denotes the size of the largest factor<br>
$\bigstar$ Product operations: $\sum_{k}(m_{k}-1)N_{k} \leq N\sum_{k}(m_{k}-1) \leq Nm^{\star}$<br>
$\bigstar$ Sum operations: $\sum_{k}N_{k} \leq$ N $\cdot$ number of elimination steps $\leq N \cdot n$<br>
$\bigstar$ Total work is linear in N + $m^{\star}$<br>
$\bigstar$ $N_{k} = \left | Val(X_{k}) \right | = O(d^{r_{k}})$ where<br>
-- d = max($\left | Val(X_{i}) \right |$) denotes d values in their scope e.g. d = 2 for binary variables<br>
-- $r_{k} = \left | Val(X_{i}) \right |$ denotes the cardinality of the scope of the $k^{th}$ factor, in other word, $r_{k}$ variables in the $k^{th}$ factors<br><br>

For example
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_2_3.png"/></center>
</p>
<p align="justify">
We can find $\tau_{5}$ has the most variables.<br><br>

<b>Complexity and Elimination Order</b><br>
Which factors do we need to multiply in to eliminate G?
$$\sum_{G} \phi_L(L, G)\phi_G(G, I, D)\phi_H(H, G, J)$$

So, in this step, the scope of factor is 6 variables: L, G, I, D, H, J.<br><br>

For another example
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_2_4.png"/></center>
</p>
<p align="justify">
If we eliminate A first, the scope of variable is $\{ A, B_{1}, \cdots, B_{k} \}$, says k+1. But, if we eliminate $B_{1}$ first, the scope is $\{ A, B_{1}, C \}$.<br><br>

<b>Summary</b><br>
$\bigstar$ Complexity of variable elimination linear in<br>
-- size of the model (number of factors, number of variables)<br>
-- size of the largest factor generated<br>
$\bigstar$ Size of factor is exponential in its scope<br>
$\bigstar$ Complexity of algorithm depends on heavily on elimination order<br><br>
</p>

#### 2.2.3 Graph-Based Perspective on Variable Elimination
<p align="justify">
Consider an initial network, we deduce a set of factors
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_3_1.png"/></center>
</p>
<p align="justify">
$$\phi_{J}(J, L, S)\phi_{L}(L, G)\phi_{S}(S, I)\phi_{G}(G, I, D)\phi_{H}(H, G, J)\phi_{I}(I)\phi_{D}(C, D)\phi_{C}(C)$$

A structure of the graph that corresponds to the set of factors is like right one. We notice that, for any v-structure, we have to add an edge after turning all edges undirected. This is called moralization.<br><br>

For example, <b>How many edges should be added to the graph to make it moralized?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_3_2.png"/></center>
</p>
<p align="justify">
<b>Answer:</b> 2.<br>
There are two v-structures in the graph, one at A and the other at D. So we have to moralize this graph by adding two edges.<br><br>

<b>Elimination as Graph Operations</b><br>
We eliminate all variables except J in order
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_3_3.png"/></center>
</p>
<p align="justify">
After eliminating I, we introduce a new edge between G and S, because S and G are in a same factor $\tau_{3}(S, G)$. <b>All variables that connect to I are connected after I is eliminated.</b> The new edge is called <b>fill edge</b>.<br><br>

<b>Induced Graph</b><br>
$\bigstar$ The induced graph $I_{\phi, \alpha}$ over factors $\phi$ and ordering $\alpha$:<br>
-- Undirected graph<br>
-- $X_{i}$ and $X_{j}$ are connected if they appeared in the same factor in a run of the VE algorithm using $\alpha$ as the ordering.<br><br>

<b>Cliques in Induced Graph</b><br>
A <b>clique</b> is a fully connected graph.<br><br>

$\bigstar$ Theorem: Every factor produced during VE is a clique in the induced graph.<br><br>

For example, we eliminate 5 variables, says C, D, I, H, G in order.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_3_4.png"/></center>
</p>
<p align="justify">
After each elimination, we have a clique, e.g. C-D.<br><br>

$\bigstar$ Theorem: Every (maximal) clique in the induced graph is a factor produced during VE.<br><br>

<b>Induced Width</b><br>
$\bigstar$ The <b>width of an induced graph</b> is the number of nodes in the largest clique in the graph minus 1.<br>
$\bigstar$ <b>Minimal induced width</b> of a graph K is $\min_{\alpha}width(I_{K, \alpha})$<br>
$\bigstar$ Provides a <b>lower bound</b> on best performance of VE to a model factorizing over K.<br><br>

<b>Summary</b><br>
$\bigstar$ Variable elimination can be viewed as transformations on undirected graph<br>
-- Elimination connects all node's current neighbors<br>
$\bigstar$ Cliques in resulting induced graph directly correspond to algorithm's complexity<br><br>
</p>

#### 2.2.4 Finding Elimination Orderings
<p align="justify">
<b>Finding Elimination Orderings</b><br>
$\bigstar$ Theorem: For a graph H, determining whether there exists an elimination ordering for H with induced width $\leq$ K is NP-Complete.<br>
$\bigstar$ Note: This NP-hardness result is distinct from the NP-hardness result of inference<br>
-- Even given the optimal ordering, inference may still be exponential.<br>
$\bigstar$ Greedy search using heuristic cost function<br>
-- At each point, eliminate node with smallest cost<br>
$\bigstar$ Possible cost functions:<br>
-- min-neighbors: # neighbors in current graph<br>
-- min-weight: weight (# value) of factor formed<br>
-- min-fill: number of new fill edges<br>
-- weighted min-fill: total weight of new fill edges (edge weight = product of weights of the 2 nodes)<br><br>

$\bigstar$ Theorem: The induced graph is triangulated.<br>
-- No loops of lenghth > 3 without a bridge<br><br>

For example, here is a loop of length > 3 without a bridge
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_4_1.png"/></center>
</p>
<p align="justify">
Assume we eliminate A, will add a fill edge between B and D.<br><br>

$\bigstar$ Can find elimination ordering by finding a low-width triangulation of original graph $H_{\phi}$.<br><br>

<b>Summary</b><br>
$\bigstar$ Finding the optimal elimination ordering is NP-hard<br>
$\bigstar$ Simple heuristics that try to keep induced graph small often provide reasonable performance<br><br>
</p>

#### 2.2.5 Quiz
<p align="justify">
<b>1. Intermediate Factors</b><br>
Consider running variable elimination on the following Bayesian network over binary variables.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_5_1.png"/></center>
</p>
<p align="justify">
<b>Which of the nodes, if eliminated first, results in the largest intermediate factor?</b> By largest factor we mean the factor with the largest number of entries.<br>
<b>Answer:</b> $X_{5}$.<br><br>

<b>2. Elimination Orderings</b><br>
<b>Which of the following characteristics of the variable elimination algorithm are affected by the choice of elimination ordering?</b> You may select 1 or more options.<br>
A. Size of the largest intermediate factor<br>
B. Which marginals can be computed correctly<br>
C. Runtime of the algorithm<br>
D. Memory usage of the algorithm<br>
<b>Answer:</b> A, C, D.<br><br>

<b>3. Marginalization</b><br>
<b>3.1</b><br>
Suppose we run variable elimination on a Bayesian network where we eliminate all the variables in the network. <b>What number will the algorithm produce?</b><br>
<b>Answer:</b> 1.<br><br>

<b>3.2</b><br>
Suppose we run variable elimination on a Markov network where we eliminate all the variables in the network. <b>What number will the algorithm produce?</b><br>
<b>Answer:</b> Z, the partition function for the network.<br><br>

<b>4. Intermediate Factors</b><br>
We have a graph
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_5_2.png"/></center>
</p>
<p align="justify">
<b>4.1</b><br>
If we perform variable elimination on the graph with the variable ordering B, A, C, F, E, D. <b>What is the intermediate factor produced by the third step (just before summing out C)?</b><br>
<b>Answer:</b> $\psi(C, D, F)$.<br><br>

<b>4.2</b><br>
If we perform variable elimination on the graph with the variable ordering F, E, D, C, B, A. <b>What is the intermediate factor produced by the third step (just before summing out D)?</b><br>
<b>Answer:</b> $\psi(B, C, D)$.<br><br>

<b>4.3 Induced Graphs</b><br>
If we perform variable elimination on the graph shown below with the variable ordering B, A, C, F, E, D. <b>what is the induced graph for the run?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_2_5_3.png"/></center>
</p>
<p align="justify">
<b>Answer:</b> (a).<br><br>

<b>5. *Time Complexity of Variable Elimination</b><br>
Consider a Bayesian network taking the form of a chain of n variables, $X_{1} \rightarrow X_{2} \rightarrow \cdots \rightarrow X_{n}$, where each of the $X_{i}$ can take on k values. <b>What is the computational cost of running variable elimination on this network if we eliminate the $X_{i}$ in order</b> (i.e., first $X_{1}$, then $X_{2}$ and so on)?<br>
<b>Answer:</b> $O(nk^{2})$.<br><br>

<b>6. Time Complexity of Variable Elimination</b><br>
Suppose we eliminate all the variables in a Markov network using the variable elimination algorithm. <b>Which of the following could affect the runtime of the algorithm?</b> You may select 1 or more options.<br>
A. Number of values that each variable can take<br>
B. Number of factors in the network<br>
C. Number of variables in the network<br>
<b>Answer:</b> A, B, C.<br><br>
</p>

### 2.3 Message Passing in Cluster Graphs
<p align="justify">
An alternative class of algorithms to the variable elimination algorithms is the class of Message Passing algorithms. This is in some ways closely related to variable elimination. It also offers us additional flexibility in how we do a summation and factor product steps so as to potentially come up with a lower complexity than would be required by even the minimal elimination ordering.<br><br>
</p>

#### 2.3.1 Belief Propagation Algorithm
<p align="justify">
<b>Cluster Graph</b><br>
Imagine based on a Markov network, we have a cluster graph
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_1_1.png"/></center>
</p>
<p align="justify">
So, in this graph, we have 4 clusters. For example, cluster 1 contains A and B and cluster 2 has a jurisdiction over B and C. <b>These clusters are going to talk to each other and they are going to try to convince each other that what they think about a variable under their jurisdiction is correct.</b> For example, cluster 1 is going to talk to cluster 2 about the variable B and it's going to tell cluster 2 what it thinks about B, so cluster 2 becomes more informed about the distribution over B. Besides, each cluster has its own initial information, namely the fatcor in original graph. The variables are going to communicate by via these things called <b>messages</b>.<br><br>

Now we are going to call these clusters $\Psi$, so cluster 1 is $\Psi_{1}(A, B)$.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_1_2.png"/></center>
</p>
<p align="justify">
Imagine cluster 2 sends a message to cluster 1. Because they start to talk to each other, so the initial message is 1, namely $\delta_{2, 1}$ = 1, which means a message from 2 to 1. For cluster A, it's going to talk to cluster 4 with more information.
$$\delta_{1, 4} = \sum_{B} \delta_{2, 1}(B)\Psi_{1}(A, B)$$

We call $\delta_{2, 1}$ incoming message and $\Psi_{1}(A, B)$ initial belief of initial factors.<br><br>

In this message passing process, <b>what would be the right definition for the message $\delta_{1, 2}$?</b><br>
<b>Answer:</b>
$$\delta_{1, 2}(B) = \sum_{A}\psi_{1}(A,B)\delta_{4,1}(A)$$

So, this general process is what keeps the message-passing algorithm going and each cluster is going to send message to its adjacent clusters.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_1_3.png"/></center>
</p>
<p align="justify">
Notice the message that 3 sends to 4 doesn't take into consideration the information that it got from 4. Because we'd like to avoid this case in which 3 thought about this first, but now 4 is reinforcing 3 by telling it to 3 again and the beliefs are just going to go up. And so what happens here is that we deliberately only, restrict attention to evidence that. <b>Messages come in from other sources.</b><br><br>

So, what is a cluster grpah?<br>
$\bigstar$ Undirected graph such that:<br>
-- nodes are clusters $C_{i} \subseteq \{ X_{1}, \cdots, X_{n} \}$<br>
-- edge between $C_{i}$ and $C_{j}$ associated with sepset $S_{i, j} \subseteq C_{i} \cap C_{j}$, where $C_{i}$ is a the jurisdiction of cluster I, $S_{i, j}$ is the communication between two adjacent clusters in the cluster graph.<br>
$\bigstar$ Given set of factors $\Phi$, we assign each $\phi_{k}$ to a cluster $C_{\alpha(k)}$ s.t. $Scope[ \phi_{k} ] \subseteq C_{\alpha(k)}$.<br>
$\bigstar$ Define
$$\psi_{i}(C_{i}) = \prod_{k: \alpha(k)=i} \phi_{k}$$

For example, we have a set of factors. So, we have a possible cluster graph with an initial belief of factors.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_1_4.png"/></center>
</p>
<p align="justify">
<b>Message Passing</b><br>
$$\delta_{i \rightarrow j}(S_{i, j}) = \sum_{C_{i}-S_{i, j}} \psi_{i} \times \prod_{k \in (N_{i}-\{ j \})} \delta_{k \rightarrow i}$$

Where $N_{j}$ means j's neighbors.<br><br>

Here is an example,
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_1_5.png"/></center>
</p>
<p align="justify">
$$\delta_{1 \rightarrow 4}(B) = \sum_{A,C} \psi_{1}(A, B, C) \times \delta_{2 \rightarrow 1}(C)$$

$$\delta_{4 \rightarrow 1}(B) = \sum_{E} \psi_{4}(B, E) \times \delta_{2 \rightarrow 4}(B) \times \delta_{5 \rightarrow 4}(E) \times \delta_{3 \rightarrow 4}(B)$$

<b>Belief Propagation Algorithm</b><br>
$\bigstar$ Assign each factor $\phi_{k} \in \Phi$ to cluster $C_{\alpha(k)}$<br>
$\bigstar$ Construct initial potentials $\psi_{i}(C_{i}) = \prod_{k: \alpha(k)=i} \phi_{k}$<br>
$\bigstar$ Initialize all messages to 1<br>
$\bigstar$ Repeat<br>
-- Select edge (i, j) and pass message
$$\delta_{i \rightarrow j}(S_{i, j}) = \sum_{C_{i}-S_{i, j}} \psi_{i} \times \prod_{k \in (N_{i}-\{ j \})} \delta_{k \rightarrow i}$$

$\bigstar$ Compute beliefs $\beta_{i}(C_{i}) = \psi_{i} \times \prod_{k \in N_{i}} \delta_{k \rightarrow i}$.<br><br>

There are two key techniques: <b>when to stop</b> and <b>how to select an edge</b>.<br><br>

<b>Summary</b><br>
$\bigstar$ Graph of clusters connected by sepsets.<br>
$\bigstar$ Adjacent clusters pass information to each other about variables in sepset.<br>
-- Messages from i to j summaries everything i knows except information obtained from j<br>
$\bigstar$ Algorithm may not converge<br>
$\bigstar$ The resulting beliefs are pseudo-marginals<br>
$\bigstar$ Nevertheless, very useful in practice.<br><br>
</p>

#### 2.3.2 Properties of Cluster Graphs
<p align="justify">
<b>Family Preservation</b><br>
$\bigstar$ Given set of factors $\Phi$, we assign each $\phi_{k}$ to a cluster $C_{\alpha(k)}$, s.t. $Scope[ \phi_{k} ] \subseteq C_{\alpha(k)}$<br>
$\bigstar$ For each factor $\phi_{k} \in \Phi$, there exists a cluster $C_{i}$ s.t. $Scope[\phi_{k}] \subseteq C_{i}$<br><br>

<b>Running Intersection Property</b><br>
$\bigstar$ For each pair of clusters $C_{i}$, $C_{j}$ and variable $X \in C_{i} \cap C_{j}$ there exists a unique path between $C_{i}$ and $C_{j}$ for which all clusters and sepsets contain X.<br><br>

For example, here is a cluster graph. X is one of variables in $C_{5} \cap C_{7}$.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_2_1.png"/></center>
</p>
<p align="justify">
Then, there must be a path, suppose $C_{5}$ - $C_{3}$ - $C_{7}$, in which all clusters and sepsets contain X.<br><br>

We illustrate why exist and unique.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_2_2.png"/></center>
</p>
<p align="justify">
For exist, suppose a sepset between $C_{3}$ and $C_{7}$ doesn't contain X, it will be impossible for $C_{5}$ to communicate with $C_{7}$.<br><br>

For uniqueness, suppose we have two paths: $C_{5}$ - $C_{3}$ - $C_{7}$ and $C_{5}$ - $C_{2}$ - $C_{3}$ - $C_{7}$. $C_{5}$ can pass message about X to $C_{2}$, $C_{2}$ can pass it to $C_{3}$ and $C_{3}$ can do the same thing to $C_{5}$. So, this forms a feedback loop, which gives rise to extreme and skewed probability.<br><br>

Imagine we have another variable Y, which is strongly correlated to X and we have a path $C_{5}$ - $C_{2}$ - $C_{3}$ for Y.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_2_3.png"/></center>
</p>
<p align="justify">
Then, consider $C_{3}$ sends information to $C_{5}$ about X. $C_{5}$ translates that to information about Y. That information about Y goes back to $C_{3}$ via $C_{2}$ and increases the probability in X. <b>Belief Propagation does poorly when we have strong correlations because of feedback loops.</b><br><br>

We have an alternaive definition about Running Intersection Property<br>
$\bigstar$ Equivalently: For any X, the set of clusters and sepsets containing X form a tree.<br><br>

<b>Which of the following cluster graphs satisfy the running intersection property?</b> Mark all that apply.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_2_4.png"/></center>
</p>
<p align="justify">
<b>Answer:</b> (b), (c).<br><br>

<b>Bethe Cluster Graph</b><br>
$\bigstar$ For each $\phi_{k} \in \Phi$, a factor cluster $C_{k}$ = Scope[$\phi_{k}$]<br>
$\bigstar$ For each $X_{i}$ a singleton cluster {$X_{i}$}<br>
$\bigstar$ Edge $C_{k}$ --- $X_{i}$ if $X_{i} \in C_{k}$<br><br>

For example,
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_2_5.png"/></center>
</p>
<p align="justify">
<b>Which of the following is the Bethe cluster graph for a network with factors $\phi_1(A,B)$, $\phi_2(B,C)$, $\phi_3(A,C)$?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_2_6.png"/></center>
</p>
<p align="justify">
<b>Answer:</b> (c).<br>
(a) Incorrect because we do not have a factor $\phi(A,B,C)$.<br>
(b) Incorrect because the variables in the sepsets along the edges are incorrect. This violates the running intersection property. Also violates the family intersection property.<br>
(c) Correct because sepsets are right and the cluster graph is made of singleton and pairwise factors.<br>
(d) Incorrect because Bethe cluster graph is a bipartite graph made of singleton and pairwise factors.<br><br>

<b>Summary</b><br>
$\bigstar$ Cluster graph must satisfy two properties:<br>
-- family preservation: allow $\Phi$ to be encoded<br>
-- running intersection: connect all information about any variable, but without feedback loops<br>
$\bigstar$ Bethe cluster is often first default<br>
$\bigstar$ Richer cluster graph structures can offer different tradeoffs worth computational cost and preservation of dependencies.<br><br>
</p>

#### 2.3.3 Properties of Belief Propagation
<p align="justify">
<b>Calibration</b><br>
$\bigstar$ Cluster beliefs
$$\beta_{i}(C_{i}) = \psi_{i} \times \prod_{k \in N_{i}} \delta_{k \rightarrow i}$$

For example, consider a cluster graph
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_3_1.png"/></center>
</p>
<p align="justify">
For cluster 1, its belief
$$\beta_{1}(A, B) = \psi_{1}(A, B) \times \delta_{4 \rightarrow 1}(A) \times \delta_{2 \rightarrow 1}(B)$$

$\bigstar$ A cluster graph is <b>calibrated</b> if every pair of adjacent clusters $C_{i}$, $C_{j}$ agree on their sepset $S_{i, j}$
$$\sum_{C_{i}-S_{i, j}} \beta_{i}(C_{i}) = \sum_{C_{j}-S_{i, j}} \beta_{j}(C_{j})$$

<b>Convergence $\rightarrow$ Calibration</b><br>
$\bigstar$ Convergence: message at next time is equal to that at current step
$$\delta_{i \rightarrow j}(S_{i, j}) = \delta'_{i \rightarrow j}(S_{i, j})$$

Proof
$$\beta_{i}(C_{i}) = \psi_{i} \times \prod_{k \in N_{i}} \delta_{k \rightarrow i} = \psi_{i} \times \delta_{j \rightarrow i}(S_{i, j}) \times \prod_{k \in (N_{i} - \{ j \})} \delta_{k \rightarrow i}$$

$$\delta'_{i \rightarrow j}(S_{i, j}) = \sum_{C_{i}-S_{i, j}}(\psi_{i} \times \prod_{k \in (N_{i} - \{ j \})} \delta_{k \rightarrow i}) = \sum_{C_{i}-S_{i, j}} \frac{\beta_{i}(C_{i})}{\delta_{j \rightarrow i}(S_{i, j})}$$

$$\delta_{i \rightarrow j}(S_{i, j})\delta_{j \rightarrow i}(S_{i, j}) = \sum_{C_{i}-S_{i, j}} \beta_{i}(C_{i})$$

Similarly, we deduce for j
$$\delta_{i \rightarrow j}(S_{i, j})\delta_{j \rightarrow i}(S_{i, j}) = \sum_{C_{j}-S_{i, j}} \beta_{j}(C_{j})$$

$$\sum_{C_{i}-S_{i, j}} \beta_{i}(C_{i}) = \sum_{C_{j}-S_{i, j}} \beta_{j}(C_{j})$$

That is to say, the clsuter graph is calibrated. Besides, we have a new notation $\mu_{i, j}$, called sepset belief.
$$\mu_{i, j}(S_{i, j}) = \delta_{j \rightarrow i}\delta_{i \rightarrow j}$$

<b>What are the consequences of this analysis?</b><br>
A. The beliefs of neighboring clusters agree on all variables.<br>
B. The beliefs of neighboring clusters agree on all variables in the sepset.<br>
C. The beliefs of neighboring clusters agree on all variables they have in common.<br>
D. The beliefs of neighboring clusters agree on all variables in either cluster.<br>
<b>Answer:</b> B.<br>
The sepset is required to be a subset of the intersection of variables in neighboring clusters, not the intersection. This analysis shows only that beliefs agree on the sepset variables.<br><br>

<b>Reparameterization</b><br>
Recall cluster belief
$$\beta_{i}(C_{i}) = \psi_{i} \times \prod_{k \in N_{i}} \delta_{k \rightarrow i}$$

sepset belief
$$\mu_{i, j}(S_{i, j}) = \delta_{j \rightarrow i}\delta_{i \rightarrow j}$$

We put cluster beliefs ands sepset beliefs on the cluster graph
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_3_2.png"/></center>
</p>
<p align="justify">
We notice each message, says $\delta_{4 \rightarrow 1}$ appears twice in this cluster graph.
$$\frac{\prod_{i} \beta_{i}}{\prod_{i, j} \mu_{i, j}} = \frac{\prod_{i} (\psi_{i} \prod_{j \in N_{i}} \delta_{j \rightarrow i})}{\prod_{i, j} \delta_{i \rightarrow j}} = \prod_{i} \psi_{i} = \widetilde{P}_{\Phi}(X_{1}, \cdots, X_{n})$$

This is the unnormalized measure, which means no information lost.<br><br>

<b>Summary</b><br>
$\bigstar$ At convergence of BP, cluster graph beliefs are calibrated:<br>
-- beliefs at adjacent clusters agree on sepsets<br>
$\bigstar$ Cluster graph beliefs are an alternative, calibrated parameterization of the original unnormalized density<br>
-- no information is lost by message passing.<br><br>
</p>

#### 2.3.4 Quiz
<p align="justify">
<b>1. Cluster Graph Construction</b><br>
Consider the pairwise MRF, H, shown below with potentials over {A,B}, {B,C}, {A,D}, {B,E}, {C,F}, {D,E} and {E,F}.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_4_1.png"/></center>
</p>
<p align="justify">
<b>Which of the following is/are valid cluster graph(s) for H?</b> (A cluster graph is valid if it satisfies the running intersection property and family preservation. You may select 1 or more options).
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_4_2.png"/></center>
</p>
<p align="justify">
<b>Answer:</b> (b), (d) (e).<br>
(a): This graph does not satisfy family preservation. For example, the potential over {C, F} cannot be assigned to any cluster.<br>
(c): This graph does not satisfy family preservation. For example, the potential over {B, C} cannot be assigned to any cluster.<br>
(e): This is a valid cluster graph because a single cluster with all variables is always a valid cluster graph for any distribution. It however, does not admit efficient inference, since we would have to sum over (multiple) variables in order to extract any marginals from the cluster belief.<br><br>

<b>2. Family Preservation</b><br>
Suppose we have a factor $P(A \mid C)$ that we wish to include in our sum-product message passing inference. We should:<br>
A. Assign the factor to all cliques that contain A or C<br>
B. Assign the factor to all cliques that contain A and C<br>
C. Assign the factor to one cliques that contain A or C<br>
D. Assign the factor to one cliques that contain A and C<br>
<b>Answer:</b> D.<br>
Family Preservation explains that the proper construction of a clique tree (cluster graph) requires assigning each factor to one cluster whose scope contains the scope of the factor.<br><br>

<b>3.</b><br>
Suppose we wish to perform inference over the Markov network M as shown below. Each of the variables $X_{i}$ are binary, and the only potentials in the network are the pairwise potentials $\phi_{i,j}(X_i, X_j)$, with one potential for each pair of variables $X_{i}$, $X_{j}$ connected by an edge in M.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_3_4_3.png"/></center>
</p>
<p align="justify">
<b>3.1 Message Passing in a Cluster Graph</b><br>
<b>Which of the following expressions correctly computes the message $\delta_{3 \rightarrow 6}$ that cluster $C_{3}$ will send to cluster $C_{6}$ during belief propagation?</b> Assume that the variables in the sepsets are equal to the intersection of the variables in the adjacent cliques.<br>
<b>Answer:</b>
$$\delta_{3 \rightarrow 6} (X_5) = \sum_{X_2}\phi_{2,5}(X_2,X_5) \delta_{2\rightarrow 3}(X_2) \delta_{4 \rightarrow 3}(X_2) \delta_{7 \rightarrow 3}(X_5)$$

<b>3.2 Message Passing Computation</b><br>
If the initial factors in the Markov network M are of the form as shown in the table below, regardless of the specific value of i, j (we basically wish to encourage variables that are connected by an edge to share the same assignment), <b>compute the message $\delta_{3 \rightarrow 6}$</b>, assuming that it is the first message passed during in loopy belief propagation. Assume that the messages are all initialized to the 1 message, i.e. all the entries are initially set to 1.<br>
Separate the entries of the message with spaces. Order the entries by lexicographic variable order: for example, if the message is over one variable $X_{i}$, then enter in $\delta_{3 \rightarrow 6} (X_{i} = 0)$ $\delta_{3 \rightarrow 6} (X_{i} = 1)$. If the message is over two variables $X_{i}$, $X_{j}$, where i < j, enter the answer in the order $\delta_{3 \rightarrow 6} (X_{i} = 0, X_{j} = 0)$ $\delta_{3 \rightarrow 6} (X_{i} = 0, X_{j} = 1)$ $\delta_{3 \rightarrow 6} (X_{i} = 1, X_{j} = 0)$ $\delta_{3 \rightarrow 6} (X_{i} = 1, X_{j} = 1)$<br>
<table class="a">
  <tr><th>$X_{i}$</th><th>$X_{j}$</th><th>$\phi(X_{i}, X_{j})$</th></tr>
  <tr><td>1</td><td>1</td><td>10</td></tr>
  <tr><td>1</td><td>0</td><td>1</td></tr>
  <tr><td>0</td><td>1</td><td>1</td></tr>
  <tr><td>0</td><td>0</td><td>10</td></tr>
</table><br>
</p>
<p align="justify">
<b>Answer:</b> 11 11.<br><br>

<b>3.3 *Extracting Marginals at Convergence</b><br>
Given that you can renormalize the messages at any point during belief propagation and still obtain correct marginals, consider the message $\delta_{3 \rightarrow 6}$ that you computed. Use this observation to compute the final and possibly approximate marginal probability $P(X_{4} = 1, X_{5}=1)$ in cluster $C_{6}$ at convergence (as extracted from the cluster beliefs), giving your answer to 2 decimal places.<br>
<b>Answer:</b> 0.45.<br>
During all iterations, we never change le value of $\phi_{4, 5}(X_{4}, X_{5})$, so $\frac{10}{10+1+1+10}$ = 0.45.<br><br>
</p>

### 2.4 Clique Trees
#### 2.4.1 Clique Tree Algorithm - Correctness
<p align="justify">
<b>Message Passing in Tree</b><br>
Consider a Markov network as well as its cluster graph with pairwise potential for each cluster.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_1_1.png"/></center>
</p>
<p align="justify">
So, we put message for each sepset.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_1_2.png"/></center>
</p>
<p align="justify">
<b>Correctness</b><br>
If we want to calculate cluster belief of cluster 3
$$\beta_{3}(C, D) = \psi_{3} \times \delta_{2 \rightarrow 3} \times \delta_{4 \rightarrow 3}$$

$$= \psi_{3} \times (\sum_{B} \psi_{2} \times \delta_{1 \rightarrow 2}) \times \sum_{E} \psi_{4}$$

$$= \psi_{3} \times (\sum_{B} \psi_{2} \times \sum_{A} \psi_{1}) \times \sum_{E} \psi_{4}$$

That is to say, we have four factor products $\psi_{1}$, $\psi_{2}$, $\psi_{3}$, $\psi_{4}$ and we have all information of variables except variables in cluster 3, namely A, B, E. In other wors, we have a factor product and a marginalization over unnecessary variables. This is logically equal to variable elimination. So we confirm the correctness.<br><br>

<b>Clique Tree</b><br>
$\bigstar$ Undirected tree such that:<br>
-- nodes are clusters $C_{i} \subseteq \{ X_{1}, \cdots, X_{n} \}$<br>
-- edge between $C_{i}$ and $C_{j}$ associated with sepset $S_{i, j} = C_{i} \cap C_{j}$<br><br>

<b>Family Preservation</b><br>
$\bigstar$ Given set of factors $\Phi$, we assign each $\phi_{k}$ to a cluster $C_{\alpha(k)}$, s.t. $Scope[ \phi_{k} ] \subseteq C_{\alpha(k)}$<br>
$\bigstar$ For each factor $\phi_{k} \in \Phi$, there exists a cluster $C_{i}$ s.t. $Scope[\phi_{k}] \subseteq C_{i}$<br><br>

<b>Running Intersection Property</b><br>
$\bigstar$ For each pair of clusters $C_{i}$, $C_{j}$ and variable $X \in C_{i} \cap C_{j}$ there exists a unique path between $C_{i}$ and $C_{j}$ for which all clusters and sepsets contain X.<br>
$\bigstar$ For each pair of clusters $C_{i}$, $C_{j}$ and variable $X \in C_{i} \cap C_{j}$, in the unique path between $C_{i}$ and $C_{j}$, all clusters and sepsets contain X.<br><br>

For example, we have a cluster graph and $C_{6}$ and $C_{7}$ contain a common variable.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_1_3.png"/></center>
</p>
<p align="justify">
Then $C_{4}$ and $C_{3}$ must contain X.<br><br>

In contrast, if X is eliminated when we pass the message $C_{i} \rightarrow C_{j}$, X doesn't appear in the $C_{j}$ side of the tree. For example, edge between $C_{3}$ and $C_{5}$ has no X, $C_{5}$ must have no X, too.<br><br>

<b>Summary</b><br>
$\bigstar$ Belief propagation can be run over a tree-structured cluster graph<br>
$\bigstar$ In this case, computation is a variant of variable elimination<br>
$\bigstar$ Resulting beliefs are guaranteed to be correct marginals<br><br>
</p>

#### 2.4.2 Clique Tree Algorithm - Computation
<p align="justify">
<b>Message Passing in Tree</b><br>
We have a Markov network as well as its cluster graph
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_2_1.png"/></center>
</p>
<p align="justify">
Once we compute $\delta_{1 \rightarrow 2}(B)$, it is converged because $\psi_{1}$ never changes. So cluster 1 pass the message of B to cluster 2, then $\delta_{2 \rightarrow 3}(B)$ is converged. This applies for other clusters. So, we notice one propagation from left to right and one propagagtion form right to left makes the whole cluster graph converged. We call <b>forward propagation algorithm</b>. Then we compute cluster beliefs $\beta_{i}(C_{i})$ represents an unnormalized measure.<br><br>

<b>Convergence of Message Passing</b><br>
$\bigstar$ Once $C_{i}$ receives a final message from all neighbors except $C_{j}$, then $\delta_{i \rightarrow j}$ is also final (will never change).<br>
$\bigstar$ Messages from leaves are immediately final<br>
$\bigstar$ Can pass messages from leaves inward<br>
$\bigstar$ If messages are passed in the right order, only need to pass 2(K-1) messages, where K is the number of total clusters.<br><br>

For example, we have a cluster grapha and we give a possible right order for two directions.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_2_2.png"/></center>
</p>
<p align="justify">
<b>Answering Queries</b><br>
$\bigstar$ Posterior distribution queries on variables that appear together in clique<br>
-- Sum out irrelevant variables from any clique containing those variables<br>
$\bigstar$ Introducing new evidence Z = z and querying X. If X appears in clique with Z<br>
-- Multiply clique that contains X and Z with indicators function $1_{Z=z}$<br>
-- Sum out irrelevant variables and renormalize<br>
$\bigstar$ Introducing new evidence Z = z and querying X. If X does not share a clique with Z<br>
-- Multiply $1_{Z=z}$ into some clique containing Z<br>
-- Propagate messages along path to clique containing X<br>
-- Sum out irrelevant variables and normalize<br><br>

For example, we want to query E given A = a
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_2_3.png"/></center>
</p>
<p align="justify">
So, we multiply $1_{A=a}$ into clique 1 in order to remove all inconsistent factors in cluster 1. Then, we propagate the message to cluster 4 and compute the marginalization with normalization<br><br>

If $\psi_{1}$ changes, $\delta_{1 \rightarrow 2}$, $\delta_{2 \rightarrow 3}$, $\delta_{3 \rightarrow 4}$ will change, while $\delta_{4 \rightarrow 3}$, $\delta_{3 \rightarrow 2}$, $\delta_{2 \rightarrow 1}$ will not change.<br><br>

<b>Summary</b><br>
$\bigstar$ In clique tree with K cliques, if messages are passed starting at leaves, 2(K-1) messages suffice to compute all beliefs.<br>
$\bigstar$ Can compute marginals over all variables at only twice the cost of variables elimination.<br>
$\bigstar$ By storing messages, inference can be reused in cremental queries.<br><br>
</p>

#### 2.4.3 Clique Trees and Independence
<p align="justify">
<b>RIP and Independence</b><br>
$\bigstar$ For an edge (i, j) in T, let:<br>
-- $W_{<(i, j)}$ = all variables that appear only on $C_{i}$ side of T<br>
-- $W_{<(j, i)}$ = all variables that appear only on $C_{j}$ side of T<br>
-- Variables on both sides are in the sepsets $S_{i, j}$<br>
$\bigstar$ Theorem: T satisfies RIP (Running Intersection Property) if and only if, for every (i, j)
$$P_{\Phi} \models (W_{<(i, j)} \perp W_{<(j, i)} \mid S_{i, j})$$

For example, we have a Markov network as well as its cluster graph
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_3_1.png"/></center>
</p>
<p align="justify">
Consider variable G and S, so, C, I, D are separated from H, L, J given G, S.
$$P_{\Phi} \models (\{ C, I, D \} \perp \{ H, J, L \} \mid \{ G, S \})$$

<b>Implications</b><br>
$\bigstar$ Each sepset needs to separate graph into two conditionally independent parts<br><br>

For example, we have a complete bipartite graph
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_3_2.png"/></center>
</p>
<p align="justify">
The minimum sepset has a size at least min(k, m) in order to completely seperate two sets of variables.<br><br>

Another example, Ising model
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_3_3.png"/></center>
</p>
<p align="justify">
The minimum sepset is $\{ A_{2, 1}, A_{1, 2} \}$ with a size of 2. The largest sepset has a size at least n for a $n \times n$ grid.<br><br>

<b>Summary</b><br>
$\bigstar$ Correctness of clique tree inference relies on running intersection property<br>
$\bigstar$ Running intersection property implies separation in original distribution<br>
$\bigstar$ Implies minimal complexity incurred by any clique tree:<br>
-- Related to minimal induced width of graph<br><br>
</p>

#### 2.4.4 Clique Trees and VE
<p align="justify">
<b>Variable Elimination & Clique Tree</b><br>
$\bigstar$ Variable elimination<br>
-- Each step creates a factor $\lambda_{i}$ through factor product<br>
-- A variable is eliminated in $\lambda_{i}$ to generate new factor $\tau_{i}$<br>
-- $\tau_{i}$ is used in computing other factors $\lambda_{j}$<br>
$\bigstar$ Clique tree view<br>
-- Intermediate factros $\lambda_{i}$ are cliques<br>
-- $\tau_{i}$ are 'messages' generated by clique $\lambda_{i}$ and transmitted to another clique $\lambda_{j}$<br><br>

<b>Clique Tree from VE</b><br>
$\bigstar$ VE defines a graph<br>
-- Cluster $C_{i}$ for each factor $\lambda_{i}$ used in the computation<br>
-- Draw edge $C_{i} - C_{j}$ if the factor generated from $\lambda_{i}$ is used in the computation of $\lambda_{j}$<br><br>

For example, in a Markov network we eliminate variables C, D, I, H, G, S, L one by one.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_4_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ Eliminate C: $\tau_{1}(D) = \sum_{C}\phi_{C}(C)\phi_{C, D}(C, D)$<br>
-- $\phi_{C}(C)\phi_{C, D}(C, D)$ is $\lambda_{1}$ (factor product) and $\tau_{1}(D)$ is a new factor generated. So, we draw a cluster 1 for factor $\phi_{C}(C)$ and $\phi_{C, D}(C, D)$<br><br>

$\bigstar$ Eliminate D: $\tau_{2}(G, I) = \sum_{D}\phi_{G}(G, I, D)\tau_{1}(D)$<br>
-- $\phi_{G}(G, I, D)\tau_{1}(D)$ is $\lambda_{2}$. We draw cluster 2 for factors $\phi_{G}(G, I, D)\tau_{1}(D)$ and an edge between cluster 1 and cluster 2, because $\tau_{1}(D)$ is used in this computation.<br><br>

$\bigstar$ Eliminate I: $\tau_{3}(G, S) = \sum_{I}\phi_{I}(I)\phi_{S}(S,I)\tau_{2}(G, I)$<br>
-- $\phi_{I}(I)\phi_{S}(S,I)\tau_{2}(G, I)$ is $\lambda_{3}$. Cluster 3 is for factors $\phi_{I}(I)\phi_{S}(S,I)\tau_{2}(G, I)$. Because $\tau_{2}(G, I)$ is used, we draw an edge between cluster 2 and cluster 3.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_4_2.png"/></center>
</p>
<p align="justify">
$\bigstar$ Eliminate H: $\tau_{4}(G, J) = \sum_{H}\phi_{H}(H,G,J)$<br>
-- Cluster 4 for factors $\phi_{H}(H,G,J)$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_4_3.png"/></center>
</p>
<p align="justify">
$\bigstar$ Eliminate G: $\tau_{5}(J, L, S) = \sum_{G}\phi_{L}(L, G)\tau_{3}(G, S)\tau_{4}(G, J)$<br>
-- Cluster 5 for factors $\phi_{L}(L, G)\tau_{3}(G, S)\tau_{4}(G, J)$. An edge between cluster 5 and cluster 3 and another edge between cluster 5 and cluster 4.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_4_4.png"/></center>
</p>
<p align="justify">
$\bigstar$ Eliminate S: $\tau_{6}(J, L) = \sum_{S}\phi_{J}(J, L, S)\tau_{5}(J, L, S)$<br>
-- Cluster 6 for factors $\phi_{J}(J, L, S)\tau_{5}(J, L, S)$. An edge between cluster 6 and cluster 5
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_4_5.png"/></center>
</p>
<p align="justify">
$\bigstar$ Eliminate L: $\tau_{7}(J) = \sum_{L}\tau_{6}(J, L)$<br>
-- Cluster 7 for factors $\tau_{6}(J, L)$. An edge between cluster 7 and cluster 6.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_4_6.png"/></center>
</p>
<p align="justify">
$\bigstar$ Remove redundant cliques: <b>those whose scope is a subset of adjacent clique's scope</b><br>
-- remove cluster 7 and cluster 6
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_4_7.png"/></center>
</p>
<p align="justify">
<b>Properties of Tree</b><br>
$\bigstar$ VE process induces a tree<br>
-- In VE, each intermediate factor $\tau_{i}$ is used only once<br>
-- Hence, each cluster 'passes' a factor (message) to exactly one other cluster<br>
$\bigstar$ Tree is family preserving<br>
-- Each of the original factors $\phi_{i} \in \Phi$ must be used in some elimination step<br>
-- And therefore must be contained in scope of associated $\lambda_{i}$<br>
$\bigstar$ Tree obeys running intersection property<br>
-- If $X \in C_{i}$ and $X \in X_{j}$ then X is in each cluster in the unique path between $C_{i}$ and $C_{j}$.<br>
-- For example, G is in cluster 2 and cluster 4, so there is a unique path between cluster 2 and cluter 4.<br><br>

<b>Running Intersection Property</b><br>
$\bigstar$ Theorem: If T is a tree of clusters induced by VE, then T obeys RIP.<br><br>

<b>Summary</b><br>
$\bigstar$ A run of variable elimination implicitly defines a correct clique tree<br>
-- We can 'simulate' a run of VE to define cliques and connections between them<br>
$\bigstar$ Cost of variable elimination is the same as passing messages in one direction in tree<br>
$\bigstar$ Clique trees use dynamic programming (storing messages) to compute marginals over all variables at only twice the cost of VE<br><br>
</p>

#### 2.4.5 Quiz
<p align="justify">
<b>1.</b><br>
Here is a cluster tree
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_5_1.png"/></center>
</p>
<p align="justify">
<b>1.1 Message Ordering</b><br>
<b>which of the following starting message-passing orders is/are valid?</b> (Note: These are not necessarily full sweeps that result in calibration. You may select 1 or more options.)<br>
A. $C_{4} \rightarrow C_{3}, C_{5} \rightarrow C_{3}, C_{2} \rightarrow C_{3}$<br>
B. $C_{4} \rightarrow C_{3}, C_{3} \rightarrow C_{5}, C_{3} \rightarrow C_{2}, C_{1} \rightarrow C_{2}$<br>
C. $C_{4} \rightarrow C_{3}, C_{5} \rightarrow C_{3}, C_{3} \rightarrow C_{2}, C_{1} \rightarrow C_{2}$<br>
D. $C_{1} \rightarrow C_{2}, C_{2} \rightarrow C_{3}, C_{5} \rightarrow C_{3}, C_{3} \rightarrow C_{4}$<br>
<b>Answer:</b> C, D.<br><br>

<b>1.2 Message Passing in a Clique Tree</b><br>
In the clique tree above, what is the correct form of the message from clique 3 to clique 2, $\delta_{3 \rightarrow 2}$, where $\psi_i(C_i)$ is the initial potential of clique i?<br>
<b>Answer:</b> $\sum_{G,H} \psi_{3}(C_3) \times \delta_{4 \rightarrow 3} \times \delta_{5 \rightarrow 3}$.<br><br>

<b>2. Clique Tree Properties</b><br>
Consider the following Markov Network (a) over potentials $\phi_{A,B}$, $\phi_{B,C}$, $\phi_{A,D}$, $\phi_{B,E}$, $\phi_{C,F}$, $\phi_{D,E}$, and $\phi_{E,F}$:
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_4_5_2.png"/></center>
</p>
<p align="justify">
Which of the following properties are necessary for a valid clique tree for the above network (a), but are NOT satisfied by graph (b):<br>
A. Famility preservation<br>
B. The number of nodes in a clique tree containing a variable should be exactly the number of factors in the Markov network that contain the same variable<br>
C. No loops<br>
D. Running intersection property<br>
<b>Answer:</b> A, C.<br>
The graph (b) does satisfy the running intersection property, assuming that the sepsets are equal to the intersection of the variables in the adjacent cliques.<br><br>

<b>3. Cluster Graphs vs. Clique Trees</b><br>
Suppose that we ran sum-product message passing on a cluster graph G for a Markov network M and that the algorithm converged. <b>Which of the following statements is true only if G is a clique tree and is not necessarily true otherwise?</b><br>
A. The sepsets in G are the product of the two messages passed between the clusters adjacent to the sepset.<br>
B. All the options are true for cluster graphs in general.<br>
C. G is calibrated.<br>
D. The beliefs and sepsets of G can be used to compute the joint distribution defined by the factors of M.<br>
E. If there are E edges in G, there exists a message ordering that guarantees convergence after passing 2E messages.<br>
<b>Answer:</b> E.<br>
This is a property specific to clique trees. We can select one of the cliques to be the root clique, and pass messages away from the root clique to all other cliques. Then, we can pass messages from all other cliques towards the root clique and we are guaranteed to have calibrated the tree. In a cluster graph however, depending on the potentials, convergence may take longer.<br><br>

<b>4. Clique Tree Calibration</b><br>
<b>Which of the following is true?</b> You may select more than one option.<br>
A. If there exists a pair of adjacent cliques that are max-calibrated, then a clique tree is max-calibrated.<br>
B. It is true that adjacent cliques have to be max-calibrated, but all adjacent pairs need to be max-calibrated, not just any two.<br>
C. After we complete one upward pass of the max-sum message passing algorithm, the clique tree is max-calibrated.<br>
D. If a clique tree is max-calibrated, then within each clique, all variables are max-calibrated with each other.<br>
E. If a clique tree is max-calibrated, then all pairs of adjacent cliques are max-calibrated.<br>
<b>Answer:</b> D, E.<br>
B: The beliefs are max-calibrated only after we do a downward pass.<br>
E is the condition that makes a clique tree max-calibrated. All adjacent cliques have to agree over their sepset beliefs.<br><br>
</p>

### 2.5 Loopy Belief Propagation
#### 2.5.1 BP In Practice
<p align="justify">
<b>Misconception Revisited</b><br>
Here is a Markov network as well as a figure about distance to true marginals as iterations
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_1_1.png"/></center>
</p>
<p align="justify">
We have strong potentials for factors. Par example, C highly agrees with B, B highly agrees with A and A highly agrees with D, but D highly disagrees with C. Besides, there is a conflict between two directions of messages passing: C - B - A - D and C - D, so there is much oscillation. <b>Tight loops, strong potentials and conflicting directions is probably the worst example for belief propagagtion.</b><br><br>

<b>Different Variant of BP</b><br>
$\bigstar$ Synchronous BP<br>
-- all messages are updated in parallel
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_1_2.png"/></center>
</p>
<p align="justify">
In fact, synchronous BP performs not well enough, because the right figure tells us not 100% messages are converged.<br><br>

$\bigstar$ Asynchronous BP<br>
-- messages are updated one at a time
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_1_3.png"/></center>
</p>
<p align="justify">
But different order has different result.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_1_4.png"/></center>
</p>
<p align="justify">
<b>Observations</b><br>
$\bigstar$ Convergence is a local property<br>
-- some messages converge soon<br>
-- others may never converge<br>
$\bigstar$ Synchronous BP converges considerably worse than asynchronous<br>
$\bigstar$ Message passing order makes a difference to extent and rate of convergence<br><br>

<b>Informed Message Scheduling</b><br>
$\bigstar$ Tree reparameterization (TRP)<br>
-- Pick a tree and pass message to calibrate<br>
-- trees must cover all edges<br>
-- improve performance if trees are large (spanning trees)<br>
$\bigstar$ Residual belief propagation (RBP)<br>
-- Pass messages between two clusters whose beliefs over the sepset disagree the most<br>
-- priority queue is helpful<br><br>

<b>Smoothing (Damping) Messages</b><br>
$$\delta_{i \rightarrow j} \leftarrow \sum_{C_{i} - S_{i, j}} \psi_{i} \prod_{k \neq j} \delta_{k \rightarrow i}$$

$$\delta_{i \rightarrow j} \leftarrow \lambda (\sum_{C_{i} - S_{i, j}} \psi_{i} \prod_{k \neq j} \delta_{k \rightarrow i}) + (1 - \lambda) \delta_{i \rightarrow j}^{old}$$

$\bigstar$ Dampens oscillations in message<br><br>

<b>What will (probably) happen to the oscillations in the network, and the convergence of the network, as λ decreases from 1 to 0?</b><br>
A. Oscillations will increase, and so will the time to convergence.<br>
B. Oscillations will decrease, and so will the time to convergence.<br>
C. Oscillations will decrease. The time to convergence will decrease up to a point, then start increasing.<br>
D. Oscillations will decrease. The time to convergence will increase up to a point, then start decreasing.<br>
<b>Answer:</b> C.<br>
At λ=0, there is no damping, while at λ=1, there is full damping, i.e., no messages will be updated. As λ goes from 1 to 0, damping will increase, so the amplitude of the oscillations will tend to decrease. Damping will help the convergence of the network to a point, but after some point, increased damping will tend to slow the convergence down (consider the extreme case of λ=1, in which no messages will be updated, so the algorithm will never converge).<br><br>

Here is a comparison between three methods: Synchronous (red), Asynchronous without smoothing (green) and asynchronous with smoothing (blue). Notice y-axis represente a percentage of converged messages.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_1_5.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ To achieve BP convergence, two main tricks<br>
-- Damping<br>
-- Intelligent message ordering<br>
$\bigstar$ Convergence doesn't guarantee correctness<br>
$\bigstar$ Bad cases for BP - both convergence & accuracy<br>
-- Strong potentials pulling in different directions<br>
-- Tight loops<br>
$\bigstar$ Some new algorithms have better convergence<br>
-- Optimization-based view to inference<br><br>
</p>

#### 2.5.2 Loopy BP and Message Decoding
<p align="justify">
<b>Message Coding & Decoding</b><br>
Imagine k bits through a noisy channel
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_2_1.png"/></center>
</p>
<p align="justify">
<b>Channel Capacity</b><br>
We have three kinds of channels: binary symmetric channel, binary erasure channel and gaussian channel
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_2_2.png"/></center>
</p>
<p align="justify">
We have a Gaussian channel with capacity $0.5log(1 + \frac{E(X^{2})}{\sigma^{2}})$, and an amplifier that doubles the expected value of the signal $E(X^{2})$, but also scales its noise, parameterized by $\sigma$, by a factor of $\sqrt{2}$. <b>What would using this circuit do to the capacity of the channel?</b><br>
A. It will increase by a factor of $\sqrt{2}$<br>
B. It will decrease by a factor of $\sqrt{2}$<br>
C. It will stay the same.<br>
D. It is impossible to tell because it depends on the power curve of the new circuit.<br>
<b>Answer:</b> C.<br>
The increase in $E(X^{2})$ would be canceled out exactly by the increased noise, resulting in no change in the capacity.<br><br>

<b>Shannon's Theorem</b><br>
Shannon's Theorem, related the notion of channel capacity and bit error probability in a way defines an extremely sharp boundary between codes that are feasible and codes that are infeasible.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_2_3.png"/></center>
</p>
<p align="justify">
X-axis is the rate of the code $\frac{k}{n}$. This is a rate in multiple channel capacity so this says once you define the channel capacity we can look at the rate of a code and each code could be in a different point of the spectrum in terms of the rate.<br><br>

Y-axis is the bit error probability. So obviously, lower is better, in terms of bit error probability. In the attainable region, we could construct codes that achieve any point in this space. Conversely, forbidden region is not obtainable, which means we could not construct a code that had a rate above a certain value and bit era probability that was below a certain value.<br><br>

<b>How close to C can we get?</b><br>
But the question is how can we achieve something that's close To the Shannon limit. On the x axis, we have the signal to noise ratio, measured in db. And on the y axis, we have the log of the bit error probability.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_2_4.png"/></center>
</p>
<p align="justify">
<b>Turbocodes: The Idea</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_2_5.png"/></center>
</p>
<p align="justify">
<b>Iterations of Turbo Decoding</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_2_6.png"/></center>
</p>
<p align="justify">
<b>Low-Density Parity Checking Codes</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_2_7.png"/></center>
</p>
<p align="justify">
We have seen this kind of cluster graph before, when we called it a Bethe graph. <b>What are the $U_{i}$?</b><br>
A. Clusters with multiple variables.<br>
B. Messages shared between the clusters.<br>
C. Singleton clusters.<br>
D. Messages that are fixed because the clique tree is calibrated once it has converged.<br>
<b>Answer:</b> C.<br><br>

<b>Decoding as Loopy BP</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_5_2_8.png"/></center>
</p>
<p align="justify">
<b>Turbo-Codes & LDPCs</b><br>
$\bigstar$ 3G and 4G mobile telephony standards<br>
$\bigstar$ Mobile television system from Qualcom<br>
$\bigstar$ Digital video broadcasting<br>
$\bigstar$ Satellite communication system<br>
$\bigstar$ New NASA mission (e.g. Mars Orbiter)<br>
$\bigstar$ Wireless metropolitan networks standard<br><br>

<b>Summary</b><br>
$\bigstar$ Loopy BP rediscovered by coding practitioners<br>
$\bigstar$ Understanding turbocodes as loopy BP led to development of many new and better codes<br>
-- Current codes coming closer and closer to Shannon limit<br>
$\bigstar$ Resurgence of interest in BP led to much deeper understanding of approximate inference in graphical models<br>
-- Many new algorithms<br><br>
</p>

### 2.6 MAP Algorithms
#### 2.6.1 Max Sum Message Passing
<p align="justify">
<b>Product $\rightarrow$ Summation</b><br>
We can convert product into summation by logarithm.
$$P_{\Phi}(x) \propto \prod_{k}\phi_{k}(D_{k})$$

$$arg\max \prod_{k}\phi_{k}(D_{k})$$

$$\theta_{k}(D_{k}) = \log \phi_{k}(D_{k})$$

$$arg\max \sum_{k}\theta_{k}(D_{k}) = arg \max \theta(X_{1}, \cdots, X_{n})$$

For example
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_6_1_1.png"/></center>
</p>
<p align="justify">
<b>Max-Sum Elimination in Chains</b><br>
Here is an example of Markov network
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_6_1_2.png"/></center>
</p>
<p align="justify">
Our factor product is changed to a summation after logarithm
$$\theta(A, B, C, D, E) = \theta_{1}(A, B) + \theta_{2}(B, C) + \theta_{3}(C, D) + \theta_{4}(D, E)$$

$$\max_{D}\max_{C}\max_{B}\max_{A} \theta(A, B, C, D, E)$$

<b>Which factor(s) would be optimal to max over first?</b><br><br>

<b>Answer:</b> $\theta_{1}(A, B)$, $\theta_{4}(D, E)$.<br><br>

Similar to factor product, we have a table to represent hwo to sum two factors
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_6_1_3.png"/></center>
</p>
<p align="justify">
Factor marginalization becomes fatcor maximization
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_6_1_4.png"/></center>
</p>
<p align="justify">
For example, $\theta(a^{1}, c^{1}) = \max (\theta(a^{1}, b^{1}, c^{1}), \theta(a^{1}, b^{2}, c^{1}))$.<br><br>

If we max over $\theta_{1}(A, B)$ first
$$= \max_{D}\max_{C}\max_{B} (\theta_{2}(B, C) + \theta_{3}(C, D) + \theta_{4}(D, E) + \max_{A} \theta_{1}(A, B)) = \max_{D}\max_{C}\max_{B} (\theta_{2}(B, C) + \theta_{3}(C, D) + \theta_{4}(D, E) + \lambda_{1}(B))$$

$$= \max_{D}\max_{C} (\theta_{3}(C, D) + \theta_{4}(D, E) + \max_{B} (\theta_{2}(B, C) + \lambda_{1}(B))) = \max_{D}\max_{C} (\theta_{3}(C, D) + \theta_{4}(D, E) + \lambda_{2}(C))$$

$$= \max_{D} (\theta_{4}(D, E) + \max_{C} (\theta_{3}(C, D) + \lambda_{2}(C))) = \max_{D} (\theta_{4}(D, E) + \lambda_{3}(D)) = \lambda_{4}(E)$$

$\lambda_{4}(E=e) = \max_{A, B, C, D} \theta(A, B, C, D, E=e)$ is called max-marginal.<br><br>

<b>Max-Sum in Clique Tree</b><br>
For the markov network, we have a clique tree. Each cluster represent a potential $\theta_{i}$. Similarly, we have message passing.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_6_1_5.png"/></center>
</p>
<p align="justify">
<b>Convergence of Message Passing</b><br>
$\bigstar$ Once $C_{i}$ receives a final message from all neighbors except $C_{j}$, then $\lambda_{i \rightarrow j}$ is also final (will never change).<br>
$\bigstar$ Messages from leaves are immediately final.<br><br>

<b>Max-Sum BP at Convergence</b><br>
$\bigstar$ Beliefs at each clique are max-marginals
$$\beta_{i}(C_{i}) = \theta_{i}(C_{i}) + \sum_{k} \lambda_{k \rightarrow i} = \max_{W_{i}} \theta(C_{i}, W_{i}), \quad W_{i} = \{ X_{1}, \cdots, X_{n} \} - C_{i}$$

$\bigstar$ Calibration: cliques agree on shared variables
$$\max_{C_{i} \rightarrow S_{i, j}} \beta_{i}(C_{i}) = \max_{C_{j} \rightarrow S_{i, j}} \beta_{j}(C_{j})$$

For example, cluster 1 and cluster 2 have a common variable B
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_6_1_6.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ The same clique tree algorithm used for sum-product can be used for max-sum<br>
$\bigstar$ As in sum-product, convergence is achieved after a single up-down pass<br>
$\bigstar$ Result is a <b>max-marginal</b> at each clique C:<br>
-- For each assignment c to C, what is the scope of the best completion to c<br><br>
</p>

#### 2.6.2 Finding a MAP Assignment
<p align="justify">
<b>Decoding a MAP Assignment</b><br>
$\bigstar$ Easy if MAP is unique<br>
-- Single maximizing assignment at each clique<br>
-- Whose value is the $\theta$ value of the MAP assignement<br>
-- Due to calibration, choices at all cliques must agree.<br><br>

For example, we have a same $\theta$ value for these three cliques
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_6_2_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ If MAP assignement is not unique, we may have multiple choices at some cliques<br>
$\bigstar$ Arbitrary tie-breaking may not produce a MAP assignement<br>
$\bigstar$ Two options:<br>
-- Slightly perturb parameters to make MAP unique<br>
-- Use traceback procedure that incrementally builds a MAP assignement, one variable at a time<br><br>
</p>

#### 2.6.3 Quiz
<p align="justify">
<b>1. Real-World Applications of MAP Estimation</b><br>
Suppose that you are in charge of setting up a soccer league for a bunch of kindergarten kids, and your job is to split the N children into K teams. The parents are very controlling and also uptight about which friends their kids associate with. So some of them bribe you to set up the teams in certain ways.<br><br>

The parents' bribe can take two forms: For some children i, the parent says "I will pay you $A_{ij}$ dollars if you put my kid i on the same team as kid j"; in other cases, the parent of child i says "I will pay you $B_{i}$ dollars if you put my kid on team k." In our notation, this translates to factor $f_{i, j}(x_{i}, x_{j}) = A_{ij} \cdot \mathbf{1}_{x_{i} = x_{j}}$ or $g_{i}(x_{i}) = B_{i} \cdot \mathbf{1}_{x_{i} = k}$, respectively, where $x_{i}$ is the assigned team of child i and $\mathbf{1}$ is the indicator function. More formally, if we define $x_{i}$ to be the assigned team of child i, the amount of money you get for the first type of bribe will be $f_{i, j}(x_{i}, x_{j})$.<br><br>

Being greedy and devoid of morality, you want to make as much money as possible from these bribes. <b>What are you trying to find?</b><br>
A. $arg\max_{\bar{x}} \prod_{i} g_{i}(x_{i})$<br>
B. $arg\max_{\bar{x}} \prod_{i} g_{i}(x_{i}) \cdot \prod_{i, j} f_{i, j}(x_{i}, x_{j})$<br>
C. $arg\max_{\bar{x}} \sum_{i} g_{i}(x_{i}) + \sum_{i, j} f_{i, j}(x_{i}, x_{j})$<br>
D. $arg\max_{\bar{x}} \sum_{i} g_{i}(x_{i})$<br>
<b>Answer:</b> C.<br><br>

<b>2. *Decoding MAP Assignments</b><br>
You want to find the optimal solution to the above problem using a clique tree over a set of factors ϕ. <b>How could you accomplish this such that you are guaranteed to find the optimal solution?</b> (Ignore issues of tractability, and assume that if you specify a set of factors ϕ, you will be given a valid clique tree of minimum tree width.)<br>
A. Set $\phi_{i, j} = f_{i, j}$, $\phi_{i} = g_{i}$, get the clique tree, run sum product message passing and decode the marginals<br>
B. Set $\phi_{i, j} = f_{i, j}$, $\phi_{i} = g_{i}$, get the clique tree over this set of factors, run max-sum message passing on this clique tree, and decode the marginals<br>
C. Set $\phi_{i, j} = e^{f_{i, j}}$, $\phi_{i} = e^{g_{i}}$, get the clique tree, run sum-product message passing and decode marginals<br>
D. Set $\phi_{i, j} = e^{f_{i, j}}$, $\phi_{i} = e^{g_{i}}$, get the clique tree over this set of factors, run max-sum message passing on this clique tree, and decode the marginals<br>
<b>Answer:</b> D.<br>
We want to compute
$$arg\max_{\bar{x}} \sum_{i} g_{i}(x_{i}) + \sum_{i, j} f_{i, j}(x_{i}, x_{j}) = arg\max_{\bar{x}} \log[\prod_{i}e^{g_{i}(x_{i})} \cdot \prod_{i, j}e^{f_{i, j}(x_{i}, x_{j})}]$$

Since maximizing $\log(z)$ over z is the same as maximizing z over z, we can simply compute
$$arg\max_{\bar{x}} \prod_{i}e^{g_{i}(x_{i})} \cdot \prod_{i, j}e^{f_{i, j}(x_{i}, x_{j})}$$

which is what max-sum message passing returns. So setting the potentials appropriately and running clique tree inference (which is exact) is guaranteed to get the optimal solution.<br><br>

(Remember that max-sum message passing involves taking a log-transform of the factors first, and summing up log-transformed factors is equivalent to multiplying them together; don't be tricked by the "sum"!)<br><br>
</p>

### 2.7 Other MAP Algorithms
#### 2.7.1 Tractable MAP Problems
<p align="justify">
<b>Correspondence / data association</b><br>
Imagine a bipartite graph
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_7_1_1.png"/></center>
</p>
<p align="justify">
$$
X_{ij} =
\begin{cases}
	1, \quad i - j \\
	0, \quad otherwise
\end{cases}
$$

$i - j$ means i matched to j. $\theta_{ij}$ = quality of 'match' between i and j.<br><br>

$\bigstar$ Find highest scoring matching<br>
-- maximize $\sum_{ij} \theta_{ij}X_{ij}$<br>
-- subject to mutual exclusion constraint<br>
$\bigstar$ Easily solved using matching algorithms<br>
$\bigstar$ Many applications<br>
-- matching sensor readings to objects<br>
-- matching features in two related images<br>
-- matching mentions in text to entities<br><br>

<b>Associative potentials</b><br>
Imagine we have two binary variables as well as a table of potentials
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_7_1_2.png"/></center>
</p>
<p align="justify">
We have a constraint a + d $\geq$ b + c.<br><br>

$\bigstar$ Arbitrary network over binary variables using only singleton $\theta_{i}$ and supermodular pairwise potentials $\theta_{ij}$<br>
-- Exact solution using algorithms for finding minimum cuts in graph<br>
$\bigstar$ Many related variants admit efficient exact or approximate solutions<br>
-- Metric MRFs<br><br>

<b>Cardinality Factors</b><br>
$\bigstar$ A factor over arbitrarily many binary variables $X_{1}, \cdots, X_{k}$<br>
$\bigstar$ Scope($X_{1}, \cdots, X_{n}$) = f($\sum_{i} X_{i}$)<br>
$\bigstar$ Example applications:<br>
-- soft parity constraints<br>
-- prior on # pixels in a given category<br>
-- prior on # of instances assigned to a given cluster<br><br>

<b>Sparse Pattern Factors</b><br>
$\bigstar$ A factor over variables $X_{1}, \cdots, X_{k}$<br>
-- Scope($X_{1}, \cdots, X_{k}$) specified for some small # of assignements $x_{1}, \cdots, x_{k}$<br>
-- Constant for all other assignements<br>
$\bigstar$ Examples: give higher score to combinations that occur in real data<br>
-- In spelling, letter combinations that occur in dictionary<br>
-- $5 \times 5$ image patches that appear in natural images<br><br>

<b>Convexity Factors</b><br>
$\bigstar$ Ordered binary variables $X_{1}, \cdots, X_{k}$<br>
$\bigstar$ Convexity constraints<br>
$\bigstar$ Examples<br>
-- Convexity of 'parts' in image segmentation<br>
-- Contiguity of word labeling in text<br>
-- Temporal contiguity of subactivities<br><br>

<b>Summary</b><br>
$\bigstar$ Many specialized models admit tractable MAP solution<br>
-- Many do not have tractable algorithms for computing marginals<br>
$\bigstar$ These specialized models are useful<br>
-- On their own<br>
-- As a component in a larger model with other types of factors<br><br>
</p>

#### 2.7.2 Dual Decomposition - Intuition
<p align="justify">
<b>Problem Formulation</b><br>
$\bigstar$ Singleton factors $\theta_{i}(x_{i})$<br>
$\bigstar$ Large factors $\theta_{F}(x_{F})$
$$MAP(\theta) = \max_{x} (\sum_{i=1}^{n} \theta_{i}(x_{i}) + \sum_{F} \theta_{F}(x_{F}))$$

<b>Divide and Conquer</b><br>
$$MAP(\theta) = \max_{x}(\sum_{i=1}^{n}(\theta_{i}(x_{i}) + \sum_{F: i \in F}\lambda_{F_{i}}(x_{i})) + \sum_{F}(\theta_{F}(x_{F}) - \sum_{i \in F}\lambda_{F_{i}}(x_{i})))$$

$$L(\lambda) = \sum_{i=1}^{n} \max_{x_{i}}(\theta_{i}(x_{i}) + \sum_{F: i \in F}\lambda_{F_{i}}(x_{i})) + \sum_{F}\max_{x_{F}}(\theta_{F}(x_{F}) - \sum_{i \in F}\lambda_{F_{i}}(x_{i}))$$

$L(\lambda)$ is upper bound on MAP(\theta) for any setting of $\lambda$.<br><br>

For example
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_7_2_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ Slaves don't have to be factors in original model<br>
-- Subsets of factors that admit tractable solution to local maximization task<br><br>

For example
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_7_2_2.png"/></center>
</p>
<p align="justify">
$\bigstar$ In pairwise networks, often divide factors into set of disjoint trees<br>
-- Each edge factor assigned to exactly one tree<br>
$\bigstar$ Other tractable classes of factor sets<br>
-- Matchings<br>
-- Associative models<br><br>
</p>

#### 2.7.3 Dual Decomposition - Algorithm
<p align="justify">
<b>Dual Decomposition Algorithm</b><br>
$$\bar{\theta}_{i}^{\lambda} = \theta_{i}(x_{i}) + \sum_{F: i \in F} \lambda_{F_{i}}(x_{i}), \quad \bar{\theta}_{F}^{\lambda} = \theta_{F}(x_{F}) - \sum_{i \in F} \lambda_{F_{i}}(x_{i})$$

$\bigstar$ Initialize all $\lambda$ to be 0<br>
$\bigstar$ Repeat for t = 1, 2, ...<br>
-- Locally optimize all slaves:
$$x_{F}^{*} = arg\max_{x_{F}} \bar{\theta}_{F}^{\lambda}(x_{F}), \quad x_{i}^{*} = arg\max_{x_{i}} \bar{\theta}_{i}^{\lambda}(x_{i})$$

-- For all F and i $\in$ F, if $x_{Fi}^{*} \neq x_{i}^{*}$, then
$$\lambda_{Fi}(x_{i}^{*}) := \lambda_{Fi}(x_{i}^{*}) - \alpha_{t}, \quad \lambda_{Fi}(x_{Fi}^{*}) := \lambda_{Fi}(x_{Fi}^{*}) + \alpha_{t}$$

where $\alpha_{t} > 0$<br><br>

<b>Dual Decomposition Convergence</b><br>
$\bigstar$ Under weak conditions on $\alpha_{t}$, the $\lambda$ are guaranteed to converge<br>
-- $\sum_{t} \alpha_{t} = \infty$<br>
-- $\sum_{t} \alpha_{t}^{2} < \infty$<br>
$\bigstar$ Convergence is to a unique global optimum, regardless of initialization<br><br>

<b>At Convergence</b><br>
$\bigstar$ Each slave has a locally optimal solution over its own variables<br>
$\bigstar$ Solutions may not agree on shared variables<br>
$\bigstar$ If all salves agree, the shared solution is a <b>guaranteed MAP assignement</b><br>
$\bigstar$ Otherwise, we need to solve the decoding problem to construct a joint assignment<br><br>

<b>Options for Decoding $x^{*}$</b><br>
$\bigstar$ Several heuristics<br>
-- If we use decomposition into spanning trees, can take MAP solution of any tree<br>
-- Have each slave vote on $X_{i}$'s in its scope & for each $X_{i}$ pick value with most votes<br>
-- Weighted average of sequence of message sent regarding each $X_{i}$<br>
$\bigstar$ Score $\theta$ is easy to evaluate<br>
$\bigstar$ Best to generate many candidates and pick the one with highest score<br><br>

<b>Upper Bound</b><br>
$\bigstar$ $L(\lambda)$ is upper bound on MAP($\theta$)
$$score(x) \leq MAP(\theta) \leq L(\lambda)$$

$$MAP(\theta) - score(x) \leq L(\lambda) - score(x)$$

We hope $L(\lambda) - score(x)$ is small enough.<br><br>

<b>Important Design Choices</b><br>
$\bigstar$ Division of problem into slaves<br>
-- Larger slaves (with more factors) improve convergence and often quality of answers<br>
$\bigstar$ Selecting locally optimal solutions for slaves<br>
-- Try to move toward faster agreement<br>
$\bigstar$ Adjusting the step size $\alpha_{t}$<br>
$\bigstar$ Methods to construct candidate solutions<br><br>

<b>Summary: Algorithm</b><br>
$\bigstar$ Dual decomposition is general-purpose algorithm for MAP inference<br>
-- Divide model into tractabel components<br>
-- Solve each one locally<br>
-- Passes 'message' to induce them to agree<br>
$\bigstar$ Any tractable MAP subclass can be used in this setting<br><br>

<b>Summary: Theory</b><br>
$\bigstar$ Formally: a subgradient optimization algorithm on dual problem to MAP<br>
$\bigstar$ Provides important guarantees<br>
-- Upper bound on distance to MAP<br>
-- Conditions that guarantee exact MAP solution<br>
$\bigstar$ Even some analysis for which decomposition into slaves is better<br><br>

<b>Summary: Practice</b><br>
$\bigstar$ Pros<br>
-- Very general purpose<br>
-- Best theoretical guarantees<br>
-- Can use very fast, specialized MAP subroutines for solving large model components<br>
$\bigstar$ Cons<br>
-- Not the fastest algorithm<br>
-- Lots of tunable parameters / design choices<br><br>
</p>

### 2.8 Sampling Methods
#### 2.8.1 Simple Sampling
<p align="justify">
<b>Sampling-Based Estimation</b><br>
We have a dataset D = {x[1], ..., x[M]} which is sampled IID (independent identically distributed) from P. Take toss for an example, P(X = 1) = p. Now we want to estimate p, so estimator for p:
$$T_{D} = \frac{1}{M} \sum_{m=1}^{M} x[m]$$

More generally, for any distribution P, function f:
$$E_{P}[f] \approx \frac{1}{M} \sum_{m=1}^{M} f(x[m])$$

Function f is on sample D, $E_{P}$ is called empirical expectation.<br><br>

<b>Sampling from Discrete Distribution</b><br>
Consider k values for X and each x has a probability
$$Val(X) = \{ x^{1}, \cdots, x^{k} \}, \quad P(x^{i}) = \theta^{i}$$

We can put the probabilities into an interval of [0, 1].
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_1_1.png"/></center>
</p>
<p align="justify">
So, each sub-interval corresponds to a value of x. For example, if we generate a value of probability randomly between $\theta^{1}+\theta^{2}$ and $\theta^{1}+\theta^{2}+\theta^{3}$, we will assign $x^{3}$ to this random point.<br><br>

Recall our estimator
$$T_{D} = \frac{1}{M} \sum_{m=1}^{M} x[m]$$

<b>Hoeffding bound</b><br>
$$P_{D}(T_{D} \notin [p-\epsilon, p+\epsilon]) \leq 2e^{-2M\epsilon^{2}}$$

For additive bound $\epsilon$ on error with probability > $1 - \delta$
$$P_{D}(T_{D} \notin [p-\epsilon, p+\epsilon]) \leq 2e^{-2M\epsilon^{2}} < \delta \rightarrow M \geq \frac{1}{2\epsilon^{2}}\ln\frac{2}{\delta}$$

<b>Chernoff bound</b><br>
$$P_{D}(T_{D} \notin [p(1-\epsilon), p(1+\epsilon)]) \leq 2e^{-\frac{Mp\epsilon^{2}}{3}}$$

For multiplicative bound $\epsilon$ on error with probability > $1 - \delta$
$$M \geq \frac{3}{p\epsilon^{2}}\ln\frac{2}{\delta}$$

<b>Forward Sampling from a BN</b><br>
Consider a Bayesian network
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_1_2.png"/></center>
</p>
<p align="justify">
We can generate a series of samples
\begin{matrix}
d^{0} & i^{1} & g^{1} & s^{0} & l^{1}\\
 & & & &\\
d^{1} & i^{1} & g^{2} & s^{1} & l^{0}\\
 & & \vdots & &
\end{matrix}

Forward means a topological order, namely, sampling parents before children.<br><br>

<b>Forward Sampling for Querying</b><br>
$\bigstar$ Goal: Estimate P(Y = y)<br>
-- Generate samples from BN<br>
-- Compute fraction where Y = y<br><br>

<b>Queries with Evidence</b><br>
$\bigstar$ Goal: Estimate $P(Y=y \mid E=e)$<br>
$\bigstar$ Rejection sampling algorithm<br>
-- Generate samples from BN<br>
-- Throw away all those where $E \neq e$<br>
-- Compute fraction where Y = y<br><br>

Expected fraction of samples kept ~ P(e)<br>

<b># samples needed grows exponentially with # of observed variables</b>, so this cost much.<br><br>

<b>Summary</b><br>
$\bigstar$ Generating samples from a BN is easy<br>
$\bigstar$ $(\epsilon, \delta)$-bounds exist, but usefulness is limited:<br>
-- Additive bounds: useless for low probability events<br>
-- Multiplicative bounds: # of samples grows as $\frac{1}{P(y)}$<br>
$\bigstar$ With evidence, # of required samples grows exponentially with # of observed variables<br>
$\bigstar$ <b>Forward sampling is generally infeasible for MNs</b><br><br>
</p>

#### 2.8.2 Markov Chain Monte Carlo
<p align="justify">
<b>Markov Chain</b><br>
$\bigstar$ A Markov chain defines a probabilistic transition model T(x $\rightarrow$ x') over states x:<br>
-- for all x:
$$\sum_{x'} T(x \rightarrow x') = 1$$

For example,
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_2_1.png"/></center>
</p>
<p align="justify">
The current state is 0, the next state is one of {-1, 0, 1} with a probability {0.25, 0.5, 0.25} respectively.<br><br>

<b>Temporal Dynamics</b><br>
Given a current state, we can calculate a probability for next state
$$P^{(t+1)}(X^{(t+1)} = x') = \sum_{x} P^{(t)}(X^{(t)} = x)T(x \rightarrow x')$$

For example<br>
<table class="a">
  <tr><th></th><th>-2</th><th>-1</th><th>0</th><th>1</th><th>2</th></tr>
  <tr><td>$P^{(0)}$</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr>
  <tr><td>$P^{(1)}$</td><td>0</td><td>0.25</td><td>0.5</td><td>0.25</td><td>0</td></tr>
  <tr><td>$P^{(2)}$</td><td>0.0625</td><td>0.25</td><td>0.375</td><td>0.25</td><td>0.0625</td></tr>
</table><br>
</p>
<p align="justify">
For t = 0,
$$P^{(0)}(X^{(0)} = 0) = 1$$

For t = 1
$$P^{(1)}(X^{(1)} = 0)$$
$$= P^{(0)}(X^{(0)} = -1)T(-1 \rightarrow 0) + P^{(0)}(X^{(0)} = 0)T(0 \rightarrow 0) + P^{(0)}(X^{(0)} = 1)T(1 \rightarrow 0)$$
$$= 0 + 1 \cdot 0.5 + 0 = 0.5$$

$$P^{(1)}(X^{(1)} = 1)$$
$$= P^{(0)}(X^{(0)} = 0)T(0 \rightarrow 1) + P^{(0)}(X^{(0)} = 1)T(1 \rightarrow 1) + P^{(0)}(X^{(0)} = 2)T(2 \rightarrow 1)$$
$$= 1 \times 0.25 + 0 + 0 = 0.25$$

For t = 2
$$P^{(2)}(X^{(2)} = 0)$$
$$= P^{(1)}(X^{(1)} = 0)T(0 \rightarrow 0) + P^{(1)}(X^{(1)} = 1)T(1 \rightarrow 0) + P^{(1)}(X^{(1)} = -1)T(-1 \rightarrow 0)$$
$$= 0.5 \cdot 0.5 + 2 \times 0.25 \times 0.25 = 0.375$$

$$P^{(2)}(X^{(2)} = 1)$$
$$= P^{(1)}(X^{(1)} = 0)T(0 \rightarrow 1) + P^{(1)}(X^{(1)} = 1)T(1 \rightarrow 1) + P^{(1)}(X^{(1)} = 2)T(2 \rightarrow 1)$$
$$= 0.5 \cdot 0.25 + 0.25 \times 0.5 + 0 = 0.25$$

$$P^{(2)}(X^{(2)} = 2)$$
$$= P^{(1)}(X^{(1)} = 1)T(1 \rightarrow 2) + P^{(1)}(X^{(1)} = 2)T(2 \rightarrow 2) + P^{(1)}(X^{(1)} = 3)T(3 \rightarrow 2)$$
$$= 0.25 \cdot 0.25 + 0 + 0 = 0.0625$$

<b>Stationary Distribution</b><br>
$$P^{(t)}(x') \approx P^{(t+1)}(x') = \sum_{x} P^{(t)}(x)T(x \rightarrow x')$$

We use $\pi(x)$ to denote a stationary distribution
$$\pi(x') = \sum_{x} \pi(x)T(x \rightarrow x')$$

For example
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_2_2.png"/></center>
</p>
<p align="justify">
We can write a stationary distribution for each variable
$$
\left\{\begin{matrix}
\pi(x^{1}) = 0.25 \cdot \pi(x^{1}) + 0.5 \cdot \pi(x^{3})\\
\pi(x^{2}) = 0.7 \cdot \pi(x^{2}) + 0.5 \cdot \pi(x^{3})\\
\pi(x^{3}) = 0.75 \cdot \pi(x^{1}) + 0.3 \cdot \pi(x^{2})
\end{matrix}\right.
$$

Besides,
$$\pi(x^{1}) + \pi(x^{2}) + \pi(x^{3}) = 1$$

So, we have a solution
$$
\left\{\begin{matrix}
\pi(x^{1}) = 0.2\\
\pi(x^{2}) = 0.5\\
\pi(x^{3}) = 0.3
\end{matrix}\right.
$$

<b>Regular Markov Chains</b><br>
$\bigstar$ A Markov chain is regular if there exists k such that, for every pair (x, x'), the probability of getting from x to x' in exactly k steps is > 0<br>
-- which means we take a k first and we cannot take different k for different pair<br>
$\bigstar$ Theorem: A regular Markov chain converges to a unique stationary distribution regardless of start state<br>
-- regularity is sufficient not necessary<br>
$\bigstar$ Sufficient conditions for regularity:<br>
-- Every two states are connected with a positive transition probability<br>
-- For every state, there is a self-transition<br><br>

<b>Summary</b><br>
$\bigstar$ A Markov chain defines a dynamical system from which we can sample trajectories<br>
$\bigstar$ Under certain conditions (e.g. regularity), this process is guaranteed to converge to a stationary distribution at the limit<br>
$\bigstar$ This allows us to sample from a distribution indirectly and thereby provides a mechanism for sampling from an intractable distribution<br><br>
</p>

#### 2.8.3 Using a Markov Chain
<p align="justify">
<b>Using a Markov Chain</b><br>
$\bigstar$ Goal: compute $P(x \in S)$<br>
-- but P is too hard to sample from directly<br>
$\bigstar$ Construct a (regular) Markov chain T whose unique stationary distribution is P<br>
$\bigstar$ Sample $x^{0}$ from some $P^{0}$<br>
$\bigstar$ For t = 0, 1, 2, ...<br>
-- Generate $x^{t+1}$ from $T(x^{t} \rightarrow x')$<br><br>

$\bigstar$ We only want to use samples that are sampled from a distribution close to P<br>
$\bigstar$ At early iterations, $P^{t}$ is usually far from P<br>
$\bigstar$ Start collecting samples only after the chain has run long enough to 'mix' ($P^{t}$ close to $\pi$)<br><br>

<b>Mixing</b><br>
$\bigstar$ How do you know if a chain has mixed or not?<br>
-- In general, you can never 'prove' a chain has mixed<br>
-- But in many cases you can show that it has not<br>
$\bigstar$ How do you know a chain has not mixed?<br>
-- Conmpute chain statistics in different windows wthin a sample run of the chain<br>
-- and across different runs initialized differently<br><br>

<b>Using the Samples</b><br>
$\bigstar$ Once the chain mixes, all samples $x^{(t)}$ are from the stationary distribution $\pi$.<br>
-- so we can (and should) use all $x^{t}$ for t > $T_{min}$<br>
$\bigstar$ However, nearby samples are correlated<br>
-- so we shouldn't overestimate the quality of our estimate by simply counting samples. (not iid)<br>
$\bigstar$ The faster a chain mixes, the less correlated (more useful) the samples<br><br>

<b>MCMC Algorithm</b><br>
For c = 1, ..., C<br>
&nbsp;&nbsp;&nbsp;&nbsp;Sample $x^{c, 0}$ from $P^{0}$<br>
Repeat until mixing<br>
&nbsp;&nbsp;&nbsp;&nbsp;For c = 1, ..., C<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generate $x^{c, t+1}$ from $T(x^{c, t} \rightarrow x')$<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute window statistics in different chains to determine mixing<br>
&nbsp;&nbsp;&nbsp;&nbsp;t = t + 1<br>
Repeat until sufficient samples<br>
&nbsp;&nbsp;&nbsp;&nbsp;D := $\varnothing$<br>
&nbsp;&nbsp;&nbsp;&nbsp;For c = 1, ..., C<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generate $x^{c, t+1}$ from $T(x^{c, t} \rightarrow x')$<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$D := D \cup \{ x^{(c, t+1)} \}$<br>
&nbsp;&nbsp;&nbsp;&nbsp;t = t + 1<br>
Let D = {x[1], ..., x[M]}<br>
&nbsp;&nbsp;&nbsp;&nbsp;$E_{P}[f] \approx \frac{1}{M} \sum_{m=1}^{M} f(x[m])$<br><br>

<b>Summary</b><br>
$\bigstar$ Pros:<br>
-- Very general purpose<br>
-- Often easy to implement<br>
-- Good theoretical guarantees as $t \rightarrow \infty$<br>
$\bigstar$ Cons:<br>
-- Lots of tunable parameters / design choices<br>
-- Can be quite slow to converge<br>
-- Difficult to tell whether it's working<br><br>
</p>

#### 2.8.4 Gibbs Sampling
<p align="justify">
<b>Gibbs Chain</b><br>
$\bigstar$ Target distribution $P_{\Phi}(X_{1}, \cdots, X_{n})$<br>
$\bigstar$ Markov chain state sapce: complete assignment x to X = {$X_{1}, \cdots, X_{n}$}<br>
$\bigstar$ Transition model given starting state x:<br>
-- For i = 1, ..., n: Sample $x_{i} \sim P_{\Phi}(X_{i} \mid x_{-i})$, where $x_{-i}$ means assignment to all {$X_{1}, \cdots, X_{n}$} except $X_{i}$<br>
-- Set x' = x<br><br>

For example, we have 3 binary variables and their initial values are {0, 0, 0} and the sample order is 1, 2, 3.<br>
<table class="a">
  <tr><th>$X_{1}$</th><th>$X_{2}$</th><th>$X_{3}$</th><th>Prob</th></tr>
  <tr><td>0</td><td>0</td><td>0</td><td>$x_{1} \sim P(x_{1} \mid x_{2}=0, x_{3}=0)$</td></tr>
  <tr><td>1</td><td>0</td><td>0</td><td>$x_{2} \sim P(x_{2} \mid x_{1}=1, x_{3}=0)$</td></tr>
  <tr><td>1</td><td>0</td><td>0</td><td>$x_{3} \sim P(x_{3} \mid x_{1}=1, x_{2}=0)$</td></tr>
  <tr><td>1</td><td>0</td><td>0</td><td></td></tr>
</table><br>
</p>
<p align="justify">
After sampling $x_{1}$, we get a assignement $x_{1} = 1$. After the whole sample, we get a new x' = {1, 0, 0} to replace x.<br><br>

Another example, a Bayesian network with $L = l^{0}$ and $S = s^{1}$ observed.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_4_1.png"/></center>
</p>
<p align="justify">
Our factor product
$$\widetilde{P}_{\Phi}(D, I, G \mid l^{0}, s^{1})$$

Suppose our initial assignment (usually arbitrary)
$$d^{0}, i^{0}, g^{0}$$

So, we sample D
$$P(D \mid i^{0}, g^{0}, l^{0}, s^{1}) \rightarrow D = d^{1} \rightarrow d^{1}, i^{0}, g^{0}$$

We sample I
$$P(I \mid d^{1}, g^{0}, l^{0}, s^{1}) \rightarrow I = i^{1} \rightarrow d^{1}, i^{1}, g^{0}$$

We sample G
$$P(G \mid d^{1}, i^{1}, l^{0}, s^{1}) \rightarrow G = g^{3} \rightarrow d^{1}, i^{1}, g^{3}$$

Finally, we have a new assignment {$d^{1}, i^{1}, g^{3}$}.<br><br>

<b>Computational Cost</b><br>
$\bigstar$ For i = 1, ..., n: Sample $x_{i} \sim P_{\Phi}(X_{i} \mid x_{-i})$
$$P_{\Phi}(X_{i} \mid x_{-i}) = \frac{P_{\Phi}(X_{i}, x_{-i})}{P_{\Phi}(x_{-i})} = \frac{\frac{1}{Z}\widetilde{P}_{\Phi}(X_{i}, x_{-i})}{\frac{1}{Z}\widetilde{P}_{\Phi}(x_{-i})} = \frac{\widetilde{P}_{\Phi}(X_{i}, x_{-i})}{\widetilde{P}_{\Phi}(x_{-i})}$$

<b>How do we compute $\frac{\widetilde{P}_{\Phi}(X_{i}, x_{-i})}{\widetilde{P}_{\Phi}(x_{-i})}$?</b><br>
$\bigstar$ Sum the numerator over $X_{i}$ to get the denominator<br>
$\bigstar$ Multiply all the factors together to get the numerator<br>
The chain rule allows us to compute the numerator by simply multiplying all factors together (operations are linear in the number of factors). We can get the denominator by simply summing out $X_{i}$ from the numerator (which is linear in the number of values of $X_{i}$. Therefore it's always tractable.<br><br>

For example, we sample A based a Markov network
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_4_2.png"/></center>
</p>
<p align="justify">
We set A' in order to distinguish A in numerator because of different assignments for A.<br><br>

Besides, we can reduce $\phi_{2}(b, c)\phi_{3}(c, d)$ for both numerator and denominator to reduce computational cost.
$$\frac{\phi_{1}(A, b)\phi_{4}(A, d)}{\sum_{A'}\phi_{1}(A', b)\phi_{4}(A', d)}$$

The numerator $\phi_{1}(A, b)\phi_{4}(A, d)$ denotes factors involving A and denominator means a normalizing constant $\propto \phi_{1}(A, b)\phi_{4}(A, d)$.<br><br>

<b>Computational Cost Revisited</b><br>
$\bigstar$ For i = 1, ..., n: Sample $x_{i} \sim P_{\Phi}(X_{i} \mid x_{-i})$
$$P_{\Phi}(X_{i} \mid x_{-i}) = \frac{P_{\Phi}(X_{i}, x_{-i})}{P_{\Phi}(x_{-i})} = \frac{\frac{1}{Z}\widetilde{P}_{\Phi}(X_{i}, x_{-i})}{\frac{1}{Z}\widetilde{P}_{\Phi}(x_{-i})} = \frac{\widetilde{P}_{\Phi}(X_{i}, x_{-i})}{\widetilde{P}_{\Phi}(x_{-i})} \propto \prod_{j: X_{i} \in Scope[C_{j}]} \phi_{j}(X_{i}, x_{j, -i})$$

which means we only care about $X_{i}$ and its neighbors in some a graph.<br><br>

<b>Gibbs Chain and Regularity</b><br>
Consider a deterministic model with XOR. Imagine y = 1 is observed
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_4_3.png"/></center>
</p>
<p align="justify">
Suppose our initial state is $X_{1} = 0, X_{2} = 1$. So. we sample $X_{1}$ given $X_{2}$ and Y
$$P(X_{1} \mid X_{2} = 1, Y = y)$$

Because of deterministic dependence, the only possible value for $X_{1}$ is 0. Similarly, the only value for $X_{2}$ is 1. Therefore, no matter how many samples, we can only have one assignment. This is a classical example of non-maxing chain.<br><br>

$\bigstar$ If all factors are positive, Gibbs chain is regular.<br>
$\bigstar$ However, mixing can still be very slow.<br><br>

<b>Summary</b><br>
$\bigstar$ Converts the hard problem of inference to a sequence of 'easy' sampling steps.<br>
$\bigstar$ Pros:<br>
-- Probably the simplest Markov chain for PGMs<br>
-- Computationally efficient to sample<br>
$\bigstar$ Cons:<br>
-- Often slow to mix, especially when probabilities are peaked<br>
-- Only applies if we can sample from product of factors<br><br>
</p>

#### 2.8.5 Metropolis Hastings Algorithm
<p align="justify">
Markov chain Monte Carlo sampling is a general paradigm for generating samples from distribution through which it is otherwise difficult to perhaps interactable to generate samples directly. Markov chains gives us the general way of of approaching this problem, but the framework leaves open the question of where the Markov chain comes from. That is how do we design a Markov chain that has the desired stationary distribution. Gibbs chain is a general solution to this problem in the context of graphical models, but Gibbs chain has limitations, in terms of its convergence rates for certain types of graphical models. So what happens if we have a graphical model for which the Gibbs chain doesn't have good convergence properties? How do we design a Markov change for that? The answer is Metropolis Hastings Algorithm<br><br>

<b>Revsersible Chain</b><br>
$$\pi(x)T(x \rightarrow x') = \pi(x')T(x' \rightarrow x)$$

<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_5_1.png"/></center>
</p>
<p align="justify">
which means a probability of transition i to j is equal to that j to i. We call this <b>detailed balance</b>.<br><br>

$\bigstar$ Theorem: If detailed balanced holds and T is regular Markov chain, then T has a unique stationary distribution $\pi$.<br>
$\bigstar$ Proof:
$$\sum_{x}\pi(x)T(x \rightarrow x') = \sum_{x} \pi(x')T(x' \rightarrow x) = \pi(x') \sum_{x} T(x' \rightarrow x) = \pi(x')$$

This is exactly the definition of stationary distribution. $\sum_{x} T(x' \rightarrow x) = 1$ means all probabilities from x' to its neighbors sum up to 1.<br><br>

<b>Metropolis Hastings Chain</b><br>
$\bigstar$ Proposal distribution $Q(x \rightarrow x')$<br>
-- a transition model<br>
-- x can be far away from x'<br>
$\bigstar$ Acceptance probability $A(x \rightarrow x')$<br>
-- a credit for checking if x to x' is reasonable<br><br>

At each state x, sample x' from $Q(x \rightarrow x')$<br>
Accept proposal with probability $A(x \rightarrow x')$<br>
&nbsp;&nbsp;&nbsp;&nbsp;If proposal accepted, move to x'<br>
&nbsp;&nbsp;&nbsp;&nbsp;Otherwise stay at x<br>
$$
\begin{cases}
  T(x \rightarrow x') = Q(x \rightarrow x')A(x \rightarrow x'), \quad \text{if} \quad x' \neq x \\
  T(x \rightarrow x) = Q(x \rightarrow x) + \sum_{x' \neq x}Q(x \rightarrow x')(1 - A(x \rightarrow x')), \quad \text{if} \quad x' = x
\end{cases}
$$

<b>Acceptance Probability</b><br>
Recall revsersible chain
$$\pi(x)T(x \rightarrow x') = \pi(x')T(x' \rightarrow x)$$

We construct our acceptance probability
$$\pi(x)Q(x \rightarrow x')A(x \rightarrow x') = \pi(x')Q(x' \rightarrow x)A(x' \rightarrow x) \quad x \neq x'$$

$$\frac{A(x \rightarrow x')}{A(x' \rightarrow x)} = \frac{\pi(x')Q(x' \rightarrow x)}{\pi(x)Q(x \rightarrow x')} = \rho$$

$$A(x \rightarrow x') = \min[1, \rho]$$

For example, consider a Markov chain over a state space $s_{1}, \cdots, s_{m}$ and assume we wish to sample from the stationary distribution $\pi(s_{i}) \propto q^{i}$ for q < 1. Let $d(s_{i}, s_{j})$ be the distance between $s_{i}$ and $s_{j}$. Let $Q(s_{i}, s_{j}) \propto p^{-d(s_{i}, s_{j})}$, where p < 1. For j > i, what is the accpetance probability $A(s_{i} \rightarrow s_{j})$ will give rise to a legal MH chain with this stationary distribution?<br><br>

<b>Answer:</b> $q^{j - i}$.<br>
$$A(s_{i} \rightarrow s_{j}) = \min[1, \frac{\pi(s_{j})Q(s_{j} \rightarrow s_{i})}{\pi(s_{i})Q(s_{i} \rightarrow s_{j})}] = \min[1, \frac{q^{j}p^{-d(s_{j}, s_{i})}}{q^{i}p^{-d(s_{i}, s_{j})}}] = \min[1, q^{j-i}]$$

<b>Choice of Q</b><br>
$\bigstar$ Q must be reversible:<br>
-- $Q(x \rightarrow x') > 0 \Leftrightarrow Q(x' \rightarrow x) > 0$<br>
$\bigstar$ Opposing forces<br>
-- Q should try to spread out, to improve mixing<br>
-- But then acceptance probability often low<br><br>

<b>MCMC for Matching</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_5_2.png"/></center>
</p>
<p align="justify">
<b>HM for Matching: Augmenting Path</b><br>
1) randomly pick one variable $X_{i}$<br>
2) sample $X_{i}$, pretending that all values are available<br>
3) pick the variable whose assignment was taken (conflict), and return to step 2.<br>
$\bigstar$ When step 2 creates no conflict, modify assignment to flip augmenting path
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_5_3.png"/></center>
</p>
<p align="justify">
<b>Example Results</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_5_4.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ MH is a general framework for building Markov chains with a particular stationary distribution<br>
-- Requires a proposal distribution<br>
-- Acceptance computed via detailed balance<br>
$\bigstar$ Tremendous flexibility in designing proposal distributions that explore the space quickly<br>
-- But proposal distribution makes a big difference<br>
-- and finding a good one is not always easy<br><br>
</p>

#### 2.8.6 Quiz
<p align="justify">
<b>1. Forward Sampling</b><br>
One strategy for obtaining an estimate to the conditional probability P(y | e) is by using forward sampling to estimate P(y, e) and P(e) separately and then computing the ratio. We can use the Hoeffding Bound to obtain a bound on both the numerator and the denominator. Assume M is large. <b>When does the resulting bound provide meaningful guarantees?</b> Think about the difference between the true value and our estimate. Recall that we need $M \geq \frac{1}{2\epsilon^{2}}\ln\frac{2}{\delta}$ to get an additive error bound $\epsilon$ that holds with probability $1 - \delta$ for our estimate.<br>
A. It provides a meaningful guarantee, but only when $\delta$ is small relative to P(e) and P(y, e)<br>
B. It always provides meaningful guarantees.<br>
C. It provides a meaningful guarantee, but only when $\epsilon$ is small relative to P(e) and P(y, e)<br>
D. It never provides a meaningful guarantee.<br>
<b>Answer:</b> C.<br>
A: This is incorrect because when $\delta$ is small for M samples, $\epsilon$ is large, which means that we are very far from the true value.<br>
C: When $\epsilon$ isn't small with respect to P(y, e) and P(e) the value of the estimated ratio $\frac{P(y, e)}{P(e)}$ can be far from the true value of P(y | e) even if the absolute value of $\epsilon$ and hence the absolute error in estimating P(e) and P(y, e) is small.<br><br>

<b>2. Rejecting Samples</b><br>
Consider the process of rejection sampling to generate samples from the posterior distribution P(X | e). If we want to obtain M samples, <b>what is the expected number of samples that would need to be drawn from P(X)?</b><br>
A. $\frac{M}{1 - P(e)}$<br>
B. $\frac{M}{P(e)}$<br>
C. $M \cdot (1 - P(X \mid e))$<br>
D. $M \cdot P(X \mid e)$<br>
E. $M \cdot (1 - P(e))$<br>
F. $M \cdot P(e)$<br>
<b>Answer:</b> B.<br>
A: Let's say we start with A samples. $A \cdot (1 - P(e))$ will give us the number of samples that we rejected. If $A =  \frac{M}{1 - P(e)}$, then we are not keeping M samples at the end. We are actually keeping A−M samples, since this formula makes M the number of samples we rejected.<br>
C: We want to be able to compute the number of samples that we will reject. The posterior distribution will not give us the samples we will discard.<br><br>

<b>3. Stationary Distributions</b><br>
Consider the simple Markov chain shown in the figure below.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_6_1.png"/></center>
</p>
<p align="justify">
By definition, a stationary distribution π for this chain must satisfy <b>which of the following properties?</b> You may select 1 or more options.<br>
A. $\pi(x_{1}) = \pi(x_{2})  = \pi(x_{3})$<br>
B. $\pi(x_{3}) = 0.3\pi(x_{1}) + 0.7\pi(x_{3})$<br>
C. $\pi(x_{3}) = 0.4\pi(x_{1}) + 0.5\pi(x_{2})$<br>
D. $\pi(x_{1}) = 0.2\pi(x_{1}) + 0.4\pi(x_{2}) + 0.4\pi(x_{3})$<br>
E. $\pi(x_{1})+ \pi(x_{2}) + \pi(x_{3}) = 1$<br>
F. $\pi(x_{1}) = 0.2\pi(x_{1}) + 0.3\pi(x_{3})$<br>
<b>Answer:</b> E, F.<br><br>

<b>4. *Gibbs Sampling in a Bayesian Network</b><br>
Suppose we have the Bayesian network shown in the image below.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_8_6_2.png"/></center>
</p>
<p align="justify">
If we are sampling the variable $X_{23}$ as a substep of Gibbs sampling, <b>what is the closed form equation for the distribution we should use over the value $X'_{23}$?</b> By closed form, we mean that all computation such as summations are tractable and that we have access to all terms without requiring extra computation.<br>
<b>Answer:</b>
$$\frac{P(x'_{23} \mid x_{22}, x_{24})P(x_{15} \mid x'_{23}, x_{14}, x_{9}, x_{25})}{\sum_{x''_{23}} P(x''_{23} \mid x_{22}, x_{24})P(x_{15} \mid x''_{23}, x_{14}, x_{9}, x_{25})}$$

<b>5. Gibbs Sampling</b><br>
Suppose we are running the Gibbs sampling algorithm on the Bayesian network $X \rightarrow Y \rightarrow Z$. If the current sample is $(x_{0}, y_{0}, z_{0})$ and we sample y as the first substep of the Gibbs sampling process, with <b>what probability will the next sample be $(x_{0}, y_{0}, z_{0})$ in the first substep?</b><br>
A. $P(y_{1} \mid x_{0}, z_{0})$<br>
B. $P(y_{1} \mid x_{0})$<br>
C. $P(x_{0}, y_{1}, z_{0})$<br>
D. $P(x_{0}, z_{0} \mid y_{1})$<br>
<b>Answer:</b> A.<br><br>

<b>6. Collecting Samples</b><br>
Assume we have a Markov chain that we have run for a sufficient burn-in time, and now wish to collect samples and use them to estimate the probability that $X_{i} = 1$. <b>Can we collect and use every sample from the Markov chain after the burn-in?</b><br>
A. No, once we collect one sample, we have to continue running the chain in order to "re-mix" it before we get another sample.<br>
B. Yes, and if we collect m consecutive samples, we can use the Hoeffding bound to provide (high-probability) bounds on the error in our estimated probability.<br>
C. No, Markov chains are only good for one sample; we have to restart the chain (and burn-in) before we can collect another sample.<br>
D. Yes, that would give a correct estimate of the probability. However, we cannot apply the Hoeffding bound to estimate the error in our estimate.<br>
<b>Answer:</b> D.<br>
Once the chain has mixed, we can collect samples continuously. The samples won't be independent, but they will still provide a correct estimate of the marginal probability.<br><br>

<b>7. Markov Chain Mixing</b><br>
<b>Which of the following classes of chains would you expect to have the shortest mixing time in general?</b><br>
A. Markov chains with many distinct and peaked probability modes.<br>
B. Markov chains for networks with nearly deterministic potentials.<br>
C. Markov chains with distinct regions in the state space that are connected by low probability transitions.<br>
D. Markov chains where state spaces are well connected and transitions between states have high probabilities.<br>
<b>Answer:</b> D.<br>
D: This is correct because if you are able to move around the state space, you are more likely to mix in quickly.
</p>

### 2.9 Inference in Temporal Models
#### 2.9.1 Inference in Temporal Models
<p align="justify">
<b>DBN Template Specification</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_9_1_1.png"/></center>
</p>
<p align="justify">
<b>Ground Bayesian Network</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_9_1_2.png"/></center>
</p>
<p align="justify">
$\bigstar$ Can unroll DBN for given trajectory and run inference over ground network.<br><br>

<b>Plate Model</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_9_1_3.png"/></center>
</p>
<p align="justify">
$\bigstar$ Can unroll plate model for given set of objects and run inference over ground network.<br><br>

<b>Belief State Tracking</b><br>
Belief state tracking is keeping track over the state of the system as it evolves.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_9_1_4.png"/></center>
</p>
<p align="justify">
This is a traditional probabilistic inference task because it corresponds to asking what is our probability distribution over the state of time t, given the observations that the agent has had access to up to time t. 
$$\sigma^{(t)}(S^{(t)}) = P(S^{(t)} \mid \mathbf{o}^{(1:t)})$$

where $\mathbf{o}^{(1:t)}$ denotes all observations from 1 to t. Besides, we want to compute a probability over $S^{t+1}$ given all observations from 1 to t without observation at t+1.
$$\sigma^{(\cdot t+1)}(S^{(t+1)}) = P(S^{(t+1)} \mid \mathbf{o}^{(1:t)})$$

$$= \sum_{S^{(t)}} P(S^{(t+1)} \mid S^{(t)}, \mathbf{o}^{(1:t)})P(S^{(t)} \mid \mathbf{o}^{(1:t)})$$

In fact, $S^{(t+1)}$ and $\boldsymbol{o}^{(1:t)}$ are independent given $S^{(t)}$: $P(S^{(t+1)} \mid S^{(t)}, \mathbf{o}^{(1:t)}) = P(S^{(t+1)} \mid S^{(t)})$
$$= \sum_{S^{(t)}} P(S^{(t+1)} \mid S^{(t)})\sigma^{(t)}(S^{(t)})$$

Now we want to compute $\sigma^{(t+1)}(S^{(t+1)})$
$$\sigma^{(t+1)}(S^{(t+1)}) = P(S^{(t+1)} \mid \mathbf{o}^{(1:t)}, \boldsymbol{o}^{(t+1)})$$

We apply Bayes' rule
$$= \frac{P(\boldsymbol{o}^{t+1} \mid S^{(t+1)}, \mathbf{o}^{(1:t)}) P(S^{(t+1)} \mid \mathbf{o}^{(1:t)})}{P(\mathbf{o}^{(t+1)} \mid \mathbf{o}^{(1:t)})}$$

Once again, $\boldsymbol{o}^{t+1}$ and $\mathbf{o}^{(1:t)}$ are independent given $S^{(t+1)}$.
$$= \frac{P(\boldsymbol{o}^{t+1} \mid S^{(t+1)}) \sigma^{(\cdot t+1)}(S^{(t+1)})}{P(\mathbf{o}^{(t+1)} \mid \mathbf{o}^{(1:t)})}$$

The denominator is normalizing constants, which can be derived by computing the numerator then normalizing.<br><br>

<b>Computational Issues</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_9_1_5.png"/></center>
</p>
<p align="justify">
$\bigstar$ Minimal sepset must seperate future from past $\Rightarrow$ must involve at least all of the persistent variables<br><br>

<b>What is the set of variables that separates t = 0 from t = 2?</b><br><br>

<b>Answer:</b> Weather, Velocity, Location, Failure.<br><br>

<b>Entanglement</b><br>
We're thinking of belief state tracking.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_9_1_6.png"/></center>
</p>
<p align="justify">
Our goal is to maintain a belief state over t=3, and we're trying to think about how can we maintain this probability distribution $\sigma^{(t=3)}$ in a way that doesn't involve maintaining a full explicit joint distribution over the variables t=3.<br><br>

We quickly realize that we have no choice, because there are no conditional independencies within t=3. Consider two variables Weather and Failure at t=3, we can find an active trial between them, e.g. Weather t=3 $\rightarrow$ Weather t=2 $\rightarrow$ Weather t=1 $\rightarrow$ Failure t=2 $\rightarrow$ Failure t=3. In other word, variables Weather and Failure at t=3 are not conditionally independent of each other.<br><br>

This entanglement process occurs very rapidly over the course of tracking a belief state in a dynamic Bayesian network, which eventually means that the belief state, if we want to maintain the exact belief state, is just fully correlated, in most cases. This is a computational consequence of the fact that we have a very large Bayesian network.<br><br>

<b>Summary</b><br>
$\bigstar$ Inference in template and temporal models can be done by unrolling the ground network and using standard methods<br>
$\bigstar$ Temporal models also raise new inference tasks, such as real-time tracking, which require that we adapt our methods<br>
$\bigstar$ Moreover, ground network is often large and densely connected, requiring careful algorithm design and use of approximate methods<br><br>
</p>

#### 2.9.2 Quiz
<p align="justify">
<b>1. Unrolling DBNs</b><br>
<b>Which independencies hold in the unrolled network for the following 2-TBN for all t?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_9_2_1.png"/></center>
</p>
<p align="justify">
(Hint: it may be helpful to draw the unrolled DBN for several slices)<br>
A. $(Weather^{t} \perp Location^{t} \mid Velocity^{t}, Obs^{1, \cdots, t})$<br>
B. $(Failure^{t} \perp Location^{t} \mid Obs^{1, \cdots, t})$<br>
C. $(Weather^{t} \perp Velocity^{t} \mid Weather^{t-1}, Obs^{1, \cdots, t})$<br>
D. $(Weather^{t} \perp Velocity^{t} \mid Obs^{1, \cdots, t})$<br>
E. $(Failure^{t} \perp Velocity^{t} \mid Obs^{1, \cdots, t})$<br>
<b>Answer:</b> C.<br><br>

<b>2. *Limitations of Inference in DBNs</b><br>
<b>What makes inference in DBNs difficult?</b><br>
A. Standard clique tree inference cannot be applied to a DBN<br>
B. As t grows large, we generally lose all independencies in the ground network<br>
C. As t grows large, we generally lose independencies of the form $(X^{t} \perp Y^{t} \mid Z^{t})$<br>
D. In many networks, maintaining an exact belief state over the variables requires a full joint distribution over all variables in each time slice<br>
<b>Answer:</b> C, D.<br><br>

<b>3. Entanglement in DBNs</b><br>
<b>Which of the following are consequences of entanglement in Dynamic Bayesian Networks over discrete variables?</b><br>
A. The size of an exact representation of the belief state is exponentially large in the number of variables.<br>
B. The belief state factorizes in the unrolled DBN if the belief state factorizes in the 2-TBN for the DBN.<br>
C. All variables in the unrolled DBN become correlated.<br>
D. The size of an exact representation of the belief state is quadratic in the number of variables.<br>
<b>Answer:</b> A.<br>
A: This is true, since the only way to represent the belief state exactly is to maintain a full joint distribution.<br>
B: This is not a consequence of entanglement. In fact, even if the belief state factorizes in the 2-TBN for the DBN, it is unlikely to factorize in the unrolled network due to entanglement.<br>
C: This is not true; only variables in the belief state become correlated.<br>
D: This is generally not true, unless we represent the belief state using a Gaussian distribution; a Gaussian distribution is completely determined by the mean vector and covariance matrix, the latter having a size that is quadratic in the number of variables.<br><br>
</p>

### 2.10 Inference Summary
#### 2.10.1 Inference: Summary
<p align="justify">
<b>Marginals vs MAP</b><br>
$\bigstar$ Marginals<br>
-- less fragile<br>
-- Confidence in answers<br>
-- Support decision making<br>
-- For approximate inference, errors are often attenuated<br>
$\bigstar$ MAP<br>
-- Coherent joint assignment<br>
-- More tractable model classes<br>
-- Some theoretical guarantees<br>
-- For approximate inference, ability to gauge whether algorithm is working<br><br>

<b>Algorithms for Marginals</b><br>
$\bigstar$ Exact inference<br>
-- variable elimination in clique tree<br>
-- problem is small enough that exact inference fits in memory<br>
$\bigstar$ Loopy message passing<br>
$\bigstar$ Sampling methods<br><br>

<b>What kind of algorithm is message passing in a clique tree?</b><br>
A. Approximate if all the sepsets are strict subsets of the intersections of the variables in the connected cliques.<br>
B. Exact if and only if the clique tree is a poly tree.<br>
C. Exact<br>
D. Approximate if and only if the clique tree is a poly tree.<br>
<b>Answer:</b> C.<br>
Recall that in a clique tree, message passing converges to the correct marginals after upward and downward passes.<br><br>

<b>Algorithms for MAP</b><br>
$\bigstar$ Exact inference<br>
$\bigstar$ Optimization methods<br>
-- dual decomposition<br>
$\bigstar$ Search-based methods (including MCMC)
-- hill-climbing<br><br>

<b>Factors in Approximate Inference</b><br>
$\bigstar$ Connectivity structure<br>
$\bigstar$ Strength of influence<br>
$\bigstar$ Opposing influences<br>
$\bigstar$ Multiple peaks in likelihood<br>
-- Hill climbing search algorithms that optimize the likelihood function will be particularly vulnerable to factors.<br><br>

<b>So, now what?</b><br>
$\bigstar$ Indentify 'problem regions' in network<br>
$\bigstar$ Try to make inference in these regions more exact<br>
-- Large clusters in cluster graph<br>
-- Proposal moves over multilple variables<br>
-- Large 'slave' in dual decomposition<br><br>
</p>

#### 2.10.2 Quiz
<p align="justify">
<b>1. Reparameterization</b><br>
Suppose we have a calibrated clique tree T and calibrated cluster graph G for the same Markov network, and have thrown away the original factors. Now we wish to reconstruct the joint distribution over all the variables in the network only from the beliefs and sepsets. <b>Is it possible for us to do so from the beliefs and sepsets in T? Separately, is it possible for us to do so from the beliefs and sepsets in G?</b><br>
A. It is possible in T but not in G<br>
B. It is possible in both T and G<br>
C. It is possible in T or G<br>
D. It is possible in G but not in T<br>
<b>Answer:</b> B.<br><br>

<b>2. *Markov Network Construction</b><br>
Consider the unrolled network for the plate model shown below, where we have n students and m courses. Assume that we have observed the grade of all students in all courses. In general, what does a pairwise Markov network that is a minimal I-map for the conditional distribution look like? (Hint: the factors in the network are the CPDs reduced by the observed grades. We are interested in modeling the conditional distribution, so we do not need to explicitly include the Grade variables in this new network. Instead, we model their effect by appropriately choosing the factor values in the new network.)
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_10_2_1.png"/></center>
</p>
<p align="justify">
A. A fully connected graph with instantiations of the Difficulty and Intelligence variables.<br>
B. Impossible to tell without more information on the exact grades observed.<br>
C. A fully connected bipartite graph where instantiations of the Difficulty variables are on one side and instantiations of the Intelligence variables are on the other side.<br>
D. A graph over instantiations of the Difficulty variables and instantiations of the Intelligence variables, not necessarily bipartite; there could be edges between different Difficulty variables, and there could also be edges between different Intelligence variables.<br>
E. A bipartite graph where instantiations of the Difficulty variables are on one side and instantiations of the Intelligence variables are on the other side. In general, this graph will not be fully connected.<br>
<b>Answer:</b> C.<br><br>

<b>3. **Clique Tree Construction</b><br>
Consider a pairwise Markov network that consists of a graph with m variables on one side and n on the others. This graph is bipartite but fully connected, in that each of the m variables on the one side is connected to all and only the n variables on the other side. Define the size of a clique to be the number of variables in the clique. There exists a clique tree $T^{*}$ for the pairwise Markov network such that the size of the largest clique in $T^{*}$ is the smallest amongst all possible clique trees for this network. <b>What is the size of the largest sepset in $T^{*}$?</b><br><br>

Note: if you're wondering why we would ever care about this, remember that the complexity of inference depends on the number of entries in the largest factor produced in the course of message passing, which in turn, is affected by the size of the largest clique in the network, amongst other things.<br><br>

Hint: Use the relationship between sepsets and conditional independence to derive a lower bound for the size of the largest sepset, then construct a clique tree that achieves this bound.<br>
<b>Answer:</b> min(m, n).<br><br>

<b>4. Uses of Variable Elimination</b><br>
<b>Which of the following quantities can be computed using the sum-product variable elimination algorithm?</b> (In the options, let X be a set of query variables, and E be a set of evidence variables in the respective networks.) You may select 1 or more options.<br>
A. The most likely assignment to the variables in a Bayesian network.<br>
B. The partition function for a Markov network<br>
C. P(X) in a Bayesian network<br>
D. The partition function for a Markov network with evidence<br>
E. P(X | E=e) in a Markov network<br>
F. P(X | E=e) in a Bayesian network<br>
<b>Answer:</b> B, C, D, E, F.<br><br>

<b>5. *Time Complexity of Variable Elimination</b><br>
Consider a Bayesian network taking the form of a chain of n variables, $X_{1} \rightarrow X_{2} \rightarrow \cdots \rightarrow X_{n}$, where each of the $X_{i}$ can take k values. Assume we eliminate the $X_{i}$ starting from $X_{2}$ going to $X_{3}, \cdots X_{n}$ and then back to $X_{1}$. <b>What is the computational cost of running variable elimination with this ordering?</b><br>
<b>Answer:</b> $O(nk^{3})$.<br><br>

<b>6. *Numerical Issues in Belief Propagation</b><br>
In practice, one of the issues that arises when we propagate messages in a clique tree is that when we multiply many small numbers, we quickly run into the precision limits of floating-point numbers, resulting in arithmetic underflow. One possible approach for addressing this problem is to renormalize each message, as it's passed, such that its entries sum to 1. Assume that we do not store the renormalization factor at each step. <b>Which of the following statements describes the consequence of this approach?</b><br>
A. This does not change the results of the algorithm: when the clique tree is calibrated, we can obtain from it both the partition function and the correct marginals.<br>
B. We will be unable to extract the partition function, but the variable marginals that are obtained from renormalizing the beliefs at each clique will still be correct.<br>
C. This renormalization will give rise to incorrect marginals at calibration.<br>
D. Calibration will not even be achieved using this scheme.<br>
<b>Answer:</b> B.<br><br>

<b>7. Convergence in Belief Propagation</b><br>
Suppose we ran belief propagation on a cluster graph G and a clique tree T for the same Markov network that is a perfect map for a distribution P. Assume that both G and T are valid, i.e., they satisfy family preservation and the running intersection property. <b>Which of the following statements regarding the algorithm are true?</b> You may select 1 or more options.<br>
A. Assuming the algorithm converges, if a variable X appears in two clusters in G, the marginals P(X) computed from the two cluster beliefs must agree.<br>
B. If the algorithm converges, the final clique beliefs in T, when renormalized to sum to 1, are true marginals of P.<br>
C. If the algorithm converges, the final cluster beliefs in G, when renormalized to sum to 1, are true marginals of P.<br>
D. Assuming the algorithm converges, if a variable X appears in two cliques in T, the marginals P(X) computed from the the two clique beliefs must agree.<br>
E. Belief propagation always converges on T<br>
<b>Answer:</b> B, D, E.<br><br>

<b>8. Metropolis-Hastings Algorithm</b><br>
Assume we have an n×n grid-structured MRF over the variables $X_{i, j}$. Let $X_{i} = \{ X_{i, 1}, \cdots, X_{i, n} \}$ and $\bf{X_{-i}} = {\cal X} - {\bf{X_i}}$. Consider the following instance of the Metropolis-Hastings algorithm: at each step, we take our current assignment $x_{-i}$ and use exact inference to compute the conditional probability $P(X_{i} \mid x_{-1})$. We then sample $x'_{i}$ from this posterior distribution, and use that as our proposal. <b>What is the correct acceptance probability for this proposal?</b> Hint: what is the relationship between this and Gibbs sampling?<br>
A. 1<br>
B.
$$\frac{P(x'_{i}, x_{-i})}{P(x_{i}, x_{-i})}$$
C.
$$\frac{P(x_{i}, x_{-i})}{P(x'_{i}, x_{-i})}$$
D. 
$$\frac{P(x_{i} \mid x_{-i})}{p(x'_{i} \mid x_{-i})}$$
E.
$$\frac{p(x'_{i} \mid x_{-i})}{P(x_{i} \mid x_{-i})}$$
<b>Answer:</b> A.<br><br>

<b>9. *Value of Information</b><br>
<b>In the influence diagram on the right, when does performing LabTest have value? That is, when would you want to observe the LabTest variable?</b>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/2_10_2_2.png"/></center>
</p>
<p align="justify">
Hint: Think about when information is valuable in making a decision.<br>
A. When there is some treatment t such that V(D, t) is different for different diseases D<br>
B. When there is some disease d such that
$$arg\max_{t}V(d,t) \neq arg\max_{t} \sum_{d} P(d)V(d, t)$$
C. When there is some lab value l such that
$$arg\max_{t} \sum_{d} P(d \mid l)V(d, t) \neq arg\max_{t} \sum_{d} P(d)V(d, t)$$
D. When P(D | L) is different from P(D)<br>
<b>Answer:</b> C.<br><br>

<b>10. *Belief Propagation</b><br>
Say you had a probability distribution $P_{\Phi}$ encoded in a set of factors $\Phi$, and that you constructed a loopy cluster graph C to do inference in it. While you were performing loopy belief propagation on this graph, lightning struck and your computer shut down; to your horror, when you booted it back up, the only information you could recover were the graph structure C and the cluster beliefs at the current iteration. (For each cluster, the cluster belief is its initial potential multiplied by all incoming messages. You don’t have access to the sepset beliefs, the messages, or the original factors $\Phi$.) Assume the lightning struck before you had finished, i.e., the graph is not yet calibrated. Can you still recover the original distribution $P_{\Phi}$ from this? Why?<br>
A. We can reconstruct the original distribution by taking the product of cluster beliefs and normalizing it.<br>
B. We can reconstruct the (unnormalized) original distribution by taking the ratio of the product of cluster beliefs to sepset beliefs, and the sepset beliefs can be obtained by marginalizing the cluster beliefs.<br>
C. We can't reconstruct the (unnormalized) original distribution because we don't have the sepset beliefs to compute the ratio of the product of cluster beliefs to sepset beliefs.<br>
D. We can't reconstruct the original distribution because we were preforming loopy belief propagation, and the reparameterization property doesn't hold when it's loopy.<br>
<b>Answer:</b> C.<br><br>
</p>


## 3. Learning
### 3.1 Learning: Overview
#### 3.1.1 Learning: Overview
<p align="justify">
<b>Learning</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_1_1_1.png"/></center>
</p>
<p align="justify">
<b>Structure, Data</b><br>
We have four cases: known structure with complete data, unknow structure and complete data, known structure with complete data and unknown structure with incomplete data. Our goal is to produce a CPD corresponded to the network (if unknow, infer it)
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_1_1_2.png"/></center>
</p>
<p align="justify">
<b>Latent Variables, Incomplete Data</b><br>
In our final network, a variable H is a hidden variable. We haven't observed it.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_1_1_3.png"/></center>
</p>
<p align="justify">
<b>How would you characterize hidden Markov models?</b><br>
A. Known structure, fully observed.<br>
B. Known structure, partially observed.<br>
C. Unknown structure, fully observed.<br>
D. Unknown structure, partially observed.<br>
<b>Answer:</b> B.<br>
You know all the variables, but you do not get to observe the state.<br><br>

<b>PGM Learning Task I</b><br>
$\bigstar$ Goal: Answer general probabilistic queries about new instances<br>
$\bigstar$ Simple metrics: Training set likelihood<br>
-- $P(D: M) = \prod_{m} P(d[m]: M)$<br>
-- We assume trainning data are i.i.d and training set D is from a model M, then we calculate its likelihood<br>
$\bigstar$ But we really care about new data<br>
-- evaluate on test set likelihood - P(D' : M)<br>
-- generalized performance<br><br>

<b>PGM Learning Task II</b><br>
$\bigstar$ Goal: Specific prediction task on new instances<br>
-- predict target variables y from observed variables x<br>
-- e.g. image segmentation, speech recognition<br>
$\bigstar$ Often care about specialized objective<br>
-- e.g. pixel-level segmentation accuracy<br>
$\bigstar$ Often convient to select model to optimize<br>
-- likelihood $\prod_{m} P(d[m]: M)$ or<br>
-- conditional likelihood $\prod_{m} P(y[m] \mid x[m]: M)$<br>
$\bigstar$ Model evaluated on 'true' objective over test data<br><br>

<b>Is training set likelihood a good metric to evaluate a learned model with, when performing structure learning?</b><br>
A. Yes.<br>
B. No.<br>
C. Yes for Bayes Nets, No for Markov Nets.<br>
D. Yes with Bayes Nets with discrete CPTs, no otherwise.<br>
<b>Answer:</b> B.<br>
Because you are really more interested in the likelihood in the test set. This is because it is a better estimate of the real error in the model - the likelihood of the model in the training set is likely to be rather optimistic.<br><br>

<b>PGM Learning Task III</b><br>
$\bigstar$ Goal: Knowledge discovery of $M^{*}$<br>
-- distinguish direct vs indirect dependencies<br>
-- possibly directionality of edges<br>
-- presence and location of hidden variables<br>
$\bigstar$ Often train using likelihood<br>
-- poor surrogate for structural accuracy<br>
-- not use likelihood of the test set as the sole objective for evaluating model performance<br>
$\bigstar$ Evaluate by comparing to prior knowledge<br><br>

<b>Avoid Overfitting</b><br>
$\bigstar$ Selecting M to optimize training set likelihood overfits to statistical noise<br>
$\bigstar$ Parameter overfitting<br>
-- parameters fit random noise in training data<br>
-- use regularization / parameter priors<br>
$\bigstar$ Structure overfitting<br>
-- training likelihood always increases for more complex structure<br>
-- bound or penalize model complexity<br><br>

<b>Selecting Hyperparameters</b><br>
$\bigstar$ Regularization for overfitting involves hyperparameters<br>
-- parameters priors<br>
-- complexity penality<br>
$\bigstar$ Choice of hyperparameters makes a big difference to performance<br>
$\bigstar$ Must be selected on validation set (cross-validation)<br><br>

<b>Should you pick hyper parameters on the test data?</b><br>
A. No<br>
B. Yes<br>
C. Only if you include the test data in the validation set.<br>
D. Yes, if you have very limited data.<br>
<b>Answer:</b> A.<br><br>

<b>Why PGM Learning</b><br>
$\bigstar$ Predictions of structured objects (sequences, graphs, trees)<br>
-- exploit correlations between several predicted variables<br>
$\bigstar$ Can incorporate prior knowledge into model<br>
$\bigstar$ Learning single model for multiple tasks<br>
$\bigstar$ Framework for knowledge discovery<br><br>
</p>

### 3.2 Maximum Likelihood Parameter Estimation in BNs
#### 3.2.1 Maximum Likelihood Estimation
<p align="justify">
<b>Biased Coin Example</b><br>
P is a Bernoulli distribution:
$$P(X = 1) = \theta, \quad P(X=0) = 1 - \theta$$

$$D = \{ x[1], ..., x[M] \} \text{ sample IID from P}$$

$\bigstar$ Tosses are independent of each other<br>
$\bigstar$ Tosses are sampled from the same distribution (identically distributed)<br>

Based on the information above, we construct a plate model
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_2_1_1.png"/></center>
</p>
<p align="justify">
<b>Which of the following is true of the plate model shown here?</b><br>
A. Each coin toss has the same CPD.<br>
B. θ is considered a random variable.<br>
C.There is a different θ for each coin toss.<br>
D. None of the above.<br>
<b>Answer:</b> A, B.<br>
We are treating the continuous parameterθ as a random variable, though we haven't dealt much with continuous variables. The CPD ofX is a function ofX andθ.<br><br>

Then we can write our CPD
$$
P(x[m] \mid \theta) =
\begin{cases}
  \theta, \quad x[m]=x^{1}\\
  1-\theta, \quad x[m]=x^{0}
\end{cases}
$$

<b>Maximum Likelihood Estimation</b><br>
$\bigstar$ Goal: find $\theta \in [0, 1]$ that predicts D well<br>
$\bigstar$ Predictions equality = likelihood of D give $\theta$
$$L(\theta : D) = P(D \mid \theta) = \prod_{m=1}^{M} P(x[m] \mid \theta)$$

For example, we have 5 tosses
$$L(\theta: H, T, T, H, H)$$

Then the likelihood function is
$$L(\theta: D) = P(D \mid \theta) = P(H \mid \theta) \cdot P(T \mid \theta) \cdot P(T \mid \theta) \cdot P(H \mid \theta) \cdot P(H \mid \theta) = \theta^{3}(1-\theta)^{2}$$

We plot a figure of this function with regard to $\theta$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_2_1_2.png"/></center>
</p>
<p align="justify">
$$\theta^{*} = \arg\max_{\theta} L(\theta: D) = \arg\max_{\theta}\theta^{3}(1-\theta)^{2} = 0.6$$

<b>Maximum Likelihood Estimator</b><br>
$\bigstar$ Observations: $M_{H}$ heads and $M_{T}$ tails<br>
$\bigstar$ Find $\theta$ maximizing likelihood<br>
$$L(\theta: M_{H}, M_{T}) = \theta^{M_{H}}(1-\theta)^{M_{T}}$$

$\bigstar$ Equivalent to maximizing log-likelihood<br>
$$l(\theta: M_{H}, M_{T}) = M_{H}\log\theta + M_{T}\log(1-\theta)$$

$\bigstar$ Differentiating the log-likelihood and solving for $\theta$<br>
$$\hat{\theta} = \frac{M_{H}}{M_{H} + M_{T}}$$

<b>Sufficient Statistics</b><br>
$\bigstar$ For computing $\theta$ in the coin toss example, we only need $M_{H}$ and $M_{T}$ since<br>
$$L(\theta: D) = \theta^{M_{H}}(1-\theta)^{M_{T}}$$

$\bigstar$ $M_{H}$ and $M_{T}$ are sufficient statistics<br>
$\bigstar$ A function s(D) is a <b>sufficient statistic</b> from instances to a vector in $R^{k}$ if for any two datasets D and D' and any $\theta \in \Theta$ we have<br>
$$\sum_{x[i] \in D} s(x[i]) = \sum_{x[i] \in D'} s(x[i]) \quad \Rightarrow \quad L(\theta: D) = L(\theta: D')$$

In other word, we try to use a smaller set of parameters/notations/statistics to describe a large dataset<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_2_1_3.png"/></center>
</p>
<p align="justify">
<b>Sufficient Statistic for Multinomial</b><br>
$\bigstar$ For a dataset D over variables X with k values, the sufficient statistics are counts {$M_{1}$, ..., $M_{k}$} where $M_{i}$ is the number of times that X[m] = $x^{i}$ in D<br>
$\bigstar$ Sufficient statistic s(x) is a tuple of dimension k<br>
$$s(x^{i}) = (0,..., 0, 1, 0, ..., 0)$$

There is only one non-zero in this vector, namely in $i^{th}$ position. If we sum all $s(x_{i})$ up, we can get sufficient statistics<br>
$$\sum_{m}s(x[m]) = \{ M_{1}, ..., M_{k} \}$$

Based on sufficient statistics, we can get a likelihood function<br>
$$L(\theta:  D) = \prod_{i=1}^{k} \theta_{i}^{M_{i}}$$

<b>Sufficient Statistic for Gaussian</b><br>
$\bigstar$ Gaussian distribution<br>
$$P(X) \sim N(\mu, \sigma^{2}) \quad \text{if} \quad p(X) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2} (\frac{x-\mu}{\sigma})^{2}}$$

$\bigstar$ Rewrite as<br>
$$p(X) = \frac{1}{\sigma\sqrt{2\pi}}e^{-x^{2}\frac{1}{2\sigma^{2}} + x\frac{\mu}{\sigma^{2}} - \frac{\mu^{2}}{2\sigma^{2}}}$$

$\bigstar$ Sufficient statistic for Gaussian<br>
$$s(x) = \{1, x, x^{2} \}$$

$\bigstar$ Sufficient statistics<br>
$$S(D) = \{ M, \sum_{m}X[m], \sum_{m}X[m]^{2} \}$$

<b>Maximum Likelihood Estimation</b><br>
$\bigstar$ MLE Principle: choose $\theta$ to maximize $L(D: \Theta)$<br>
$\bigstar$ Multinomial MLE<br>
$$\hat{\theta^{i}} = \frac{M_{i}}{\sum_{i=1}^{m} M_{i}}$$

$\bigstar$ Gaussian MLE<br>
Empirical mean<br>
$$\hat{\mu_{}} = \frac{1}{M} \sum_{m}x[m]$$

Empirical standard deviation<br>
$$\hat{\sigma_{}} = \sqrt{\frac{1}{M} \sum_{m} (x[m] - \hat{\mu})^{2}}$$

<b>Summary</b><br>
$\bigstar$ Maximum likelihood estimation is a simple principle for parameter selection given D<br>
$\bigstar$ Likelihood function uniquely determined by sufficent statistics that summarize D<br>
$\bigstar$ MLE has closed form solution for many parametric distributions<br><br>
</p>

#### 3.2.2 MLE for BNs
<p align="justify">
<b>MLE for Bayesian Networks</b><br>
Consider a simple Bayesian Network in a form of plate model<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_2_2_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ Parameters<br>
-- we have two parameters $\theta_{X}$ and $\theta_{Y \mid X}$ to describe variable X and Y<br>
$$\{ \theta_{x}: x \in Val(X) \}, \quad \{ \theta_{y \mid x}: x \in Val(X), y \in Val(X) \}$$

So we write likelihood function<br>
$$L(\Theta: D) = \prod_{m=1}^{M} P(x[m], y[m]: \theta) = \prod_{m=1}^{M} P(x[m]: \theta) P(y[m] \mid x[m]: \theta)$$

We seperate these two factors<br>
$$= \prod_{m=1}^{M} P(x[m]: \theta) \prod_{m=1}^{M} P(y[m] \mid x[m]: \theta)$$

Each fatcor has a specific $\theta$<br>
$$= \prod_{m=1}^{M} P(x[m]: \theta_{X}) \prod_{m=1}^{M} P(y[m] \mid x[m]: \theta_{Y \mid X})$$

So, we divide a likelihood into a product of local likelihood<br><br>

$\bigstar$ likelihood for Bayesian network<br>
$$L(\Theta: D) = \prod_{m} P(x[m]: \Theta) = \prod_{m}\prod_{i} P(x_{i}[m] \mid U_{i}[m] : \Theta) = \prod_{i}\prod_{m} P(x_{i}[m] \mid U_{i}[m] : \Theta)$$
$$= \prod_{i} L_{i}(D: \Theta_{i})$$

If $\theta_{X_{i} \mid U_{i}}$ are <b>disjoint</b> (CPDs don't share a parameter), then MLE can be computed by maximizing each <b>local likelihood</b> seperately.<br><br>

<b>MLE for Table CPDs</b><br>
$$\prod_{m=1}^{M} P(x[m] \mid u[m]: \theta) = \prod_{m=1}^{M} P(x[m] \mid u[m]: \theta_{X \mid U}) = \prod_{x, u} \prod_{m: x[m]=x, u[m]=u} P(x[m] \mid u[m]: \theta_{X \mid U})$$

Besides<br>
$$P(x[m]=x \mid u[m]=u : \theta_{X \mid U}) = \theta_{X \mid U}$$

So, we continue to go deeper<br>
$$= \prod_{x, u} \prod_{m: x[m]=x, u[m]=u} \theta_{x \mid u} = \prod_{x, u} \theta_{x \mid u}^{M[x, u]}$$

Finally, the estimated parameter<br>
$$\hat{\theta_{}}_{x \mid u} = \frac{M[x, u]}{\sum_{x'} M[x', u]} = \frac{M[x, u]}{M[u]}$$

<b>Shared Parameters</b><br>
Image a markov chain with state transition<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_2_2_2.png"/></center>
</p>
<p align="justify">
$$L(\theta: S^{0:T}) = \prod_{t=1}^{T} P(S^{t} \mid S^{t-1}: \theta) = \prod_{i, j} \prod_{t: S^{t}=s^{i}, S^{t+1}=s^{j}} P(S^{t+1} \mid S^{t}: \theta_{S' \mid S}) = \prod_{i, j} \prod_{t: S^{t}=s^{i}, S^{t+1}=s^{j}} \theta_{s^{i} \rightarrow s^{j}}$$
$$= \prod_{i, j} \theta_{s^{i} \rightarrow s^{j}}^{M[s^{i} \rightarrow s^{j}]}$$

where<br>
$$M[s^{i} \rightarrow s^{j}] = \left | \{ t: S^{t} = s^{i}, S^{t+1} = s^{j} \} \right |$$

Finally, the estimated parameter<br>
$$\hat{\theta_{}}_{s^{i} \rightarrow s^{j}} = \frac{M[s^{i} \rightarrow s^{j}]}{M[s^{i}]}$$

We take a more complex example<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_2_2_3.png"/></center>
</p>
<p align="justify">
$$L(\Theta : S^{0:T}, O^{0:T}) = \prod_{t=1}^{T} P(S^{t} \mid S^{t-1}: \theta_{S' \mid S}) \prod_{t=1}^{T} P(O^{t} \mid S^{t} : \theta_{O' \mid S'}) = \prod_{i, j} \theta_{s^{i} \rightarrow s^{j}}^{M[s^{i} \rightarrow s^{j}]} \prod_{i, k} \theta_{o^{k} \mid s^{i}}^{M[o^{k}, s^{i}]}$$

where<br>
$$M[s^{i} \rightarrow s^{j}] = \left | \{t: S^{t} = s^{i}, S^{t+1} = s^{j} \} \right |$$
$$M[o^{k}, s^{i}] = \left | \{t: S^{t} = s^{i}, O^{t} = o^{k} \} \right |$$

<b>Summary</b><br>
$\bigstar$ For BN with disjoint sets of parameters in CPDs, likelihood decomposes as product of local likelihood functions, one per variable<br>
$\bigstar$ For table CPDs, local likelihood further decomposes as product of likelihood for multinomials, one for each parent combination<br>
$\bigstar$ For networks with shared CPDs, sufficient statistics accumulate over all uses of CPD<br><br>

<b>Fragmentation & Overfitting</b><br>
$$\hat{\theta_{}}_{x \mid u} = \frac{M[x, u]}{\sum_{x'} M[x', u]} = \frac{M[x, u]}{M[u]}$$

$\bigstar$ Number of 'buckets' increases exponentially with |U|<br>
$\bigstar$ For large |U|, most 'buckets' will have very few instances<br>
-- <b>very poor parameters estimates</b><br>
$\bigstar$ With limited data, we often get better generalization with simpler structure.<br><br>
</p>

#### 3.2.3 Quiz
<p align="justify">
<b>1.</b><br>
Suppose that you are playing Dungeons & Dragons, and you suspect that the 4-sided die that your Dungeon Master is using is biased. In the past 60 times that you have attacked with your dagger and the 4-sided die was rolled to calculate how many hit points of damage you inflicted, 20 times it has come up 1, 15 times it has come up 2, 15 times it has come up 3, and 10 times it has come up 4. Let $\theta_{1}$ be the true probability of the die landing on 1, and similarly for $\theta_{2}$, $\theta_{3}$, and $\theta_{4}$.<br><br>

<b>1.1 Computing Sufficient Statistics</b><br>
You want to estimate these parameters from the past 60 rolls that you observed using a simple multinomial model. <b>Which of the following is a sufficient statistic for this data?</b><br>
A. vector with with four components, with the $i^{th}$ component being the number of times you dealt i hit points worth of damage.<br>
B. The total number of times you viciously attacked a monster with your dagger (i.e., the total number of times that the dice was rolled).<br>
C. The total amount of damage you inflicted with your trusty dagger (i.e., the sum of all die rolls).<br>
D. None of these are sufficient statistics.<br>
<b>Answer:</b> A.<br>
A sufficient statistic is a function of the data that summarizes the relevant information for computing the likelihood. The sufficient statistics for a multinomial model are the "counts" of each possible result. The number of times each digit was rolled allows us to compute the likelihood function.<br><br>

<b>1.2 MLE Parameter Estimation</b><br>
In the context of the previous question, <b>what is the unique Maximum Likelihood Estimate (MLE) of the parameter $\theta_{1}$?</b> Give your answers rounded to the nearest ten-thousandth (i.e. 1/6 should be 0.1667).<br>
<b>Answer:</b> 0.3333.<br><br>

<b>2. Likelihood Functions</b><br>
For a Naive Bayes network with one parent node, X, and 3 children nodes, $Y_{1}$, $Y_{2}$, $Y_{3}$, what is the expressions for the likelihood, decomposed in terms of the local likelihood functions?<br>
<b>Answer:</b>
$$L(\theta: D) = (\prod_{m=1}^{M} P(x[m]: \theta_{X})) (\prod_{m=1}^{M} P(y_{1}[m] \mid x[m]: \theta_{Y_{1} \mid X})) (\prod_{m=1}^{M} P(y_{2}[m] \mid x[m]: \theta_{Y_{2} \mid X}))$$
$$(\prod_{m=1}^{M} P(y_{3}[m] \mid x[m]: \theta_{Y_{3} \mid X}))$$

<b>3. MLE for Naive Bayes</b><br>
Using a Naive Bayes model for spam classification with the vocabulary V={"SECRET", "OFFER", "LOW", "PRICE", "VALUED", "CUSTOMER", "TODAY", "DOLLAR", "MILLION", "SPORTS", "IS", "FOR", "PLAY", "HEALTHY", "PIZZA"}. We have the following example spam messages SPAM = {"MILLION DOLLAR OFFER", "SECRET OFFER TODAY", "SECRET IS SECRET"} and normal messages, NON-SPAM = {"LOW PRICE FOR VALUED CUSTOMER", "PLAY SECRET SPORTS TODAY", "SPORTS IS HEALTHY", "LOW PRICE PIZZA"}.<br><br>

We create a multinomial naive Bayes model for the data given above. This can be modeled as a parent node taking values SPAM and NON-SPAM and a child node for each word in the vocabulary. The θ values are estimated based on the number of times that a word appears in the vocabulary.<br><br>

<b>3.1</b><br>
Give the MLE for $\theta_{SPAM}$. Enter the value as a decimal rounded to the nearest ten-thousandth (0.xxxx).<br>
<b>Answer:</b> 9/21 = 0.4286.<br>
V has unique word. SPAM has 3 words appearing twice or more: SECRET, SECRET, OFFER. NON-SPAM also has 3 words: SPORTS, LOW, PRICE. All samples are 15 + 3 + 3 = 21<br><br>

<b>3.2</b><br>
Using the same data and model above, give the MLE for $\theta_{SECRET \mid SPAM}$. Enter the value as a decimal rounded to the nearest ten-thousandth (0.xxxx).<br>
<b>Answer:</b> 3/9 = 0.3333.<br><br>

<b>3.3 </b><br>
Using the same data and model above, give the MLE for$\theta_{SECRET \mid NON-SPAM}$. Enter the value as a decimal rounded to the nearest ten-thousandth (0.xxxx).<br>
<b>Answer:</b> 1/15 = 0.0667.<br><br>

<b>4. Learning Setups</b><br>
Consider the following scenario: You have been given a dataset that contains patients and their gene expression data for 10 genes. You are also given a 0/1 label where 1 means that patient has disease A and 0 means the patient does not.<br>
Your goal is to learn a classification algorithm that could predict these labels with high accuracy. You split the data into three sets:<br>
1: Set of patients used for fitting the classifier parameters (e.g., the weights and bias of a logistic regression classifier).<br>
2: Set of patients used for tuning the hyperparameters of the classifier (e.g., how much regularization to apply).<br>
3: Set of patients used to assess the performance of the classifier.<br>
<b>What are these sets called?</b><br>
A. 1: Training Set, 2: Test Set, 3: Validation Set<br>
B. 1: Validation Set, 2: Test Set, 3: Training Set<br>
C. 1: Training Set, 2: Validation Set, 3: Test Set<br>
D. 1 & 2: Training Set, 3: Validation Set.<br>
<b>Answer:</b> C.<br><br>

<b>5. Constructing CPDs</b><br>
Assume that we are trying to construct a CPD for a random variable whose value labels a document (e.g., an email) as belonging to one of two categories (e.g., spam or non-spam). We have identified K words whose presence (or absence) in the document each changes the distribution over labels (e.g., the presence of the word "free" is more likely to indicate that the email is spam). Assume that we have M labeled documents that we use to estimate the parameters for the CPD of the label given indicator variables representing the appearance of words in the document. We plan to use maximum likelihood estimation to select the parameters of this CPD.<br><br>

<b>5.1</b><br>
IfM = 1000 and K = 30, which of the following CPD types are most likely to provide the best generalization performance to unseen data? Mark all that apply.<br>
A. A table CPD<br>
B. A sigmoid CPD<br>
C. A linear Gaussian CPD<br>
D. None of these CPDs would work<br>
<b>Answer:</b> B.<br>
With a sigmoid CPD the number of parameters that will need to be learned is K = 30 (plus 1 for the bias term) and thus M = 1000 instances are sufficient to get a reasonable maximum likelihood estimation of the parameters and hence the distribution.<br><br>

<b>5.2</b><br>
For the same scenario as described in the previous question, if M = 100000 and K = 3, which of the following CPD types is most likely to provide the best generalization performance to unseen data?<br>
A. A sigmoid CPD<br>
B. A table CPD<br>
C. A linear Gaussian CPD<br>
D. A tree CPD with K = 3 leaves<br>
<b>Answer:</b> B.<br>
In this scenario, a table CPD has $2^{3} - 1$ free parameters, so we have enough instances to get a good estimate of the distribution for this type of CPD.<br><br>
</p>

### 3.3 Bayesian Parameter Estimation for BNs
#### 3.3.1 Bayesian Estimation
<p align="justify">
<b>Limitations of MLE</b><br>
$\bigstar$ Two teams play 10 times, and the first wins 7 of the 10 matches<br>
-- Probability of the first team winning = 0.7<br>
$\bigstar$ A coin is tossed 10 times, and comes out 'heads' 7 of the 10 tosses<br>
-- probability pf heads = 0.7<br>
$\bigstar$ A coin is tossed 10000 times, and comes out 'head s' 7000 of the 10000 tosses<br>
-- probability of heads = 0.7<br>
-- more plausible<br>
<b>Maximum likelihood estimation has absolutely no ability to distinguish between these three scenarios.</b> Between the case of a familiar setting such as a coin versus an unfamiliar event such as the two teams playing, on the one hand, and between the case where we toss a coin ten times verses tossing a coin ten thousand times. Neither of these distinction is apparent in the maximum likelihood estimate.<br><br>

<b>Parameter Estimation as a PGM</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_1_1.png"/></center>
</p>
<p align="justify">
We're going to see that theta is in fact, a random variable by itself. It's a continuous valued random variable. which in this case, in the case of a coin toss, takes on value in the space, 01 but in either case it is a random variable, and therefore something over which will maintain and probability distribution. This is factored at the heart of the Bayesian formalism.<br><br>

Anything about which we are uncertain we should view as a random variable over which we have a distribution that is updated over time as data is acquired<br><br>

$\bigstar$ Given a fixed $\theta$, tosses are independent<br>
$\bigstar$ If $\theta$ is unknown, tosses are marginally independent<br>
-- for example, suppose we have observed X[1] = head whith an unknown $\theta$, X[1] = head will influence other tosses<br>
-- each toss tells us something about $\theta$<br><br>

<b>Bayesian Inference</b><br>
$\bigstar$ Joint probabilistic model<br>
$$P(x[1], ..., x[M], \theta) = P(x[1], ..., x[M] \mid \theta) P(\theta) = P(\theta)\prod_{i=1}^{M}P(x[i] \mid \theta)$$

In fact, $\prod_{i=1}^{M}P(x[i] \mid \theta)$ is a likelihood function<br>
$$= P(\theta) \theta^{M_{H}} (1-\theta)^{M_{T}}$$

The posterior probability give our data x[1], ..., x[M]<br>
$$P(\theta \mid x[1], ..., x[M]) = \frac{P(x[1], ..., x[M] \mid \theta) P(\theta)}{P(x[1], ..., x[M])}$$

$P(\theta)$ is called prior probability. The denominator is a constant normalization relative to $\theta$. In other word, if we know how to calculate the numerator, we can derive the denominator by integrating out over all possible $\theta$.<br><br>

<b>Dirichlet Distribution</b><br>
$\bigstar$ $\theta$ is a multinomial distribution over k values<br>
$\bigstar$ Dirichlet distribution $\theta \sim \text{Dirichlet}(\alpha_{1}, ..., \alpha_{k})$<br>
-- Where<br>
$$P(\theta) = \frac{1}{Z} \prod_{i=1}^{k} \theta_{i}^{\alpha_{i}-1}, \quad Z = \frac{\prod_{i=1}^{k} \Gamma(\alpha_{i})}{\Gamma(\sum_{i=1}^{k} \alpha_{i})}, \quad \Gamma(x) = \int_{0}^{\infty} t^{x-1}e^{-t} dt$$

$\bigstar$ Intuitively, hyperparameters $(\alpha_{1}, ..., \alpha_{k})$ correspond to the number of samples we have seen.<br><br>

We plot several Dirichlet distributions with different $\alpha_{H}$ and $\alpha_{T}$<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_1_2.png"/></center>
</p>
<p align="justify">
We can notice that a mix of $\alpha_{H}$ and $\alpha_{T}$ determines the position of the peak and total $\alpha = \alpha_{H} + \alpha_{T}$ determines how sharp it is.<br><br>

<b>Dirichlet Priors & Posteriors</b><br>
We compute posterior $P(\theta \mid D)$<br>
$$P(\theta \mid D) \propto P(D \mid \theta) P(\theta)$$

Note D is multinomial<br>
Likelihood $P(D \mid \theta)$
$$P(D \mid \theta) = \prod_{i=1}^{k} \theta_{i}^{M_{i}}$$

Prior $P(\theta)$<br>
$$P(\theta) \propto \prod_{i=1}^{k} \theta_{i}^{\alpha_{i}-1}$$

$\bigstar$ If $P(\theta)$ is Dirichlet and the likelihood is multinomial, then the posterior is also Dirichlet<br>
-- Prior is Dir($\alpha_{1}, ..., \alpha_{k}$)<br>
-- Data counts are $M_{1}, ..., M_{k}$<br>
-- posterior is Dir($\alpha_{1}+M_{1}, ..., \alpha_{k}+M_{k}$)<br>
$\bigstar$ Dirichlet is a <b>conjugate prior</b> for the multinomial<br>
-- Prior and posterior have the same form.<br><br>

The Beta distribution is a special case of the Dirichlet distribution in the sense that:<br>
A. k = 2<br>
B. None of the above.<br>
C. The Dirichlet distribution is an appropriate prior for parameters of a multinomial distribution; the beta distribution is an appropriate prior for parameters of a binomial distribution.<br>
D. $\alpha_{i}$ in the beta distribution is the same as $\alpha_{i-1}$ in the Dirichlet distribution.<br><br>

<b>Answer:</b> A, C.<br>
The binomial distribution is a special case of the multinomial distribution, where again k = 2.<br><br>

<b>Summary</b><br>
$\bigstar$ bayesian learning treats parameters as <b>random variables</b><br>
-- <b>learning is then a special case of inference</b><br>
$\bigstar$ Dirichlet distribution is conjugate to mutinomial<br>
-- posterior has the same form as prior<br>
-- can be updated in closed form using sufficient statistics from data<br><br>
</p>

#### 3.3.2 Bayesian Prediction
<p align="justify">
<b>Bayesian Prediction</b><br>
Consider a model with parameters which satisfiy the Dirichlet distribution<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_2_1.png"/></center>
</p>
<p align="justify">
$$P(X) = \int_{\theta} P(X \mid \theta)P(\theta) d\theta$$

$$P(X = x^{i} \mid \theta) = \frac{1}{Z} \int_{\theta} \theta_{i} \cdot \prod_{j} \theta^{\alpha_{j}-1} d\theta = \frac{\alpha_{i}}{\sum_{j} \alpha_{j}}$$

$\bigstar$ Dirichlet hyperparameters correspond to the number of samples we have seen.<br><br>

For example, we want to predict X[M+1] with a Bayesian network<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_2_2.png"/></center>
</p>
<p align="justify">
$$P(x[M+1] \mid x[1], ..., x[M]) = \int_{\theta} P(x[M+1] \mid x[1], ..., x[M], \theta) P(\theta \mid x[1], ..., x[M]) d\theta$$

In fact x[M+1] is independent of x[1], ..., x[M] given $\theta$<br>
$$= \int_{\theta} P(x[M+1] \mid \theta)P(\theta \mid x[1], ..., x[M]) d\theta$$

Note $P(\theta \mid x[1], ..., x[M])$ is a Dirichlet distribution with hyperparameters ($\alpha_{1}+M_{1}, ..., \alpha_{k} +M_{k}$)<br><br>

So, the probability that vriable X[M+1] take some value, say $x^{i}$<br>
$$P(X[M+1] = x^{i} \mid \theta, x[1], ..., x[M]) = \frac{\alpha_{i} + M_{i}}{\sum_{i} \alpha_{i} + \sum_{i} M_{i}} = \frac{\alpha_{i} + M_{i}}{\alpha + M}$$

$\bigstar$ Equivalent sample size $\alpha = \alpha_{1} + ... + \alpha_{k}$<br>
-- large $\alpha$ $\Rightarrow$ more confidence in our prior<br>
-- The equivalent sample size represents the number of samples that we would have seen. If we double all of our $\alpha$, then we're going to let the $M_{i}$'s effect on our estimate a lot less than for smaller values of alpha<br><br>

<b>Example: Binomial Data</b><br>
$\bigstar$ Prior: uniform for $\theta$ in [0, 1]<br>
-- which corresponds to Dirichlet distribution with hyperparameters $(\alpha_{1} = 1, \alpha_{0} = 1)$<br>
$$P(\theta) = \frac{1}{Z} \prod_{k} \theta_{k}^{\alpha_{k}-1}$$

Suppose we have 4 one and 1 zeos (e.g. 4 heads and 1 tail)<br>
$$(M_{1}, M_{0}) = (4, 1)$$

So, we plot the Dirichlet distribution<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_2_3.png"/></center>
</p>
<p align="justify">
Then, we want to predict a probability for X[6] = 1<br>
$\bigstar$ MLE method<br>
$$P(X[6] = 1) = \frac{4}{5}$$

$\bigstar$ Bayesian estimate<br>
$$P(X[6] = 1) = \frac{\alpha_{1} + M_{1}}{\alpha + M} = \frac{1 + 4}{2 + 5} = \frac{5}{7}$$

<b>Effect of Priors</b><br>
$\bigstar$ Prediction of P(X=1) after seeing data with $M_{1} = \frac{1}{4}M_{0}$ as a function of sample size M<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_2_4.png"/></center>
</p>
<p align="justify">
For fixed ratio $\frac{\alpha_{1}}{\alpha_{0}}$, we have a 0.5 probability when 0 data observed, this is reasonable. Low $\alpha$ is closer to data estimates. High $\alpha$ takes more data to be close to the empirical fractions of heads versus tails.<br><br>

For fixed equivalent sample size, we start with different priors ($\alpha_{1}, \alpha_{0}$), the posterior probability will be converged to 0.2 (empirical estimate) with different data.<br><br>

$\bigstar$ In real data, Bayesian estimates are less sensitive to noise in the data.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_2_5.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ Bayesian prediction combines sufficient statistics from <b>imaginary Dirichlet samples</b> and <b>real data samples</b>.<br>
$\bigstar$ Asymptotically, the same as MLE<br>
$\bigstar$ But Dirichlet hyperparameters determine both the prior beliefs and their strength<br><br>
</p>

#### 3.3.3 Bayesian Estimation for BNs
<p align="justify">
<b>Bayesian Estimation in BNs</b><br>
Consider a Bayesian network with paramters
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_3_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ Instances are independent given parameters<br>
-- (X[m'], Y[m']) are d-seperated from (X[m], Y[m]) given $\theta$<br>
$\bigstar$ Parameters for individual variables are independent a priori<br>
$$P(\theta) = \prod_{i} P(\theta_{X_{i} \mid Parent(X_{i})})$$

$\bigstar$ Posteriors of $\theta$ are independent given complete data<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_3_2.png"/></center>
</p>
<p align="justify">
$$\theta_{X} \perp \theta_{Y \mid X} \mid D$$

-- complete data d-seperate parameters for different CPDs<br>
-- $P(\theta_{X}, \theta_{Y \mid X} \mid D) = P(\theta_{X} \mid D) P(\theta_{Y \mid X} \mid D)$<br>
-- As in MLE, we can solve each estimation problem seperately<br><br>

-- Also holds for parameters wthin families<br>
-- Note context specific independence between $\theta_{Y \mid x^{1}}$ and $\theta_{Y \mid x^{0}}$ when given X's and Y's.<br><br>

If we are using table CPDs, <b>which independence property can help us further split the likelihood function?</b><br>
A. $(\theta_{Y \mid x^{1}}, \theta_{Y \mid x^{0}} \perp X)$<br>
B. $(\theta_{Y \mid x^{1}} \perp \theta_{Y \mid x^{0}} \mid X)$<br>
C. $(\theta_{Y \mid x^{1}} \perp \theta_{Y \mid x^{0}} \mid X, Y)$<br>
D. $(\theta_{Y \mid x^{1}} \perp \theta_{Y \mid x^{0}})$<br>
<b>Answer:</b> C.<br>

$\bigstar$ Posteriors of $\theta$ can be computed independently<br>
-- For multinomial $\theta_{X \mid u}$ if prior is Dirichlet ($\alpha_{x^{1} \mid u}, ..., \alpha_{x^{k} \mid u}$)<br>
-- posterior is Dirichlet ($\alpha_{x^{1} \mid u} + M[x^{1}, u], ..., \alpha_{x^{k} \mid u} + M[x^{k}, u]$)<br><br>

<b>Assessing Priors for BNs</b><br>
$\bigstar$ We need hyperparameter $\alpha_{x \mid u}$ for each node X, value x, and parent assignment u.<br>
-- Prior network with parameters $\Theta_{0}$<br>
-- Equivalent sample size parameter $\alpha$<br>
-- $\alpha_{x \mid u} := \alpha \cdot P(x, u \mid \Theta_{0})$<br><br>

For examole, condsider a simple Bayesian network composed of two binary variables X $\rightarrow$ Y<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_3_3.png"/></center>
</p>
<p align="justify">
Suppose the parameter $\Theta_{0}$ follows a uniform distribution. For variable X<br>
$$\theta_{X} \sim \text{Dirichlet}(\frac{\alpha}{2}, \frac{\alpha}{2})$$

For variable Y<br>
$$\theta_{Y \mid x^{1}} \sim \text{Dirichlet}(\frac{\alpha}{4}, \frac{\alpha}{4}), \quad \theta_{Y \mid x^{0}} \sim \text{Dirichlet}(\frac{\alpha}{4}, \frac{\alpha}{4})$$

<b>Case study</b><br>
$\bigstar$ ICU-Alarm network<br>
-- 37 variables<br>
-- 504 parameters<br>
$\bigstar$ Experiment<br>
-- Sample instance from network<br>
-- Relearn parameters<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_3_3_4.png"/></center>
</p>
<p align="justify">
Which of the following statements are true regarding the generalization performance (test-set log-likelihood, or relative entropy) for different priors? Mark all that apply.<br>
A. As $M \rightarrow \infty$, the prior matters less and less.<br>
B. A strong prior will always do worse than a weak prior, as long at the weak prior does not have $\alpha$ = 0.<br>
C. If M is small relative to the number of parameters in a CPD, using MLE is generally a bad idea.<br>
D. MLE will always perform worse than even the strongest prior.<br>
<b>Answer:</b> A, C.<br><br>

<b>Summary</b><br>
$\bigstar$ In Bayesian networks, if parameters are indepedent a priori, then also independent in the posterior<br>
$\bigstar$ For multinomial BNs, estimation uses sufficient statistics M[x, u]<br>
-- MLE<br>
$$\hat{\theta_{}}_{x \mid u} = \frac{M[x, u]}{M[u]}$$

-- Bayesian (Dirichlet)<br>
$$P(x \mid u, D) = \frac{\alpha_{x \mid u} + M[x, u]}{\alpha_{u} + M[u]}$$

$\bigstar$ Bayesian methods require choice of prior<br>
-- can be elicited as prior network and equivalent sample size<br><br>
</p>

#### 3.3.4 Quiz
<p align="justify">
<b>1. BDe Priors</b><br>
The following is a common approach for defining a parameter prior for a Bayesian network, and is referred to as the BDe prior. Let $P_{0}$ be some distribution over possible assignments $x_{1}, ..., x_{n}$, and select some fixed $\alpha$. For a node X with parents U we define $\alpha_{x \mid u} = \alpha P_{0}(x, u)$.<br><br>

For this question, assume X takes one of m values and that X has k parents, each of which takes d values. If we choose $P_{0}$ to be the uniform distribution, then <b>what is the value of $\alpha_{x \mid u}$?</b><br>
<b>Answer:</b> $\frac{\alpha}{md^{k}}$.<br><br>

<b>2. Learning with a Dirichlet Prior</b><br>
Suppose we are interested in estimating the distribution over the English letters. We assume an alphabet that consists of 26 letters and the space symbol, and we ignore all other punctuation and the upper/lower case distinction. We model the distribution over the 27 symbols as a multinomial parametrized by $\theta = (\theta_{1}, ..., \theta_{27})$, where $\sum_{i}\theta_{i} = 1$ and all $\theta_{i} \geq 0$.<br><br>

Now we go to Stanford's Green library and repeat the following experiment: randomly pick up a book, open a page, pick a spot on the page, and write down the nearest symbol that is in our alphabet. We use X[m] to denote the letter we obtain in the $m^{th}$ experiment.<br><br>

<b>2.1</b><br>
In the end, we have collected a dataset $D = \{ x[1], ..., x[2000] \}$ consisting of 2000 symbols, among which "e" appears 260 times. We use a Dirichlet prior over $\theta$, i.e. $P(\theta) = \text{Dirichlet}(\alpha_{1}, ..., \alpha_{27})$ where each $\alpha_{i}$ = 10. <b>What is the predictive probability that letter "e" occurs with this prior? (i.e., what is P(X[2001] = 'e' | D)?</b> Write your answer as a decimal rounded to the nearest ten thousandth (0.xxxx).<br>
<b>Answer:</b> (10+260)/(10*27+2000) = 0.1189.<br><br>

<b>2.2</b><br>
n the setting of the previous question, suppose we had collected M = 2000 symbols, and the number of times "a" appeared was 100, while the number of times "p" appeared was 87. Now suppose we draw 2 more samples, X[2001] and X[2002]. If we use $\alpha_{i} = 10$ for all i, <b>what is the probability of P(X[2001] = 'p', X[2002] = 'a' | D)?</b> (round your answer to the nearest millionth, 0.xxxxxx)<br>
<b>Answer:</b> (10+87)/(10*27+2000)*(10+100)/(10*27+2001) = 0.002070.<br><br>

<b>3. *Learning with a Dirichlet Prior</b><br>
In the setting of previous two questions, suppose we have collected M symbols, and let $\alpha = \sum_{i} \alpha_{i}$ (we no longer assume that each $\alpha = 10$). <b>In which situation(s) does the Bayesian predictive probability using the Dirichlet prior ( i.e., P(X[M+1] | D)) converge to the MLE estimation for any distribution overM?</b> You may select 1 or more options.<br>
A. M $\rightarrow$ 0 and $\alpha$ is fixed and non-zero<br>
B. None of the above<br>
C. Both $\alpha$ and M are fixed and non-zero for some fixed distribution over $\alpha$<br>
D. $\alpha \rightarrow 0$ and M is fixed<br>
E. $M \rightarrow \infty$ and $\alpha$ is fixed<br>
<b>Answer:</b> D, E.<br>
</p>

### 3.4 Parameter Estimation in MNs
#### 3.4.1 Maximum Likelihood for Log-Linear Models
<p align="justify">
<b>Log-Likelihood for markov Nets</b><br>
Consider a Markov Network with 3 variables<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_4_1_1.png"/></center>
</p>
<p align="justify">
Given a dataset (a, b, c), the distribution over this graph is<br>
$$P(a, b, c) = \frac{1}{Z} \phi_{1}(a, b) \phi_{2}(b, c)$$

Partition function is a function about $\theta$<br><br>

The log-likelihood function is<br>
$$l(\theta : D) = \sum_{m} (\ln\phi_{1}(a[m], b[m]) + \ln\phi_{2}(b[m], c[m]) - \ln Z(\theta))$$

We change the terms by calculating how many entries containing a, b for $\phi_{1}$ and how many entries containing b, c for $\phi_{2}$<br>
$$= \sum_{a, b} M[a, b]\ln\phi_{1}(a, b) + \sum_{b, c} M[b, c]\ln\phi_{2}(b, c) - M\ln Z(\theta)$$

Which of the following statements is true regarding the difficulty of optimizing this objective?<br>
A. This objective is easy to optimize because Z is constant and we can optimize each $\phi$ separately.<br>
B. This objective is hard to optimize because Z is not constant and couples the optimization of the different $\phi$.<br>
C. This objective is hard to optimize because the variable b appears in both summations and so they are coupled.<br>
D. This objective is easy to optimize because the number of entries in each factor $\phi$ grows exponentially in its scope.<br>
<b>Answer:</b> C.<br><br>

$$Z(\theta) = \sum_{a, b, c} \phi_{1}(a, b)\phi_{2}(b, c)$$

$\bigstar$ Partition function couples the parameters<br>
-- No decomposition of likelihood<br>
-- No closed form solution to derive a sufficient statistics<br><br>

<b>Example: Log-Likelihood Function</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_4_1_2.png"/></center>
</p>
<p align="justify">
<b>Log-Likelihood for Log-Linear Model</b><br>
We have a dataset $D = \{X_{1}, ..., X_{n} \}$ and k features f.<br>
$$P(X_{1}, ..., X_{n}: \theta) = \frac{1}{Z(\theta)} e^{\sum_{i=1}^{k} \theta_{i}f_{i}(D_{i})}$$

The log-likelihood function is<br>
$$l(\theta : D) = \sum_{i} \theta_{i}(\sum_{m} f_{i}(x[m])) - M\ln Z(\theta)$$

$f_{i}(x[m])$ means a feature f is applied to $m^{th}$ instance<br><br>

The log-partition function is<br>
$$\ln Z(\theta) = \ln \sum_{x} e^{\sum_{i} \theta_{i}f_{i}(x)}$$

<b>The Log-Partition Function</b><br>
$\bigstar$ Theorem:<br>
-- Vector at first derivative<br>
$$\frac{\partial}{\partial \theta_{i}}\ln Z(\theta) = E_{\theta}[f_{i}] = \sum_{x} P_{\theta}(x)f_{i}(x)$$

which means expectation of $f_{i}$ relative to $P_{\theta}$<br><br>

-- Matrix (Hessian)<br>
$$\frac{\partial^{2}}{\partial\theta_{i}\partial\theta_{j}} \ln Z(\theta) = Cov_{\theta}[f_{i}; f_{j}]$$

$\bigstar$ Proof:<br>
$$\frac{\partial}{\partial \theta_{i}}\ln Z(\theta) = \frac{1}{Z(\theta)} \sum_{x} \frac{\partial}{\partial\theta_{i}} e^{\sum_{j}\theta_{j}f_{j}(x)} = \frac{1}{Z(\theta)} \sum_{x} f_{i}(x) e^{\sum_{j}\theta_{j}f_{j}(x)} = \sum_{x} \frac{1}{Z(\theta)} e^{\sum_{j}\theta_{j}f_{j}(x)} f_{i}(x)$$
$$= \sum_{x} P_{\theta}(x)f_{i}(x)$$

Besides, Hessian of $\ln Z(\theta)$ is a convex function<br>
-- a convex function g satisfies $g(\alpha x + (1-\alpha)y) \leq \alpha g(x) + (1-\alpha)g(y)$ in the interval [x, y]<br><br>

Our log-likelihood function is a concave function<br>
$$l(\theta : D) = \sum_{i} \theta_{i}(\sum_{m} f_{i}(x[m])) - M\ln Z(\theta)$$

Because $- M\ln Z(\theta)$ is a concave function and $\sum_{i} \theta_{i}(\sum_{m} f_{i}(x[m]))$ is a linear function about $\theta$.<br><br>

$\bigstar$ Log Likelihood function<br>
-- No local optima<br>
-- Easy to optimize<br><br>

<b>Maximum Likelihood Estimation</b><br>
$$\frac{1}{M}l(\theta : D) = \sum_{i} \theta_{i}(\frac{1}{M} \sum_{m} f_{i}(x[m])) - \ln Z(\theta)$$

A derivative<br>
$$\frac{\partial}{\partial \theta_{i}} \frac{1}{M}l(\theta : D) = E_{D}[f_{i}(X)] - E_{\theta}[f_{i}]$$

$E_{D}[f_{i}(X)]$ denotes an empirical expectation of f in D and $E_{\theta}[f_{i}]$ shows an expectation of f in $P_{\theta}$<br><br>

$\bigstar$ Theroem: $\hat{\theta_{}}$ is the MLE if and only if<br>
$$E_{D}[f_{i}(X)] = E_{\hat{\theta_{}}}[f_{i}]$$

<b>Computation: Gradient Ascent</b><br>
$$\frac{\partial}{\partial \theta_{i}} \frac{1}{M}l(\theta : D) = E_{D}[f_{i}(X)] - E_{\theta}[f_{i}]$$

$\bigstar$ use gradient ascent<br>
-- typically L-BFGS is a quasi-Newton method<br>
$\bigstar$ For gradient, need expected features counts (the critical computation cost):<br>
-- in data $E_{D}[f_{i}(X)]$<br>
-- relative to current model $E_{\theta}[f_{i}]$<br>
$\bigstar$ Requires inference at each gradient step<br><br>

<b>Example: Ising Model</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_4_1_3.png"/></center>
</p>
<p align="justify">
Energy function is composed of a pairwise terms and a singleton terms<br>
$$E(x_{1}, ..., x_{n}) = -\sum_{i < j} w_{i, j} x_{i}x_{j} - \sum_{i} u_{i}x_{i}$$

Gradiant of log-likelihood function is composed of data expectation and model expextation<br>
$$\frac{\partial}{\partial \theta_{i}} \frac{1}{M}l(\theta : D) = E_{D}[f_{i}(X)] - E_{\theta}[f_{i}]$$

For singleton parameters $u_{i}$<br>
$$\frac{\partial}{\partial u_{i}} \frac{1}{M} l(\theta: D) = E_{D} [f_{i}(X)] - E_{\theta}[f_{i}] = \frac{1}{M} \sum_{m} x_{i}[m] - (p_{\theta}(X_{i}=1) + (-1) \cdot P_{\theta}(X_{i} = -1))$$

For pari-wise parameters $w_{i, j}$<br>
$$\frac{\partial}{\partial w_{i, j}} \frac{1}{M} l(\theta: D) = E_{D} [f_{i}(X)] - E_{\theta}[f_{i}]$$
$$= \frac{1}{M} \sum_{m} x_{i}[m]x_{j}[m] - (P_{\theta}(X_{i} = 1, X_{j} = 1) +(-1)(-1) P_{\theta}(X_{i} = -1, X_{j} = -1)$$
$$+ (1)(-1) P_{\theta}(X_{i} = 1, X_{j} = -1) + (-1)(1) P_{\theta}(X_{i} = -1, X_{j} = 1))$$

$$= \frac{1}{M} \sum_{m} x_{i}[m]x_{j}[m] - (P_{\theta}(X_{i} = 1, X_{j} = 1) + P_{\theta}(X_{i} = -1, X_{j} = -1) - P_{\theta}(X_{i} = 1, X_{j} = -1)$$
$$- P_{\theta}(X_{i} = -1, X_{j} = 1))$$

<b>Summary</b><br>
$\bigstar$ Partition function couples parameters in likelihood<br>
$\bigstar$ No closed form solution, but convex optimization<br>
-- solved using gradient ascent (usually L-BFGS)<br>
$\bigstar$ Gradient computation requires inference at each gradient step to compute expected feature counts<br>
$\bigstar$ Features are always within clusters in cluster-graph or clique tree due to family preservation<br>
-- One calibration suffices for all feature expectations.<br><br>
</p>

#### 3.4.2 Maximum Likelihood for Conditional Random Fields
<p align="justify">
<b>Estimation for CRFs</b><br>
$$P_{\theta} (Y \mid x) = \frac{1}{Z_{x}(\theta)} \tilde{P}_{\theta}(x, Y), \quad Z_{x}(\theta) = \sum_{Y} \tilde{P}_{\theta}(x, Y)$$

Dataset is a set of pairs: observed features and targets<br>
$$D = \{(x[m], y[m] \}_{m=1}^{M}$$

The log-conditional likelihood<br>
$$l_{Y \mid X}(\theta : D) = \sum_{m=1}^{M} \ln P_{\theta}(y[m], \mid x[m], \theta)$$

For one pair of data (x[m], y[m])<br>
$$l_{Y \mid X} (\theta: (x[m], y[m])) = (\sum_{i}\theta_{i}f_{i}(x[m], y[m])) - \ln Z_{x[m]}(\theta)$$

The gradient
$$\frac{\partial}{\partial \theta_{i}} \frac{1}{M} l_{Y \mid X} (\theta: D) = \frac{1}{M} \sum_{m=1}^{M} (f_{i}(x[m], y[m]) - E_{\theta}[f_{i}(x[m], Y)])$$

Note that, $x[m]$ is fixed, so $E_{\theta}[f_{i}(x[m], Y)]$ is only about Y.<br><br>

We take image segmentation as an example<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_4_2_1.png"/></center>
</p>
<p align="justify">
s, t denote a superpixel and s, t are near to each other.<br><br>

The gradient of log-conditional likelihood<br>
$$\frac{\partial}{\partial \theta_{i}} l_{Y \mid X} (\theta: (x[m], y[m])) = f_{i}(x[m], y[m]) - E_{\theta}[f_{i}(x[m], Y)]$$

For the fisrt feature parameter $\theta_{1}$<br>
$$\frac{\partial}{\partial \theta_{1}} = \sum_{s} \mathbf{1}_{y_{s}[m]=g}G_{s}[m] - \sum_{s} P_{\theta}(Y_{s} = g \mid x[m])G_{s}[m]$$

For the second feature parameter $\theta_{2}$<br>
$$\frac{\partial}{\partial \theta_{2}} = \sum_{(s, t) \in N} \mathbf{1}_{y_{s}[m] = y_{t}[m]} - \sum_{(s, t) \in N} P_{\theta}(Y_{s} = Y_{t} \mid x[m])$$

Consider a log-linear model over two sets of random variables, X and Y. The model is defined via a set of features $f_{i}(X_{i}, Y_{i})$, such that $Y_{i}$ is nonempty. We consider two training regimes, one where we train the model as an MRF, to optimize $l_{Y, X} (\theta: D)$ and one where we train it as a CRF, to optimize $l_{Y \mid X} (\theta: D)$, <b>Which of the following hold?</b><br>
A. The sufficient statistics and the likelihood functions for both models are the same.<br>
B. The sufficient statistics for both models are the same but the likelihood functions are different.<br>
C. The likelihood functions are the same but sufficient statistics are different.<br>
D. Neither of the two are same for both models.<br>
<b>Answer:</b> B.<br>
Since the feature set is the same, the sufficient statistics are the same. However, the likelihood function for an MRF is based on a joint probability, while the likelihood function for a CRF is based on a conditional probability, so the likelihoods could be different.<br><br>

<b>Computation</b><br>
$\bigstar$ MRF<br>
$$\frac{\partial}{\partial \theta_{i}} \frac{1}{M}l(\theta : D) = E_{D}[f_{i}(X)] - E_{\theta}[f_{i}]$$

-- Requires <b>1</b> inference at each gradient step for the term $E_{\theta}[f_{i}]$<br><br>

$\bigstar$ CRF<br>
$$\frac{\partial}{\partial \theta_{i}} \frac{1}{M} l_{Y \mid X} (\theta: D) = \frac{1}{M} \sum_{m=1}^{M} (f_{i}(x[m], y[m]) - E_{\theta}[f_{i}(x[m], Y)])$$

-- Requires 1 inference for each x[m] at each gradient step. In other word, M inferences at one gradient step, where M is the number of training instance<br><br>

<b>However</b><br>
$\bigstar$ For inference of P(Y | X), we need to compute distribution only over Y.<br>
$\bigstar$ If we learn an MRF, need to compute P(Y, X), which may be much more complex.<br><br>

<b>Summary</b><br>
$\bigstar$ CRF learning very similar to MRF learning<br>
-- Likelihood function is concave<br>
-- Optimized using gradient ascent (usually L-BFGS)<br>
$\bigstar$ Gradient computation requires inference: one per gradient step, data instance<br>
-- c.f. once per gradient step for MRFs<br>
$\bigstar$ But conditional model is often much simpler, so inference cost for CRF, MRF is not the same.<br><br>
</p>

#### 3.4.3 MAP Estimation for MRFs and CRFs
<p align="justify">
<b>Gaussian Parameter Prior</b><br>
$$P(\theta: \sigma^{2}) = \prod_{i=1}^{k} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{\theta_{i}^{2}}{2\sigma^{2}}}$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_4_3_1.png"/></center>
</p>
<p align="justify">
This is a 0-mean variante gaussian. $\sigma$ dictates how firmly we believe that the parameter is close to 0. As $\sigma$ increases, parameters will tend to be further from zero. The $\sigma^{2}$ is called <b>hyperparameter</b>.<br><br>

<b>Laplacian Parameter Prior</b><br>
$$P(\theta : \beta) = \prod_{i=1}^{k} \frac{1}{2\beta} e^{-\frac{\left | \theta_{i} \right |}{\beta}}$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_4_3_2.png"/></center>
</p>
<p align="justify">
$\beta$ is a hyperparameter. As $\beta$ increases, fewer parameters will be exactly 0. As $\beta$ increases, parameters will tend to be further from zero.<br><br>

<b>MAP Estimation & Regularization</b><br>
We have two parameter prior<br>
$$P(\theta: \sigma^{2}) = \prod_{i=1}^{k} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{\theta_{i}^{2}}{2\sigma^{2}}}, \quad P(\theta : \beta) = \prod_{i=1}^{k} \frac{1}{2\beta} e^{-\frac{\left | \theta_{i} \right |}{\beta}}$$

MAP estimation<br>
$$\arg\max_{\theta}P(D, \theta) = \arg\max_{\theta} P(D \mid \theta)P(\theta)$$

where $P(D \mid \theta)$ is the likelihood and $P(\theta)$ is the prior probability.<br><br>

We take a log-form
$$= \arg\max_{\theta} (l(\theta : D) + \log P(\theta))$$

Look at what these negative logarithms $-\log P(\theta)$ look like in the context of these two priors<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_4_3_3.png"/></center>
</p>
<p align="justify">
For a Gaussian Distribution, the negative logarithm is simply a quadratic, so we have a quadratic penalty that pushes the parameters towards zero. This is exactly L2 regularization in Machine Learning.<br><br>

For a Laplacian distribution, the penalty pushes the parameter towards 0 in a way that depends on the absolute value of the parameter. This is equivalently to L1 regularization.<br><br>

Although, both of two penalties push the parameters towards zero, there is a quite difference in terms of the behavior. For L2, when the parameters far away from zero, the quadratic penalty's actually quite severe and it pushes the parameter down dramatically. But as the parameter gets small, there isn't a lot of push at the point to get the parameters to hit to zero exactly. In other word, models that use L2 regularization are going to be dense. That is, a lot of the theta I's are going to be non-zero. For L1, there's a consistent push, towards zero regardless of where in the parameter space we are. So L1 regularization makes models sparse. In fact, the sparser a model is, the easier it is to the inference. So, L1 regularization or the corresponding Laplacian penalty is often used to make the model both more comprehensible as well as computationally more efficient.<br><br>

<b>Summary</b><br>
$\bigstar$ In undirected models, parameters coupling prevents efficient Bayesian estimation.<br>
$\bigstar$ however, can still use parameter priors to avoid overfitting of MAD.<br>
$\bigstar$ Typical priors are L1, L2.<br>
-- drive parameters towards 0.<br>
$\bigstar$ L1 provably induces sparse solutions<br>
-- performances feature selection / structure learning<br><br>
</p>

#### 3.4.4 Quiz
<p align="justify">
<b>1. *MN Parameter Estimation and Inference</b><br>
Consider the process of gradient-ascent training for a log-linear model with k features, given a data set D with M instances. Assume for simplicity that the cost of computing a single feature over a single instance in our data set is constant, as is the cost of computing the expected value of each feature once we compute a marginal over the variables in its scope. Also assume that we can compute each required marginal in constant time after we have a calibrated clique tree.<br>
Assume that we use clique tree calibration to compute the expected sufficient statistics in this model and that the cost of doing this is c. Also, assume that we need r iterations for the gradient process to converge. <b>What is the cost of this procedure?</b> Recall that in big-O notation, same or lower-complexity terms are collapsed.<br>
A. $O(r(Mc + k))$<br>
B. $O(Mk + r(c+k))$<br>
C. $O(r(Mk+c))$<br>
D. $O(Mk+rc)$<br>
<b>Answer:</b> B.<br>
Before we start the gradient ascent process, we compute the empirical expectation for each of the k features by summing over their values for each of the M instances in our data set D; the cost of this is Mk. Then, at each iteration, we use clique tree calibration at a cost c and extract the expected sufficient statistics from calibrated beliefs and update each of the k parameters $\theta_{i}$. Thus, the cost per iteration is c+k, and the total cost for r iterations is r(c+k). Together with the initial computation of empirical expectations, we get a total cost of $O(Mk+r(c+k))$.<br><br>

<b>2. *CRF Parameter Estimation</b><br>
Consider the process of gradient-ascent training for a CRF log-linear model with k features, given a data set D with M instances. Assume for simplicity that the cost of computing a single feature over a single instance in our data set is constant, as is the cost of computing the expected value of each feature once we compute a marginal over the variables in its scope. Also assume that we can compute each required marginal in constant time after we have a calibrated clique tree.<br>
Assume that we use clique tree calibration to compute the expected sufficient statistics in this model, and that the cost of running clique tree calibration is c. Assume that we need r iterations for the gradient process to converge.<br>
What is the cost of this procedure? Recall that in big-O notation, same or lower-complexity terms are collapsed.<br>
A. $O(r(Mc + k))$<br>
B. $O(Mk + rc)$<br>
C. $O(r(Mk+c))$<br>
D. $O(Mk+r(c+k))$<br>
<b>Answer:</b> A.<br>
When training the CRF, at each iteration we need to perform clique tree calibration and compute the expected value of each of the k features M times; thus, the computation at each iteration required M(c+k) operations. Note, however, that we can actually do better: if we aggregate the probabilities from these M clique trees into a single clique tree and then compute the feature value using the aggregated clique tree, we get Mc+k operations per iteration; the procedure is correct due to linearity of expectations.<br><br>

<b>3. Parameter Learning in MNs vs BNs</b><br>
Compared to learning parameters in Bayesian networks, learning in Markov networks is generally<br>
A. equally difficult, as both require an inference step at each iteration.<br>
B. less difficult because we must separately optimize decoupled portions of the likelihood function in a Bayes Net, while we can optimize portions together in a Markov network.<br>
C. more difficult because we cannot push in sums to decouple the likelihood function, allowing independent parallel optimizations, as we can in Bayes Nets.<br>
D. equally difficult, though MN inference will be better by a constant factor difference in the computation time as we do not need to worry about directionality.<br>
<b>Answer:</b> C.<br>
One trick that often makes Bayes Net learning more efficient is our ability to optimize each CPD independently after we have obtained our expected counts. Markov Net learning cannot be decoupled, as the partition function couples all parameters in Markov Nets.<br><br>
</p>

### 3.5 Structure Learning: Overview and Scoring Functions
#### 3.5.1 Structure Learning Overview
<p align="justify">
<b>Why Structure Learning</b><br>
$\bigstar$ To learn model for new queries, when domain expertise is not perfect<br>
$\bigstar$ For structure discovery, when inferring network structure is goal itself.<br><br>

<b>Importance of Accurate Structure</b><br>
Consider the true network $G^{*}$ and the learned network $G_{1}$ and $G_{2}$. Let's define $P^{*}$ such that $G^{*}$ is the perfect map for $P^{*}$.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_5_1_1.png"/></center>
</p>
<p align="justify">
<b>Which of the following statements hold?</b><br>
A. $G_{2}$ can be used to learn $P^{*}$ correctly; $G_{1}$ cannot.<br>
B. Both $G_{1}$ and $G_{2}$ can be used to learn $P^{*}$ correctly.<br>
C. Neither $G_{1}$ nor $G_{2}$ can be used to learn $P_{*}$ correctly.<br>
D. $G_{1}$ can be used to learn $P^{*}$ correctly; $G_{2}$ cannot.<br>
<b>Answer:</b> A.<br><br>

$\bigstar$ For $G_{1}$, we miss an arc<br>
-- Incorrect independencies<br>
-- Correct distribution $P^{*}$ cannot ne learned<br>
-- But could generaliza better<br><br>

$\bigstar$ For $G_{2}$, we add an arc<br>
-- Spurious dependencies<br>
-- Can correctly learn $P^{*}$<br>
-- Increases the number of parameters<br>
-- Worse generalization or worse performance on unseen instances<br><br>

<b>Score-Based Learning</b><br>
Define scoring function that evaluates how well a structure matches the data.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_5_1_2.png"/></center>
</p>
<p align="justify">
Search for a structure that maximizes the score.<br><br>
</p>

#### 3.5.2 Likelihood Scores
<p align="justify">
<b>Likelihood Score</b><br>
$\bigstar$ Find (G, $\theta$) that maximizes the likelihood<br>
$$\text{score}_{L}(G: D) = l((\hat{\theta_{}}, G): D)$$

$\theta_{}$ = MLE of the parameters given G and D.<br><br>

<b>Example</b><br>
We have two different graphs as well as their likelihood score<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_5_2_1.png"/></center>
</p>
<p align="justify">
We calculate their difference<br>
$$\text{score}_{L}(G_{1}: D) - \text{score}_{L}(G_{0}: D) = \sum_{m} (\log\hat{\theta_{}}_{y[m] \mid x[m]} - \log\hat{\theta_{}}_{y[m]})$$

With sufficient statistics M[x, y] and M[y]<br>
$$= \sum_{x, y} M[x, y]\log\hat{\theta_{}}_{y \mid x} - \sum_{y} M[y] \log\hat{\theta_{}}_{y}$$

We difine $\hat{P_{}}$ as an empirical distribution, such that $M[x, y] = M \hat{P_{}}[x, y]$<br>
$$= M\sum_{x, y} \hat{P_{}}(x, y) \log \hat{P_{}}(y \mid x) - M\sum_{y} \hat{P_{}}(y) \log(\hat{P_{}})(y)$$

Because $\sum_{x} \hat{P_{}}(x, y) = \hat{P_{}}(y)$<br>
$$= M(\sum_{x, y} \hat{P_{}}(x, y) \log \hat{P_{}}(y \mid x) - \sum_{x, y} \hat{P_{}}(x, y) \log\hat{P_{}}(y))$$

$$= M(\sum_{x, y} \hat{P_{}}(x, y) \log \frac{\hat{P_{}}(x, y)}{\hat{P_{}}(x)\hat{P_{}}(y)}) = M \cdot \mathbf{I}_{\hat{P_{}}}(X; Y)$$

We call $\mathbf{I}_{\hat{P_{}}}(X; Y)$ <b>mutual information</b>.<br><br>

Given two binary random variables, X and Y, <b>what are the minimum and maximum possible values for the mutual information I(X, Y)?</b><br>
A. min I(X, Y) = -1, max I(X, Y) = 1<br>
B. min I(X, Y) = 0, max I(X, Y) = $\infty$<br>
C. min I(X, Y) = 0, max I(X, Y) = 1<br>
D. min I(X, Y) = 0, max I(X, Y) = 2<br>
<b>Answer:</b> C.<br>
The minimum mutual information occurs when the joint distribution over the values of X and Y is uniform, and is equal to 0. The maximum mutual information is achieved by a number of different distributions, such as $P(x^{0}, y^{0}) = P(x^{1}, y^{1}) = 0.5$ and is equal to 1. In general, we can achieve the maximum mutual information if one variable is a deterministic function of the other variable.<br><br>

<b>General Decomposition</b><br>
$\bigstar$ The likelihood score decompose as a mutual information minus an entropy<br>
$$\text{score}_{L}(G: D) = M \sum_{i=1}^{n} \mathbf{I}_{\hat{P_{}}}(X_{i}; \mathbf{Par}_{X_{i}}^{G}) - M\sum_{i} \mathbf{H}_{\hat{P_{}}}(X_{i})$$

where mutual information<br>
$$\mathbf{I}_{P}(X; Y) = \sum_{x, y}P(x, y)\log\frac{P(x, y)}{P(x)P(y)}$$

entropy is independent of G<br>
$$\mathbf{H}_{P}(X) = \sum_{x}P(x)\log P(x)$$

Score is higher if $X_{i}$ is correlated with its parents.<br><br>

<b>Limitations of Likelihood Score</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_5_2_2.png"/></center>
</p>
<p align="justify">
$$\text{score}_{L}(G_{1}: D) - \text{score}_{L}(G_{0}: D) = M \mathbf{I}_{\hat{P_{}}}(X; Y)$$

$\bigstar$ Mutula information is always $\geq$ 0<br>
$\bigstar$ Equal to 0 if and only if X, Y are independent<br>
-- In empirical distribution<br>
-- statistical fluctuations<br><br>

Assume that X and Y are drawn from a generating distribution in which $X \perp Y$. Which of the following can you conclude about $I_{\hat{P_{}}}(X, Y)$, the mutual information calculated from the empirical distribution?<br>
A. $I_{\hat{P_{}}}(X, Y)$ can be large<br>
B. $I_{\hat{P_{}}}(X, Y) > 0$ some of the time<br>
C. $I_{\hat{P_{}}}(X, Y) > 0$ a small fraction of the time<br>
D. $I_{\hat{P_{}}}(X, Y) = 0$<br>
<b>Answer:</b> B.<br>
In any given random sample from the generating distribution, it is highly unlikely that we will calculate $I_{\hat{P_{}}}(X, Y) = 0$. The lower bound on $I_{\hat{P_{}}}(X, Y)$ still holds, so we conclude that most of the time $I_{\hat{P_{}}}(X, Y) > 0$<br><br>

$\bigstar$ Adding edges can't hurt, and almost always helps<br>
$\bigstar$ Score maximized for fully connected network<br><br>

<b>Avoiding Overfitting</b><br>
$\bigstar$ Restricting the hypothesis space<br>
-- restrict the number of parents or the number of parameters<br>
$\bigstar$ Scores that penalize complexity<br>
-- explicitly<br>
-- Bayesian score averages over all possible parameter values<br><br>

<b>Summary</b><br>
$\bigstar$ Likelihood score computes log-likelihood of D relative to G, using MLE parameters<br>
-- Parameters optimized for D<br>
$\bigstar$ Nice information-theoretic interpretation in terms of (in)dependencies in G<br>
$\bigstar$ Guaranteed to overfit the training data (if we don't impose constraints)<br><br>
</p>

#### 3.5.3 BIC and Asymptotic Consistency
<p align="justify">
<b>Penalizing Complexity</b><br>
$$\text{score}_{BIC}(G: D) = l(\hat{\theta_{}}_{G} : D) - \frac{\log M}{2}\text{Dim}[G]$$

The first term is just the likelihood score $\text{score}_{L}(G: D)$. M denotes the number of training instances and Dim[G] shows the number of independent parameters.<br><br>

The first term, the log likelihood, tries to push towards a better fit to the training data, whereas the second term, the the penalty term, is going to try and keep the number of independent parameters and therefore the network complexity down.<br><br>

$\bigstar$ Tradeoff between fit to data and model complexity<br>
$\bigstar$ Negation of the score.<br>
-- If we negate this entire term that is often called MVL criteron where MVL stands for Minimum Description Length. And so, in fact, this notion of minimum description length is an information theoretic justification.<br>
$\bigstar$ BIC stands for Bayesian Information Criteria<br><br>

<b>Asymptotic behavior</b><br>
$$l(\hat{\theta_{}}_{G} : D) - \frac{\log M}{2}\text{Dim}[G]$$

$$M \sum_{i=1}^{n} \mathbf{I}_{\hat{P_{}}}(X_{i}; \mathbf{Par}_{X_{i}}^{G}) - M\sum_{i} \mathbf{H}_{\hat{P_{}}}(X_{i})- \frac{\log M}{2}\text{Dim}[G]$$

The second term is independent of G. In other word, the complexity of G (dense or sparse graph) doesn't affect it.<br><br>

Which of the following are true as $M \rightarrow \infty$?<br>
A. $\text{score}_{BIC}$ will favor models which fit the data better.<br>
B. $\text{score}_{BIC}$ will favor models which have fewer edges but more parameters per random variable.<br>
C. $\text{score}_{BIC}$ excessively penalize models which overfit the data, leading to underfitting.<br>
D. $\text{score}_{BIC}$ lead to overfitting to the data.<br>
<b>Answer:</b> A.<br>
The first two terms, which increase the score for adding edges between highly variables which depend on each other, increase linearly in M, which the complexity penalty increases more slowly as log(M). Thus, as $M \rightarrow \infty$, the score will reward more complex graphs that fit the data better.<br><br>

$\bigstar$ Mutual information grows linearly with M while complexity grows logarithmically with M<br>
-- as M grows, more empjasis is given to fit to data.<br><br>

<b>Consistency</b><br>
$$M \sum_{i=1}^{n} \mathbf{I}_{\hat{P_{}}}(X_{i}; \mathbf{Par}_{X_{i}}^{G}) - M\sum_{i} \mathbf{H}_{\hat{P_{}}}(X_{i})- \frac{\log M}{2}\text{Dim}[G]$$

$\bigstar$ As $M \rightarrow \infty$, the true structure $G^{*}$ (or any I-equivalent structure) maximizes the score.<br>
-- Asymptotically, spurious edges will not contribute to likelihood and will be penalized<br>
-- Required edges will be added due to linear growth of likelihood term compared to logarithmic growth of model complexity<br><br>

We just learned that using $\text{score}_{BIC}$ does not tend to lead to overfitting to the data. Does it tend to underfit the data, i.e. does it tend to lead to models which are missing edges or dependencies which are present in the generating distribution as $M \rightarrow \infty$?<br>
A. No<br>
B. Yes<br>
C. No general result can be stated because it depends on the form of the generating distribution.<br>
D. No - instead it will add spurious edges.<br>
<b>Answer:</b> A.<br>
If a dependency is present in the generating distribution, it will probably be added to the learned model as $M \rightarrow \infty$ because the reward for adding the edge, given by the mutual information term of the score, grows linearly in M while the penalty for additional model complexity grows as $\log(M)$.<br><br>

<b>Summary</b><br>
$\bigstar$ BIC score explicitly penalizes model complexity (the number of independent parameters)<br>
-- Its negation is often called MDL<br>
$\bigstar$ BIC is asymptotically consistant<br>
-- If data generated by $G^{*}$, networks I-equivalent to $G^{*}$ will have highest score as M grows to $\infty$<br><br>
</p>

#### 3.5.4 Bayesian Scores
<p align="justify">
<b>Bayesian Score</b><br>
$$P(G \mid D) = \frac{P(D \mid G) P(G)}{P(D)}$$

$P(D \mid G)$ denotes marginal likelihood, $P(G)$ is prior over structure and $P(D)$ represents marginal probability of data. Besides, $P(D)$ is independent of G.<br><br>

We can deduce the Bayesian score using the two terms in the numerator<br>
$$\text{score}_{B}(G :  D) = \log P(D \mid G) + \log P(G)$$

This score is going to avoid overfitting because it uses the prior of graph. Although that prior can play a role, actually it turns out to be a significantly less important role than the first of these terms, which is the marginal likelihood.<br><br>

The marginal likelihood<br>
$$P(D \mid G) = \int P(D \mid G, \theta_{G}) P(\theta_{G} \mid G) d\theta_{G}$$

where, $P(D \mid G, \theta_{G})$ is a likelihood but we fix at all $\theta_{G}$ instead of the maximum $\hat{\theta_{}}$ in MLE. $P(\theta_{G} \mid G)$ denote a prior over parameters.<br><br>

<b>Marginal Likelihood Intuition</b><br>
$$P(D \mid G) = P(x[1], ..., x[M] \mid G)$$

Aplly chain rule for probabilities, a product of probabilities of X[i] given its previous instances<br>
$$
\begin{matrix}
P(x[1] \mid G)\\
P(x[2] \mid x[1], G)\\
\cdots\\
P(x[M] \mid x[1], ..., x[M-1], G)
\end{matrix}
$$

Each of these is effectively making a prediction over an unseen instance say $X[M]$ given $X[1], ..., X[M-1]$. So we can think of it as almost doing some kind of cross validation or generalization or estimate of generalization ability because we're estimating the ability to predict an unseen instance given the previous ones.<br><br>

The probability of D in some sense incorporates into it, some kind of analysis of generalization ability. This is exactly the same thing as standard likelihood.<br><br>

The maximum likelihood set of parameters $\hat{\theta_{}}_{G}$ depends on G and we cannot break it down because $\hat{\theta_{}}_{G}$ has already been incorporating into D, all of the instances including the unseen ones, which is why the maximum tends to overfit.<br><br>

<b>Marginal Likelihood: BayesNets</b><br>
$$P(D \mid G) = \prod_{i} \prod_{u_{i} \in Val(Par_{X_{i}}^{G})} \frac{\Gamma(\alpha_{X_{i} \mid u_{i}})}{\Gamma(\alpha_{X_{i} \mid u_{i}} + M[u_{i}])}  \prod_{x_{i}^{j} \in Val(X_{i})} \frac{\Gamma(\alpha_{x_{i}^{j} \mid u_{i}} + M[x_{i}^{j}, u_{i}])}{\Gamma(\alpha_{x_{i}^{j} \mid u_{i}})}$$

$X_{i}$ denote each variable. $\alpha_{X_{i}}$ is Dirichlet prior parameters. $M[u_{i}]$ is sufficient statistics.<br>

$$\Gamma(x) = \int_{0}^{\infty} t^{x-1}e^{-t} dt, \quad \Gamma(x) = x \cdot \Gamma(x-1)$$

<b>Marginal Likelihood Decomposition</b><br>
$$\log P(D \mid G) = \sum_{i} \text{FamScore}_{B}(X_{i} \mid Pa_{X_{i}}^{G}: D)$$

This can be computed quite fast.<br><br>

<b>Structure Priors</b><br>
$$\text{score}_{B}(G :  D) = \log P(D \mid G) + \log P(G)$$

$\bigstar$ Structure prior P(G)<br>
-- Uniform prior: P(G) $\propto$ constant<br>
-- Prior penalizing the number of edges: $P(G) \propto c^{\left | G \right |}, \quad 0< c< 1$<br>
-- Prior penalizing the number of parameters<br>
$\bigstar$ Normalizing constant P(G) across network is similar and can thus be ignored<br><br>

<b>Parameters Priors</b><br>
$\bigstar$ Parameter prior $P(\theta \mid G)$ is usually the BDe prior<br>
-- $\alpha$: equivalent sample size<br>
-- $B_{0}$: network representing prior probability of events<br>
-- Set<br>
$$\alpha(x_{i}, pa_{i}^{G}) = \alpha P(x_{i}, pa_{i}^{G} \mid B_{0})$$

Note: $pa_{i}^{G}$ are not the same as parents of $X_{i}$ in $B_{0}$<br><br>

$B_{0}$ is a network representing the prior probability of joint values of the random variables in our model so our Dirichlet prior parameters $\alpha(x_{i}, pa_{i}^{G})$ are $\alpha P_{B_{0}}(x_{i}, pa_{i}^{G})$, where $\alpha$ in the latter expression is our equivalent sample size. <b>Which of the following is an advantage of this encoding of the parameter priors given G?</b><br>
A. We can use the same $B_{0}$ for all candidate graph structures G<br>
B. We can calibrate a clique tree for $B_{0}$ for each candidate structure G, allowing us to compute marginal probabilities efficiently.<br>
C. The Dirichlet parameters $\alpha(x_{i}, pa_{i}^{G})$ onverge to correct values as $M \rightarrow \infty$<br>
D. The Dirichlet parameters $\alpha(x_{i}, pa_{i}^{G})$ onverge to correct values as $\alpha \rightarrow \infty$<br>
<b>Answer:</b> A.<br>
$B_{0}$ does not depend on the candidate graphs G.<br><br>

$\bigstar$ A single network provides priors for all candidate networks<br>
$\bigstar$ Unique prior with the property that I-equivalent networks have the same Bayesian score.<br><br>

<b>BDe and BIC</b><br>
$\bigstar$ As $M \rightarrow \infty$, a network G with Dirichlet priors satisfies<br>
$$\log P(D \mid G) = l(\hat{\theta_{}}_{G} : D) - \frac{\log M}{2}\text{Dim}[G] + O(1)$$

$l(\hat{\theta_{}}_{G} : D)$ is log-likelihood of Data given MLE $\hat{\theta_{}}_{G}$, this is just the likelihood score. M denotes the number of training instances. $\text{Dim}[G]$ is the number of independent parameters. $O(1)$ is a constant, which means it does't play a role in our socre as M increases.<br><br>

In fact, the first two terms is BIC score<br>
$$\text{score}_{BIC}(G: D) = l(\hat{\theta_{}}_{G} : D) - \frac{\log M}{2}\text{Dim}[G]$$

$M \rightarrow \infty$, score is consistent.<br><br>

<b>Summary</b><br>
$\bigstar$ Bayesian score averages over parameters to avoid overfitting<br>
$\bigstar$ Most often instantiated as BDe<br>
-- BDe requires assessing prior network<br>
-- Can naturally incorporate prior knowledge<br>
-- I-equivalent networks have some score<br>
$\bigstar$ Bayesian score<br>
-- Asymptotically equivalent to BIC<br>
-- Asymptotically consistent<br>
-- But for small M, BIC tends to underfit<br><br>
</p>

#### 3.5.5 Quiz
<p align="justify">
<b>1.</b><br>
Consider the following 4 graphs:<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_5_5_1.png"/></center>
</p>
<p align="justify">
<b>1.1 Likelihood Scores</b><br>
<b>Which of the following statements about the likelihood scores of the different graphs is/are true? </b> You may choose more than 1 option.<br>
A. $\text{Score}_{L}(G1: D) = \text{Score}_{L}(G3: D)$ for every dataset D<br>
B. $\text{Score}_{L}(G1: D) \geq \text{Score}_{L}(G4: D)$ for every dataset D<br>
C. $\text{Score}_{L}(G4: D) \geq \text{Score}_{L}(G2: D)$ for every dataset D<br>
D. $\text{Score}_{L}(G2: D) \geq \text{Score}_{L}(G3: D)$ for every dataset D<br>
<b>Answer:</b> A, D.<br>
For A: I-equivalent graphs will have the same likelihood score, as the ranges of distributions that they can express are the same.<br>
For B, C, D: The more complex a graph, the more its likelihood score is. G4 is an I-map of G1, namely, every independence relation in G4 is also in G1, hence, G4 can represent all distributions that G1 can. So, likelihood score of G4 isn't fewer than that of G1. Similarly, $\text{Score}_{L}(G2: D) \geq \text{Score}_{L}(G3: D)$<br><br>

<b>1.2 BIC Scores</b><br>
Consider the same 4 graphs as in the previous question, but now think about the BIC score. <b>Which of the following statements is/are true?</b>.<br>
A. $\text{Score}_{BIC}(G1: D) = \text{Score}_{BIC}(G3: D)$ for every dataset D<br>
B. $\text{Score}_{BIC}(G2: D) \neq \text{Score}_{BIC}(G3: D)$ for every dataset D<br>
C. $\text{Score}_{BIC}(G1: D) \geq \text{Score}_{BIC}(G4: D)$ for every dataset D<br>
D. $\text{Score}_{BIC}(G1: D) \geq \text{Score}_{BIC}(G4: D)$ for every dataset D<br>
<b>Answer:</b> A.<br>
For A: I-equivalent graphs have the same likelihood score, and have the same complexity (in terms of the number of independent parameters). Hence, they have the same BIC score.<br>
For B: G2 is an I-map of G3, there will be always $\text{Score}_{L}(G2: D) \geq \text{Score}_{L}(G3: D)$. But BIC score is essentially the likelihood score minus a penalty for more complex models, these two components are in opposition. So, it is possible $\text{Score}_{BIC}(G2: D) = \text{Score}_{BIC}(G3: D)$ for some dataset.<br>
For C, D: G4 is an I-map of G1, there will be always $\text{Score}_{L}(G4: D) \geq \text{Score}_{L}(G1: D)$, for large dataset G4 is a much better model than G1, the likelihood score could dominate; for small dataset, the BIC score would be in G1's favor.<br><br>

<b>1.3 Likelihood Guarantees</b><br>
Consider graphs G2 and G3.<br>
We have a dataset D generated from some probability distribution P, and the likelihood scores for G2 and G3 are $\text{Score}_{L}(G2: D)$ and $\text{Score}_{L}(G3: D)$, respectively<br><br>

Let $\theta_{D, 2}^{*}$ and $\theta_{D, 3}^{*}$ be the maximum likelihood parameters for each network, taken with respect to the dataset D.<br><br>

Now let $L(X: G, \theta)$ represent the likelihood of dataset X given the graph G and parameters $\theta$, so 
$$\text{Score}_{L}(G2: D) = L(D: G2, \theta_{D, 2}^{*}), \quad \text{Score}_{L}(G3: D) = L(D: G3, \theta_{D, 3}^{*})$$

Suppose that $L(D: G2, \theta_{D, 2}^{*}) > L(D: G3, \theta_{D, 3}^{*})$.<br><br>

If we draw a new dataset E from the distribution P, <b>which of the following statements can we guarantee?</b> If more than one statement holds, choose the more general statement.<br>
A. $L(E: G2, \theta_{D, 2}^{*}) \leq L(E: G3, \theta_{D, 3}^{*})$<br>
B. $L(E: G2, \theta_{D, 2}^{*}) \neq L(E: G3, \theta_{D, 3}^{*})$<br>
C. $L(E: G2, \theta_{D, 2}^{*}) < L(E: G3, \theta_{D, 3}^{*})$<br>
D. $L(E: G2, \theta_{D, 2}^{*}) > L(E: G3, \theta_{D, 3}^{*})$<br>
E. None of them<br>
<b>Answer:</b> E.<br>
$\theta_{D, 2}^{*}$ and $\theta_{D, 3}^{*}$ correspond to the ML estimation from dataset D. Given the new dataset E, they might not be the ML estimation parameters any longer.<br><br>

<b>2. Hidden Variables</b><br>
Consider the case where the generating distribution has a naive Bayes structure, with an unobserved class variable C and its binary-valued children $X_{1}, ..., X_{100}$. Assume that C is strongly correlated with each of its children (that is, distinct classes are associated with fairly different distributions over each $X_{i}$.<br><br>

<b>2.1</b><br>
Now suppose we try to learn a network structure directly on $X_{1}, ..., X_{100}$, <b>without including C</b> in the network. <b>What network structure are we likely to learn</b> if we have 10,000 data instances, and we are using table CPDs with the <b>likelihood score</b> as the structure learning criterion?<br>
A. Some connected network over $X_{1}, ..., X_{100}$ that is not fully connected nor empty.<br>
B. The empty network, i.e., a network consisting of only the variables but no edges between them.<br>
C. A fully connected network, i.e., one with an edge between every pair of nodes.<br>
<b>Answer:</b> C.<br>
Because of $X_{i} \perp X_{j} \mid C$, $X_{i} \perp X_{j}$ doesn't hold when C is not given. Hence, no variable is independent of any other variable in the network over $X_{1}, ..., X_{100}$.<br><br>

<b>2.2</b><br>
Now suppose that we use the BIC score instead of the likelihood score in the previous question. <b>What network structure are we likely to learn</b> with the same 10,000 data instances?<br>
A. Some connected network over $X_{1}, ..., X_{100}$ that is not fully connected nor empty.<br>
B. The empty network, i.e., a network consisting of only the variables but no edges between them.<br>
C. A fully connected network, i.e., one with an edge between every pair of nodes.<br>
<b>Answer:</b> A.<br>
Even though a fully connected network may be the best representation for the true underlying distribution, we don't have enough data to learn it, and the BIC structure penalty will not allow the learning of a network with such high complexity, given only 10,000 instances.<br><br>
</p>

### 3.6 Searching Over Structures
#### 3.6.1 Learning Tree Structured Networks
<p align="justify">
<b>Optimization Problem</b><br>
$\bigstar$ Input:<br>
-- Training data<br>
-- Scoring function (including priors, if need)<br>
-- Set of possible structues<br>
$\bigstar$ Output: A network thta maximizes the socre<br>
$\bigstar$ Key Property: Decomposability<br>
-- for sake of computational efficiency<br>
$$\text{score}(G: D) = \sum_{i} \text{score} (X_{i} \mid Pa_{X_{i}}^{G}: D)$$

<b>Learning Trees / Forests</b><br>
$\bigstar$ Forest<br>
-- at most one parent per variable<br>

$\bigstar$ Why trees?<br>
-- elegant math<br>
-- efficient optimization<br>
-- sparse parameterization<br><br>

Which of the following is a desirable consequence of the sparse parameterization of tree structured networks when learning a structure?<br>
A. The learning algorithm is resistant to overfitting.<br>
B. The learning algorithm will tend to underfit the data.<br>
C. The learned structure is less likely to be composed of a large number of disconnected trees.<br>
D. The learned structure is more likely to be composed of a large number of disconnected trees.<br>
<b>Answer:</b> A.<br><br>

<b>Learning Forests</b><br>
$\bigstar$ p(i) = parent of $X_{i}$ or 0 if $X_{i}$ has no parent<br>
$$\text{score}(G: D) = \sum_{i} \text{score} (X_{i} \mid Pa_{X_{i}}^{G}: D)$$

So, we unroll this expression with variables with parents and variables without parents<br>
$$= \sum_{i: p(i) > 0} \text{score}(X_{i} \mid X_{p(i)}: D) + \sum_{i: p(i) = 0} \text{score}(X_{i}: D)$$

Then, we merge all variables without parents into the variables with parents<br>
$$= \sum_{i: p(i) > 0}(\text{score}(X_{i} \mid X_{p(i)}: D) -  \text{score}(X_{i} : D)) + \sum_{i=1}^{n} \text{score}(X_{i}: D)$$

We notice, the second term $\sum_{i=1}^{n} \text{score}(X_{i}: D)$ is same for all trees. Besides, we can regard the second term as a score of 'empty' network, whicn means no edges and every variable is independent of each other; the first term is an improvement over this 'empty' network because of some dependencies.<br><br>

$\bigstar$ Score = sum of edges scores + constant<br><br>

<b>Learning Forest I</b><br>
$\bigstar$ Set weight<br>
$$w(i \rightarrow j) = \text{Score}(X_{j} \mid X_{i}) - \text{Score}(X_{j})$$

$\bigstar$ For likelihood score, this weight is actually the mutual information and all edge weights are non-negative.<br>
$$w(i \rightarrow j) = M \mathbf{I}_{\hat{P_{}}}(X_{i}; X_{j})$$

-- Optimal structure is always a tree<br><br>

$\bigstar$ For BIC or BDe, weights can be negative due to a penalizing term<br>
-- Optimal structure might be a forest.<br><br>

<b>Learning Forest II</b><br>
$\bigstar$ A score satisfies score equivalence if I-equivalent structures have the same score<br>
-- such scores include likelihood, BIC and BDe<br>
$\bigstar$ For such a score, we can show $w(i \rightarrow j) = w(j \rightarrow i)$, and use an undirected graph<br>
-- This is logical, because the mutual information has no sign of direction.<br><br>

<b>Learning Forest III -- Algorithm for equivalent scores</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_6_1_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ Define undirected graph with nodes {1, ..., n}<br>
$\bigstar$ Set $w(i, j) = \max[\text{Score}(X_{j} \mid X_{i}) - \text{Score}(X_{j}), 0]$<br>
$\bigstar$ Find forest with maximal weights<br>
-- standard algorithms for max-weight spanning trees (e.g. Prime or Kruskal) in $O(n^{2})$ time<br>
-- remove all edges of weight 0 to produce a forest<br><br>

When do we have to make sure to include the zero term in the expression for $w(i, j) = \max[\text{Score}(X_{j} \mid X_{i}) - \text{Score}(X_{j}), 0]$? Pick up all<br>
A. We are using the likelihood score.<br>
B. We are learning an undirected model.<br>
C. We are using the BIC score.<br>
D. We are using the Bayesian score.<br>
<b>Answer:</b> C, D.<br>
For the Bayesian and BIC scores, we have a penalty for the complexity of the structure, so it is possible that adding an edge can lead to a decrease in the score. Another way of saying that is that $w(i, j)$ can be negative. For likelihood scores, additional edges can only increase the score, so $w(i, j)$ is non-negative.<br><br>

<b>Learning Forest: Example</b><br>
Tree learned from data of Alarm network<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_6_1_2.png"/></center>
</p>
<p align="justify">
$\bigstar$ Not every edge in tree is in the original network<br>
$\bigstar$ Inferred edges are undirected - can't determine direction<br><br>

<b>Summary</b><br>
$\bigstar$ Structure learning is an optimization over the combinatorial space of graph structures<br>
$\bigstar$ Decomposability $\Rightarrow$ network score is a sum of terms for different families<br>
$\bigstar$ optimal tree-structured network can be found using standard MST algorithms<br>
$\bigstar$ Computation takes quadratic time<br><br>
</p>

#### 3.6.2 Learning General Graphs: Heuristic Search
<p align="justify">
<b>Beyond Trees</b><br>
$\bigstar$ Problem is not obvious for general networks<br>
-- Example: allowing two parents, greedy algorithm is no longer guaranteed to find the optimal network<br>
$\bigstar$ Theorem:<br>
-- Finding maximal scoring network structure with at most k parents for each variable is NP-hard for k > 1.<br><br>

<b>Heuristic Search</b><br>
We have an initial network then we try add an edge or remove an edge in order to get a bigger score.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_6_2_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ Search operatiors<br>
-- local steps: edegs addition, deletion, reversal<br>
-- global steps<br>
$\bigstar$ Search techniques:<br>
-- greedy hill-climbing<br>
-- best first search<br>
-- simulated annealing<br><br>

<b>Search: Greedy Hill Climbing</b><br>
$\bigstar$ Start with a given network<br>
-- empty network<br>
-- best tree<br>
-- a random network<br>
-- prior knowledge<br>
$\bigstar$ At each iteration<br>
-- consider score for all possible changes<br>
-- apply change that most improves the score<br>
$\bigstar$ Stop when no modification improves score<br><br>

<b>Greedy Hill Climbing Pitfalls</b><br>
$\bigstar$ Greedy hill-climbing can get stuck in<br>
-- local maxima<br>
-- plateaux: small perturbations of the graph structure lead to no or very small changes in the score. Typically because equivalent networks are often neighbors in the search space.<br><br>

<b>Why Edges Reversal</b><br>
In order to avoid local maxima<br><br>

Given that we can define edge reversal as a sequence of two other operations, edge deletion and edge addition (of the original edge in the reverse direction), <b>why do we need to include it our candidate perturbations of the graph structure in greedy hill climbing?</b><br>
A. Removing the edge as a discrete step in greedy hill climbing will tend to decrease the score, so that initial step of deleting the edge will not be taken.<br>
B. The structure with the original edge is guaranteed to be a local minimum with respect to removing the edge.<br>
C. The BDe and BIC scores will penalize edge removal so that the initial step of deleting the edge is unlikely to be taken.<br>
D. Removing the edge is fine, but we will not be allowed to add the reverse edge back in because the resulting structure will be too similar to the starting structure.<br>
<b>Answer:</b> C, D.<br>
Edge deletion can be a suboptimal step (i.e. lead to a decrease in the structure score) so it will not be taken in greedy hill climbing.<br><br>

<b>A pretty Good, Simple Algorithm</b><br>
Greedy hill-climbing, augmented with<br>
$\bigstar$ Random restarts:<br>
-- when we got stuck, take some number of random steps then start climbing again<br>
$\bigstar$ Tabu list:<br>
-- keep a list of K steps most recently taken<br>
-- search cannot reverse any of these steps<br><br> 

For example,<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_6_2_2.png"/></center>
</p>
<p align="justify">
<b>Summary</b><br>
$\bigstar$ Useful for building better predictive models:<br>
-- when domain experts don't know the structure<br>
-- for knowledge discovery<br>
$\bigstar$ Finding highest-scoring structure is NP-hard<br>
$\bigstar$ Typically solved using simple heuristic search<br>
-- local steps: edge addition, deletion, reversal<br>
-- hill-climbing with tabu list and random restarts<br>
$\bigstar$ But there are better algorithms<br><br>
</p>

#### 3.6.3 Learning General Graphs: Search and Decomposability
<p align="justify">
<b>Naive Computational Analysis</b><br>
$\bigstar$ operations per search step<br>
-- n(n-1) possible edges<br>
-- three operations: delete, revere or add<br>
$\bigstar$ Cost per network evaluation<br>
-- O(n) components in score<br>
-- compute sufficient statistics O(M)<br>
-- acylicity check: O(m) the number of edges<br>
$\bigstar$ Total: $O(n^{2}(Mn+m))$ per search step<br><br>

<b>Exploiting Decomposability</b><br>
Consider a Bayesian network and we want to add an edge from B to D<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_6_3_1.png"/></center>
</p>
<p align="justify">
The original graph's score<br>
$$\text{Score} = \text{Score}(A \mid \{\}) + \text{Score}(B \mid \{\}) + \text{Score}(C \mid \{A, B \}) + \text{Score}(D \mid \{C\})$$

The new graph's score<br>
$$\text{Score'} = \text{Score}(A \mid \{\}) + \text{Score}(B \mid \{\}) + \text{Score}(C \mid \{A, B \}) + \text{Score}(D \mid \{B, C\})$$

In fact, only the last term changes, so we don't need to recalculate the first 3 terms. We can get a score difference<br>
$$\Delta \text{Score}(D) = \text{Score}(D \mid \{B, C\}) - \text{Score}(D \mid \{C\})$$

We can generalize to other operations<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_6_3_2.png"/></center>
</p>
<p align="justify">
Furthermore, if we contunue to delete an edge form B to C, we have a score difference $\Delta \text{Score}(C)$<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_6_3_3.png"/></center>
</p>
<p align="justify">
In fact, $\Delta \text{Score}(C)$ has already been calculated in the previous steps<br><br>

To recompute scores, only need to re-score families that changeed in the last move.<br><br>

<b>Computational Cost</b><br>
$\bigstar$ Cost per move<br>
-- 1-2 families affected by move<br>
-- compute $O(n)$ $\Delta\text{scores}$<br>
-- each takes $O(M)$ time<br>
$\bigstar$ Priority queue of operations sorted by $\Delta\text{scores}$<br>
-- $O(n\log n)$<br><br>

What is the cost of each step of structure search given these optimizations? Assume M = size of dataset, n = number of nodes.<br>
A. $O(Mn^{3})$<br>
B. $O(Mn+n\log n)$<br>
C. $O(Mn\log n)$<br>
D. $Mn^{2}\log n$<br>
<b>Answer:</b> B.<br>
We can avoid recomputing the values of most of the $\Delta\text{scores}$ after taking a move. The cost of updating these deltas is $O(Mn)$ because we have to update $O(n)$ scores, and we have to calculate sufficient statistics for each. Furthermore, if we keep the scores in a priority queue, it only takes $O(n\log n)$ to select a move (it takes $O(\log n)$ to insert an updated delta score, and we have to do that n times; getting the best move takes $O(1)$ time). Thus the time required for each move is $O(Mn+n\log n)$.<br><br>

<b>More Computational Efficiency</b><br>
$\bigstar$ Most plausible families are variations on a theme<br>
$\bigstar$ Reuse and adapt previously computed sufficient statistics<br>
$\bigstar$ Restrict in advance the set of operatiors considered in the search<br>

<b>Summary</b><br>
$\bigstar$ Even heuristic structure search can get expensive for large n<br>
$\bigstar$ Can exploit decomposability to get orders of magnitude reduction in cost<br>
$\bigstar$ Can also exploit recurrence of plausible families in search<br><br>
</p>

#### 3.6.4 Quiz
<p align="justify">
<b>1.</b><br>
You are a detective tracking down a serial robber who has already stolen from 1,000 victims, thereby giving you a large enough training set.<br><br>

No one else has been able to catch him (or her), but you are certain that there is a method (specifically, a Bayesian network) in this madness.<br><br>

<b>1.1 Detective</b><br>
You decide to model the robber's activity with a <b>tree-structured network</b> (meaning that each node has <b>at most</b> one parent): this network has (observed) variables such as the location of the previous crime, the gender and occupation of the previous victim, the current day of the week, etc., and a single unobserved variable, which is the location of the next robbery. The aim is to predict the location of the next robbery.<br><br>

Unfortunately, you have forgotten all of your classical graph algorithms, but fortunately, you have a copy of The Art of Computer Programming (volume 4B) next to you.<br><br>

<b>Which graph algorithm do you look up to help you find the optimal tree-structured network?</b>Assume that the structure score we are using satisfies score decomposability and score equivalence.<br>
A. Finding the maximum-weight undirected spanning forest (i.e., a set of undirected edges such that there is at most one path between any pair of nodes).<br>
B. Finding the shortest path between all pairs of points.<br>
C. Finding an undirected spanning forest with the largest diameter (i.e., the longest distance between any pair of nodes).<br>
D. Finding a directed spanning forest with the largest diameter (i.e., the longest distance between any pair of nodes).<br>
E. Finding the size of the largest clique in a graph.<br>
<b>Answer:</b> A.<br>
The tree-structured Bayesian network that we eventually want to construct is directed, However, if we have score equivalence, finding the maximum-weight undirected spanning forest is equivalent to finding the maximum-weight directed spanning forest and is easier to implement.<br><br>

<b>1.2 *Recovering Directionality</b><br>
Once again, assume that our structure score satisfies score decomposability and score equivalence. After we find the optimal undirected spanning forest (containing n nodes), <b>how can we recover the optimal directed spanning forest (and catch the robber)?</b> If more than one option is correct, pick the faster option; if the options take the same amount of time, pick the more general option.<br>
A. Pick any arbitrary root, and direct all edges away from it. This takes O(n) time.<br>
B. Pick any arbitrary direction for each edge, which takes O(n) time. Because of score equivalence, all possible directed versions of the optimal undirected spanning forest have the same score, so this is valid.<br>
C. Evaluate all possible directions for the edges. While there are at most $2^{n}$ possible sets of edge directions, we can exploit score decomposability to find the best directed spanning forest in O(n) time.<br>
D. Evaluate all possible directions for the edges by iterating over them. This takes $O(2^{n})$ time, since there are at most $2^{n}$ possible sets of edge directions in the spanning forest.<br>
<b>Answer:</b> A.<br>
Not all possible sets of edge directions give rise to a valid tree: for example, the undirected tree A−B−C might give rise to the directed graph $A \rightarrow B \leftarrow C$,<br>
No matter which root we pick, the resulting trees are in the same I-equivalence class; in fact, there are no valid directed trees that cannot be obtained with this procedure. Because of score equivalence, it does not matter which root we pick.<br><br>

<b>1.3 *Augmenting Trees</b><br>
It turns out that the tree-structured network we learnt in the preceding questions was not sufficient to apprehend the robber, allowing him to claim his 1001th victim.<br><br>

Not one to be discouraged, you decide to increase the expressiveness of your network.<br><br>

Assume that we now want to learn a hybrid naive-Bayes/tree-structured network, where we have a single class variable C as well as the variables $X_{1}, ..., X_{n}$.<br><br>

In this model, each $X_{i}$ has C as a parent, and there is also a tree connecting the $X_{i}$; that is, each $X_{i}$ in addition to C, may also have up to one other parent $X_{j}$.<br><br>

For our baseline network $G_{0}$, we are going to use the naive Bayes network, in which each $X_{i}$ has only C as a parent.<br><br>

We are thus aiming to optimize the difference in likelihood scores
$${\rm Score}_L(G : {\cal D}) - {\rm Score}_L(G_0 : {\cal D})$$
Where where D is the training dataset.<br><br>

If we use the appropriate spanning tree algorithm to find the optimal forest structure, <b>what is the correct edge weight to use for $w_{j \rightarrow i}$?</b> In these options, M=1001 is the size of our training dataset, and $I_{\hat{P_{}}}(A, B)$ is the mutual information in the empirical distribution of the variables in set A with the variables in set B.<br>
A. $M \cdot (I_{\hat{P_{}}}(X_{i}; X_{j}, C) - I_{\hat{P_{}}}(X_{i}; C))$<br>
B. $M \cdot (I_{\hat{P_{}}}(X_{i}; X_{j}, C) - I_{\hat{P_{}}}(X_{i}; C)) - H_{\hat{P_{}}}(X_{i})$<br>
C. $M \cdot (I_{\hat{P_{}}}(X_{i}; X_{j}, C) - I_{\hat{P_{}}}(X_{i}; X_{j})) - H_{\hat{P_{}}}(X_{i})$<br>
D. $M \cdot I_{\hat{P_{}}}(X_{i}; X_{j}, C)$<br>
E. $M \cdot I_{\hat{P_{}}}(X_{i}; X_{j}, C) - H_{\hat{P_{}}}(X_{i})$<br>
F. $M \cdot (I_{\hat{P_{}}}(X_{i}; X_{j}, C) - I_{\hat{P_{}}}(X_{i}; X_{j}))$<br>
<b>Answer:</b> A.<br><br>

<b>1.4 Trees vs. Forests</b><br>
Congratulations! Your hybrid naive-Bayes/tree-structured network managed to correctly predict where the criminal would be next, allowing the police to catch him (or her) before the 1002th victim got robbed.<br><br>

The grateful populace beg you to return to studying probabilistic graphical models.<br><br>

While re-watching the video lectures, you begin to wonder if the algorithm we have been using to learn tree-structured networks can produce a forest, rather than a single tree.<br><br>

Assume that we use the likelihood score, and also assume that the maximum spanning forest algorithm breaks ties (between equal-scoring trees) arbitrarily.<br><br>

Which of the following is true? In this question, interpret "forest" to mean a set of two or more disconnected trees.<br>
A. It's possible for the algorithm to produce a forest even though trees will always score more highly, since the algorithm need not find the structure that is globally optimal (relative to the likelihood score).<br>
B. It's theoretically possible for the algorithm to produce a forest. However, this will only occur in very contrived and unrealistic circumstances, not in practice.<br>
C. This algorithm will never produce a forest, since there will always be a tree that has strictly higher score.<br>
D. It's possible for the algorithm to produce a forest, since there are cases in which a forest will have a higher score than any tree.<br>
<b>Answer:</b> B.<br>
A forest will be produced only if we can partition the variables into two disjoint sets A and B, such that all edges $X_{j} \rightarrow X_{i}$ with either $X_{i} \in A, X_{j} \in B$ or $X_{i} \in B, X_{j} \in A$ have weight 0. This will be the case only if all variables in A are independent of the variables in B in the empirical distribution. While this is not impossible, it is very unlikely to happen in practice.<br><br>
</p>

### 3.7 Learning With Incomplete Data
#### 3.7.1 Overview
<p align="justify">
<b>Incomplete Data</b><br>
$\bigstar$ Multiple settings<br>
-- hidden variables<br>
-- missing values<br>
$\bigstar$ Challenges<br>
-- foundational: is the learning task well defined?<br>
-- computational: how can we learn with incomplete data?<br><br>

<b>Why laten variables?</b><br>
$\bigstar$ Model sparsity<br>
Consider a Bayesian network with 7 binary variables<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_1_1.png"/></center>
</p>
<p align="justify">
So we have 17 parameters. Now, we decide H is latent and we are going to learn a network over the observable variables, which are the x's and the y's. Then we have a much more complex network.<br><br>

So, learning a model with the latent variables is by itself a a problematic situation but it may well be worth the tradeoff.<br><br>

$\bigstar$ Discovering clusters in data<br><br>

<b>Treating Missing Data</b><br>
Sample sequence: H, T, ?, ?, H, ?, H<br><br>

$\bigstar$ Case I: A coin is tossed on a table, occasionally it drops and measurements are not taken<br>
$\bigstar$ Case II: A coin is tossed, but sometimes tails are not reported<br><br>

We need to consider the missing data mechanism.<br><br>

<b>Modeling Missing Data Mechanism</b><br>
We deine several notations<br>
$\bigstar$ $\mathbf{X} = \{X_{1}, ..., X_{n} \}$ are random variables<br>
$\bigstar$ $\mathbf{O} = \{O_{1}, ..., O_{n} \}$ are <b>observability variables</b>
-- Always observed<br>
$$
O_{i} =
\begin{cases}
  1, \quad X_{i} \text{ is oberved} \\
  0, \quad \text{otherwise}
\end{cases}
$$

$\bigstar$ $\mathbf{Y} = \{Y_{1}, ..., Y_{n} \}$ new random variables<br>
-- $\text{Val}(Y_{i}) = \text{Val}(X_{i}) \cup \{?\}$<br>
-- Always observed<br>
-- $Y_{i}$ is a deterministic function of $X_{i}$ and $O_{i}$<br>
$$
Y_{i} =
\begin{cases}
  X_{i}, \quad O_{i} = o^{1} \\
  ?, \quad O_{i} = o^{0}
\end{cases}
$$

With thes variables, we can have two models<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_1_2.png"/></center>
</p>
<p align="justify">
For case I, we notice that, the targte and observed value Y depends on X and on O but there is no interaction between X and O. By comparison, case II shows X affects O.<br><br>

When can we ignore the missing data mechanism and focus only on the likelihood? In other word, when can we focus only on the green part in the diagram?<br>
$\bigstar$ Missing at Random (MAR)<br>
$$P_{missing} \models (O \perp H \mid d)$$

This expression denotes the observation variables O are independent of unobserved X given the observed value of Y.<br><br>

<b>Identifiability</b><br>
$\bigstar$ <b>Likelihood can have multiple global maxima.</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_1_3.png"/></center>
</p>
<p align="justify">
$\bigstar$ For example:<br>
-- We can rename the values of the hidden variable H<br>
-- If H has two values, likelihood has two global maxima<br><br>

$\bigstar$ With many hidden variables, there can be an exponential number of global maxima.<br>
$\bigstar$ Multiple local and global maxima can also occur with missing data (not only hidden variables)<br><br>

<b>What does having multiple global optima necessarily mean?</b> Mark all that apply.<br>
A. Even if we find a global optimum, it may not represent the true underlying parameters.<br>
B. If we find a global optima, it represents the true underlying parameters.<br>
C. Finding global optima is easy.<br>
D. Finding any global optimum is difficult.<br>
<b>Answer:</b> A.<br>
The likelihood function doesn't distinguish between the true parameters and other global optima, so there is no way to know if a given set of parameters is the true one using the likelihood function.<br><br>

<b>Likelihood for Complete Data</b><br>
Suppose we have 3 training data for a Bayesian network<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_1_4.png"/></center>
</p>
<p align="justify">
We can compute its likelihood<br>
$$L(D: \theta) = P(x[1], y[1]) \cdot P(x[2], y[2]) \cdot P(x[3], y[3])$$
$$= P(x^{0}, y^{0}) \cdot P(x^{0}, y^{1}) \cdot P(x^{1}, y^{0})$$
$$= \theta_{x^{0}} \cdot \theta_{y^{0} \mid x^{0}} \cdot \theta_{x^{0}} \cdot \theta_{y^{1} \mid x^{0}} \cdot \theta_{x^{1}} \cdot \theta_{y^{0} \mid x^{1}}$$

$\bigstar$ Likelihood decomposes by variables<br>
$\bigstar$ Likelihood decomposes within CPDs<br><br>

What happens if we don't observe X?<br>
$\theta_{Y \mid X}$ is no longer independent of $\theta_{X}$, because there is an active trial between $\theta_{Y \mid X}$ and $\theta_{X}$.<br><br>

<b>Likelihood for Incomplete Data</b><br>
x[1] and x[3] are missing<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_1_5.png"/></center>
</p>
<p align="justify">
Similarly, we compute its likelihood with ignoring x[1] and x[3]<br>
$$L(D: \theta) = P(y^{0}) \cdot P(x^{0}, y^{1}) \cdot P(y^{0})$$
$$= (\sum_{x \in \text{Val}(X)} P(x, y^{0}))^{2} \cdot P(x^{0}, y^{1})$$
$$= (\theta_{x^{0}} \cdot \theta_{y^{0} \mid x^{0}} + \theta_{x^{1}} \cdot \theta_{y^{0} \mid x^{1}})^{2} \cdot \theta_{x^{0}} \cdot \theta_{y^{1} \mid x^{0}}$$

$\bigstar$ Likelihood doesn't decompose by variables<br>
$\bigstar$ Likelihood doesn't decomposes by variables<br>
$\bigstar$ Computing likelihood requires inference<br><br>

<b>Multimodal Likelihood</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_1_6.png"/></center>
</p>
<p align="justify">
<b>Parameter Correlations</b><br>
Consider a Bayesian network with two variables as well as their own parameters<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_1_7.png"/></center>
</p>
<p align="justify">
$\bigstar$ Total of 8 data instances<br>
$\bigstar$ Some X's unobserved<br>
-- activates the v-structure<br><br>

<b>Summary</b><br>
$\bigstar$ Incomplete data arises often in pratice<br>
$\bigstar$ Raises multiple challenges & issues<br>
-- the mechanism for missingness<br>
-- identifiability<br>
-- complexity of likelihood function<br><br>
</p>

#### 3.7.2 Expectation Maximization - Intro
<p align="justify">
<b>Likelihood with Complete / Incomplete Data</b><br>
We can plot a diagram about likelihood based on complete data and incomplete data.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_2_1.png"/></center>
</p>
<p align="justify">
We notice that with incomplete data, likelihood has mutilple local maxima.<br><br>

<b>Gradient Ascent</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_2_2.png"/></center>
</p>
<p align="justify">
$\bigstar$ Follow gradient of likelihood w.r.t. parameters<br>
$\bigstar$ Line search & conjugate gradient methods for fast convergence<br><br>

$\bigstar$ Theorem:<br>
$$\frac{\partial \log P(D \mid \Theta)}{\partial \theta_{x_{i} \mid u_{i}}} = \frac{1}{\theta_{x_{i} \mid u_{i}}} \sum_{m} P(x_{i}, u_{i} \mid d[m], \Theta)$$

m denote data instance index; d[m] is evidence in $m^{th}$ instance (or observed data); ($x_{i}$, $u_{i}$) is $m^{th}$ data instance; $\Theta$ is current parameter values<br><br>

$\bigstar$ Requires computing $P(X_{i}, U_{i} \mid d[m], \Theta)$ for all i, m<br>
$\bigstar$ Can be done with clique-tree algorithm, since $X_{i}$, $U_{i}$ are in the same clique.<br><br>

<b>Gradient Ascent Summary</b><br>
$\bigstar$ Need to run inference over each data instance at every iteration<br>
$\bigstar$ Pros<br>
-- Flexible, can be extended to non table CPDs<br>
$\bigstar$ Cons<br>
-- constrained optimization: need to ensure that parameters define legal CPDs<br>
-- for reasonable convergence, need to combine with advanced method (conjugate gradient, line search)<br><br>

<b>Expectation Maximization (EM)</b><br>
$\bigstar$ Special-purpose algorithm designed for optimizing likelihood functions<br>
$\bigstar$ Intuition<br>
-- Parameter estimation is easy given complete data<br>
-- computing probability of missing data is 'easy' (= inference) given parameters<br><br>

<b>EM Overview</b><br>
$\bigstar$ Pick a starting point for parameters<br>
$\bigstar$ Iterate<br>
-- E-step (Expectation): 'Complete' the data using current parameters<br>
-- M-step (Maximization): Estimate parameters relative to data completion<br>
$\bigstar$ Guaranteed to improve $L(\theta : D)$ at each iteration<br><br>

<b>Expectation Maximization (EM)</b><br>
$\bigstar$ Expectation (E-step):<br>
-- For each data case d[m] and each family X, U compute soft completion<br>
$$P(X, U \mid d[m], \theta^{t})$$

-- Compute the <b>expected sufficient statistics</b> for each x, u<br>
$$\bar{M}_{\theta^{t}}[x, u] = \sum_{m=1}^{M} P(x, u \mid d[m], \theta^{t})$$

$\bigstar$ Maximization (M-step)<br>
-- Treat the expected sufficient statistics (ESS) as if real<br>
-- Use MLE with respect to the ESS<br>
$$\theta_{x \mid u}^{t+1} = \frac{\bar{M}_{\theta^{t}}[x, u] }{\bar{M}_{\theta^{t}}[u]}$$

Given the process we just described, select the statements that are correct.<br>
A. "use completed data as if it were real" corresponds to finding the MLE for the expected sufficient statistics.<br>
B. "complete the missing data" corresponds to finding the expected sufficient statistics for the dataset.<br>
C. "use completed data as if it were real" corresponds to finding the best parameters for all possible completions of the data.<br>
D. "complete the missing data" corresponds to finding the MAP values of the unobserved variables in each instance.<br>
<b>Answer:</b> A, B.<br><br>

<b>Example: Bayesian Clustering</b><br>
Consider a standard naive Bayesian network with a variable Class and a bunch of observed features.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_2_3.png"/></center>
</p>
<p align="justify">
Suppose Class is missing. So, we compute the sufficient statistics and optimize parameters<br>
$$\bar{M}_{\theta}[c] := \sum_{m} P(c \mid x_{1}[m], ..., x_{n}[m], \theta^{t}), \quad \theta_{c}^{t+1} = \frac{\bar{M}_{\theta}[c]}{M}$$

$$\bar{M}_{\theta}[x_{i}, c] := \sum_{m} P(c, x_{i} \mid x_{1}[m], ..., x_{n}[m], \theta^{t}), \quad \theta_{x_{i} \mid c}^{t+1} = \frac{\bar{M}_{\theta}[x_{i}, c] }{\bar{M}_{\theta}[c]}$$

<b>EM Summary</b><br>
$\bigstar$ Need to run inference over each data instance at every itertaion<br>
$\bigstar$ Pros:<br>
-- Easy to implement on top of MEL for complete data<br>
-- Makes rapid progress, especially in early iterations<br>
$\bigstar$ Cons<br>
-- Convergence slows down at later itertaions<br><br>
</p>

#### 3.7.3 Analysis of EM Algorithm
<p align="justify">
<b>More Formal Intuition</b><br>
$\bigstar$ Use current point to construct local approximation<br>
$\bigstar$ Maximize new functions in closed form<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_3_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ d: observed data in instance<br>
$\bigstar$ H: hidden variables in instance<br>
$\bigstar$ h: assignment to H<br>
$\bigstar$ Q(H): distribution over hidden variables<br><br>

The log-likelihood function looks like<br>
$$l(\theta: (d, h)) = \sum_{i=1}^{n} \sum_{(x_{i}, u_{i}) \in  \text{Val}(X_{i}, Pa_{X_{i}})} \mathbf{1}_{(d, h)}[x_{i}, u_{i}] \log \theta_{x_{i} \mid u_{i}}$$

The expectation relative to Q(H)<br>
$$E_{Q(H)}[l(\theta: (d, h))] = \sum_{i=1}^{n} \sum_{(x_{i}, u_{i}) \in  \text{Val}(X_{i}, Pa_{X_{i}})} E_{Q(H)}[\mathbf{1}_{(d, h)}[x_{i}, u_{i}]] \log \theta_{x_{i} \mid u_{i}}$$

$$= \sum_{i=1}^{n} \sum_{(x_{i}, u_{i}) \in  \text{Val}(X_{i}, Pa_{X_{i}})} Q(x_{i}, u_{i}) \log \theta_{x_{i} \mid u_{i}}$$

Now we have a set pf distributions<br>
$$Q_{m}^{t}(H[m]) = P(H[m] \mid d[m], \theta^{t})$$

Then we sum all instances up<br>
$$\sum_{m=1}^{M} E_{Q_{m}^{t}(H[m])} [l(\theta: (d[m], H[m]))] = \sum_{i=1}^{n} \sum_{(x_{i}, u_{i}) \in  \text{Val}(X_{i}, Pa_{X_{i}})} \sum_{m=1}^{M} P(x_{i}, u_{i} \mid d[m], \theta^{t}) \log \theta_{x_{i} \mid u_{i}}$$

In fact, $\sum_{m=1}^{M} P(x_{i}, u_{i} \mid d[m], \theta^{t})$ is our Expected Sufficient Statistics $\bar{M}_{\theta^{t}}[x_{t}, u_{i}]$<br>
$$= \sum_{i=1}^{n} \sum_{(x_{i}, u_{i}) \in  \text{Val}(X_{i}, Pa_{X_{i}})} \bar{M}_{\theta^{t}}[x_{t}, u_{i}] \log \theta_{x_{i} \mid u_{i}}$$

So, the final result shows we compute a log-likelihood for complete data using ESS.<br><br>

<b>EM Guarantees</b><br>
$\bigstar$ $L(D: \theta^{t+1}) \geq L(D: \theta^{t})$<br>
-- Each iteration improves the likehood<br>
$\bigstar$ if $\theta^{t+1} = \theta^{t}$, then $\theta^{t}$ is a stationary point of $L(D: \theta)$<br>
-- Usually, this means a local maxima<br><br>
</p>

#### 3.7.4 EM in Practice
<p align="justify">
<b>EM Convergence in Practice</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_4_1.png"/></center>
</p>
<p align="justify">
The log likelihood increases consistently and monotonically over iterations. Besides, at iteration 10, the log-likelihood cobverges but the parameters' values don't, so likelihood space is not the same as convergence in parameter space.<br><br>

<b>Overfitting</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_4_2.png"/></center>
</p>
<p align="justify">
$\bigstar$ Early stopping using cross validation<br>
$\bigstar$ Use MAP with parameter priors rather than MLE<br><br>

We are using EM to estimate parameters given a training set $D_{tr}$ and we wish to decide when to stop the EM iterations. <b>Which of the following should we use as a stopping criterion?</b> Mark all that apply.<br>
A. We stop when the log-likelihood on the training set $D_{tr}$ starts to decrease.<br>
B. None of the above.<br>
C. We stop when the log-likelihood on the validation set $D_{cv}$ starts to decrease.<br>
D. We stop when the log-likelihood on the test set $D_{test}$ starts to decrease.<br>
<b>Answer:</b> C.<br>
Training set performance will never decrease, and you never use your test set for selecting any parameters, only for evaluating performance.<br><br>

<b>Local Optima</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_4_3.png"/></center>
</p>
<p align="justify">
<b>Significance of Local Optima</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_4_4.png"/></center>
</p>
<p align="justify">
<b>Initialization if Critical</b><br>
$\bigstar$ Multiple random restarts<br>
$\bigstar$ From prior knowledge<br>
$\bigstar$ From the output of a simpler algorithm<br>
-- clustering, K-mean<br><br>

<b>Summary</b><br>
$\bigstar$ Convergence of likelihood $\neq$ convergence of parameters<br>
$\bigstar$ Running to convergence can lead to overftiing<br>
$\bigstar$ Local optima are unavailable, and increase with the amount of missing data<br>
$\bigstar$ Local optima can be very different<br>
$\bigstar$ Initialization is critical<br><br>
</p>

#### 3.7.5 Latent Variables
<p align="justify">
<b>Discovering User Clusters</b><br>
A naive Bayesian model<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_5_1.png"/></center>
</p>
<p align="justify">
<b>MSNBC Story Clusters</b><br>
Several categories or clusters<br>
$\bigstar$ Readers of commence and technology stories (36%)<br>
-- Email delivery isn't exactly guaranteed<br>
-- Should you buy a DVD player?<br>
-- Price low, demande high for Nintendo<br>
$\bigstar$ Readers of top promoted stories (29&)
-- 757 Crashest At Sea<br>
-- Isreal, Palestinians Agree to Direct' Talks<br>
-- Fuhrman Pleads Innoncent To Perjury<br>
$\bigstar$ Sports Readers (19%)<br>
-- Umps refusing to work is the right thing<br>
-- Cowboys are reborn in win over eagles<br>
-- Did Orioles spend money wisely?<br>
$\bigstar$ Readers of 'Softer' News (12%)<br>
-- The truth about what things cost<br>
-- Fuhrman Pleads Innoncent To Perjury<br>
-- Read Astrology<br><br>

<b>Speech Recognition HMM</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_5_2.png"/></center>
</p>
<p align="justify">
<b>3D Robot Mapping</b><br>
$\bigstar$ Input: Point cloud form laser range finder obtained by moving robot<br>
$\bigstar$ Output: 3D planar map of environment<br>
$\bigstar$ Parameters: Location & angle of walls<br>
$\bigstar$ Laten variables: Assignment of points to walls<br><br>

<b>Picking Latent Variable Cardinality</b><br>
$\bigstar$ If we use likelihood for evaluation, more values ia always better<br>
$\bigstar$ Can use score that penalizes complexity<br>
-- BIC - tends to underfit<br>
-- Extensions of BDe to incomplete data<br>
$\bigstar$ Can use metrics of cluster coherence to decide whether to add / remove clusters<br>
$\bigstar$ Bayesian methods (Dirichlet processes) can average over different cardinalities<br><br>

<b>Summary</b><br>
$\bigstar$ Latent variables are perhaps the most common scenario for incomplete data<br>
-- often a critical component in constructing models for richly structured domians<br>
$\bigstar$ Laten variables satisfy MAR, so we can use EM<br>
$\bigstar$ Serious issues with unidentifiability & multiple optima necessitate good initialization<br>
$\bigstar$ Picking variable cardinality is a key question<br><br>
</p>

#### 3.7.6 Quiz
<p align="justify">
<b>1. Missing At Random</b><br>
Suppose we are conducting a survey of job offers and salaries for Stanford graduates. We already have the major of each of these students recorded, so in the survey form, each graduating student is only asked to list up to two job offers and salaries he/she received. <b>Which of the following scenarios is/are missing at random (MAR)?</b><br>
A. The database software ignored salaries of 0 submitted by students who were taking unpaid positions with community service organizations.<br>
B. Students who accepted a low-salaried job offer tended not to reveal it.<br>
C. The person recording the information accidentally lost some of the completed survey forms.<br>
D. The person recording the information didn't care about humanities students and neglected to record their salaries.<br>
E. CS students get more offers than other majors and found selecting only two too stressful, so they neglected to list any.<br>
<b>Answer:</b> C, D, E.<br>
We say data is MAR if whether the data is missing is independent of the missing values themselves given the observed values. This is MAR because whether the data is missing depends on random loss that does not correspond to salary. In fact, this is MCAR.<br><br>

<b>2. Computing Sufficient Statistics</b><br>
Given the network and data instances shown below,
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_7_6_1.png"/></center>
</p>
<p align="justify">
how do we compute the expected sufficient statistics for a particular value of the parameters?<br>
A. $\bar{M}[x_{0}, y_{0}, z_{0}] = 3$<br>
B. $\bar{M}[x_{0}, y_{0}, z_{0}] = P(z_{0} \mid x_{0}, \theta) + P(z_{0} \mid x_{0}, y_{0}, \theta) + P(z_{0} \mid y_{0}, \theta)$<br>
C. $\bar{M}[x_{0}, y_{0}, z_{0}] = P(y_{0}, z_{0} \mid x_{0}, \theta) + P(z_{0} \mid x_{0}, y_{0}, \theta) + P(z_{0} \mid x_{1}, y_{1}, \theta) + P(z_{0} \mid x_{0}, y_{1}, \theta) + P(x_{0}, z_{0} \mid y_{0}, \theta)$
D. $\bar{M}[x_{0}, y_{0}, z_{0}] = P(y_{0}, z_{0} \mid x_{0}, \theta) + P(z_{0} \mid x_{0}, y_{0}, \theta) + P(x_{0}, z_{0} \mid  y_{0}, \theta)$<br>
<b>Answer:</b> D.<br>
The term $P(z_{0} \mid x_{1}, y_{1}, \theta)$ includes a probability over the instance $(x_{1}, y_{1}, z_{0})$ which is not consistent with the assignment for which we are trying to accumulate a count.<br><br>

<b>3. Likelihood of Observed Data</b><br>
In a Bayesian Network with partially observed training data, computing the likelihood of observed data for a given set of parameters...<br>
A. cannot be achieved by probabilistic inference, while it CAN in the case of fully observed data.<br>
B. requires probabilistic inference, while it DOES NOT in the case of fully observed data.<br>
C. requires probabilistic inference, AS IN the case of fully observed data.<br>
<b>Answer:</b> B.<br>
With missing data, inference is required to complete the expected sufficient statistics (ESS) for the expected likelihood function. Thus, inference is not needed to compute the ESS in the case of fully observed data.<br><br>

<b>4. PGM with latent variables</b><br>
Adding hidden variables to a model can significantly increase the expressiveness of a model. However, there are also some issues that arise when we try to add hidden variables. <b>For which of these problems can we learn a reasonable model by simply choosing the parameters that maximize training likelihood?</b> Assume that all variables, hidden or (partially) observed, are discrete and follow a table CPD. You may choose more than one option.<br>
A. Choosing which edges involving only the observed nodes to add to the graph.<br>
B. Given a fixed set of edges, learning the parameters in the table CPDs of each hidden node.<br>
C. Choosing the number of hidden variables to add to the graphical model.<br>
D. Given a fixed set of edges, learning the parameters in the table CPDs of observed nodes that have hidden nodes as parents.<br>
<b>Answer:</b> B, D.<br>
For A: Training set likelihood will always increase with every edge we add, so we will end up with a complete graph over all variables. This is because the addition of an edge always increases the expressiveness of a graph, in terms of the probability distributions that it can possibly represent.<br>
For C: Training set likelihood will always increase with the number of hidden variables we add, so we will end up with infinitely many hidden variables. This is because for any two integers N > n, the set of graphs with N hidden nodes is more expressive; it can represent whatever probability distributions the set of graphs with n hidden nodes can, and possibly more.<br>
For B, D: This is a standard parameter estimation with missing data problem that we can solve with methods such as the EM algorithm. While using only training likelihood runs the risk of overfitting, given a large enough training set, the parameters found are still likely to perform reasonably well.<br><br>

<b>5. Bayesian Clustering using Normal Distributions</b><br>
Suppose we are doing Bayesian clustering with K classes, and multivariate normal distributions as our class-conditional distributions.<br><br>

Let $X \in R^{n}$ represent a single data point, and $C \in \{1, 2,..., K\}$ its unobserved class.<br><br>

<b>Which of the following statement(s) is/are always true in the general case?</b><br>
A. $P(X \mid C = c) \sim N(\mu_{c}, \Sigma_{c}) , \forall c \in \{1, 2, ..., K \}$ for some class-specific parameters $\mu_{c}$ and $\Sigma_{c}$ that represent the distribution of data coming from the class c.<br>
B. $X_{i} \perp C \mid X_{-i}, \forall i \in \{1, 2, ..., n\}$, i.e., for any given data point. If we know all but one coordinate, then knowing the class that it comes from doesn't give us any information about the last coordiante<br>
C. $X_{i} \perp X_{j} \mid C, \forall i, j \in \{1, 2, ..., n\} \text{ and } i \neq j$, i.e., for any given data point. If wen know which class the data point comes from, then knowing one coordiante doesn't give us any information about another coordinate<br>
D. $P(X \mid C = c) \sim N(\mu_{c}, I_{n}) , \forall c \in \{1, 2, ..., K \}$, for some class-specific parameter $\mu_{c}$ that represents the distribution of data coming from the class c ($I_{n}$ is the $n \times n$ identity matrix)<br>
E. $P(X) \sim N(\mu, \Sigma)$ for some parameters $\mu$ and $\Sigma$ that represent the overall distribution of the data<br>
<b>Answer:</b> A.<br>
This is true only if the covariance matrices of the class-conditional distributions are diagonal. In the general case, the coordinates might be correlated, which means that knowing one coordinate can give us information about another coordinate even if we are given the class and therefore know the exact distribution of the data point.<br>
In applications of Bayesian clustering with class-conditional normal distributions, the covariance matrices of the class-conditional distributions are not necessarily restricted to be the identity matrix.<br>
A is the definition of having a multivariate normal distribution as the class-conditional distribution: given the class from which the data point came from, the distribution of the data point follows a multivariate normal distribution with mean and covariance parameters specific to its particular class.<br><br>

<b>6. Hard Assignment EM</b><br>
Continuing from the previous question, let us now fix each class-conditional distribution to have the identity matrix as its covariance matrix.<br><br>

If we use hard-assignment EM to estimate the class-dependent mean vectors, <b>which of the following can we say about the resulting algorithm?</b><br>
A. It is an instance of k-means, but using a different distance metric rather than standard Euclidean distance.<br>
B. It is equivalent to running standard k-means clustering with K clusters.<br>
C. It is an an algorithm that cannot be viewed as an instance of k-means.<br>
<b>Answer:</b> B.<br>
You will always assign vertices to their closest (in Euclidean distance) cluster centroid, just as in k-means.<br><br>

<b>7. *Hard Assignment EM</b><br>
Now suppose that we fix each class-conditional distribution to have the same diagonal matrix D as its covariance matrix, where D is not the identity matrix.<br>
If we use hard-assignment EM to estimate the class-dependent mean vectors, <b>which of the following can we say about the resulting algorithm?</b><br>
A. It is an an algorithm that cannot be viewed as an instance of k-means.<br>
B. It is equivalent to running standard k-means clustering with K clusters.<br>
C. It is an instance of k-means, but using a different distance metric rather than standard Euclidean distance.<br>
<b>Answer:</b> C.<br>
Consider a case of the cartesian plane with points $\{v_{1}, ..., v_{n}\}$. If you initialize k clusters among these points and try running the above process in 2 dimensions, consider how it relates to k-means with Euclidean distance and k-means with other distance metrics (such as a weighted distance).<br>
You will always assign vertices to their closest cluster centroid, just as in k-means. But here the definition of ``closest'' is skewed by the covariance matrix so that it does not equally depend on each dimension and is thus not a Euclidean distance.<br><br>

<b>8. EM Running Time</b><br>
Assume that we are trying to estimate parameters for a Bayesian network structured as a binary (directed) tree (not a polytree) with n variables, where each node has at most one parent. We parameterize the network using table CPDs. Assume that each variable has d values. We have a data set with M instances, where some observations in each instances are missing. What is the tightest asymptotic bound you can give for the worst case running time of EM on this data set for K iterations? In this and following questions, concentrate on the EM part of the learning algorithm only. You don't need to consider the running time of additional steps if the full learning algorithm needs any.<br>
A. $O(KMn^{2}d^{2})$<br>
B. $O(KMnd^{2})$<br>
C. Can't tell using only the information given<br>
D. $O(KMn^{2}d)$<br>
<b>Answer:</b> B.<br>
Tree structure leaves only 2 variables in the scope of each clique (rather than all n).<br>
At each iteration and for every instance, it is required to run exact inference over the given network. Using clique-tree calibration, the cost of inference is the number of cliques (n) times the size of the clique potential which is $d^{2}$ (due to the tree-structure of the network each clique can have only 2 variables in its scope).<br><br>

<b>9. EM Running Time</b><br>
Use the setting of the previous question, but now we assume that the network is a polytree, in which some variables have several parents. What is the cost of running EM on this data set for K iterations?
A. $O(KMn^{2}d^{2})$<br>
B. $O(KMnd^{2})$<br>
C. Can't tell using only the information given<br>
D. $O(KMn^{2}d)$<br>
<b>Answer:</b> C.<br>
In a polytree, there is no guarantee that our factors will only be of size $d^{2}$<br>
We cannot tell because now the factors in the clique tree can be considerably larger than $d^{2}$ (but we do not how much larger they might be).<br><br>

<b>10. *Optimizing EM</b><br>
Now, going back to the tree setting, where each node has at most one parent, assume that we are in a situation where at most 2 variables in each data instance are unobserved (not necessarily the same 2 in each instance). Can we implement EM more efficiently? If so, which of the following reduced complexities can you achieve?<br>
A. $O(KMd^{2})$<br>
B. $O(KMnd^{2})$<br>
C. No computational savings can be achieved<br>
D. $O(K(M+n)d^{2})$<br>
<b>Answer:</b> D.<br>
In this case, the cost of the E-step is $Md^{2}$ since we can easily compute the probabilities of each possible completion of instances when only up to 2 variables are missing. We can use this to compute the expected sufficient statistics (where we will be summing over the M instances and up to $d^{2}$ possible completed instances). The cost of the M-step will be $nd^{2}$ (it is equal to the number of parameter values computed).<br><br>

<b>11. *Optimizing EM</b><br>
Still in the tree setting, now assume that we are in a situation where at most 2 variables in each data instance are unobserved, but it's the same 2 each instance. Can we implement EM more efficiently? If so, which of the following reduced complexities can you achieve?<br>
A. $O(KMd^{2})$<br>
B. $O(KMnd^{2})$<br>
C. No computational savings can be achieved<br>
D. $O(K(M+n)d^{2})$<br>
<b>Answer:</b> A.<br>
In this case, most of the graph is conditionally independent of the unobserved variables, so we should be able to avoid inference over all n variables at each iteration.<br>
In this case, most of the graph is conditionally independent of the unobserved variables, so we can restrict our EM process to the sub-graph consisting of the unobserved variables and their Markov blankets, and fix parameters for the rest of the network once at the beginning, using standard MLE. Thus, the cost of updating a small subset of the parameters at each M-step will be no more than $O(d^{2})$. In the E-step, for each instance, we will run inference over a small subset of variables at a cost of $d^{2}$ per instance. Accordingly, the cost of the E-step will be $Md^{2}$.<br><br>
</p>

### 3.8 Learning: Wrapup
#### 3.8.1 Summary: Learning
<p align="justify">
<b>Learning from 10K Feet</b><br>
$\bigstar$ Hypothesis (model) space<br>
$\bigstar$ Objective function<br>
$\bigstar$ Optimization algorithm<br><br>

<b>Hypothesis Space</b><br>
$\bigstar$ What are we searching for<br>
-- Parameters<br>
-- Structures<br>
$\bigstar$ Imposing constraints<br>
-- For computational efficiency<br>
-- To reduce model capacity<br>
-- To incorporate prior knowledge<br><br>

<b>Objective Function</b><br>
$\bigstar$ Penalized likelihood<br>
-- $l((G, \theta_{G}): D) + R(G, \theta_{G})$<br>
-- Parameter prior, e.g. (MRFs - L2/L1), (BNs - Dirichlet)<br>
-- Structure complexity penalty<br>
$\bigstar$ Bayesian score: integrating parameters<br>
-- $\log P(G \mid D) = \log P(D \mid G) + \log P(G) + \text{Const}$<br>
-- $\log P(D \mid G)$ is marginal likelihood and $\log P(G)$ is graph prior<br><br>

<b>Optimization Algorithm</b><br>
$\bigstar$ Continuous<br>
-- Closed form: BNs with multinomial<br>
-- Gradient ascent: MRF learning and missing data<br>
-- EM: learning missing data<br>
$\bigstar$ Discrete<br>
-- Max spanning tree<br>
-- Hill-climbing: add, delete an edge<br>
$\bigstar$ Discrete + continuous<br><br>

<b>Hyperparameters</b><br>
$\bigstar$ Model hyperparameters<br>
-- Equivalent sample size for parameter prior<br>
-- Regularization strength for L1 and L2<br>
-- Stopping criterion for EM<br>
-- Strength of structure penalty<br>
-- Set of features<br>
-- # of values of laten variables<br>
$\bigstar$ Optimization on validation set<br>
-- Train on training data while evaluate on validation set<br>
-- Cross validation<br><br>

<b>Model Evaluation Criteria</b><br>
$\bigstar$ Log-likelihood on test set<br>
$\bigstar$ Task-specific objective<br>
-- Segmentation accuracy<br>
-- Speech recognition<br>
$\bigstar$ 'Match' with prior knowledge<br><br>

<b>Troubleshooting: Underfitting</b><br>
$\bigstar$ Training & test performance both low<br>
$\bigstar$ Solutions<br>
-- Decrease regularization<br>
-- Reduce structure penalties<br>
-- Add features vias error analysis<br><br>

<b>Troubleshooting: Overfitting</b><br>
$\bigstar$ Training performance is high but test performance is low<br>
$\bigstar$ Solutions<br>
-- Increase regularization<br>
-- Impose capacity constraints<br>
-- Reduce feature set<br><br>

<b>Troubleshooting: Optimization</b><br>
$\bigstar$ Optimization may not be converging to good / global optimum<br>
-- Can happen even if problem is convex<br>
$\bigstar$ Compare different learning rates, different random initializations<br><br>

<b>Troubleshooting: Objective Mismatch</b><br>
$$\text{Objective}(M_{1}) >> \text{Objective}(M_{2})$$
$$\text{Performance}(M_{1}) << \text{Performance}(M_{2})$$

$\bigstar$ Need to redesign objective to match desired performance criterion<br><br>

<b>Typical Learning Loop</b><br>
$\bigstar$ Design model 'template'<br>
$\bigstar$ Select hyperparameters via CV (cross validation) on training set<br>
$\bigstar$ Train on training set with chosen hyperparameters<br>
$\bigstar$ Evaluate performance on held-out set<br>
$\bigstar$ Error analysis & model redesign<br>
$\bigstar$ Report results on seperate test set<br><br>
</p>

#### 3.8.2 Quiz
<p align="justify">
<b>1. *Multiplexer CPDs</b><br>
<b>What is the form of the independence that is implied by the multiplexer CPD and that we used in our derivation of the posterior over the parameters of the simple Bayesian Network $x \rightarrow y$?</b> (i.e. the factorization of $P(\theta_{x}, \theta_{Y \mid x^{1}}, \theta_{Y \mid x^{0}} \mid D)$. Recall that a CPD is defined as a multiplexer if it has the structure $P(Y \mid A, Z_{1}, ..., Z_{k}) = \mathbf{I}_{Y = Z_{a}}$ where the values of A are the natural numbers 1 through k. Also note that the answer is specific to multiplexer CPDs and is not implied by the graph structure alone.<br>
A. $\theta_{Y \mid X} \perp X \mid \theta_{X}$<br>
B. $\theta_{Y \mid X} \perp \theta_{X}$<br>
C. $\theta_{Y \mid x^{0}}, \theta_{Y \mid x^{1}} \perp X$<br>
D. $\theta_{Y \mid x^{0}} \perp \theta_{Y \mid x^{1}} \mid X$<br>
E. $\theta_{Y \mid x^{0}} \perp \theta_{Y \mid x^{1}} \mid X, Y$<br>
<b>Answer:</b> E.<br><br>

<b>2. *Score Consistency</b><br>
Assume that the dataset D has m examples, each drawn independently from the distribution $P^{*}$, for which the graph $G^{*}$ is a perfect map. <b>What do we mean when we say that the BIC score $\text{Score}_{BIC}(G: D)$ measured with respect to D, is consistent?</b> Hint: We are looking for a definition that will always be true, not just probably be true.<br>
A. As $m \rightarrow \infty$, with probability 1 we will draw a dataset D from $P^{*}$ such that the inequality
$$\text{Score}_{BIC}(G^{*}: D) > \text{Score}_{BIC}(G: D)$$
holds for all other graphs $G \neq G^{*}$<br>
B. As $m \rightarrow \infty$, no matter which example where drawn from $P^{*}$ into the dataset D, the inequality
$$\text{Score}_{BIC}(G^{*}: D) > \text{Score}_{BIC}(G: D)$$
will always be true for all graphs G which are not I-equivalent to $G^{*}$<br>
C. As $m \rightarrow \infty$, with probability 1 we will draw a dataset D from $P^{*}$ such that the inequality
$$\text{Score}_{BIC}(G^{*}: D) > \text{Score}_{BIC}(G: D)$$
holds for all other graphs G which are not I-equivalent to $G^{*}$<br>
D. As $m \rightarrow \infty$, no matter which example where drawn from $P^{*}$ into the dataset D, the inequality
$$\text{Score}_{BIC}(G^{*}: D) > \text{Score}_{BIC}(G: D)$$
will always be true for all graphs $G \neq G^{*}$<br>
<b>Answer:</b> C.<br><br>

<b>3. EM and Convergence</b><br>
When checking for the convergence of the EM algorithm, we can choose to measure changes in either the log-likelihood function or in the parameters.<br><br>

For a generic application, we typically prefer to check for convergence using the log-likelihood function.<br><br>

However, this is not always the case, especially when the values of the parameters are important in and of themselves.<br><br>

<b>In which situations would we also be concerned about reaching convergence in terms of the parameters?</b> Do not worry about the implementation details in the following models.<br>
A. We have a graphical model in which each node represents an object part, and we are using EM to learn the parameters that specify the relations between object parts. Our end-goal is to build an image classification system that can accurately recognize the image as one of several known objects.<br>
B. We have a graphical model in which each node is a superpixel, and we are using EM to learn the parameters that specify the relations between superpixels. Our end-goal is to build an image segmentation pipeline that is highly accurate.<br>
C. We are trying to transcribe human speech by building a Hidden Markov Model (HMM) and learning its parameters with the EM algorithm. The end-goal is correctly transcribing raw audio input into words.<br>
D. We are building a graphical model for medical diagnosis, where nodes can represent symptoms, diseases, predisposing factors, and so on. Our only aim is to maximize our chances of correctly predicting diseases that patients are suffering from.<br>
E. We are building a graphical model for medical diagnosis, where nodes can represent symptoms, diseases, predisposing factors, and so on. This system will not be deployed in the clinic; our only aim is to understand how various predisposing factors can interact with each other in increasing disease risk.<br>
F. We are trying to better understand high-energy physics by using a graphical model to analyze time-series data from particle accelerators. The hope is to elucidate the types of interactions between different particle types.<br>
G. We are building a graphical model to represent a biological network, where each node corresponds to a gene. We want to learn the interactions between genes by finding the parameters that maximize the likelihood of a given training dataset of gene expression measurements. The interactions we find will then be further studied by biologists.<br>
<b>Answer:</b> E, F, G.<br><br>

<b>4. Parameter Estimation with Missing Data</b><br>
<b>The process of learning Bayesian Network parameters with missing data (partially observed instances) is more difficult than learning with complete data for which of the following reasons?</b> You may select one or more options.<br>
A. We lose local decomposition, whereby each CPD can be estimated independently.<br>
B. We require more training data, because we must throw out all incomplete instances.<br>
C. Because there can be multiple optimal values, we must always run our learning algorithm multiple times from different initializations to make sure we find ALL of them.<br>
D. While there is still always a single optimal value for the parameters, it can only be found using an iterative method.<br>
<b>Answer:</b> A.<br><br>

<b>5. Optimality of Hill Climbing</b><br>
ack and Jill come up to you one day with a worried look on their face. "All this while we've been climbing hills, trying to improve upon our graph structure," they say. "We've been considering edge deletions, reversals, and additions at each step. Today, we found that no single edge deletion, reversal, or addition could give us a higher-scoring structure. <b>Are we guaranteed that our current graph is the best graph structure?</b>"<br>
What should you tell them? You may assume that their dataset is sufficiently large, and that your answer should hold for a general graph.<br>
A. No - greedy hill-climbing will only find the true graph structure if we restrict the number of parents for each node to at most 2.<br>
B. No - greedy hill-climbing will find only local maxima of the scoring function with respect to our available moves. While it might find the true graph structure on occasion, we cannot guarantee this.<br>
C. Yes, but only if we use random restarts and tabu search.<br>
D. Yes - greedy hill-climbing provably finds the true graph structure, provided our dataset is large enough.<br>
E. Yes, but only if we extend our range of available moves to allow for pairs of edges to be changed simultaneously.<br>
F. No - greedy hill-climbing can never find the true graph structure, only local maxima of the scoring function with respect to our available moves.<br>
<b>Answer:</b> B.<br><br>

<b>6. *Latent Variable Cardinality</b><br>
Assume that we are doing Bayesian clustering, and want to select the cardinality of the hidden class variable.<b>Which of these methods can we use?</b> Assume that the structure of the graph has already been fixed. You may choose more than one option.<br>
A. Training several models, each with a different cardinality for that hidden variable. For each model, we choose the (table CPD) parameters that maximize the likelihood on the training set. We then pick the model with the highest test set likelihood.<br>
B. Training several models, each with a different cardinality for that hidden variable. For each model, we choose the (table CPD) parameters that maximize the likelihood on the training set. We then pick the model that performs the best on some external evaluation task, using a held-out validation set. For example, say we are using Bayesian clustering to classify customers visiting an online store, with the aim of giving class-specific product recommendations. We could run each model in an alpha-beta testing framework (where different customers may see the result of different models), and measure the percentage of customers that end up purchasing what each model recommends.<br>
C. If we have relevant prior knowledge, we can simply use this to set the cardinality by hand.<br>
D. Training several models, each with a different cardinality for that hidden variable. For each model, we choose the (table CPD) parameters that maximize the likelihood on the training set. We then pick the model with the highest likelihood on a held-out validation set.<br>
E. Training several models, each with a different cardinality for that hidden variable. For each model, we choose the (table CPD) parameters that maximize the likelihood on the training set. We then pick the model with the highest training set likelihood.<br>
<b>Answer:</b> B, C, D.<br><br>

<b>7. EM Stopping Criterion</b><br>
When learning the parameters $\theta \in R^{n}$ of a graphical model using the EM algorithm, an important design decision is choosing when to stop training.<br><br>

Let $l_{\text{Train}}(\theta)$, $l_{\text{Valid}}(\theta)$ and $l_{\text{Test}}(\theta)$ be the log-likelihood of the parameters $\theta$ on the training set, a held-out validation set, and the test set, respectively.<br><br>

Let $\theta^{t}$ be the parameters at the $t^{th}$ iteration of the EM algorithm.<br><br>

We can denote the change in the dataset log-likelihoods at each iteration with
$$\Delta l_{\text{Train}}^{t} = l_{\text{Train}}(\theta^{t}) - l_{\text{Train}}(\theta^{t-1})$$

and the corresponding analogues for the validation set and the test set. Likewise, let $\Delta \theta^{t} = \theta^{t} - \theta^{t-1}$ be the vector of changes in the parameters at time step t.<br><br>

<b>Which of the following would be reasonable conditions for stopping training at iteration t?</b> You may choose more than one option.<br>
A. $l_{\text{Test}}$ becomes negative<br>
B. $l_{\text{Test}}$ becomes small, i.e., it falls below a certain tolerance $\epsilon > 0$<br>
C. $\left \| \Delta \theta^{t} \right \|_{2}^{2}$ becomes small, i.e., it falls below a certain tolerance $\epsilon > 0$<br>
Note: The $l_{2}$ norm, also known as the Euclidean norm, is defined for any vector $x \in R^{n}$ as $\left \| x \right \|_{2}^{2} = \sum_{i=1}^{n} x_{i}^{2}$<br>
D. $l_{\text{Valid}}^{t}$ becomes negative<br>
<b>Answer:</b> C, D.<br><br>

<b>8. EM Parameter Selection</b><br>
Once again, we are using EM to estimate parameters of a graphical model. We use n random starting points $\{\theta_{i}^{0}\}_{i = 1, 2, ..., n}$ and run EM to convergence from each of them to obtain a set of candidate parameters $\{\theta_{i}^{0}\}_{i = 1, 2, ..., n}$. We wish to select one of these candidate parameters for use. As in the previous question, let $l_{\text{Train}}(\theta)$, $l_{\text{Valid}}(\theta)$ and $l_{\text{Test}}(\theta)$ be the log-likelihood of the parameters $\theta$ on the training set, a held-out validation set, and the test set, respectively. <b>Which of the following methods of selecting final parameters $\theta$ would be a reasonable choice?</b> You may pick more than one option.<br>
A. Pick $\theta = \arg\max_{i=1, 2, ..., n} l_{\text{Valid}}(\theta_{i})$<br>
B. Pick $\theta = \arg\max_{i=1, 2, ..., n} l_{\text{Test}}(\theta_{i})$<br>
C. Pick $\theta = \arg\max_{i=1, 2, ..., n} l_{\text{Train}}(\theta_{i})$<br>
D. Any one; the $\theta_{i}$ are all equivalent, since all of them are local maxima of the log-likelihood function<br>
<b>Answer:</b> A, C.<br><br>

<b>9. Greedy Hill-Climbing</b><br>
Your friend is performing greedy hill-climbing structure search over a network with three variables using three possible operations and the BIC score with dataset $\cal{D}$:<br>
-- Add an edge<br>
-- Delete an edge<br>
---Reverse an edge<br>
She tells you that after examining $\cal{D}$, she took a single step and got the following graph:<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_8_2_1.png"/></center>
</p>
<p align="justify">
She also tells you that for the next step she has determined that there is a unique optimal greedy operation o to take. <b>Which of the following steps could 
o be?</b> Hint: The fact that it is unique eliminates some possibilities for o<br>
A. Add edge $C \rightarrow A$<br>
B. Add edge $A \rightarrow C$<br>
C. Reverse edge $A \rightarrow B$<br>
D. Add edge $C \rightarrow B$<br>
E. Delete edge $A \rightarrow B$<br>
F. Add edge $B \rightarrow C$<br>
<b>Answer:</b> D, F.<br><br>

<b>10. Graph Structure Search</b><br>
Consider performing graph structure search using a decomposable score. Suppose our current candidate is graph G below.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_8_2_2.png"/></center>
</p>
<p align="justify">
We want to compute the changes of scores associated with applying three different operations:<br>
-- Delete the edge $A \rightarrow D$<br>
-- Reverse the edge $C \rightarrow E$<br>
-- Add the edge $F \rightarrow E$<br>
Let $\delta(G: o_{1})$, $\delta(G: o_{2})$, $\delta(G: o_{3})$ denote the score changes associated with each of these three operations, respectively. <b>Which of the following equations is/are true for all datasets $\cal{D}$?</b><br>
A. 
$$\begin{align*} \delta(G : o_2) = &\textrm{FamScore}(C,\{A,E\} : {\cal D})\ + \textrm{FamScore}(E, \emptyset : {\cal D}) \\ &- \ \textrm{FamScore}(C,\{A\} : {\cal D})\ - \ \textrm{FamScore}(E,\{C\} : {\cal D}) \end{align*}$$
B. 
$$\delta(G : o_1) = \textrm{FamScore}(D,\{B,C\} : {\cal D})\ - \ \textrm{FamScore}(D,\{A,B,C\} : {\cal D})$$
C.
$$\delta(G : o_1) = \textrm{FamScore}(D,\{B,C\} : {\cal D})$$
D.
$$\delta(G : o_3) = \textrm{FamScore}(E,\{C,F\} : {\cal D})\ - \ \textrm{FamScore}(E,\{C\} : {\cal D})$$
E.
$$\delta(G : o_3) = \textrm{FamScore}(C,\{A,E\} : {\cal D})$$
F.
$$\delta(G : o_2) = \textrm{FamScore}(C,\{A,E\} : {\cal D})\ - \ \textrm{FamScore}(C,\{A\} : {\cal D})\ - \ \textrm{FamScore}(E,\{C\} : {\cal D})$$
<b>Answer:</b> A, B, D.<br><br>

<b>11. Structure Learning with Incomplete Data</b><br>
After implementing the pose clustering algorithm in PA9, your friend tries to pick the number of pose clusters K for her data by running EM and evaluating the log-likelihood of her data for different values of K. <b>What happens to her log-likelihood as she varies K?</b><br>
A. Impossible to say - depends on the data and on what K is.<br>
B. The log-likelihood remains the same regardless of K.<br>
C. The log-likelihood (almost) always increases as K increases.<br>
D. The log-likelihood (almost) always decreases as K increases.<br>
<b>Answer:</b> C.<br><br>

<b>12. Calculating Likelihood Differences</b><br>
While doing a hill-climbing search, you run into the following two graphs, and need to choose between them using the likelihood score.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/3_8_2_3.png"/></center>
</p>
<p align="justify">
<b>What is the difference in likelihood scores, $\text{score}_{L}(G_{1}: D) - \text{score}_{L}(G_{2}: D)$, given a dataset D of size M?</b> Give your answer in terms of the entropy H and mutual information I. The subscripts below denote empirical values according to D: for example, $H_{D}(X)$ is the empirical entropy of the variable X in the dataset D.<br>
A. $M \times [\mathbf{I}_{D}(C; A, B) + \mathbf{I}_{D}(D; C) + \mathbf{I}_{D}(E; D) - \mathbf{I}_{D}(B; A) - \mathbf{I}_{D}(D; C, E)]$<br>
B. $M \times \mathbf{I}_{D}(A; B)$<br>
C. $M \times [\mathbf{I}_{D}(D; C) + \mathbf{I}_{D}(E; D) - \mathbf{I}_{D}(A; B) - \mathbf{I}_{D}(D; C, E) - \mathbf{H}_{D}(A, B, C, D, E)]$<br>
D. $M \times [\mathbf{I}_{D}(D; C) + \mathbf{I}_{D}(E; D) - \mathbf{I}_{D}(B; A) - \mathbf{I}_{D}(D; C, E)]$<br>
E. $M \times [\mathbf{I}_{D}(A; B) - \mathbf{H}_{D}(A, B)]$<br>
<b>Answer:</b> D.<br><br>
</p>


## 4. PGM Wrapup
### 4.1 Summary
#### 4.1.1 PGM Course Summary
<p align="justify">
<b>Why PGM?</b><br>
$\bigstar$ PGMs are the marriage of statistics and computer science<br>
-- Statistics: Sound probabilistic foundations<br>
-- Computer science: Data structure and algorithms for exploiting them<br><br>

<b>Declarative Representation</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/4_1_1_1.png"/></center>
</p>
<p align="justify">
<b>When PGM?</b><br>
$\bigstar$ When we have noisy data and uncertainty<br>
$\bigstar$ When we have lots of prior knowledge<br>
$\bigstar$ When we wish to reason about multiple variables<br>
$\bigstar$ When we want to construct richly structured models from modular building blocks<br><br>

<b>Intertwined Design Choices</b><br>
$\bigstar$ Representation<br>
-- affects cost of inference & learning<br>
$\bigstar$ Inference algorithm<br>
-- Used as a subroutine in learning<br>
-- Some are only usable in certain types of models<br>
$\bigstar$ Learning algorithm<br>
-- Learnability imposes modeling constraints<br><br>

<b>Example: Image Segmentation</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/4_1_1_2.png"/></center>
</p>
<p align="justify">
$\bigstar$ BNs vs MRFs vs CRFs<br>
-- Naturalness of model<br>
-- Using rich features<br>
-- Inference costs<br>
-- Training cost<br>
-- Learning with missing data<br><br>

<b>Mix & Matching: Modeling</b><br>
$\bigstar$ Mix directed & undirected edges<br>
$\bigstar$ E.g. image segmentation form unlabeled images<br>
-- Undirected edges over labels S - natural directionality<br>
-- Directed for $P(X_{i} \mid S_{i})$ - easy learning (w / o inference)<br><br>

<b>Mix & Matching: Inference</b><br>
$\bigstar$ Apply different inference algorithm to different parts of model<br>
$\bigstar$ E.g. combine approximate inference (BP or MCMC) with exact inference over subsets of variables<br><br>

<b>Mix & Matching: Learning</b><br>
$\bigstar$ Apply different learning algorithms to different parts of model<br>
$\bigstar$ E.g. combine high-accuracy, easily-trained model (e.g. SVM) for node potentials P(S | X) with CRF learning for higher order potentials<br><br>

<b>Summary</b><br>
$\bigstar$ Integrated framework for reasoning and learning in complex, uncertain domains<br>
-- Large bag of tools wthin single framework<br>
$\bigstar$ Used in a huge range of applications<br>
$\bigstar$ Much work to be done, both on applications and on foundational methods<br><br>
</p>


## 5. Honor Code
<p align="justify">
All Matlab codes are available <a href="https://github.com/chaopan95/PROJECTS/tree/master/Probabilistic-Graphical-Models-Honor-Code">here</a>.<br><br>
</p>

### 5.1 Simple BN
#### 5.1.1 Programming Assignment
<p align="justify">
<b>Constructing the network</b><br>
Your friend at the bank, hearing of your newfound expertise in probabilistic graphical models, asks you to help him develop a predictor for whether a person will make timely payments on his/her debt obligations, like credit card bills and loan payments. In short, your friend wants you to develop a predictor for credit-worthiness. He tells you that the bank is able to observe a customer’s income, the amount of assets the person has, the person’s ratio of debts to income, the person’s payment history, as well as the person’s age. He also thinks that the credit-worthiness of a person is ultimately dependent on how reliable a person is, the person’s future income, as well as the person’s ratio of debts to income. As such, he has created a skeleton Bayesian network containing the 8 relevant variables he has mentioned, and defined the possible values they can take in Credit_net.net. However he has trouble defining the connections between these variables and their CPDs, so he has asked you to help him.<br><br>

He hopes that you can help him encode into the network the following observations he has made from his experience in evaluating people’s credit-worthiness:<br>
$\bigstar$ 1. The better a person’s payment history, the more likely the person is to be reliable.<br>
$\bigstar$ 2. The older a person is, the more likely the person is to be reliable.<br>
$\bigstar$ 3. Older people are more likely to have an excellent payment history.<br>
$\bigstar$ 4. People who have a high ratio of debts to income are likely to be in financial hardship and hence less likely to have a good payment history.<br>
$\bigstar$ 5. The higher a person’s income, the more likely it is for the person to have many assets.<br>
$\bigstar$ 6. The more assets a person has and the higher the person’s income, the more likely the person is to have a promising future income.<br>
$\bigstar$ 7. All other things being equal, reliable people are more likely to be credit-worthy than unreliable people. Likewise, people who have promising future incomes, or who have low ratios of debts to income, are more likely to be credit-worthy than people who do not.<br><br>

Here is the network diagram<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/5_1_1_1.png"/></center>
</p>
<p align="justify">
Before implementing our code, it's necessary to introduce a basic data structure to represent factors or probabilities.<br><br>

<b>Data structure for factors</b><br>
Imagine three binary variables and consider a conditional probability $P(X_{3} \mid X_{1}, X_{2})$ can be represented by a factor $\phi(X_{3}, X_{1}, X_{2})$. We notice in this factor, there are 3 vriable $X_{1}$, $X_{2}$ and $X_{3}$. Besides, each variable has its own scope or cardinality. Therefore, we have a set of assignments to these 3 variables, similarly we will have a set of values for each assignment. Here one assignment denote a group of values for all variables in one factor.<br><br>

We can show such a table CPD here.<br>
<table class="a">
  <tr><th>$X_{3}$</th><th>$X_{1}$</th><th>$X_{2}$</th><th>$\phi(X_{3}, X_{1}, X_{2})$</th></tr>
  <tr><td>1</td><td>1</td><td>1</td><td>phi.val(1)</td></tr>
  <tr><td>2</td><td>1</td><td>1</td><td>phi.val(2)</td></tr>
  <tr><td>1</td><td>2</td><td>1</td><td>phi.val(3)</td></tr>
  <tr><td>2</td><td>2</td><td>1</td><td>phi.val(4)</td></tr>
  <tr><td>1</td><td>1</td><td>2</td><td>phi.val(5)</td></tr>
  <tr><td>2</td><td>1</td><td>2</td><td>phi.val(6)</td></tr>
  <tr><td>1</td><td>2</td><td>2</td><td>phi.val(7)</td></tr>
  <tr><td>2</td><td>2</td><td>2</td><td>phi.val(8)</td></tr>
</table><br>
</p>
<p align="justify">
So, a factor has such a data structure<br>
</p>

{% highlight Matlab %}
factor = struct('var', [], 'card', [], 'val', []);
{% endhighlight %}

<p align="justify">
'var' denotes all varibles in the factor, usualy indexed by 1, 2, ... etc. 'card' represents a scope for some variable, e.g. a binary variable's cardinality is 2 because of only 2 possible assignments. 'val' shows a probability value with a specific group assignment, e.g. $\phi(X_{3}=1, X_{1}=1, X_{2}=1)$. If a factor is normalized, sum over all its val is equal to 1, namely, sum($\phi.val$)<br><br>

Because the particularity of Matlab, we assignment some variable or array starting from 1 and this will appliy to all other programming assignments.<br><br>

<b>Code implementation [60 points]</b><br>
$\bigstar$ FactorProduct.m [10 points]<br>
-- This function should compute the product of two factors<br>
If two factors A, B prodcut, we will get a new factor C which keeps a same data structure as A, B. But the difference is that C.var should be a union of (A.var + B.var). Correspondingly, we have to compute C.val by multpliy A with B. For example, A is a $\phi(X_{1})$<br>
<table class="a">
  <tr><th>$X_{1}$</th><th>$\phi(X_{1})$</th></tr>
  <tr><td>1</td><td>0.11</td></tr>
  <tr><td>2</td><td>0.89</td></tr>
</table><br><br>

B is a factor $\phi(X_{2}, X_{1})$ (a conditional probability $P(X_{2} \mid X_{1})$).<br>
<table class="a">
  <tr><th>$X_{2}$</th><th>$X_{1}$</th><th>$\phi(X_{2}, X_{1})$</th></tr>
  <tr><td>1</td><td>1</td><td>0.59</td></tr>
  <tr><td>2</td><td>1</td><td>0.41</td></tr>
  <tr><td>1</td><td>2</td><td>0.22</td></tr>
  <tr><td>2</td><td>2</td><td>0.78</td></tr>
</table><br>
</p>
<p align="justify">
We want to know a joint probability when $X_{1} = x_{1}, X_{2} = x_{2}$<br>
$$\phi_{C}(X_{1} = x_{1}, X_{2} = x_{2}) = \phi_{A}(X_{1} = x_{1}) \cdot \phi_{B}(X_{2} = x_{2}, X_{1} = x_{1})$$

So, a simple algorithm to compute a product of two factors<br>
1) unify all variables as well as their cardinalities in A and B<br>
2) enumerate all assignements (e.g. one assignment is like $X_{1} = x_{1}, X_{2} = x_{2}$) and compute A.val $\times$ B.val<br><br>

So, a final C is like<br>
<table class="a">
  <tr><th>$X_{1}$</th><th>$X_{2}$</th><th>$\phi(X_{1}, X_{2})$</th></tr>
  <tr><td>1</td><td>1</td><td>0.0649</td></tr>
  <tr><td>2</td><td>1</td><td>0.1958</td></tr>
  <tr><td>1</td><td>2</td><td>0.0451</td></tr>
  <tr><td>2</td><td>2</td><td>0.6942</td></tr>
</table><br>
</p>
<p align="justify">
Although $P(X_{2} \mid X_{1}) \cdot P(X_{1}) = P(X_{2})$ according to chain rule of probability, we keep $\phi(X_{1}, X_{2})$ for sake of consistence.<br><br>

$\bigstar$ FactorMarginalization.m [10 points]<br>
-- This function should sum over the given variables in a given factor and return the resulting factor.<br>
For example, we marginalize factor $P(X_{2} \mid X_{1}$ over variable $X_{2}$<br>
$$\sum_{X_{2}} P(X_{2} \mid X_{1}) = P(X_{1})$$

So, a result is like<br>
<table class="a">
  <tr><th>$X_{1}$</th><th>$\phi(X_{1})$</th></tr>
  <tr><td>1</td><td>1</td></tr>
  <tr><td>2</td><td>1</td></tr>
</table><br>
</p>
<p align="justify">
$\bigstar$ ObserveEvidence.m [10 points]<br>
-- This function should modify a set of factors given the observed values of some of the variables, so that assignments not consistent with the observed values are set to zero (in effect, reducing them). These factors do not need to be renormalized.<br>
For example, we have observed $X_{2} = 1$, then factor $\phi(X_{2}, X_{1})$ should eliminate all rows not containing $X_{2}$ to get a new factor.<br>
<table class="a">
  <tr><th>$X_{2}$</th><th>$X_{1}$</th><th>$\phi(X_{2}, X_{1})$</th></tr>
  <tr><td>1</td><td>1</td><td>0.59</td></tr>
  <tr><td>2</td><td>1</td><td>0</td></tr>
  <tr><td>1</td><td>2</td><td>0.22</td></tr>
  <tr><td>2</td><td>2</td><td>0</td></tr>
</table><br>
</p>
<p align="justify">
$\bigstar$ ComputeJointDistribution.m [10 points]<br>
-- This function should return a factor representing the joint distribution given a set of factors that define a Bayesian network. You may assume that you will only be given factors defining valid CPDs, so no input validation is required.<br>
This function is simple to write if we have well implemented fatcor product<br>
</p>

{% highlight Matlab %}
% Empty factor
Joint = struct('var', [], 'card', [], 'val', []);
for i = 1:length(F)
    Joint = FactorProduct(Joint, F(i));
end;
{% endhighlight %}

<p align="justify">
$\bigstar$ ComputeMarginal.m [20 points]<br>
-- This function should return the marginals over input variables (the input variables are those that remain in the marginal), given a set of factors that define a Bayesian network, and, optionally, evidence.<br>
A simple algorithm to compute marginal<br>
1) compute joint distribution<br>
2) observe some evidence to set some irrelevant row 0<br>
3) marginalize joint distribution over some variables<br>
</p>

{% highlight Matlab %}
% Empty factor
M = struct('var', [], 'card', [], 'val', []);
Joint = ComputeJointDistribution(F);
Joint = ObserveEvidence(Joint, E);
Joint.val = Joint.val/sum(Joint.val);
% E + V <= Joint.var
% is not necessarily equal to the whole variables set
M = FactorMarginalization(Joint, setdiff(Joint.var, V));
{% endhighlight %}

<p align="justify">
Be careful that V + E is not equal to all variables. For example, we have a joint distribution $\phi(X_{1}, X_{2}, X_{3})$, we observe $X_{3} = 1$ and we want to marginalize over $X_{1}$. Besides, V denotes variables left in the final result, in other word, we marginalize over all variables which doesn't belong to V.<br><br>
</p>

### 5.2 Genetic Inheritance
#### 5.2.1 Programming Assignment
<p align="justify">
<b>Bayesian Network</b><br>
We sue template model to construct our Bayesian network for genetic inheritance. We have learned that a phenotype depends on its genotype and a genotype is produced by its two parent genotype.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/5_2_1_1.png"/></center>
</p>
<p align="justify">
<b>Decoupled Bayesian Network</b><br>
In a decoupled Bayesian network, a phenotype has two parents, one is a copy of genotype1 and the other is a cope of genotype 2. One anvantage of using decoupled Bayesian network is fewer parameters required to calculate.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/5_2_1_2.png"/></center>
</p>
<p align="justify">
<b>Sigmoid CPDs</b><br>
A sigmoid CPD is used to calculate a probability of phenotype with regards to more than one genotypes.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/5_2_1_3.png"/></center>
</p>
<p align="justify">
<b>Code implementation [78 points]</b><br>
$\bigstar$ phenotypeGivenGenotypeMendelianFactor.m (5 points)<br>
-- This function computes the probability of each phenotype given the different genotypes for a trait with Mendelian inheritance, meaning that each allele combination causes a specific phenotype with probability 1.<br>
Suppose we have a phenotype (binary variables) which depends on a genotype (FF, Ff, ff). If this is a dominant alleles, both FF and Ff cause a dominant phenotype; while a recessive allele ff will cause a recessive phnotype. For example, $X_{3}$ is a phenotype (1 denotes dominant, 2 denotes recessive) and its genotype $X_{1}$ (1: FF, 2: Ff, 3: ff) is dominant. Then a table CPD<br>
<table class="a">
  <tr><th>$X_{3}$</th><th>$X_{1}$</th><th>$\phi(X_{3}, X_{1})$</th></tr>
  <tr><td>1</td><td>1</td><td>1</td></tr>
  <tr><td>2</td><td>1</td><td>0</td></tr>
  <tr><td>1</td><td>1</td><td>1</td></tr>
  <tr><td>2</td><td>2</td><td>0</td></tr>
  <tr><td>1</td><td>3</td><td>0</td></tr>
  <tr><td>2</td><td>3</td><td>1</td></tr>
</table><br>
</p>
<p align="justify">
$\bigstar$ phenotypeGivenGenotypeFactor.m (5 points)<br>
-- This function computes the probability of each phenotype given the different genotypes for a trait.<br>
If people who are FF (genotype 1) have an 80% chance of having cystic fibrosis, people who are Ff (genotype 2) have a 60% chance of having the disease, and people who are ff (genotype 3) have a 10% chance of having the disease, then α1 = 0.8, α2 = 0.6, and α3 = 0.1. We take such an alpha list to generate a phenotype<br>
<table class="a">
  <tr><th>$X_{3}$</th><th>$X_{1}$</th><th>$\phi(X_{3}, X_{1})$</th></tr>
  <tr><td>1</td><td>1</td><td>0.8</td></tr>
  <tr><td>2</td><td>1</td><td>0.2</td></tr>
  <tr><td>1</td><td>1</td><td>0.6</td></tr>
  <tr><td>2</td><td>2</td><td>0.4</td></tr>
  <tr><td>1</td><td>3</td><td>0.1</td></tr>
  <tr><td>2</td><td>3</td><td>0.9</td></tr>
</table><br>
</p>
<p align="justify">
$\bigstar$ genotypeGivenAlleleFreqsFactor.m (5 points)<br>
-- This function computes the probability of each genotype given the allele frequencies in the population. You may assume that the alleles assort independently in the population, so that the probability of having a specific genotype is simply the product of the frequencies of its constituent alleles in the population.<br>
If we have an allele F and f with 0.1 and 0.9 respectively causing a dominant and recessive phenotype, then FF has 0.1*0.1 = 0.01 probability, Ff has 2*0.1*0.9 = 0.18 and ff has 0.9*0.9 = 0.81.<br><br>

$\bigstar$ genotypeGivenParentsGenotypesFactor.m (8 points)<br>
-- This function creates a factor in which the values are the probability of each genotype for a child given each possible combination of the parents’ genotypes.<br>
If a genotype's parents are Ff and Ff, then a probability that the genotype is Ff is 0.5*0.5*2 = 0.5; if parents are FF and Ft, the probability that their children is tt is 0.<br><br>

$\bigstar$ constructGeneticNetwork.m (25 points)<br>
-- This function constructs a Bayesian network for genetic inheritance. The input will be a pedigree and a vector of allele frequencies, and the output will be a struct array of factors. Look in the code for information about variable numbering.<br>
Constructing a genetic network means computing all factors, including genotype given its parents, phenotype given its genotype.<br><br>

$\bigstar$ phenotypeGivenCopiesFactor.m (5 points)<br>
-- This function computes the probability that a child has a phenotype given the alleles for the child’s maternal and paternal copies of the gene. An example of such a factor is P(James’s Phenotype | James’s Gene Copy 1, James’s Gene Copy 2) in the figure. In this factor, the values are the entries in the CPD for James’s Phenotype.<br>
3 alleles F, f, t give 6 genotypes FF, Ff, Ft, ff, ft, tt.<br><br>

$\bigstar$ constructDecoupledGeneticNetwork.m (15 points)<br>
-- This function constructs a decoupled Bayesian network for genetic inheritance; the output should be a struct array of factors. Look in the code for information about variable numbering.<br><br>

$\bigstar$ constructSigmoidPhenotypeFactor.m (10 points)<br>
-- This function constructs a factor for a phenotype variable given the 4 variables for both copies of 2 genes.<br>
$$f(X_{1}^{1}, ..., X_{n_{1}}^{1}, ..., X_{n_{m}}^{m}, Y_{1}^{1}, ..., Y_{n_{m}}^{m}) = \sum_{j=1}^{m}\sum_{i=1}^{n_{j}} w_{i}^{j} (X_{i}^{j} + Y_{i}^{j})$$
$$\text{sigmoid}(z) = \frac{e^{z}}{1+e^{z}}$$
$$z = f(X_{1}^{1}, ..., X_{n_{1}}^{1}, ..., X_{n_{m}}^{m}, Y_{1}^{1}, ..., Y_{n_{m}}^{m})$$<br>
</p>

#### 5.2.2 Companion Quiz
<p align="justify">
<b>1.</b><br>
This quiz is a companion quiz to Programming Assignment: Bayes Nets for Genetic Inheritance. Please refer to the writeup for the programming assignment for instructions on how to complete this quiz.<br>
James and Rene come to a genetic counselor because they are deciding whether to have another child or adopt. They want to know the probability that their un-born child will have cystic fibrosis.<br>
Consider the Bayesian network for cystic fibrosis. We consider a person's phenotype variable to be "observed" if the person's phenotype is known. Order the probabilities of their un-born child having cystic fibrosis in the following situations from smallest to largest:<br>
(1) No phenotypes are observed (nothing clicked)<br>
(2) Jason has cystic fibrosis<br>
(3) Sandra has cystic fibrosis<br>
<b>Answer:</b> (1)(3)(2).<br>
Since Benjamin's phenotype and genotype are not observed in all of these situations, the probability that he will have cystic fibrosis (CF) is equivalent to the probability that James and Rene's unborn child will have CF. Observing that Benjamin's cousin has CF makes Benjamin more likely to have CF because CF is a genetic disease. Observing that Benjamin's brother has CF makes Benjamin more likely to have CF than when observing that Benjamin's cousin has CF because Benjamin's brother is a more closely-related relative than his cousin is.<br><br>

<b>2.</b><br>
James never knew his father Ira because Ira passed away in an accident when James was a few months old. Now James comes to the genetic counselor wanting to know if Ira had cystic fibrosis. The genetic counselor wants your help in determining the probability that Ira had cystic fibrosis. Consider the Bayesian network for cystic fibrosis. We consider a person's phenotype variable to be "observed" if the person's phenotype is known. Order the probabilities of Ira having had cystic fibrosis in the following situations from smallest to largest:<br>
(1) No phenotypes are observed (nothing clicked)<br>
(2) Benjamin has cystic fibrosis<br>
(3) Benjamin and Robin have cystic fibrosis.<br>
<b>Answer:</b> (1)(3)(2).<br>
Observing that Ira's grandson has cystic fibrosis (CF) makes Ira more likely to have CF because CF is a genetic disease. Observing that Ira's wife also has CF partially explains away why Ira has CF.<br><br>

<b>3.</b><br>
Recall that, for a trait with 2 alleles, the CPD for genotype given parents' genotypes has 27 entries, and 18 parameters were needed to specify the distribution. How many parameters would be needed if the trait had 3 alleles instead of 2?<br>
<b>Answer:</b> 5*6*6 = 180.<br>
There are 6 possible genotypes for each parent and for the child, so the size of the CPD is 6×6×6=216. Since the probability of having a genotype is fully defined if the probabilities for having the other genotypes are known, there are 216−(6×6)=180 parameters.<br><br>

<b>4.</b><br>
You will now gain some intuition for why decoupling a Bayesian network can be worthwhile. Consider a <b>non-decoupled Bayesian network</b> for cystic fibrosis with <b>3 alleles</b> over the pedigree that was used in section 2.4 and 3.3. How many parameters are needed to specify all probability distributions across the entire network?<br>
<b>Answer:</b> 191.<br>
There are 6 parameters for a phenotype given genotype factor, 5 parameters for a genotype given allele frequency factor, and 180 parameters for a child genotype given parents' genotypes factor. Since each type of factor has the same parameters, regardless of where it occurs in the network, the total number of parameters is 6+5+180=191.<br><br>

<b>5.</b><br>
Now consider the <b>decoupled Bayesian network</b> for cystic fibrosis with <b>3 alleles</b> over the pedigree that was used in section 2.4 and 3.3. How many parameters are needed to specify all of the probability distributions across the entire network? Hint: A child cannot inherit an allele that is not present in either parent, so there aren't as many degrees of freedom here as there might be without that context-specific information.<br>
<b>Answer:</b> 20.<br>
There are 9 parameters for a phenotype given genotype factor, 2 parameters for a copy of gene given allele frequency factor, and 9 parameters for a child copy of gene given parent's copies of gene factor. Since each type of factor has the same parameters, regardless of where it occurs in the network, the total number of parameters is 9+2+9=20.<br><br>

<b>6.</b><br>
Consider the decoupled Bayesian network for cystic fibrosis with three alleles that you constructed in section 3.3. We consider a person's gene copy variable to be "observed" if the person's allele for that copy of the gene is known.<br>
James and Rene are debating whether to have another child or adopt a child. They are concerned that, if they have a child, the child will have cystic fibrosis because both of them have one F allele observed (their other gene copy is not observed), even though neither of them have cystic fibrosis. You want to give them advice, but they refuse to tell you whether anyone else in their family has cystic fibrosis. What is the probability (NOT a percentage) that their unborn child will have cystic fibrosis?<br>
<b>Answer:</b> 0.4672.<br><br>

<b>7.</b><br>
Consider a Bayesian network for spinal muscular atrophy (SMA), in which there are multiple genes and 2 phenotypes.<br>
Let n be the number of genes involved in SMA and m be the maximum number of alleles per gene. How many parameters are necessary if we use a table CPD for the probabilities for phenotype given copies of the genes from both parents?<br>
<b>Answer:</b> $O(m^{2n})$.<br>
There are two alleles per gene, so there are $O(m^{2})$ allele combinations per gene. Therefore, there are $O(m^{2n})$ parameters for $n$ genes.<br><br>

<b>8.</b><br>
Consider the Bayesian network for spinal muscular atrophy (SMA), in which there are multiple genes and two phenotypes.<br>
Let n be the number of genes involved in SMA and m be the maximum number of alleles per gene. How many parameters are necessary if we use a sigmoid CPD for the probabilities for phenotype given copies of the genes from both parents?<br>
<b>Answer:</b> $O(mn)$.<br>
Each gene has up to m alleles, and there is an indicator for each allele for each copy of the gene. Therefore, if there were one gene, there would be O(2m)=O(m) parameters. Since there are n genes, there are O(mn) possible parameters.<br><br>

<b>9.</b><br>
Consider genes A and B that might be involved in spinal muscular atrophy. Assume that A has 2 alleles $A_{1}$ and $A_{2}$, and B has 2 alleles, $B_{1}$ and $B_{2}$. Which of the following relationships between A and B can a sigmoid CPD capture?<br>
A. Neither gene A nor gene B contribute to SMA.<br>
B. Gene A contributes to SMA, but gene B does not contribute to SMA and thus does not affect the effects of gene A on SMA.<br>
C. Allele $A_{1}$ and allele $B_{1}$ make a person equally more likely to have SMA, but when both are present the effect on SMA is the same as when only one is present.<br>
D. Alleles $A_{1}$ and $B_{1}$ each independently make a person likely to have SMA.<br>
E. Allele $A_{1}$ and allele $B_{1}$ make a person more likely to be have SMA when both of these alleles are present, but neither affect SMA otherwise.<br>
F. Allele $A_{1}$ makes a person more likely to have SMA, while allele $B_{1}$ independently makes a person less likely to have SMA.<br>
G. When the allele are $A_{1}$ and $B_{2}$ or $A_{2}$ and $B_{1}$ the person has SMA; otherwise the person does not have SMA.<br>
<b>Answer:</b> A, B, D, F.<br>
A: A sigmoid CPD can capture this by giving alleles for copies of gene A as well as alleles for copies of gene B weights with value zero.<br>
B: A sigmoid CPD can capture this by giving the alleles for copies of gene A positive weights and the alleles for copies of gene B zero weights.<br>
C: This OR relationship cannot be captured by a sigmoid CPD because interaction terms between the alleles are not present.<br>
D: Since their contributions are independent, a sigmoid CPD that weights the alleles for each gene based on the extent of their contribution would capture this perfectly.<br>
E: This AND relationship cannot be captured by a sigmoid CPD because interaction terms between the alleles are not present.<br>
F: A sigmoid CPD can capture this by making the weights for the inidicators for allele $A_{1}$ positive while making the weights for the indicators for allele $B_{1}$ negative.<br>
G: This XOR relationship means that the effect of the allele for gene A depends on which allele for gene B is present; since the sigmoid CPD does not have interactive terms, it will not be able to capture this.<br><br>

<b>10.</b><br>
Consider the Bayesian network for spinal muscular atrophy that we provided in spinalMuscularAtrophyBayesNet.net. We consider a person's gene copy variable to be "observed" if the person's allele for that copy of that gene is known.<br>
Now say that Ira and Robin come to the genetic counselor because they are debating whether to have a biological child or adopt and are concerned that their child might have spinal muscular atrophy. They have some genetic information, but because sequencing is still far too expensive to be affordable for everyone, their information is limited to only a few genes and to only 1 chromosome in each pair of chromosomes.<br>
Order the probabilities of their un-born child having spinal muscular atrophy in the following situations from smallest to largest:
(1) No genetic information or phenotypes are observed (nothing clicked)<br>
(2) Ira and Robin each have at least 1 M allele<br>
(3) Ira and Robin each have at least 1 M allele and at least 1 B allele.<br>
<b>Answer:</b> (1)(2)(3).<br>
Since James is unobserved, the probability that he will have spinal muscular atrophy (SMA) is equivalent to the probability that Ira and Robin's unborn child will have SMA. Observing that Ira and Robin each have an allele that is involved in causing SMA makes James more likely to have SMA than if no variables were observed. Observing that Ira and Robin each have alleles for 2 genes that are involved in causing SMA makes James even more likely to have SMA than if only 1 allele for 1 gene were observed.<br><br>

<b>11.</b><br>
Consider the Bayesian network for spinal muscular atrophy that we provided in spinalMuscularAtrophyBayesNet.net.<br>
No longer interested in finding out whether his father had cystic fibroisis, James comes to the genetic counselor with another question: Did his father have spinal muscular atrophy? The genetic counselor now wants your help in figuring this out. This time, however, James has other information for you: both he and Robin have spinal muscular atrophy. What is the probability (NOT a percentage) that Ira had spinal muscular atrophy?<br>
<b>Answer:</b> 0.3541.<br>
Since Ira's wife has spinal muscular atrophy (SMA), this helps explain away why his child has SMA, so Ira is more likely to have SMA than he would be if no phenotypes were observed but is less likely to have SMA than he would be if only James were observed to have SMA.<br><br>
</p>

### 5.3 Markov Networks for Optical Character Recognition
#### 5.3.1 Programming Assignment
<p align="justify">
Markov network is designed for this task. One image represents one characters. Besides, we can always observe one image but we cannot observe its character. We have to establish one model to predict the character and words (a sequence of characters). We have three factors to use: singleton factor, pairwise factor and triplet factors. Singleton factor can represent image-character, but pairwise factor and triplet factor are used to describe a relationship between 2 neighbor characters and 3 consecutive characters because a word is continuous.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/5_3_1_1.png"/></center>
</p>
<p align="justify">
Since I is always observed, Markov networks like these can be seen as modeling the conditional distribution P(C | I) in lieu of the joint distribution P(C, I). They are therefore also known as conditional random fields (CRFs).<br><br>

<b>Code implementation</b><br>
$\bigstar$ ComputeSingletonFactors.m<br>
-- This function takes a list of images and the provided image model. You should fill out the code to generate one factor for every image provided. The factor values should be given by the vector returned from ComputeImageFactor on each image.<br><br>

$\bigstar$ ComputePairwiseFactors.m<br>
-- This function is just like ComputeEqualPairwiseFactors, but it also uses the pairwiseModel we provided. For the factor values, an assignment of character i to the first character and character j to the second should have a score of pairwiseModel(i,j).<br><br>

$\bigstar$ ComputeTripletFactors.m<br>
-- This function is just like ComputePairwiseFactors, except that it uses tripletList.mat to compute factors over triplets.<br><br>

$\bigstar$ ComputeSimilarityFactor.m<br>
-- This function accepts a list of all the images in a word and two indices, i and j. It should return one factor: the similarity factor between the ith and jth images.<br><br>

$\bigstar$ ComputeAllSimilarityFactors.m<br>
-- This function should compute a list of every similarity factor for the images in a given word. That is, you should use ComputeSimilarityFactor for every i, j pair (i $\neq$ j) in the word and return the resulting list of factors.<br><br>

$\bigstar$ ChooseTopSimilarityFactors.m<br>
-- This function should take in an array of all the similarity factors and a parameter F, and return the top F factors based on their similarity score. Ties may be broken arbitrarily.<br><br>
</p>

### 5.4 Decision Making
#### 5.4.1 Programming Assignment
<p align="justify">
<b>Calculating Expected Utility Given A Decision Rule</b><br>
U is a utility node, D is decision node and others are Bayesian network nodes.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/5_4_1_1.png"/></center>
</p>
<p align="justify">
$\delta$-rule is a deterministic CPD. For example, D depends only on variable T (test for ARVD). Our decision rule is to have a surgery if $T = t^{2}$ and to not have a surgery if $T = t^{1}$.<br>
$$
\begin{bmatrix}
d^{1}t^{1} \rightarrow 1\\
d^{2}t^{1} \rightarrow 0\\
d^{1}t^{2} \rightarrow 0\\
d^{2}t^{2} \rightarrow 1
\end{bmatrix}
$$

In the figure above, utility node has 2 parents O (outcome) and D (decision rule), which determine together a value of utility. For example,<br>
$$
\begin{bmatrix}
d^{1}o^{1} \rightarrow 0.15\\
d^{2}o^{1} \rightarrow 0.7\\
d^{1}o^{2} \rightarrow 0.1\\
d^{2}o^{2} \rightarrow 0.05
\end{bmatrix}
$$

When we multiply these factors, we are multiplying the utility for each joint assignment to $\text{Pa}_{U}$ by its probability. The result of summing out over the joint assignments is just the expected utility. In other words, marginalizing the product of probability factors and a utility factor yields an expected utility:<br>
$$EU[I[\sigma]] = \sum_{\chi \cup D} \prod_{X \in \chi} P(X \mid \text{Parent}_{X}) \delta_{D}U = \sum_{D, \text{Parent}_{D}} \delta_{D} \sum_{\chi-\text{Parent}_{D}} \prod_{X \in \chi} P(X \mid \text{Parent}_{X})U$$
$$= \sum_{D, \text{Parent}_{D}} \delta_{D} \cdot \mu_{-D}(D, \text{Parent}_{D})$$

$\mu_{-D}(D, \text{Parent}_{D})$ doesn't depend on D, so it can be precomputed to avoid much repetition.<br><br>

An optimal decision<br>
$$\delta_{D}^{*}(\text{Parent}_{D}) = \arg\max_{d \in D} \mu_{-D}(D, \text{Parent}_{D})$$

For example, consider a simple case where we have a decision node D with {$d^{0}$, $d^{1}$} and D has a single parent X with {$x^{0}$, $x^{1}$}. If the expected utility is calculated like<br>
$$
\mu_{-D}(D, X) =
\begin{bmatrix}
x^{0}d^{0} \rightarrow 10\\
x^{0}d^{1} \rightarrow 1\\
x^{1}d^{0} \rightarrow 2\\
x^{1}d^{1} \rightarrow 5
\end{bmatrix}
$$

Therefore, an optimal decision is D = $d^{0}$ when X = $x^{0}$ and D = $d^{1}$ when X = $x^{1}$<br><br>

<b>Code implementation (65 pts)</b><br>
$\bigstar$ SimpleCalcExpectedUtility.m (5 pts)<br>
This function takes an influence diagram with a single decision node (which has a fully specified decision rule) and a single utility node, and returns the expected utility.<br><br>

$\bigstar$ CalculateExpectedUtilityFactor.m (15 Pts)<br>
-- This function takes an influence diagram I that has a single decision node D and returns the expected utility factor of I with respect to D.<br><br>

$\bigstar$ OptimizeMEU.m (15 Pts)<br>
-- This function takes an influence diagram I that has a single decision node D and returns the maximum expected utility and a corresponding optimal decision rule for D. You should use your implementation of CalculateExpectedUtilityFactor in this function.<br><br>

$\bigstar$ OptimizeWithJointUtility.m (15 Pts)<br>
-- This function takes an influence diagram with a single decision node D and possibly many utility nodes and returns the MEU and corresponding optimal decision rule for D. Hint - you will have to write a function that implements the Factor Sum operation. If you wrote this function for Programming Assignment 4, feel free to use it here!<br><br>

$\bigstar$ OptimizeLinearExpectations.m (15 Pts)<br>
-- This function takes an influence diagram I, with a single decision node and possibly with multiple utility nodes and returns an optimal decision rule along with the corresponding expected utility. Verify that the MEU and optimal decision rule is the same as that returned by OptimizeWithJointUtility.<br><br>
</p>

#### 5.4.2 Companion Quiz
<p align="justify">
<b>1.</b><br>
We have provided an instantiated influence diagram FullI (complete with a decision rule for D) in the file FullI.mat. <b>What is the expected utility for this influence diagram?</b> Please round to the nearest tenth (i.e., 1 decimal place), do not include commas, and do not write the number in scientific notation.<br>
<b>Answer:</b> -686.0.<br><br>

<b>2.</b><br>
Run ObserveEvidence.m on FullI to account for the following: We have been informed that variable 3 in the model, which models an overall genetic risk for ARVD, has value 2 (indicating the presence of genetic risk factors). Then run SimpleCalcExpectedUtility on the modified influence diagram. <b>What happened to the expected utility?</b> (Hint -- ObserveEvidence does not re-normalize the factors so that they are again valid CPDs unless the normalize flag is set to 1. -- If you do not use the normalize flag, you can use NormalizeCPDFactors.m to do the normalization.)<br>
A. It substantially decreased.<br>
B. It substantially increased.<br>
C. It did not change.<br>
D. The expected utility might or might not change because there is some randomness in the process for determining the expected utility.<br>
<b>Answer:</b> A.<br>
It decreased from -685.9 to -729.2<br><br>

<b>3.</b><br>
<b>Why can we explicitly enumerate all the possible decision rules while we often cannot enumerate over all possible CPDs?</b><br>
A. All choices have a probability of either 0 or 1, where in a general CPD, choices could take on any value in [0, 1].<br>
B. In an influence diagram, each decision node cannot have more than 1 parent, while in a general Bayes net, a node can have many parents.<br>
C. We can actually always enumerate over all possible CPDs.<br>
D. If there is one choice in a decision rule, at least one choice must have a 0 probability, where in a general CPD, no entries are restricted to having 0 probabilities.<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
Let a decision node D take on d possible values. Let it have m parents that can each take on n possible values. <b>How many possible decision rules $\delta_{D}$ are there?</b><br>
<b>Answer:</b> $d^{n^{m}}$.<br><br>

<b>5.</b><br>
Consider an influence diagram with 1 decision node D that can take on d values. Let D have m parents that can each take on n values. Assume that running sum-product inference takes O(S) time. <b>What is the run-time complexity of running OptimizeMEU on this influence diagram?</b>
<b>Answer:</b> $O(S+dn^{m})$.<br><br>

<b>6.</b><br>
<b>In which of the following situations does it make sense to use OptimizeWithJointUtility instead of OptimizeLinearExpectations?</b><br>
A. When the scopes of the utility factors are large compared to the scopes of the other (random variable) factors.<br>
B. When there are large factors in the random-variables part of the influence diagram, making inference over the network slow, and there are only a few utility factors, each involving a small number of variables.<br>
C. When every random variable in the network is a parent of at least one other utility factor.<br>
D. When the bottleneck in inference is in enumerating the large number of possible assignments to the parents of the utility variables, and each utility variable has a disjoint set of parents.<br>
<b>Answer:</b> B.<br><br>

<b>7.</b><br>
In the field below, enter the dollar value of the test T1, rounded to the nearest cent (e.g., "1.23" means that you would pay $1.23 for the test; any more than that, and your net utility will be lower than if you didn't perform any test). Do not precede with the amounts with dollar signs.<br>
<b>Answer:</b> 155.97.<br><br>

<b>8.</b><br>
In the field below, enter the dollar value of the test T2, rounded to the nearest cent (e.g., "1.23" means that you would pay $1.23 for the test; any more than that, and your net utility will be lower than if you didn't perform any test). Do not precede with the amounts with dollar signs.<br>
<b>Answer:</b> 2.82.<br><br>

<b>9.</b><br>
In the field below, enter the dollar value of the test T3, rounded to the nearest cent (e.g., "1.23" means that you would pay $1.23 for the test; any more than that, and your net utility will be lower than if you didn't perform any test). Do not precede with the amounts with dollar signs.<br>
<b>Answer:</b> 846.15.<br><br>
</p>

### 5.5 Exact Inference
#### 5.5.1 Programming Assignment
<p align="justify">
<b>Belief Propagation Algorithm</b><br>
1) Construct a clique tree from a given set of factors $\Phi$.<br>
2) Assign each factor $\phi \in \Phi$ to a clique $C_{\alpha(k)}$ such that $\text{Scope}[\phi_{k}] \subseteq C_{\alpha(k)}$. $\alpha(k)$ returns the index of the clique to which $\phi_{k}$ is assigned.<br>
3) Compute initial potentials $\psi_{i}(C_{i}) = \prod_{k: \alpha(k) = i} \phi_{k}$<br>
4) Designate an arbitrary clique as the root, and pass messages upwards from the leaves towards the root clique.<br>
5) Pass messages from the root down towards the leaves.<br>
6) Compute the beliefs for each clique $\beta_{i}(C_{i}) = \psi_{i} \times \prod_{k \in N_{i}} \delta_{k \rightarrow i}$<br><br>

<b>Code implementation (100 pts)</b><br>
$\bigstar$ ComputeInitialPotentials.m (10 points)<br>
-- This function should take in a set of factors (possibly already reduced by evidence, which is handled earlier in CreateCliqueTree) and a clique tree skeleton corresponding to those factors, and returns a cliqueTree with the .val fields correctly filled in. Concretely, it should assign each factor to a clique, and for each clique do a factor product over all factors assigned to it. Note that no messages should be computed in this function.<br><br>

$\bigstar$ GetNextCliques.m (10 points)<br>
-- This function looks at all the messages that have been passed so far, and returns indices i and j such that clique Ci is ready to pass a message to its neighbor Cj. Do not return i and j if Ci has already passed a message to Cj.<br><br>

$\bigstar$ CliqueTreeCalibrate.m (20 points)<br>
-- This function should perform clique tree calibration by running the sum-product message passing algorithm. It takes in an uncalibrated CliqueTree, calibrates it (i.e., setting the .val fields of the cliqueList to the final beliefs βi(Ci)), and returns it. To avoid numerical underflow, normalize each message as it is passed, such that it sums to 1. For consistency with our autograder, do not normalize the initial potentials nor the final cluster beliefs. This function also takes in an isMax flag that toggles between sum-product and max-sum message passing. For this part, it will always be set to 0.<br><br>

$\bigstar$ ComputeExactMarginalsBP.m (15 points)<br>
-- This function should take a network, a set of initial factors, and a vector of evidence and compute the marginal probability distribution for each individual variable in the network. A network is defined by a set of factors. You should be able to extract all the network information that you need from the list of factors. Marginals for every variable should be normalized at the end, since they represent valid probability distributions. As before, this function takes in an isMax flag, which should be set to 0 for now (at this point, you need to write the function for only when isMax=0).<br><br>

$\bigstar$ FactorMaxMarginalization.m (10 points)<br>
-- Similar to FactorMarginalization (but with sums replaced by maxima), this function takes in a factor and a set of variables to marginalize out. For each assignment to the remaining variables, it finds the maximum factor value over all possible assignments to the marginalized variables.<br><br>

$\bigstar$ CliqueTreeCalibrate.m (20 points)<br>
-- This function should perform clique tree calibration by running the max-sum message passing algorithm when the isMax flag is set to 1. It takes in an uncalibrated CliqueTree, does a log-transform of the values in the factors/cliques using natural log, max-calibrates it (i.e., setting the .val fields of the cliqueList to the final beliefs βi(Ci)), and returns it in log-space. We are working in log-space, so do not normalize each message as it is passed. For consistency with our autograder, do not normalize the initial potentials nor the final cluster beliefs. This function takes in an isMax flag that toggles between sum-product and max-sum message passing. For this part, it will always be set to 1, but make sure that it still does sum-product message passing correctly when isMax=0.<br><br>

$\bigstar$ ComputeExactMarginalsBP.m (5 points)<br>
-- This function should take a network, a set of initial factors, and a vector of evidence and compute the max-marginal for each individual variable in the network (including variables for which there is evidence). Max-marginals for every variable should not be normalized at the end. Leave the max-marginals in log-space; do not re-exponentiate them. As before, this function takes in an isMax flag, which will be set to 1 now; make sure it still works for computing the (non-max) marginals when it is set to 0.<br><br>
</p>

### 5.6 Sampling Methods
#### 5.6.1 Programming Assignment
<p align="justify">
<b>Code implementation (28 points)</b><br>
$\bigstar$ BlockLogDistribution.m: (5 points)<br>
-- This is the function that produces the sampling distribution used in Gibbs sampling and (possibly) versions of Metropolis-Hastings.<br>
</p>
{% highlight Matlab %}
idx = unique([G.var2factors{V}]);
factors = F(idx);

assignments = repmat(A, d, 1);
assignments(:, V) = repmat((1:d)', 1, length(V));

for i = 1:length(factors)
    factor = factors(i);
    val = GetValueOfAssignment(factor, assignments(:, factor.var));
    LogBS = LogBS + log(val);
end;
{% endhighlight %}

<p align="justify">
$\bigstar$ GibbsTrans.m (5 points)<br>
-- This function defines the transition process in the Gibbs chain as described above<br>
Each variable has a new assignment given other assignments of other variables.<br>
</p>
{% highlight Matlab %}
LogBS = BlockLogDistribution(i, G, F, A);
BS = exp(LogBS);
prob = BS/sum(BS);
A(i) = randsample(G.card(i), 1, true, prob);
{% endhighlight %}

<p align="justify">
MCMCInference.m PART 1 (3 points)<br>
-- This function defines the general framework for conducting MCMC inference.<br>
</p>
{% highlight Matlab %}
A = Trans(A, G, F);
all_samples(i+1, :) = A;
{% endhighlight %}

<p align="justify">
MHUniformTrans.m (5 points)<br>
-- This function defines the transition process associated with the uniform proposal distribution in Metropolis-Hastings.<br>
In this case, proposal distribution Q is a uniform distribution. Besides, we need an acceptance probability to determine whether we accept the proposal.<br>
$$A(x \rightarrow x') = \min[1, \frac{\pi(x')Q(x' \rightarrow x)}{\pi(x)Q(x \rightarrow x')}]$$
</p>
{% highlight Matlab %}
delta = LogProbOfJointAssignment(F, A_prop) -...
    LogProbOfJointAssignment(F, A);

p_acceptance = min(1, exp(delta));
{% endhighlight %}

<p align="justify">
MHSWTrans.m (Variant 1) (3 points)<br>
-- This function defines the transition process associated with the Swendsen-Wang proposal distribution in Metropolis-Hastings.<br><br>

MHSWTrans.m (Variant 2) (3 points)<br>
-- Now implement the second variant of SW.<br><br>

MCMCInference.m PART 2 (4 points)<br>
-- Flesh this function out to run our Swendsen-Wang variants in addition to Gibbs.<br><br>
</p>

#### 5.6.2 Companion Quiz
<p align="justify">
<b>1.</b><br>
Let’s run an experiment using our Gibbs sampling method. As before, use the toy image network and set the on-diagonal weight of the pairwise factor (in ConstructToyNetwork.m) to be 1.0 and the off-diagonal weight to be 0.1. Now run Gibbs sampling a few times, first initializing the state to be all 1’s and then initializing the state to be all 2’s. <b>What effect does the initial assignment have on the accuracy of Gibbs sampling? Why does this effect occur?</b><br>
A. The initial state has a significant impact on the result of our sampling as Gibbs will never switch variables because the pairwise potentials enforce strong agreement so we are in a local optima.<br>
B. The initial state has a significant impact on the result of our sampling, which makes sense as strong correlation makes mixing time long and we remain close to the initial assignment for a long time.<br>
C. The initial state has a significant impact on the result as, though our chain mixes quickly, it will mix to a distribution far from the actual distribution and close to the initial assignment.<br>
D. The initial state is not an important factor in our result as Gibbs can make large moves of multiple variables to quickly escape this bad state.<br>
<b>Answer:</b> B.<br><br>

<b>2.</b><br>
Set the on-diagonal weight of our toy image network to 1 and off-diagonal weight to .2. Now visualize multiple runs with each of Gibbs, MHUniform, Swendsen-Wang variant 1, and Swendsen-Wang variant 2 using VisualizeMCMCMarginals.m (see TestToy.m for how to do this). <b>How do the mixing times of these chains compare? How do the final marginals compare to the exact marginals? Why?</b><br>
A. All variants perform poorly in the case of strong pairwise potentials. All algorithms are subject to positive feedback loops with the tight loops in our grid and strong pairwise agreement potentials, preventing appropriate mixing.<br>
B. The Swendsen-Wang variants outperform the other approaches, with faster mixing and better final marginals. This is likely due to the block-flipping nature of Swendsen-Wang which allows us to flip blocks and quickly mix in environments with strong agreeing potentials.<br>
C. Gibbs outperforms the other variants in this instance. Gibbs has some issues with strong pairwise potentials, but is not nearly as bad as MH where blocks end up stuck with the same level so we cannot mix appropriately.<br>
D. Having strong pairwise potentials enforcing agreement is not a problem for any of these sampling methods and all perform equally well -- mixing quickly and ending up close to the final marginals.<br>
<b>Answer:</b> B.<br><br>

<b>3.</b><br>
Set the on-diagonal weight of our toy image network to .5 and off- diagonal weight to .5. Now visualize multiple runs with each of Gibbs, MHUniform, Swendsen-Wang variant 1, and Swendsen-Wang variant 2 using VisualizeMCMCMarginals.m (see TestToy.m for how to do this). <b>How do the mixing times of these chains compare? How do the final marginals compare to the exact marginals? Why?</b><br>
A. All variants perform equally well. They all mix quickly and have very low variance throughout their runs -- remaining close to the true marginals. This is because the pairwise marginals do not force us into preferring agreement when we should not.<br>
B. Gibbs performs poorly relative to the other variants -- exhibiting slower mixing time and marginals further from the exact ones. This difference is likely due to the Gibbs strong global dependence that prevents it from acting appropriately unless all variables are relatively well synced to their true marginals.<br>
C. Swendsen-Wang outperforms the other variants, though all perform relatively well. SW is better because its larger block moves allow for faster mixing and mean it reaches marginal estimates closer to the true marginals faster.<br>
D. Gibbs and MHUniform perform very well and are somewhat better than the Swendsen-Wang variants. This is because the first two variants use local moves so the local marginals remained consistently close the the true marginals, while SW allows big swings over multiple variables that perturb the distribution.<br>
<b>Answer:</b> D.<br><br>

<b>4.</b><br>
When creating our proposal distribution for Swendsen-Wang, if you set all the $q_{i, j}$'s to zero, <b>what does Swendsen-Wang reduce to?</b><br>
A. Switching $q_{i, j}$ to 0 is equivalent to the first variant of Swendsen-Wang.<br>
B. Switching $q_{i, j}$ to 0 is equivalent to MH-Uniform.<br>
C. Switching $q_{i, j}$ to 0 leaves us without a valid proposal distribution ans id not a feasible sampling algorithm.<br>
D. Switching $q_{i, j}$ to 0 is equivalent to randomized variant of Gibbs sampling where we are allowed to take a random, rather than fixed, order.<br>
<b>Answer:</b> D.<br>
</p>

### 5.7 Learning CRFs for Optical Character Recognition
#### 5.7.1 Programming Assignment
<p align="justify">
<b>Learning CRF Parameters For OCR</b><br>
$\bigstar$ The Log-Linear Representation<br>
$$P(Y \mid x:\theta) = \frac{1}{Z_{x}(\theta)} exp\{\sum_{i=1}^{k} \theta_{i}f_{i} \}, \quad Z_{x}(\theta) = \sum_{Y}  exp\{\sum_{i=1}^{k} \theta_{i}f_{i} \}$$

Note that the partition function requires summing over an exponential number of assignments to the variables.<br><br>

In our CRF, we have three kinds of factors<br>
-- $f_{i, c}^{C}(Y_{i})$: sigle character<br>
-- $f_{i, j, c, d}^{I}(Y_{i}, x_{ij})$: a single character is associated with an image pixel<br>
-- $f_{i, c, d}^{P}(Y_{i}, Y_{i+1})$: a pair of adjacent characters<br><br>

Our goal is to maximize the log-likelihood of the parameters given the data. We use log-likelihood to avoid numeric overflow<br>
$$\text{nll}(x, Y, \theta) \equiv \log(Z_{x}(\theta)) - \sum_{i=1}^{k}\theta_{i}f_{i}(Y, x) + \frac{\lambda}{2}\sum_{i=1}^{k} \theta_{i}^{2}$$

The gradient<br>
$$\frac{\partial}{\partial \theta_{i}} \text{nll}(x, Y, \theta) = E_{\theta}[f_{i}] - E_{D}[f_{i}] + \lambda \theta_{i}$$

$E_{\theta}[f_{i}]$ is the expectation of feature values with respect to the model parameters; $E_{D}[f_{i}]$ the expectation of the feature values with respect to the given data instance $D \equiv (X, y)$.<br><br>

After calibrating one clique tree, we will get a set of cluster messages, which can represent our factors. But we have to be careful that these cluster messages might be unnormalized. In this case, we want to compute the partition function, so the cluster message shouldn't be normalized.<br><br>

<b>Code implementation (100 Points)</b><br>
$\bigstar$ StochasticGradientDescent.m (10 Points)<br>
This function runs stochastic gradient descent for the specified number of iterations and returns the value of θ upon termination.<br><br>

$\bigstar$ LRSearchLambdaSGD.m (10 Points) <br><br>

$\bigstar$ LogZ (10 points): This test checks that you have correctly modified CliqueTreeCalibrate.m (originally from PA4) to compute the variable logZ. Be sure to read the comment at the top: we have modified the code for you to also keep track of message that are not normal- ized (in the variable unnormalizedMessages). You will need this to compute logZ.<br><br>

$\bigstar$ CRFNegLogLikelihood (30 points): This test checks that you correctly compute nll, the first return value of InstanceNegLogLikelihood.m.<br><br>

$\bigstar$ CRFGradient (40 points): This test checks that you correctly compute grad, the second return value of InstanceNegLogLikelihood.m.<br>
</p>
{% highlight Matlab %}
lambda = modelParams.lambda;
% inference P(Y|x, theta)
[F, FeatureCounts] = GenerateFactors(y, theta, featureSet,...
    modelParams);
P = CreateCliqueTree(F);
[P, logZ] = CliqueTreeCalibrate(P, 0);
[F] = ComputeNormalizedP(P, F);

WeightFeatureCounts = theta.*FeatureCounts;

[ModelFeatureCounts] = GenerateModelFeatureCounts(F, featureSet);

RegulazrizationCost = (lambda/2)*(theta*theta');
RegularizationGradient = lambda*theta;

NLL = logZ - sum(WeightFeatureCounts) + RegulazrizationCost;
Grad = ModelFeatureCounts - FeatureCounts + RegularizationGradient;

nll = NLL;
grad = Grad;
{% endhighlight %}
<p align="justify">
<br>
</p>

### 5.8 Learning Network Structure
#### 5.8.1 Programming Assignment
<p align="justify">
<b>Learning with Known Skeletal Structure</b><br>
We want to fit the paramters for Gaussian distribution given training data<br>
$$X \mid U \sim N(\beta_{1}U_{1} + \beta_{2}U_{2} + ... + \beta_{n}U_{n} + \beta_{n+1}, \sigma^{2})$$
$$E_{\hat{P_{}}}[X] = \beta_{1}E_{\hat{P_{}}}[U_{1}] +  \beta_{2}E_{\hat{P_{}}}[U_{2}] + ... +  \beta_{n}E_{\hat{P_{}}}[U_{n}] + \beta_{n+1}$$
$$E_{\hat{P_{}}}[X \cdot U_{i}] = \beta_{1}E_{\hat{P_{}}}[U_{1} \cdot U_{i}] +  \beta_{2} E_{\hat{P_{}}}[U_{2} \cdot U_{i}] + ... +  \beta_{n} E_{\hat{P_{}}}[U_{n} \cdot U_{i}] + \beta_{n+1} E_{\hat{P_{}}}[U_{i}] $$

<b>Model Learning</b><br>
We apply log-likelihood method<br>
$$\sum_{i=1}^{N} \log P(O_{1} = o_{1}^{(i)}, ..., O_{10} = o_{10}^{(i)})$$
$$P(O_{1}, ..., O_{10}) = \sum_{k=1}^{2} P(C = k, O_{1}, ..., O_{10}) = \sum_{k=1}^{2} P(C = k) \prod_{i=1}^{10} P(O_{i} \mid C = k, O_{parent(i)})$$

<b>Learning Graph Structures</b><br>
We can take use of training data to generate a graph by likelihood score. Given a scoring function that satisfies decomposability and score-equivalence, we can compute the score, or weight, between all pairs of variables, and find the maximum spanning tree. Using the likelihood score, we set the edge weights to be:<br>
$$w_{i \rightarrow j} = \text{Score}_{L}(O_{i} \mid O_{j}: D) - \text{Score}_{L}(O_{i}: D)  = M \cdot I_{\hat{P_{}}} (O_{i}, O_{j})$$

where M is the number of instances and $I_{\hat{P_{}}}$ is the mutual information with respect to distribution $\hat{P_{}}$.<br><br>

<b>Code implementation (100 Points)</b><br>
$\bigstar$ FitGaussianParameters.m (5 points):<br>
-- in this function, you will implement the algorithm for fitting maximum likelihood parameters $\mu$ and $\sigma$ of the Gaussian distribution given the samples.<br><br>

$\bigstar$ FitLinearGaussianParameters.m (15 points):<br>
-- In this function you will implement the algorithm for fitting linear Gaussian parameters in the general form.<br><br>

$\bigstar$ ComputeLogLikelihood.m (20 points):<br>
-- This function computes, given the model P, cou- pled with the graph structure G, and the dataset, the log likelihood of the model over the dataset.<br><br>

$\bigstar$ LearnCPDsGivenGraph.m (25 points):<br>
-- This is the function where we learn the parameters.<br><br>

$\bigstar$ ClassifyDataset.m (15 points):<br>
-- This function takes a dataset, a learned model P , and a graph structure G.<br><br>

$\bigstar$ LearnGraphStructure.m (10 points):<br>
-- Learning the tree-structured graph from data should be quite straightforward given the two functions GaussianMutualInformation.m and MaxSpanningTree.m we provide.<br><br>

$\bigstar$ LearnGraphAndCPDs.m (10 points):<br>
-- This function learns the parameters, as well as the graph structure, for each class of data. For learning the graph structure, you should call the function LearnGraphStructure.m you just implemented, and use the provided function ConvertAtoG.m to convert the maximum spanning tree to the desired graph structure G.<br><br>
</p>

### 5.9 Learning with Incomplete Data
#### 5.9.1 Programming Assignment
<p align="justify">
<b>The EM algorithm for Bayesian Clustering</b><br>
-- E-step:<br>
In the E-step our goal is to do soft assignment of poses to classes. Thus, for each pose, we infer the conditional probabilities for each of the K class labels using the current model parameters. This can be done by first computing the joint probability of the class assignment and pose (body parts), which decomposes as follows.<br>
$$P(C = k, O_{1}, ..., O_{10}) = P(C = k) \prod_{i=1}^{10} P(O_{i} \mid C = k, O_{parent(i)})$$

In this programming, we have no idea about variable class, so we have to compute its conditional probability<br>
$$P(C = k \mid O_{1}, ..., O_{10})$$

Then we pick up the class with the highest probability. Besides, this conditional probability is also our expected sufficient statistics (ESS).<br><br>

-- M-step:<br>
We seek the maximum likelihood CLG parameters given ESS.<br><br>

<b>HMM action models</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/PGM/5_9_1_1.png"/></center>
</p>
<p align="justify">
One action can be composed of more than 1 pose (each pose has 10 bodies parts). Besides, each pose can be regarded as a state, but the state isn't observed (in other word, hidden). So, we take HMM model to predict action class. In this programming, we have 3 action categories: clap, high kick, low kick.<br><br>

The HMM action model defines a joint distribution over the hidden state variables and the poses comprising an action of length m:<br>
$$P(S_{1}, ..., S_{m}, P_{1}, ..., P_{m}) = P(S_{1}) P(P_{1} \mid S_{1})\prod_{i=2}^{m} P(S_{i} \mid S_{i-1}) P(P_{i} \mid S_{i})$$

In HMM model, we have to specify 3 factors:<br>
-- prior distribution over the initial states $P(S_{1})$<br>
$$P(S_{1} = s_{1}) = \frac{M_{1}[s_{1}]}{\sum_{k=1}^{K} M_{1}[k]}$$
-- the transition model P(S' | S), S' is behind of S<br>
$$P(S' = s' \mid S = s) = \frac{M[s', s]}{\sum_{k=1}^{K} M[k, s]}$$
-- the emission model for pose $P_{j}$<br>
$$P(P_{j} \mid S) = \prod_{i=1}^{10} P(O_{i} \mid S, O_{parent(i)})$$

<b>Code implementation (100 Points)</b><br>
$\bigstar$ EM cluster.m (35 points)<br>
-- This is the function where we run the overall EM procedure for clustering. This function takes as input the dataset of poses, a known graph structure G, a set of initial probabilities, and the maximum number of iterations to be run. Descriptions of the input and output parameters are given as comments in the code. Follow the comments and structure given in the code to complete this function. Note that we start with the M-step first.<br><br>

$\bigstar$ EM HMM.m (35 points)<br>
-- This is the function where you will implement the EM procedure to learn the parameters of the action HMM. Many parts of this function should be the same as portions of EM cluster.m. This function takes as input the dataset of actions, the dataset of poses, a known graph structure G, a set of initial probabilities, and the number of iterations to be run. Descriptions for the input and output parameters are given as comments in the code. Follow the comments and structure given in the code to complete this function. Note that we start with the M-step first.<br><br>

$\bigstar$ RecognizeActions.m (10 points)<br>
-- In this function, you should train an HMM for each action class using the training set datasetTrain. Then, classify each of the instances in the test set datasetTest and compute the classification accuracy. Details on the datasetTrain and datasetTest data structures can be found in Section 6.<br><br>

$\bigstar$ RecognizeUnknownActions.m (20 points)<br>
-- In this function, you should train a model for each action class using the training set datasetTrain3. Then, classify each of the instances in the test set datasetTest3 and save the classifier predictions by calling SavePrediction.m. This function is left empty for you to be creative in your approach (we give some suggestions below). When you are done, write a short description of your method in YourMethod.txt, which is submitted along with your predictions in the submit script. Make sure you execute this function before running the submit script so that the saved predictions are ready for submission. Your score for this part will be determined by the percentage of unknown action instances you successfully recognize. If you obtain an accuracy of x%, you will receive 0.2x points for this part.<br><br>
</p>


## 6. References
<p align="justify">
[1] <a href="https://books.google.fr/books?hl=en&lr=&id=7dzpHCHzNQ4C&oi=fnd&pg=PR9&dq=probabilistic+graphical+models+koller&ots=pw6DBk0VyJ&sig=8LUY13c_ryEbmderzKBc5EdilZk&redir_esc=y#v=onepage&q=probabilistic%20graphical%20models%20koller&f=false"> Koller, Daphne, and Nir Friedman. Probabilistic graphical models: principles and techniques. MIT press, 2009.</a><br>
</p>
