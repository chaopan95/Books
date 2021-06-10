---
layout: post
title:  "Deep Learning Basis"
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


## 1. Neural Network
### 1.1 MLP
#### 1.1.1 Dnese Layer
<p align="justify">
<b>Linear Regression</b><br>
$\bigstar$ Loss function: MSE (mean square error)
$$L(w) = \frac{1}{n}\sum_{i=1}^{n}(w^{T}x_{i}-y_{i})^{2} = \frac{1}{n}\left \| Xw-y \right \|^{2} \rightarrow \min_{w}$$

$\bigstar$ Analytical solution is available, but inverting a matrix is hard for high-dimensional data.
$$
\begin{aligned}
& (\frac{1}{n}\left \| Xw-y \right \|^{2})^{'} = \frac{2}{n}X^{T}(Xw-y) = \frac{2}{n}(X^{T}Xw-X^{T}y) = 0 \\
& w = (X^{T}X)^{-1}X^{T}y
\end{aligned}
$$

<b>Logistic Regression</b><br>
Given $x \in R^{n}$, we want
$$\hat{y_{}} = P(y = 1 \mid x) \rightarrow \hat{y_{}} \in [0, 1]$$

$\bigstar$ Parameters: $w \in R^{n}, b \in R$<br><br>

$\bigstar$ Output
$$\hat{y_{}} = \sigma(w^{T}x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

$\bigstar$ Squared loss function is not a good Loss (error) function
$$L(\hat{y_{}}, y) = \frac{1}{2} (\hat{y_{}} - y)^{2}$$
With squared loss function, we won't have a convex optimization problem because of local optimum. Gradient descend doesn't work well.<br><br>

$\bigstar$ Loss function for losgistic regression
$$L(\hat{y_{}}, y) = -[y \log\hat{y_{}} + (1-y)\log(1-\hat{y_{}})]$$
Intuitively, if y = 1, we push $\hat{y_{}}$ bigger to have a small loss; if y = 0, we push $\hat{y_{}}$ smaller.<br><br>

$\bigstar$ Cost Function<br>
-- Given training set {($x^{1}$, $y^{1}$), ..., ($x^{m}$, $y^{m}$)}, want $\hat{y^{i}} \approx y^{i}$ (ground truth).<br>
-- Loss function is for a single training example; while cost function is for the entire dataset.
$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y^{i}}, y^{i}) = -\frac{1}{m} \sum_{i=1}^{m} [y^{i} \log\hat{y^{i}} + (1-y^{i}) \log(1-\hat{y^{i}})]$$

$\bigstar$ Our goal is to minimize cost function<br>
-- Minimizing the loss corresponds with maximizing $\log P(y \mid x)$
$$\arg\min_{w, b} J(w, b)$$

$\bigstar$ Gradient Descent
$$
\begin{aligned}
& w = w - \alpha \frac{\partial J(w, b)}{\partial w} \\
& b = b - \alpha \frac{\partial J(w, b)}{\partial b}
\end{aligned}
$$

$\bigstar$ Logistic Regression Gradient Descent
$$
\begin{aligned}
& z = w^{T}x + b \\
& \hat{y_{}} = a = \sigma(z) \\
& L(a, y) = -[y \log a + (1-y) \log(1-a)] \\
& 
\begin{aligned}
\frac{\partial L(a, y)}{\partial z} &= - [y \frac{1}{a} - (1- y)\frac{1}{1-a}] \frac{\partial a}{\partial z}  \\
&= - [y \frac{1}{a} - (1- y)\frac{1}{1-a}] a(1-a) \\
&= a-y
\end{aligned}
\end{aligned}
$$

Consider n features for one example and m training examples, in vector implementation, X has a shape of $n \times m$, W is $n \times 1$, Y is $1 \times m$ and b is a scalar.
$$
\begin{aligned}
& X =
\begin{bmatrix}
x_{11} & \cdots  & x_{1m}\\
\vdots  & \ddots  & \vdots\\
x_{n1} & \cdots & x_{nm}
\end{bmatrix}, \quad Y = 
\begin{bmatrix}
y_{11} & \cdots & y_{1m}
\end{bmatrix} \\
& W = 
\begin{bmatrix}
w_{11} \\
\vdots \\
w_{n1}
\end{bmatrix}, \quad b = \begin{bmatrix} R \end{bmatrix} \\
& Z = W^{T}X + b \\
& A = \sigma(Z) \\
& dZ = A - Y \\
& dW = \frac{1}{m} X (dZ)^{T} \\
& db = \frac{1}{m} \sum_{i=1}^{m}dZ
\end{aligned}
$$

<b>Quiz:</b> Suppose a sample with 10 examples and 5 features. How many elements will matrix X contain?<br>
<b>Answer:</b> 10*(5+1)=60.<br><br>

<b>Neural Networks Representation</b><br>
$\bigstar$ Notations<br>
-- X is a matrix in which each column is one training example.<br>
-- $a^{[2]}$ denotes the activation vector of the 2nd layer.<br>
-- $a^{[2](12)}$ denotes the activation vector of the 2nd layer for the 12th training example.<br>
-- $a_{4}^{[2]}$ is the activation output by the 4th neuron of the 2nd layer<br><br>

Number of layers is equal to number of hidden layers + 1 output layer. For example, 2 layers NN with one input (3 features) in next picture
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/DLB/1_1_1_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ Forward Propagation<br>
-- Vectorized implementation for layer l, where $l \in [1, L]$<br>
-- At l-th layer, $W^{[l]}$ has a shape of ($n^{[l]}$, $n^{[l-1]}$) and $b^{[l]}$ has a shape of ($n^{[l]}$, 1), where $n^{[l]}$ is the number if neurons in l-th layer. Specifically, l = 0 for input layer.<br>
-- e.g. $W^{[1]}$ is (4, 3), $b^{[1]}$ is (4, 1), $W^{[2]}$ is (1, 4), $b^{[2]}$ is (1, 1), $Z^{[1]}$ (4, m)<br><br>

For a single neuron
$$
\begin{aligned}
& a_{1}^{[1]} = \sigma(z_{1}^{[1]}) \\
& z_{1}^{[1]} = w_{1}^{[1]T} x + b_{1}^{[1]} \\
& 
x =
\begin{bmatrix}
x_{1}\\
x_{2}\\
x_{3}
\end{bmatrix} \\
& 
w_{1}^{[1]T} =
\begin{bmatrix}
w_{1,1} & w_{1,2} & w_{1, 3}
\end{bmatrix} \\
& 
b_{1}^{[1]} =
\begin{bmatrix}
R
\end{bmatrix}
\end{aligned}
$$

Conventionally, we use column vector to represent x and w. Note that b is a scalar for a single neuron. We stack each neuron in a same layer vertically.<br>
$$
\begin{aligned}
&
\begin{bmatrix}
z_{1}^{[1]}\\
z_{2}^{[1]}\\
z_{3}^{[1]}\\
z_{4}^{[1]}
\end{bmatrix} =
\begin{bmatrix}
w_{1}^{[1]T}\\
w_{2}^{[1]T}\\
w_{3}^{[1]T}\\
w_{4}^{[1]T}
\end{bmatrix}
\begin{bmatrix}
x_{1}\\
x_{2}\\
x_{3}
\end{bmatrix} +
\begin{bmatrix}
b_{1}^{[1]}\\
b_{2}^{[1]}\\
b_{3}^{[1]}\\
b_{4}^{[1]}
\end{bmatrix} \\
& Z^{[1]} = W^{[1]}x + b^{[1]} \\
& A^{[1]} = \sigma(Z^{[1]}) \\
& Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} \\
& A^{[2]} = \sigma(Z^{[2]})
\end{aligned}
$$

Therefore, $W^{[1]}$ has a shape of $4 \times 3$ and $b^{[1]}$ has a shape of $4 \times 1$ (scalar); while $W^{[2]}$ has a shape of $1 \times 4$ and $b^{[2]}$ has a shape of $1 \times 1$.<br><br>

If we have more than 1 input, x will become X with a shape of $3 \times m$.
$$
\begin{aligned}
& Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \quad A^{[l]} = g^{[l]}(Z^{[l]}) \\
&
X =
\begin{bmatrix}
| & | & ... & |\\
x^{(1)} & x^{(2)} & ... & x^{(m)}\\
| & | & ... & |
\end{bmatrix}, \quad
x^{(i)} =
\begin{bmatrix}
x_{1}^{(i)}\\
x_{2}^{(i)}\\
x_{3}^{(i)}
\end{bmatrix} \\
&
A^{[1]} =
\begin{bmatrix}
| & | & ... & |\\
a^{[1](1)} & a^{[1](2)} & ... & a^{[1](m)}\\
| & | & ... & |
\end{bmatrix} \\
& a^{[1](i)} = \sigma(z^{[1](i)}) \\
&
z^{[1](i)} =
\begin{bmatrix}
z_{1}^{[1](i)}\\
z_{2}^{[1](i)}\\
z_{3}^{[1](i)}\\
z_{4}^{[1](i)}
\end{bmatrix} =
\begin{bmatrix}
w_{1}^{[1]T}\\
w_{2}^{[1]T}\\
w_{3}^{[1]T}\\
w_{4}^{[1]T}
\end{bmatrix}
\begin{bmatrix}
x_{1}^{(i)}\\
x_{2}^{(i)}\\
x_{3}^{(i)}
\end{bmatrix} +
\begin{bmatrix}
b_{1}^{[1]}\\
b_{2}^{[1]}\\
b_{3}^{[1]}\\
b_{4}^{[1]}
\end{bmatrix} \\
& Z^{[1]} = W^{[1]}X + b^{[1]} \\
& A^{[1]} = g^{[1]}(Z^{[1]}) \\
& Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} \\
& A^{[2]} = g^{[2]}(Z^{[2]}) = \hat{Y_{}}
\end{aligned}
$$

$\bigstar$ Backward Propagation
$$
\begin{aligned}
& dA^{[l]} = (W^{[l+1]})^{T} dZ^{[l+1]} \\
& dZ^{[l]} = dA^{[l]} * [g^{[l]}(Z^{[l]})]' \quad \text{where * means element-wise multiplication} \\
& dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^{T} \\
& db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l]}
\end{aligned}
$$

$$
\begin{aligned}
& L(\hat{Y_{}}, Y) = -[Y \log\hat{Y_{}} + (1 - Y)\log(1 - \hat{Y_{}})] \\
& dZ^{[2]} = A^{[2]} - Y \\
& dW^{[2]} = \frac{1}{m} dZ^{[2]} (A^{[1]})^{T} \\
& db^{[2]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[2]} \\
& dZ^{[1]} = dA^{[1]} [g^{[1]}(dZ^{1})]' = (W^{[2]})^{T}dZ^{[2]} [g^{[1]}(dZ^{1})]' \\
& dW^{[1]} = \frac{1}{m} dZ^{[1]} X^{T} \\
& db^{[1]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[1]}
\end{aligned}
$$

$\bigstar$ Hyperparameters<br>
-- Learning rate $\alpha$<br>
-- number of iterations<br>
-- number of hidden layers<br>
-- number of hidden units<br>
-- choice of activation function
</p>
{% highlight Python %}
A = np.random.randn(4,3)
B = np.sum(A, axis=1, keepdims=True)  # B.shape is (4, 1)

dw2 = 1/m * np.dot(dZ2, A1.T)  # Backprop
db2 = 1/m * np.sum(dZ2, axis=1, keepdims=False)
dW1 = 1/m * np.dot(dZ1, X.T)
db1 = 1/m * np.sum(dZ1, axis=1, keepdims=False)
{% endhighlight %}
<p align="justify">
<b>Jacobin</b><br>
$\bigstar$ Let's take a composition of two vectors functions<br>
-- g(x): $(x_{1},  ..., x_{n}) \rightarrow (g_{1}, ..., g_{m})$<br>
-- f(g): $(g_{1}, ..., g_{m}) \rightarrow (f_{1}, ..., f_{k})$ this will be useful for RNN<br>
-- h(x) = f(g(x)): $(x_{1}, ..., x_{n}) \rightarrow (h_{1}, ..., h_{k}) = (f_{1}, ..., f_{k})$<br>
$\bigstar$ The matrix of partial derivatives $\frac{\partial h_{i}}{\partial x_{j}}$ is called the Jacobian:
$$
\begin{aligned}
& J^{h} =
\begin{pmatrix}
\frac{\partial h_{1}}{\partial x_{1}} & \cdots & \frac{\partial h_{1}}{\partial x_{n}}\\
\vdots & \ddots & \vdots\\
\frac{\partial h_{k}}{\partial x_{1}} & \cdots & \frac{\partial h_{k}}{\partial x_{n}}
\end{pmatrix} \\
& J_{i, j}^{h} = \frac{\partial h_{i}}{\partial x_{j}} = \sum_{l}\frac{\partial f_{i}}{\partial g_{l}}\frac{\partial g_{l}}{\partial x_{j}} = \sum_{l}J_{i, l}^{f} \cdot J_{l, j}^{g} \\
& J^{h} = J^{f} \cdot J^{g}
\end{aligned}
$$

$\bigstar$ Tensor Derivative
$$
\begin{aligned}
&
\begin{aligned}
C &= AB \\
&=
\begin{pmatrix}
a_{1, 1} & a_{1, 2}\\
a_{2, 1} & a_{2, 2}
\end{pmatrix} \cdot
\begin{pmatrix}
b_{1, 1} & b_{1, 2}\\
b_{2, 1} & b_{2, 2}
\end{pmatrix} \\
&=
\begin{pmatrix}
a_{1, 1}b_{1, 1}+a_{1, 2}b_{2, 1} & a_{1, 1}b_{1, 2}+a_{1, 2}b_{2, 2}\\
a_{2, 1}b_{1, 1}+a_{2, 2}b_{2, 1} & a_{2, 1}b_{1, 2}+a_{2, 2}b_{2, 2}
\end{pmatrix}
\end{aligned}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial C}{\partial A} &= \{ \frac{\partial c_{i, j}}{\partial a_{k, l}} \}_{i, j, k, l} \\
&=
\begin{pmatrix}
\frac{\partial c_{1, 1}}{\partial A} & \frac{\partial c_{1, 2}}{\partial A}\\
\frac{\partial c_{2, 1}}{\partial A} & \frac{\partial c_{2, 2}}{\partial A}
\end{pmatrix} \\
&=
\begin{pmatrix}
\begin{bmatrix}
b_{1, 1} & b_{2, 1}\\
0 & 0
\end{bmatrix} &
\begin{bmatrix}
b_{1, 2} & b_{2, 2}\\
0 & 0
\end{bmatrix}\\
\begin{bmatrix}
0 & 0\\
b_{1, 1} & b_{2, 1}
\end{bmatrix} &
\begin{bmatrix}
0 & 0\\
b_{1, 2} & b_{2, 2}
\end{bmatrix}
\end{pmatrix}
\end{aligned}
$$

<b>Quiz:</b> How many dimensions will a derivative of a 3-d tensor by a 4-d tensor have?<br>
<b>Answer:</b> 7.
</p>

#### 1.1.2 ReLU Layer
<p align="justify">
<b>Why do we need non-linear activation functions?</b><br>
A combination of two linear functions is still linear function. So, we will lose representation of neural network.<br><br>

$$
\begin{aligned}
&
g(z) =
\begin{cases}
  z, \quad z > 0 \\
  0, \quad \text{otherwise}
\end{cases} \\
&
g(z)' =
\begin{cases}
  1, \quad z > 0 \\
  0, \quad \text{otherwise}
\end{cases}
\end{aligned}
$$
$\bigstar$ Fast to compute<br>
$\bigstar$ Gradient do not vanish for x > 0<br>
$\bigstar$ Provide faster convergence in practice<br>
$\bigstar$ Not zero-centered<br>
$\bigstar$ If not activated (x < 0), never update<br>
$\bigstar$ ReLU (rectified linear unit) is a default activation function.
</p>

#### 1.1.3 Leaky ReLU
<p align="justify">
$\bigstar$ Will not die<br>
$\bigstar$ $a \neq 1$, because a = 1 loses activation functionality.
$$
\begin{aligned}
f(x) &=
\begin{cases}
x, \quad & x \geq 0 \\
ax, \quad & x < 0
\end{cases} \\
&= \max(ax, x), \quad \text{where } a \in (0, 1)
\end{aligned}
$$
</p>

#### 1.1.4 Sigmoid Layer
<p align="justify">
$$
\begin{aligned}
& \text{Sigmoid}(Z) = \pi(Z) = \frac{1}{1 + e^{-Z}} \\
& \frac{\partial \pi(Z)}{\partial Z} = \pi(Z) (1 - \pi(Z)) \leq \frac{1}{4}
\end{aligned}
$$
$\bigstar$ Sigmoid neurons can saturate and lead to <b>vanishing gradients</b><br>
$\bigstar$ Not zero-centered (actually 0.5-centered)<br>
$\bigstar$ $e^{x}$ is computationally expensive.<br>
$\bigstar$ If the output is binary, sigmoid is a natural choice.
</p>

#### 1.1.5 Tanh
<p align="justify">
$$
\begin{aligned}
& \text{tanh}(x) = g(x) = \frac{2}{1 + e^{-2x}} - 1 = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} \\
& \frac{\partial}{\partial x} \text{tanh}(x) = \frac{\partial}{\partial x} g(x) = 4 \frac{e^{-2x}}{(1 + e^{-2x})^{2}} = 1 - (g(x))^{2} \leq 1
\end{aligned}
$$
$\bigstar$ Zeros-centered, which is helpful numerically<br>
$\bigstar$ Still pretty much like sigmoid<br><br>

<b>Quiz:</b> You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)*1000. What will happen?<br>
<b>Answer</b>: This will cause the inputs of the tanh to also be very large, thus causing gradients to be close to zero. The optimization algorithm will thus become slow.
</p>

#### 1.1.6 Softmax Layer
<p align="justify">
$$
\begin{aligned}
& Z = (Z_{1}, Z_{2}, ..., Z_{K}) = (w_{1}^{T}x, w_{2}^{T}x, ..., w_{K}^{T}x) \\
& \text{Softmax}(Z) = (\frac{e^{Z_{1}}}{\sum_{k=1}^{K}e^{Z_{k}}}, \frac{e^{Z_{2}}}{\sum_{k=1}^{K}e^{Z_{k}}},..., \frac{e^{Z_{K}}}{\sum_{k=1}^{K}e^{Z_{k}}}) \\
& \text{e.g. } Z = (7, -7.5, 10) \rightarrow \text{Softmax}(Z) \approx (0.05, 0, 0.95)
\end{aligned}
$$

$\bigstar$ Cross-entropy for classification as loss function is differentiable and can work as a loss function
$$
\begin{aligned}
J(w,b) &= -\sum_{i=1}^{m} \sum_{k=1}^{K} \mathbf{I}_{y_{i}=k} \log \frac{e^{Z_{i, k}}}{\sum_{j=1}^{K} e^{Z_{i, j}} } \\
&= -\sum_{i=1}^{m} \sum_{k=1}^{K} \mathbf{I}_{y_{i}=k} \log \frac{e^{w_{k}^{T}x_{i}}}{\sum_{j=1}^{K}e^{w_j^{T}x_{j}}} \\
&= -\sum_{i=1}^{m} \log\frac{e^{w_{y_{i}}^{T}x_{i}}}{\sum_{j=1}^{K}e^{w_j^{T}x_{j}}} \\
&= -\sum_{i=1}^{m} (w_{y_{i}}^{T}x_{i} - \log \sum_{j=1}^{K}e^{w_j^{T}x_{j}}) \\
&= -\sum_{i=1}^{m} (Z_{i, y_{i}} - \log \sum_{j=1}^{K} e^{Z_{i, j}}) \\
&\rightarrow \min_{w,b} 
\end{aligned}
$$

$\bigstar$ Log-softmax is better than naive log(softmax(a))<br>
-- Better numerical stability<br>
-- Easier to get derivative right<br>
-- Marginally faster to compute
</p>

#### 1.1.7 Vectorization
<p align="justify">
<b>Whenever possible, avoid explicit for-loops</b><br>

$\bigstar$ Vectorization
$$\text{z = np.dot(w, x)}$$

<b>Quiz:</b> Vectorization cannot be done without a GPU.<br>
A. False<br>
B. True<br>
<b>Answer</b>: A.<br><br>

$\bigstar$ Broadcasting
(m, n) + R = (m, n)<br>
(m, n) + (1, n) = (m, n)<br>
(m, n) + (m, 1) = (m, n)<br><br>

For example
$$
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6
\end{bmatrix} + 
\begin{bmatrix}
100 & 200 & 300
\end{bmatrix} =
\begin{bmatrix}
101 & 202 & 303\\
104 & 205 & 306
\end{bmatrix}
$$

$$
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6
\end{bmatrix} + 
\begin{bmatrix}
100\\
200
\end{bmatrix} =
\begin{bmatrix}
101 & 102 & 103\\
104 & 205 & 206
\end{bmatrix}
$$
</p>

### 1.2 CNN
#### 1.2.1 Convolution Layer
<p align="justify">
<b>Why MLP doesn't wok?</b><br>
$\bigstar$ We learn the same 'features' in different areas and don't fully utilize the training set<br>
$\bigstar$ What if cats in the test set appear in different places<br><br>

<b>Convolution</b><br>
$$
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 1 & 0 \\
1 & 0 & 1 & 0 \\
1 & 0 & 1 & 1
\end{bmatrix} *
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix} =
\begin{bmatrix}
5 & 9 & 4 \\
5 & 7 & 4 \\
4 & 6 & 8
\end{bmatrix}
$$
$\bigstar$ Sharpening kernel doesn't change an image for solid fills but adds a little intensity on the edges
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/DLB/1_2_1_1.png"/></center>
</p>
<p align="justify">
$\bigstar$ Convolutions is translation equivariant
$$
\begin{aligned}
&
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} *
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} =
\begin{bmatrix}
0 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 2
\end{bmatrix} \\
&
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix} *
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} =
\begin{bmatrix}
2 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 0
\end{bmatrix}
\end{aligned}
$$

$\bigstar$ Backpropagation for CNN<br><br>

$\bigstar$ Convolutional layer vs. Dense layer<br>
-- In convolutional layer the same kernel is used for every output neuron, this way we shared parameters of the network and train a better model<br>
-- Suppose we have $300 \times 300$ input, $300 \times 300$ output and a $5 \times5 $ kernel, we have 26 parameters in convolutional layer and $8.1\times10^{9}$ parameters in fully connected layer (each output is a perceptron).<br>
-- Convolutional layer can be viewed as a special case of a fully connected layer when all weights outside the <b>local receptive field</b> of each neuron equal to 0 and kernel parameters are shared between neurons<br><br>

$\bigstar$ $C_{out}$ kernels of $W \times H \times C_{in}$<br>
-- W: an image width<br>
-- H: an image height<br>
-- $C_{in}$: number of input channels (e.g. 3 channels RGB for color image)<br>
-- $C_{out}$: number of kernels<br>
-- Total parameters is $(W_{k} \times H_{k} \times C_{in} + 1) \times C_{out}$<br><br>

<b>Quiz:</b> Suppose you have a 10x10x3 colour image input and you want to stack two convolutional layers with kernel size 3x3 with 10 and 20 filters respectively. How many parameters do you have to train for these two layers? Don't forget bias terms!<br>
<b>Answer:</b> $(3 \times 3 \times 3 + 1) \times 10 + (3 \times 3 \times 10 + 1) \times 20 = 2100$.<br><br>

$\bigstar$ Receptive field
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/DLB/1_2_1_2.png"/></center>
</p>
<p align="justify">
<b>Quiz:</b> What receptive field do we have after stacking n convolutional layers with kernel size k×k and stride 1? Layers numeration starts with 1. The resulting receptive field will be a square, input its side as an answer.<br>
<b>Answer:</b> nk-n+1.<br><br>

Vertical edge detection
$$
\begin{bmatrix}
1 & 0 & -1\\
1 & 0 & -1\\
1 & 0 & -1
\end{bmatrix}
$$

Horizontal edge detection
$$
\begin{bmatrix}
1 & 1 & 1\\
0 & 0 & 0\\
-1 & -1 & -1
\end{bmatrix}
$$

Sobel operator
$$
\begin{bmatrix}
1 & 0 & 1\\
2 & 0 & -2\\
1 & 0 & -1
\end{bmatrix}
$$

Sharp filter
$$
\begin{bmatrix}
3 & 0 & -3\\
10 & 0 & -10\\
3 & 0 & -3
\end{bmatrix}
$$

<b>Padding</b><br>
"Valid": $n \times n$ * $f \times f$ $\rightarrow$ $(n - f +1) \times (n - f + 1)$<br>
"same": pad so that output size is the same as the input size.<br><br>

<b>Strided Convolutions</b><br>
$$(n \times n) * (f \times f) \rightarrow (\left \lfloor \frac{n + 2p - f}{s} + 1 \right \rfloor \times \left \lfloor \frac{n + 2p - f}{s} + 1 \right \rfloor)$$

For example, we have a $7 \times 7$ image with $3 \times 3$ filter under padding = 'valid' and stride = 2, the output size is<br>
$$(\frac{7 + 0 - 3}{2} + 1, \frac{7 + 0 - 3}{2} + 1) = (3, 3)$$

<b>One Layer of a Convolutional Network</b><br>
If we have 10 filters that are $3 \times 3 \times 3$ in one layer of a neural network, the number of parameters does that kayer have?<br>
$$(3 \times 3 \times 3 + 1) \times 10 = 280$$

If layer l is a convolution layer
$$f^{[l]} = \text{filter size}, \quad p^{[l]} = \text{padding}, \quad s^{[l]} = \text{stride}, \quad n_{c}^{[l]} = \text{number of filters}$$

Each filter has a size
$$f^{[l]} \times f^{[l]} \times n_{c}^{[l-1]}$$

Input for this layer
$$n_{H}^{[l-1]} \times n_{W}^{[l-1]} \times n_{c}^{[l-1]}$$

Output for this layer
$$n_{H}^{[l]} \times n_{W}^{[l]} \times n_{c}^{[l]}$$
$$n_{H}^{[l]} = \left \lfloor \frac{n_{H}^{[l-1]} +2p^{[l]} -f^{[l]}}{s^{[l]}} + 1 \right \rfloor, \quad n_{W}^{[l]} = \left \lfloor \frac{n_{W}^{[l-1]} +2p^{[l]} -f^{[l]}}{s^{[l]}} + 1 \right \rfloor$$

Activation
$$A^{[l]} \rightarrow m \times n_{H}^{[l]} \times n_{W}^{[l]} \times n_{c}^{[l]}$$

Weights
$$f^{[l]} \times f^{[l]} \times n_{c}^{[l-1]} \times n_{c}^{[l]}$$

Bias
$$n_{c}^{[l]} \rightarrow (1, 1, 1, n_{c}^{[l]})$$

<b>Quiz:</b> You have an input volume that is 63x63x16, and convolve it with 32 filters that are each 7x7, using a stride of 2 and no padding. What is the output volume?<br>
<b>Answer</b>: 29x29x32.<br><br>

<b>Quiz:</b> You have an input volume that is 15x15x8, and pad it using “pad=2.” What is the dimension of the resulting volume (after padding)?<br>
<b>Answer</b>: 19x19x8.<br><br>

<b>Quiz:</b> You have an input volume that is 63x63x16, and convolve it with 32 filters that are each 7x7, and stride of 1. You want to use a “same” convolution. What is the padding?<br>
<b>Answer</b>: 3.<br><br>

<b>Quiz:</b> You have an input volume that is 32x32x16, and apply max pooling with a stride of 2 and a filter size of 2. What is the output volume?<br>
<b>Answer</b>: 16x16x16.<br><br>

“sparsity of connections” as a benefit of using convolutional layers means each activation in the next layer depends on only a small number of activations from the previous layer<br>

</p>

#### 1.2.2 Pooling layer
<p align="justify">
$$
\begin{aligned}
&
\begin{bmatrix}
1 & 3 & 2 & 1\\
2 & 9 & 1 & 1\\
1 & 3 & 2 & 3\\
5 & 6 & 1 & 2
\end{bmatrix} \rightarrow
\begin{bmatrix}
9 & 2\\
6 & 3
\end{bmatrix} & \text{Max pooling} \\
&
\begin{bmatrix}
1 & 3 & 2 & 1\\
2 & 9 & 1 & 1\\
1 & 3 & 2 & 3\\
5 & 6 & 1 & 2
\end{bmatrix} \rightarrow
\begin{bmatrix}
\frac{15}{4} & \frac{5}{4}\\
\frac{15}{4} & 2
\end{bmatrix} & \text{Average pooling}
\end{aligned}
$$

<b>Hyperparameters</b><br>
$\bigstar$ filter size, stride, max or average pooling<br><br>
</p>

### 1.3 RNN
#### 1.3.1 Bag of words
<p align="justify">
Journal of Artificial Intelligence Research JAIR is a refereed journal, covering the areas of Artificial Inteligence, which is distributed free of charge over the Internet. Each volume of the journal is also published be Morgan Kaufman.
<table class="a">
  <tr><th>Word</th><th>Count</th></tr>
  <tr><td>journal</td><td>3</td></tr>
  <tr><td>intelligence</td><td>2</td></tr>
  <tr><td>internet</td><td>1</td></tr>
  <tr><td>$\vdots$</td><td>$\vdots$</td></tr>
</table><br>
</p>
{% highlight Python %}

{% endhighlight %}

## 2. Initialization, Regularization and Optimisation
### 2.1 Initialization
#### 2.1.1 Random Initialization
<p align="justify">
If you decide to initialize the weights and biases to be zero, each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons.<br><br>

Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?<br>
A. True<br>
B. False<br>
<b>Answer</b>: B.<br>
Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector.<br><br>

$\bigstar$ Need to break symmetry<br>
$\bigstar$ Linear models work best when inputs are normalized<br>
$\bigstar$ Neuron is a linear combination of inputs + activation<br>
$\bigstar$ Neuron output will be used by consecutive layers
</p>

#### 2.1.2 Xavier Initialization
<p align="justify">
$\bigstar$ Let's look at the neuron output before activation: $\sum_{i=1}^{n}x_{i}w_{i}$<br>
$\bigstar$ If $E(x_{i}) = E(w_{i}) = 0$ and we generate weights independently from inputs, then $E(\sum_{i=1}^{n}x_{i}w_{i}) = 0$<br>
$\bigstar$ But variance can grow with consecutive layers.<br>
$\bigstar$ Empirically this hurts convergence for deep network<br><br>

$\bigstar$ Let's look at the variance of $\sum_{i=1}^{n}x_{i}w_{i}$<br>
-- $w_{i}$ is independent of $x_{i}$
$$Var(\sum_{i=1}^{n}x_{i}w_{i}) = \sum_{i=1}^{n}Var(x_{i}w_{i})$$
$$Var(x_{i}w_{i}) = [ E(x_{i}) ]^{2}Var(w_{i}) + [ E(w_{i}) ]^{2}Var(x_{i}) + Var(x_{i})Var(w_{i})$$
$$E(x_{i}) = E(w_{i}) = 0$$
$$Var(\sum_{i=1}^{n}x_{i}w_{i}) = \sum_{i=1}^{n}Var(x_{i})Var(w_{i}) = Var(x)[ nVar(w) ]$$

We want $nVar(w) = 1$<br><br>

$\bigstar$ Let's use the fact that $Var(aw) = a^{2}Var(w)$<br>
$\bigstar$ For [nVar(aw)] to be 1, we need to multiply $N(0, 1)$ weights (Var(w) = 1) by $a = \frac{1}{\sqrt{n}}$<br>
$\bigstar$ Xavier initialization (Glorot et al)<br>
-- Multiply weights by $\frac{\sqrt{2}}{\sqrt{n_{in}+n_{out}}}$<br>
</p>

#### 2.1.3 He Initialization
<p align="justify">
$\bigstar$ Initialization for ReLU neurons (He et al)<br>
-- Multiply by $\frac{\sqrt{2}}{\sqrt{n_{in}}}$ for ReLu neurons
</p>

### 2.2 Regularization
#### 2.2.1 L1 & L2
<p align="justify">
<b>Regularization in logistic regression</b><br>
$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y^{i}}, y^{i}) + \lambda R(w), \quad \text{where } R(w) \text{ is a regularizor}$$

<b>L1 norm</b><br>
$\bigstar$ L1 norm draws w to be sparse<br>
$\bigstar$ Derive some weights exactly to 0<br>
$\bigstar$ Cannot be optimized with gradient descent<br>
$\bigstar$ Find some small subset of features most important for the problem
$$
\begin{aligned}
L_{reg}(w) &= L(w) + \lambda \left \| w \right \|_{1} \\
&= L(w) + \frac{\lambda}{2m} \sum_{j=1}^{m} \left | w_{j} \right | \\
&\rightarrow \min_{w}
\end{aligned}
$$

<b>L2 norm</b><br>
$\bigstar$ Drive all weights close to 0<br>
$\bigstar$ Can be optimised with gradient descent
$$\frac{\lambda}{2m} \left \| w \right \|_{2}^{2} = \frac{\lambda}{2m} \sum_{j = 1}^{m} w_{j}^{2} = \frac{\lambda}{2m} w^{T}w$$

<b>Quiz:</b><br> What is weight decay?<br>
A. The process of gradually decreasing the learning rate during training.<br>
B. A technique to avoid vanishing gradient by imposing a ceiling on the values of the weights.<br>
C. Gradual corruption of the weights in the neural network if it is trained on noisy data.<br>
D. A regularization technique (such as L2 regularization) that results in gradient descent shrinking the weights on every iteration.<br>
<b>Answer</b>: D.<br><br>

<b>Quiz:</b><br> What happens when you increase the regularization hyperparameter lambda?<br>
A. Weights are pushed toward becoming smaller (closer to 0)<br>
B. Weights are pushed toward becoming bigger (further from 0)<br>
C. Doubling lambda should roughly result in doubling the weights<br>
D. Gradient descent taking bigger steps with each iteration (proportional to lambda)<br>
<b>Answer</b>: A.<br><br>
</p>

#### 2.2.2 Frobenius norm
<p align="justify">
Regularization reduces overfitting by drawing weights close to 0, which is equivalent to eliminate some neurons. If z is close to 0 because of regularization, activation function tanh(z) turns to be linear.
$$
\begin{aligned}
& J(W^{[1]}, b^{[1]}, ..., W^{[L]}, b^{[L]}) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y^{i}}, y^{i}) + \frac{\lambda}{2m} \sum_{l=1}^{L} \left \| W^{[l]} \right \|_{F}^{2} \\
& \left \| W^{[l]} \right \|_{F}^{2} = \sum_{i=1}^{n^{l}} \sum_{j=1}^{n^{l-1}} (W_{ij}^{[l]})^{2}
\end{aligned}
$$
</p>

#### 2.2.3 Dropout
<p align="justify">
$\bigstar$ Regularization technique to reduce overfitting<br>
$\bigstar$ We keep neurons active (non-zero) with probability p<br>
$\bigstar$ This way we sample the network during training and change only a subset of its parameters on every iterations<br>
$\bigstar$ During testing all neurons are present but their outputs are multiplied by p to maintain the scale of inputs<br>
$\bigstar$ The author of dropout say it's similar to having an ensemble of exponentially large number of smaller networks<br><br>

<b>Dropout</b><br>
keep_prob = 0.8 means 20% chance that an edge between two neurons is hidden.<br><br>

Inverted dropout<br>
</p>
{% highlight Python %}
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3)  # a3 = a3 * d3
a3 = a3 / keep_prob
{% endhighlight %}

<p align="justify">
For test set, do not use dropout.<br><br>

<b>Why does dropout work?</b><br>
Intuition: can't rely on any one feature, so have to spread out weights.<br><br>

<b>Quiz:</b><br> Increasing the parameter keep_prob from (say) 0.5 to 0.6 will likely cause the following: (Check the two that apply)<br>
A. Increasing the regularization effect<br>
B. Reducing the regularization effect<br>
C. Causing the neural network to end up with a higher training set error<br>
D. Causing the neural network to end up with a lower training set error<br>
<b>Answer</b>: B, D.<br><br>
</p>

#### 2.2.4 Batch Normalization
<p align="justify">
<b>Batch normalization</b><br>
$\bigstar$ We know how to initialize our network to constrain variance<br>
$\bigstar$ But what if it grows during backpropagation?<br>
$\bigstar$ Batch normalization controls mean and variance of outputs before activations<br><br>

$\bigstar$ Let's normaliza $h_{i}$ -- neuron output before activations:
$$h_{i} = \gamma_{i} \frac{h_{i}-\mu_{i}}{\sqrt{\sigma^{2}}} + \beta_{i}, \quad  \frac{h_{i}-\mu_{i}}{\sqrt{\sigma^{2}}} \text{ is 0 mean, unit variance}$$

$\bigstar$ Where do $\mu_{i}$ and $\sigma^{2}$ come from?<br>
-- We can estimate them having a current training batch<br>
$\bigstar$ During testing we will use an exponential moving average over train batches:
$$0 < \alpha < 1$$

$$\mu_{i} = \alpha \cdot \text{mean}_{batch} + (1-\alpha)\mu_{i}$$

$$\sigma_{i}^{2} = \alpha \cdot \text{variance}_{batch} + (1-\alpha)\sigma_{i}^{2}$$

$\bigstar$ What about $\gamma_{i}$ and $\beta_{i}$?<br>
-- Normalization is a differentiable operation and we can apply backpropagation<br><br>

<b>Batch Normalization</b><br>
Normalizing inputs to speed up learning. We can normalize $a^{[2]}$ to train $w^{[3]}$, $b^{[3]}$ faster.<br>
$$\mu = \frac{1}{m} \sum_{i}^{m} Z^{[l](i)}, \quad \sigma^{2} = \frac{1}{m} \sum_{i=1}^{m} (Z^{[l](i)} - \mu)^{2}$$
$$Z_{\text{norm}}^{(i)} = \frac{Z^{[l](i)} - \mu}{\sqrt{\sigma^{2} + \epsilon}}$$

But hidden units are not always a distribution with 0 mean and 1 standard deviation<br>
$$\widetilde{Z}^{[l](i)} = \gamma Z_{\text{norm}}^{[l](i)} + \beta$$

$\gamma$ and $\beta$ are learnable parameters. Here's $\beta$ is different form $\beta$ in momentum.<br><br>

Adding Batch Norm to a network<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/DLB/2_2_4_1.png"/></center>
</p>
<p align="justify">
Now, besides $W^{[1]}$, $b^{[1]}$, ..., $W^{[L]}$, $b^{[L]}$, we have $\gamma^{[1]}$, $\beta^{[1]}$, ..., $\gamma^{[L]}$, $\beta^{[L]}$ to learn. Luckily, gradient descent still works for them.<br>
$$\gamma^{[l]} = \gamma^{[l]} - \alpha d\gamma^{[l]} \quad, \beta^{[l]} = \beta^{[l]} - \alpha d \beta^{[l]}$$

Evidently, mini-batch, momentum, RMSprop and Adam also work.<br><br>

Batch nnormalization works by correcting input shift during training.<br><br>

<b>Batch Normalization as regularization</b><br>
Each mini-batch is scaled by the mean / variance computed on just that mini-batch.<br>
This adds sume noise to the values $Z^{[l]}$ within that mini-batch. So similar to dropout, it adds some noise to each hidden layer's activations<br>
This has a slight regularization effect.<br><br>

When we perform Batch Norm (usually one example) at test time, we can estimate $\gamma$, $\beta$ for test data by $\gamma$, $\beta$ for training mini-batch data in a way of exponentially weighted average.<br><br>

<b>Quiz:</b><br> After training a neural network with Batch Norm, at test time, to evaluate the neural network on a new example you should:<br>
A. If you implemented Batch Norm on mini-batches of (say) 256 examples, then to evaluate on one test example, duplicate that example 256 times so that you’re working with a mini-batch the same size as during training.<br>
B. Skip the step where you normalize using $\mu$ and $\sigma^{2}$ since a single test example cannot be normalized.<br>
C. Use the most recent mini-batch’s value of $\mu$ and $\sigma^{2}$ to perform the needed normalizations.<br>
D. Perform the needed normalizations, use $\mu$ and $\sigma^{2}$ estimated using an exponentially weighted average across mini-batches seen during training.<br>
<b>Answer</b>: D.
</p>

#### 2.2.5 Data augmentation
<p align="justify">
$\bigstar$ Modern CNNs have millions of parameters<br>
$\bigstar$ But datasets are not that huge<br>
$\bigstar$ We can generate new examples applying distortion flips, rotations, color shift, scaling etc<br>
$\bigstar$ Remember: CNNs are invariant to translation<br>
$\bigstar$ color shifting
</p>

#### 2.2.6 Early stopping
<p align="justify">

</p>

#### 2.2.7 Normalizing inputs
<p align="justify">
Normalizing inputs is helpful for gradient descent.
$$
\begin{aligned}
& X = \frac{X - \mu}{\sigma} \\
& \mu = \frac{1}{m} \sum_{i=1}^{m} x^{i}, \quad \sigma^{2} = \frac{1}{m} \sum_{i=1}^{m} (x^{i} - \mu)^{2}
\end{aligned}
$$
<b>Quiz:</b> Why do we normalize the inputs x?<br>
A. It makes the parameter initialization faster<br>
B. It makes the cost function faster to optimize<br>
C. Normalization is another word for regularization--It helps to reduce variance<br>
D. It makes it easier to visualize the data<br>
<b>Answer</b>: B.<br><br>

<b>Image as a neural network input</b><br>
$\bigstar$ Normalize input pixels
$$x_{norm} = \frac{x}{255} - 0.5$$
</p>

#### 2.2.8 Dimensionality reduce
<p align="justify">

</p>

#### 2.2.9 Gradient checking
<p align="justify">
Take $W^{[1]}$, $b^{[1]}$, ..., $W^{[L]}$, $b^{[L]}$ and reshape into a big vector $\theta$<br>
Take $dW^{[1]}$, $db^{[1]}$, ..., $dW^{[L]}$, $db^{[L]}$ and reshape into a big vector $d\theta$<br>
$$J(\theta) = J(\theta_{1}, \theta_{2}, ..., \theta_{n})$$

For each $\theta_{i}$, we have to check<br>
$$d\theta_{i}^{\text{approx}} = \frac{J(\theta_{1}, ..., \theta_{i} + \epsilon, ..., \theta_{n})-J(\theta_{1}, ..., \theta_{i} - \epsilon, ..., \theta_{n})}{2\epsilon} \approx d\theta_{i}$$

In other form<br>
$$\frac{\left \| d\theta_{i}^{\text{approx}} - d\theta_{i} \right \|_{2}}{\left \| d\theta_{i}^{\text{approx}} \right \|_{2} + \left \| d\theta_{i} \right \|_{2}} < 10^{-7}$$

Don't use in training - only to debug<br>
If algorithm fails grad check, look at components to try identify bug.<br>
Remember regularization<br>
Doesn't work with dropout<br>
Run at random initialization: perhaps again after some training<br><br>
</p>

### 2.3 Optimisation
#### 2.3.1 Gradient Descent
<p align="justify">
<b>Gradient descent</b><br>
Optimization problem:
$$L(w)\rightarrow \min_{w}$$

Initialization:
$$w^{0}$$

Gradient vector:
$$\nabla L(w^{0})=(\frac{\partial L(w^{0})}{\partial w_{1}}, \frac{\partial L(w^{0})}{\partial w_{2}}, ..., \frac{\partial L(w^{0})}{\partial w_{n}})$$

$\bigstar$ Points in the direction of the steepest slop at $w^{0}$.<br>
$\bigstar$ The function has fastest decrease rate in the direction of negative gradient.<br><br>

Update w with a gradient step:
$$w^{t+1} = w^{t} - \eta_{t}\nabla L(w^{t})$$

Stop at convergence:
$$\left \| w^{t+1}-w^{t} \right \| < \epsilon$$

Optimization problem<br>
$$L(w) = \sum_{i=1}^{n}L(w; x_{i}, y_{i}) \rightarrow \min_{w}$$

$w^{0}$ -- initialization<br>
while true:<br>
&emsp;w^{t} = w^{t-1} -\eta_{t}\nabla L(w^{t-1})<br>
&emsp;if $\left\| w^{t}-w^{t-1} \right\| \leq \epsilon$, break<br><br>

For example, Mean squared error
$$\nabla L(w) = \frac{1}{n}\sum_{i=1}^{n} \nabla (w^{T}x_{i} - y_{i})^{2}$$

$\bigstar$ n gradients should be computed on each step<br>
$\bigstar$ If the dataset doesn't fit in memory, it should be read from the disk on every GD step<br><br>

$\bigstar$ Some heuristics<br>
-- How to initialize $w^{0}$<br>
-- How ro select step size $\eta_{t}$<br>
-- When to stop<br>
-- Hwo to approxiamte gradient $\nabla L(w^{t})$<br><br>

<b>Mini-batch gradient descent</b><br>
Optimization problem<br>
$$L(w) = \sum_{i=1}^{n}L(w; x_{i}, y_{i}) \rightarrow \min_{w}$$

$w^{0}$ -- initialization<br>
while true:<br>
&emsp;$i_{1}, i_{2}, ..., i_{m} $ = random indice between 1 ~ n<br>
&emsp;$g_{t} = \frac{1}{m}\sum_{j=1}^{m}\nabla L(w^{t};x_{i_{j}},y_{i_{j}})$<br>
&emsp;$w^{t+1} = w^{t} - \eta_{t}g_{t}$<br>
&emsp;if $\left\| w^{t+1}-w^{t} \right\| \leq \epsilon$, break<br><br>

$\bigstar$ Still can be used in online setting<br>
$\bigstar$ Reduce the variance of gradient approximations<br>
$\bigstar$ Careful to choose learning rate $\eta_{t}$<br>
If mini-batch size is m, this is batch gradient descent, which is suitable for smaller train set.<br>
If mini-batch size is 1, this is stochastic gradient descent, which is for huge train set.<br><br>

<b>Stochastic Gradient Descent</b><br>
Gradient descent claculate a gradient for n sample. If we have a huge amount of data, this couldn't be possible for a limited RAM. So, we introduce a stochastic gradient descent.<br>
Optimization problem<br>
$$L(w) = \sum_{i=1}^{n}L(w; x_{i}, y_{i}) \rightarrow \min_{w}$$

$w^{0}$ -- initialization<br>
while true:<br>
&emsp;i = random indice between 1 ~ n<br>
&emsp;$g_{t} = \nabla L(w^{t};x_{i},y_{i})$<br>
&emsp;$w^{t+1} = w^{t} - \eta_{t}g_{t}$<br>
&emsp;if $\left\| w^{t+1}-w^{t} \right\| \leq \epsilon$, break<br><br>

$\bigstar$ Nosiy update leads to fluctuations<br>
$\bigstar$ Needs only one example on each step<br>
$\bigstar$ Can be used in online setting<br>
$\bigstar$ Careful to choose learning rate $\eta_{t}$<br><br>

<b>Quiz:</b><br> Which of these statements about mini-batch gradient descent do you agree with?<br>
A. Training one epoch (one pass through the training set) using mini-batch gradient descent is faster than training one epoch using batch gradient descent.<br>
B. One iteration of mini-batch gradient descent (computing on a single mini-batch) is faster than one iteration of batch gradient descent.<br>
C. You should implement mini-batch gradient descent without an explicit for-loop over different mini-batches, so that the algorithm processes all mini-batches at the same time (vectorization).<br>
<b>Answer</b>: B.<br><br>

<b>Quiz:</b><br> Why is the best mini-batch size usually not 1 and not m, but instead something in-between?<br>
A. If the mini-batch size is m, you end up with batch gradient descent, which has to process the whole training set before making progress.<br>
B. If the mini-batch size is 1, you lose the benefits of vectorization across examples in the mini-batch.<br>
C. If the mini-batch size is 1, you end up having to process the entire training set before making any progress.<br>
D. If the mini-batch size is m, you end up with stochastic gradient descent, which is usually slower than mini-batch gradient descent.<br>
<b>Answer</b>: A, B.<br><br>
</p>

#### 2.3.2 Exponentially Weighted Averages
<p align="justify">
$$V_{t} = \beta V_{t-1} + (1 - \beta) \theta_{t} = \beta^{t} V_{0} + \sum_{k = 1}^{t} \beta^{t - k} (1 - \beta) \theta_{k}$$

For example, temperature in London
$$
\begin{aligned}
& \theta_{1} = 40^{\circ}\text{F} \\
& \theta_{2} = 49^{\circ}\text{F} \\
& \theta_{3} = 45^{\circ}\text{F} \\
& \vdots\\
& \theta_{180} = 60^{\circ}\text{F} \\
& \theta_{181} = 56^{\circ}\text{F} \\
& \vdots
\end{aligned}
$$

We calculate the average temperature with $\beta$ = 0.9<br>
$$
\begin{aligned}
& V_{0} = 0 \\
& V_{1} = \beta V_{0} + (1 - \beta) \theta_{1} = 0.9 \cdot 0 + 0.1 \cdot 40 = 4 \\
& V_{2} = \beta V_{1} + (1 - \beta) \theta_{2} = 0.9 \cdot 4 + 0.1 \cdot 49 = 8.5 \\
& V_{3} = \beta V_{2} + (1 - \beta) \theta_{3} = 0.9 \cdot 8.5 + 0.1 \cdot 45 = 12.15 \\
& \vdots
\end{aligned}
$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/DLB/2_3_2_1.png"/></center>
</p>
<p align="justify">
$V_{t}$ is an exponential average over $\frac{1}{1 - \beta}$ days<br>
For example, $\beta = 0.98$<br>
$$\frac{1}{1 - \beta} = \frac{1}{1 - 0.98} = 50$$

What would happen to your red curve as you vary β?<br>
-- Increasing β will shift the red line slightly to the right.<br>
-- Decreasing β will create more oscillation within the red line.<br><br>

<b>Bias correction in exponentially weighted averages</b><br>
Because there is a difference between weights and exponentially weighted averages, we need to correct it
$$\frac{V_{t}}{1 - \beta^{t}}$$

<b>Quiz:</b><br> Suppose the temperature in Casablanca over the first three days of January are the same:<br>
$$\text{jan 1st: } \theta_{1} = 10^{\circ}\text{C}$$
$$\text{jan 2nd: } \theta_{2} = 10^{\circ}\text{C}$$

Say you use an exponentially weighted average with $\beta$ = 0.5 to track the temperature:
$$v_{0}$ = 0, $v_{t} = \beta v_{t-1} + (1 - \beta) \theta_{t}$$

If $v_{2}$ is the value computed after day 2 without bias correction, and $v_{2}^{\text{corrected}}$ is the value you compute with bias correction. What are these values? (You might be able to do this without a calculator, but you don't actually need one. Remember what is bias correction doing.)<br>
A. $v_{2}$ = 10, $v_{2}^{\text{corrected}}$ = 7.5<br>
B. $v_{2}$ = 7.5, $v_{2}^{\text{corrected}}$ = 10<br>
C. $v_{2}$ = 7.5, $v_{2}^{\text{corrected}}$ = 7.5<br>
D. $v_{2}$ = 10, $v_{2}^{\text{corrected}}$ = 10<br>
<b>Answer</b>: B.<br><br>

<b>Quiz:</b><br> Which of these is NOT a good learning rate decay scheme? Here, t is the epoch number.<br>
A. $\alpha =e^{t} \cdot \alpha_{0}$<br>
B. $\alpha = 0.95^{t} \cdot \alpha_{0}$<br>
C. $\alpha = \frac{1}{\sqrt{t}} \cdot \alpha_{0}$<br>
D. $\alpha = \frac{1}{1 + 2*t} \cdot \alpha_{0}$<br>
<b>Answer</b>: A.<br><br>
</p>

#### 2.3.3 Momentum
<p align="justify">
$\bigstar$ Momentum is used to eliminate oscillation.<br>
$\bigstar$ Hyperparameter $\beta$ is usually 0.9.<br>
$\bigstar$ For each iteration, we calculate dW and db on mini-batch, then calculate their exponentially weighted averages and finally update W and b
$$
\begin{aligned}
& V_{dW} = \beta V_{dW} + (1 - \beta) dW \\
& V_{db} = \beta V_{db} + (1 - \beta) db \\
& W = W - \alpha V_{dW} \\
& b = b - \alpha V_{db}
\end{aligned}
$$
</p>

#### 2.3.4 RMSProp
<p align="justify">
$\bigstar$ RMSprop: Root Mean Square Propogation<br>
$\bigstar$ Makes hoirzontal gradient faster, vertical gradient slow
$$
\begin{aligned}
& S_{dW} = \beta S_{dW} + (1 - \beta) (dW)^{2} \quad \text{where } (dW)^{2} \text{ are element-wise square} \\
& S_{db} = \beta S_{db} + (1 - \beta) (db)^{2} \quad \text{where } (db)^{2} \text{ are element-wise square} \\
& W = W - \alpha \frac{dW}{\sqrt{S_{dW}} + \epsilon} \\
& b = b - \alpha \frac{db}{\sqrt{S_{db}} + \epsilon}
\end{aligned}
$$
</p>

#### 2.3.5 Adam
<p align="justify">
$\bigstar$ Adam combines Momentum and RMSprop<br>
$\bigstar$ Hyperparameter: $\beta_{1}$ is 0.9, $\beta_{2}$ is 0.999, $\epsilon$ is $10^{-8}$.<br>
$\bigstar$ For each iteration t
$$
\begin{aligned}
& V_{dW} = 0, \quad S_{dW} = 0, \quad V_{db} = 0, \quad S_{db} = 0 \\
& V_{dW} = \beta_{1} V_{dW} + (1 - \beta_{1}) dW, \quad V_{db} = \beta_{1} V_{db} + (1 - \beta_{1}) db \\
& S_{dW} = \beta_{2} S_{dW} + (1 - \beta_{2}) (dW)^{2}, \quad S_{db} = \beta_{2} S_{db} + (1 - \beta_{2}) (db)^{2} \\
& V_{dW}^{\text{corrected}} = \frac{V_{dW}}{1 - \beta_{1}^{t}}, \quad V_{db}^{\text{corrected}} = \frac{V_{db}}{1 - \beta_{1}^{t}} \\
& S_{dW}^{\text{corrected}} = \frac{S_{dW}}{1 - \beta_{2}^{t}}, \quad S_{db}^{\text{corrected}} = \frac{S_{db}}{1 - \beta_{2}^{t}} \\
& W = W - \alpha \frac{V_{dW}^{\text{corrected}}}{\sqrt{S_{dW}^{\text{corrected}}}+\epsilon}, \quad b = b - \alpha \frac{V_{db}^{\text{corrected}}}{\sqrt{S_{db}^{\text{corrected}}}+\epsilon}
\end{aligned}
$$
</p>

#### 2.3.6 Learning rate decay
<p align="justify">
$$
\begin{aligned}
& \alpha = \frac{\alpha_{0}}{ 1 + \text{DecayRate} * \text{EpochNumber}} \\
& \alpha =\alpha_{0}\cdot \text{DecayRate}^{\text{EpochNumber}} \\
& \alpha = \frac{k}{\sqrt{\text{EpochNumber}}} \cdot \alpha_{0}
\end{aligned}
$$
</p>


## 3. Machine Learning Strategy
### 3.1 Thoery
#### 3.1.1 Assumptions
<p align="justify">
<b>Chain of assumptions in ML</b><br>
Fit training set well on cost function<br>
Fit dev set well on cost function<br>
Fit test set well on cost function<br>
Performs well in real world<br><br>

<b>Why compare to human-level performance</b><br>
Humans are quite good at a lot of tasks. So long as ML is worse than humans, we can<br>
-- get labeled data from humans<br>
-- gain insight from manual error analysis: why did a person get this right?<br>
-- better analysis of bias/variance<br><br>

<b>Two fundamental assumptions of supervised learning</b><br>
We can fit the training set pretty well.<br>
The training set performance generalizes pretty well to the dev/test set.<br><br>

<b>Applied ML is highly iterative process.</b><br><br>
</p>

#### 3.1.2 Train/Dev/Test
<p align="justify">
<b>Train/dev/test distributions</b><br>
Choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on.<br>
Set your test set to be big enough to give high confidence in the overall performance of your system.<br><br>

<b>Train / Dev / Test sets</b><br>
For a smaller dataset, take 7:2:1; for a huge dataset, take 98:1:1<br><br>

<b>Addressing data mismatch</b><br>
Carry out manual error analysis to try to understand difference between training and dev/test sets<br>
Make training data more similar, or collect more data similar to dev/test sets.<br><br>

<b>Make sure Dev set and Test set are from a same distribution as Train set.</b><br><br>

<b>Correcting incorrect dev/test set examples</b><br>
Apply same process to your dev and test sets to make sure they continue to come from the same distribution.<br>
Consider examining examples your algorithm got right as well as ones it got wrong.<br>
Train and dev/test data may now come from slightly different distribution.<br><br>

Set up dev/test set and metric<br>
Build initial system quickly<br>
Use Bias/Variance analysis & Error analysis to prioritize next steps<br><br>

DL algorithms are quite robust to random errors in the training set.<br><br>

training set: holdout set = 7:3 or 8:2<br><br>

Besides, cross validation, e.g, 10-folder cross validation is a good measure. We split training data into 10 parts, each time we pick one part as our validation set and we repeat this 10 times. Finally, we will acquire 10 performances(loss value) and take an average.<br><br>

<b>Quiz:</b> How many times one should train a model when using cross-validaion with 5 folds?<br>
<b>Answer:</b> 5.<br><br>

<b>Cross-validation</b><br>
$\bigstar$ Requires to train models K times for K-fold CV<br>
$\bigstar$ Useful for small samples<br>
$\bigstar$ In deep learning holdout samples are usually preferred
</p>

#### 3.1.3 Variance vs. Bias
<p align="justify">
Avoid bias by<br>
training bigger model<br>
training longer / better optimization algorithms<br>
NN architecture / hyperoarameters search CNN, RNN<br><br>

Avoid variance by<br>
more data<br>
regularization<br>
NN arachitecture / hyperparameters search<br><br>

<b>Bias / Variance</b><br>
High variance (overfitting): low train set error, say 1%, but high dev/test set error, say 11%<br>
High bias (underfitting): high train set error, say 15% and high dev/test set error, say 16%<br>
High bias and high variance: high train set error, say 15% but higher dev/test set error, say 30%<br><br>

<b>Basic recipe for machine learning</b><br>
If high bias, try bigger network, train in longer time<br>
If high variance, more data, regularization<br><br>

Large model weights can indicate that model is overfitted<br>
Overfitting is a situation where a model gives lower quality for new data compared to quality on a training sample<br>
Weight penalty drives model parameters closer to zero and prevents the model from being too sensitive to small changes in features<br>
Regularization restricts model complexity (namely the scale of the coefficients) to reduce overfitting<br>
</p>

#### 3.1.4 ROC Curve
<p align="justify">
<b>Single number evaluation metric</b><br>
Precision and recall
<table class="a">
  <tr><th>Classifier</th><th>Prediction</th><th>Recall</th></tr>
  <tr><td>A</td><td>95%</td><td>90%</td></tr>
  <tr><td>B</td><td>98%</td><td>85%</td></tr>
</table><br>
</p>
<p align="justify">
$$
\begin{aligned}
& \text{Prediction} = \frac{\text{TP}}{\text{TP + FP}} = \frac{\text{Correctly predicted instances}}{\text{All positive instances}} \\
& \text{Recall} = \frac{\text{TP}}{\text{TP+FN}} = \frac{\text{Correctly predicted instance in positive instance}}{\text{All correctly predicted instances}} \\
& F_{1} \text{ score } = \frac{2}{\frac{1}{\text{Precision}} + \frac{1}{\text{Recall}}}
\end{aligned}
$$

ROC Curve: receiver operating characteristic curve.
$$
\begin{aligned}
& \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} \\
& \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}
\end{aligned}
$$
</p>

#### 3.1.5 Transfer learning
<p align="justify">
<b>When transfer learning makes sense?</b><br>
Task A and task B have the same input x<br>
We have a lot more data for Task A than Task B<br>
Low level features from A could be helpful for learning B<br><br>
</p>

#### 3.1.6 Multi-task learning
<p align="justify">
<b>When multi-task learning makes sense?</b><br>
Training on a set of tasks that could benefit form having shared lower-level features.<br>
Usually: amount of data we have for each task is quite similar<br><br>
</p>

#### 3.1.7 End-to-end learning
<p align="justify">
<b>End-to-end deep learning</b><br>
There have been some data processing systems, or learning systems that require multiple stages of processing. And what end-to-end deep learning does, is it can take all those multiple stages, and replace it usually with just a single neural network.<br><br>

For example, speech recognition example<br>
$$\text{Audio} \rightarrow \text{Features} \rightarrow \text{Phenome} \rightarrow \rightarrow \text{Words} \rightarrow \text{Transcript}$$
$$\text{Audio} \rightarrow \text{Transcript}$$

<b>Pros and cons of end-to-end deep learning</b><br>
Pros:<br>
let the data speak<br>
less hand-designing of components needed<br>
Cons:<br>
may need large amount of data<br>
exclude potentially useful hand-designed coponents<br><br>

<b>Applying end-to-end deep learning</b><br>
Key question: do we have sufficient data to learn a function of the complexity needed to map x to y?<br><br>
</p>

### 3.2 Case Study
#### 3.2.1 Bird recognition in the city of Peacetopia
<p align="justify">
<b>1.</b><br>
Problem Statement<br><br>

This example is adapted from a real production application, but with details disguised to protect confidentiality.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/DLB/3_2_1_1.png"/></center>
</p>
<p align="justify">
You are a famous researcher in the City of Peacetopia. The people of Peacetopia have a common characteristic: they are afraid of birds. To save them, you have to build an algorithm that will detect any bird flying over Peacetopia and alert the population.<br><br>

The City Council gives you a dataset of 10,000,000 images of the sky above Peacetopia, taken from the city’s security cameras. They are labelled:<br>
y = 0: There is no bird on the image<br>
y = 1: There is a bird on the image<br>
Your goal is to build an algorithm able to classify new images taken by security cameras from Peacetopia.<br><br>

There are a lot of decisions to make:<br>
What is the evaluation metric?<br>
How do you structure your data into train/dev/test sets?<br>
Metric of success<br><br>

The City Council tells you that they want an algorithm that<br>
Has high accuracy<br>
Runs quickly and takes only a short time to classify a new image.<br>
Can fit in a small amount of memory, so that it can run in a small processor that the city will attach to many different security cameras.<br><br>

Note: Having three evaluation metrics makes it harder for you to quickly choose between two different algorithms, and will slow down the speed with which your team can iterate. True/False?<br>
A. True<br>
B. False<br>
<b>Answer</b>: A.<br><br>

<b>2.</b><br>
After further discussions, the city narrows down its criteria to:<br>
"We need an algorithm that can let us know a bird is flying over Peacetopia as accurately as possible."<br>
"We want the trained model to take no more than 10sec to classify a new image.”<br>
“We want the model to fit in 10MB of memory.”<br>
If you had the three following models, which one would you choose?<br>
A.<br>
<table class="c">
  <tr><th>Test Accuracy</th><th>Runtime</th><th>Memory size</th></tr>
  <tr><td>97%</td><td>1 sec</td><td>3 MB</td></tr>
</table><br>
</p>
<p align="justify">
B.<br>
<table class="c">
  <tr><th>Test Accuracy</th><th>Runtime</th><th>Memory size</th></tr>
  <tr><td>99%</td><td>13 sec</td><td>9 MB</td></tr>
</table><br>
</p>
<p align="justify">
C.<br>
<table class="c">
  <tr><th>Test Accuracy</th><th>Runtime</th><th>Memory size</th></tr>
  <tr><td>97%</td><td>3 sec</td><td>2 MB</td></tr>
</table><br>
</p>
<p align="justify">
D.<br>
<table class="c">
  <tr><th>Test Accuracy</th><th>Runtime</th><th>Memory size</th></tr>
  <tr><td>98%</td><td>9 sec</td><td>9 MB</td></tr>
</table><br>
</p>
<p align="justify">
<b>Answer</b>: D.<br><br>

<b>3.</b><br>
Based on the city’s requests, which of the following would you say is true?<br>
A. Accuracy is an optimizing metric; running time and memory size are a satisficing metrics.<br>
B. Accuracy is a satisficing metric; running time and memory size are an optimizing metric.<br>
C. Accuracy, running time and memory size are all optimizing metrics because you want to do well on all three.<br>
D. Accuracy, running time and memory size are all satisficing metrics because you have to do sufficiently well on all three for your system to be acceptable.<br>
<b>Answer</b>: A.<br><br>

<b>4.</b><br>
Structuring your data<br>
Before implementing your algorithm, you need to split your data into train/dev/test sets. Which of these do you think is the best choice?<br>
A.<br>
<table class="a">
  <tr><th>Train</th><th>Dev</th><th>Test</th></tr>
  <tr><td>3333334</td><td>3333333</td><td>3333333</td></tr>
</table><br>
</p>
<p align="justify">
B.<br>
<table class="a">
  <tr><th>Train</th><th>Dev</th><th>Test</th></tr>
  <tr><td>6000000</td><td>1000000</td><td>3000000</td></tr>
</table><br>
</p>
<p align="justify">
C.<br>
<table class="a">
  <tr><th>Train</th><th>Dev</th><th>Test</th></tr>
  <tr><td>9500000</td><td>250000</td><td>250000</td></tr>
</table><br>
</p>
<p align="justify">
D.<br>
<table class="a">
  <tr><th>Train</th><th>Dev</th><th>Test</th></tr>
  <tr><td>6000000</td><td>3000000</td><td>1000000</td></tr>
</table><br>
</p>
<p align="justify">
<b>Answer</b>: C.<br><br>

<b>5.</b><br>
After setting up your train/dev/test sets, the City Council comes across another 1,000,000 images, called the “citizens’ data”. Apparently the citizens of Peacetopia are so scared of birds that they volunteered to take pictures of the sky and label them, thus contributing these additional 1,000,000 images. These images are different from the distribution of images the City Council had originally given you, but you think it could help your algorithm.<br>
Notice that adding this additional data to the training set will make the distribution of the training set different from the distributions of the dev and test sets. <br>
Is the following statement true or false?<br>
"You should not add the citizens' data to the training set, because if the training distribution is different from the dev and test sets, then this will not allow the model to perform well on the test set."<br>
A. True<br>
B. False<br>
<b>Answer</b>: B.<br><br>

<b>6.</b><br>
One member of the City Council knows a little about machine learning, and thinks you should add the 1,000,000 citizens’ data images to the test set. You object because:<br>
A. This would cause the dev and test set distributions to become different. This is a bad idea because you’re not aiming where you want to hit.<br>
B. The 1,000,000 citizens’ data images do not have a consistent x-->y mapping as the rest of the data (similar to the New York City/Detroit housing prices example from lecture).<br>
C. The test set no longer reflects the distribution of data (security cameras) you most care about.<br>
D. A bigger test set will slow down the speed of iterating because of the computational expense of evaluating models on the test set.<br>
<b>Answer</b>: A, C.<br><br>

<b>7.</b><br>
You train a system, and its errors are as follows (error = 100%-Accuracy):<br>
<table class="c">
  <tr><td>Training set error</td><td>4.0%</td></tr>
  <tr><td>Dev set error</td><td>4.5%</td></tr>
</table><br>
</p>
<p align="justify">
This suggests that one good avenue for improving performance is to train a bigger network so as to drive down the 4.0% training error. Do you agree?<br>
A. Yes, because having 4.0% training error shows you have high bias.<br>
B. Yes, because this shows your bias is higher than your variance.<br>
C. No, because this shows your variance is higher than your bias.<br>
D. No, because there is insufficient information to tell.<br>
<b>Answer</b>: D.<br><br>

<b>8.</b><br>
You ask a few people to label the dataset so as to find out what is human-level performance. You find the following levels of accuracy:<br>
<table class="c">
  <tr><td>Bird watching expert #1</td><td>0.3% error</td></tr>
  <tr><td>Bird watching expert #2</td><td>0.5% error</td></tr>
  <tr><td>Normal person #1 (not a bird watching expert)</td><td>1.0% error</td></tr>
  <tr><td>Normal person #2 (not a bird watching expert)</td><td>1.2% error</td></tr>
</table><br>
</p>
<p align="justify">
If your goal is to have “human-level performance” be a proxy (or estimate) for Bayes error, how would you define “human-level performance”?<br>
A. 0.0% (because it is impossible to do better than this)<br>
B. 0.3% (accuracy of expert #1)<br>
C. 0.4% (average of 0.3 and 0.5)<br>
D. 0.75% (average of all four numbers above)<br>
<b>Answer</b>: B.<br><br>

<b>9.</b><br>
Which of the following statements do you agree with?<br>
A. A learning algorithm’s performance can be better than human-level performance but it can never be better than Bayes error.<br>
B. A learning algorithm’s performance can never be better than human-level performance but it can be better than Bayes error.<br>
C. A learning algorithm’s performance can never be better than human-level performance nor better than Bayes error.<br>
D. A learning algorithm’s performance can be better than human-level performance and better than Bayes error.<br>
<b>Answer</b>: A.<br><br>

<b>10.</b><br>
You find that a team of ornithologists debating and discussing an image gets an even better 0.1% performance, so you define that as “human-level performance.” After working further on your algorithm, you end up with the following:<br>
<table class="c">
  <tr><td>Human-level performance</td><td>0.1%</td></tr>
  <tr><td>Training set error</td><td>2.0%</td></tr>
  <tr><td>Dev set error</td><td>2.1%</td></tr>
</table><br>
</p>
<p align="justify">
Based on the evidence you have, which two of the following four options seem the most promising to try? (Check two options.)<br>
A. Try decreasing regularization.<br>
B. Train a bigger model to try to do better on the training set.<br>
C. Try increasing regularization.<br>
D. Get a bigger training set to reduce variance.<br>
<b>Answer</b>: A, B.<br><br>

<b>11.</b><br>
You also evaluate your model on the test set, and find the following:<br>
<table class="c">
  <tr><td>Human-level performance</td><td>0.1%</td></tr>
  <tr><td>Training set error</td><td>2.0%</td></tr>
  <tr><td>Dev set error</td><td>2.1%</td></tr>
  <tr><td>Test set error</td><td>7.0%</td></tr>
</table><br>
</p>
<p align="justify">
What does this mean? (Check the two best options.)<br>
A. You should get a bigger test set.<br>
B. You have underfit to the dev set.<br>
C. You have overfit to the dev set.<br>
D. You should try to get a bigger dev set.<br>
<b>Answer</b>: C, D.<br><br>

<b>12.</b><br>
After working on this project for a year, you finally achieve:<br>
<table class="c">
  <tr><td>Human-level performance</td><td>0.1%</td></tr>
  <tr><td>Training set error</td><td>0.05%</td></tr>
  <tr><td>Dev set error</td><td>0.05%</td></tr>
</table><br>
</p>
<p align="justify">
What can you conclude? (Check all that apply.)<br>
A. With only 0.09% further progress to make, you should quickly be able to close the remaining gap to 0%<br>
B. This is a statistical anomaly (or must be the result of statistical noise) since it should not be possible to surpass human-level performance.<br>
C. If the test set is big enough for the 0.05% error estimate to be accurate, this implies Bayes error is ≤ 0.05<br>
D. It is now harder to measure avoidable bias, thus progress will be slower going forward.<br>
<b>Answer</b>: C, D.<br><br>

<b>13.</b><br>
It turns out Peacetopia has hired one of your competitors to build a system as well. Your system and your competitor both deliver systems with about the same running time and memory size. However, your system has higher accuracy! However, when Peacetopia tries out your and your competitor’s systems, they conclude they actually like your competitor’s system better, because even though you have higher overall accuracy, you have more false negatives (failing to raise an alarm when a bird is in the air). What should you do?<br>
A. Look at all the models you’ve developed during the development process and find the one with the lowest false negative error rate.<br>
B. Ask your team to take into account both accuracy and false negative rate during development.<br>
C. Rethink the appropriate metric for this task, and ask your team to tune to the new metric.<br>
D. Pick false negative rate as the new metric, and use this new metric to drive all further development.<br>
<b>Answer</b>: C.<br><br>

<b>14.</b><br>
You’ve handily beaten your competitor, and your system is now deployed in Peacetopia and is protecting the citizens from birds! But over the last few months, a new species of bird has been slowly migrating into the area, so the performance of your system slowly degrades because your data is being tested on a new type of data.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/DLB/3_2_1_2.png"/></center>
</p>
<p align="justify">
You have only 1,000 images of the new species of bird. The city expects a better system from you within the next 3 months. Which of these should you do first?<br>
A. Use the data you have to define a new evaluation metric (using a new dev/test set) taking into account the new species, and use that to drive further progress for your team.<br>
B. Put the 1,000 images into the training set so as to try to do better on these birds.<br>
C. Try data augmentation/data synthesis to get more images of the new type of bird.<br>
D. Add the 1,000 images into your dataset and reshuffle into a new train/dev/test split.<br>
<b>Answer</b>: A.<br><br>

<b>15.</b><br>
The City Council thinks that having more Cats in the city would help scare off birds. They are so happy with your work on the Bird detector that they also hire you to build a Cat detector. (Wow Cat detectors are just incredibly useful aren’t they.) Because of years of working on Cat detectors, you have such a huge dataset of 100,000,000 cat images that training on this data takes about two weeks. Which of the statements do you agree with? (Check all that agree.)<br>
A. Needing two weeks to train will limit the speed at which you can iterate.<br>
B. If 100,000,000 examples is enough to build a good enough Cat detector, you might be better of training with just 10,000,000 examples to gain a $approx$ 10x improvement in how quickly you can run experiments, even if each model performs a bit worse because it’s trained on less data.<br>
C. Buying faster computers could speed up your teams’ iteration speed and thus your team’s productivity.<br>
D. Having built a good Bird detector, you should be able to take the same model and hyperparameters and just apply it to the Cat dataset, so there is no need to iterate.<br>
<b>Answer</b>: A, B, C.<br><br>
</p>

#### 3.2.2 Autonomous driving
<p align="justify">
<b>1.</b><br>
To help you practice strategies for machine learning, in this week we’ll present another scenario and ask how you would act. We think this “simulator” of working in a machine learning project will give a task of what leading a machine learning project could be like!<br><br>

You are employed by a startup building self-driving cars. You are in charge of detecting road signs (stop sign, pedestrian crossing sign, construction ahead sign) and traffic signals (red and green lights) in images. The goal is to recognize which of these objects appear in each image. As an example, the above image contains a pedestrian crossing sign and red traffic lights<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/DLB/3_2_2_1.png"/></center>
</p>
<p align="justify">
Your 100,000 labeled images are taken using the front-facing camera of your car. This is also the distribution of data you care most about doing well on. You think you might be able to get a much larger dataset off the internet, that could be helpful for training even if the distribution of internet data is not the same.<br><br>

You are just getting started on this project. What is the first thing you do? Assume each of the steps below would take about an equal amount of time (a few days).<br>
A. Spend a few days getting the internet data, so that you understand better what data is available.<br>
B. Spend a few days training a basic model and see what mistakes it makes.<br>
C. Spend a few days collecting more data using the front-facing camera of your car, to better understand how much data per unit time you can collect.<br>
D. Spend a few days checking what is human-level performance for these tasks so that you can get an accurate estimate of Bayes error.<br>
<b>Answer</b>: B.<br><br>

<b>2.</b><br>
Your goal is to detect road signs (stop sign, pedestrian crossing sign, construction ahead sign) and traffic signals (red and green lights) in images. The goal is to recognize which of these objects appear in each image. You plan to use a deep neural network with ReLU units in the hidden layers.<br>
For the output layer, a softmax activation would be a good choice for the output layer because this is a multi-task learning problem. True/False?<br>
A. True<br>
B. False<br>
<b>Answer</b>: B.<br><br>

<b>3.</b><br>
You are carrying out error analysis and counting up what errors the algorithm makes. Which of these datasets do you think you should manually go through and carefully examine, one image at a time?<br>
A. 10,000 images on which the algorithm made a mistake<br>
B. 500 randomly chosen images<br>
C. 10,000 randomly chosen images<br>
D. 500 images on which the algorithm made a mistake<br>
<b>Answer</b>: D.<br><br>

<b>4.</b><br>
After working on the data for several weeks, your team ends up with the following data:<br>
100,000 labeled images taken using the front-facing camera of your car.<br>
900,000 labeled images of roads downloaded from the internet.<br>
Each image’s labels precisely indicate the presence of any specific road signs and traffic signals or combinations of them<br>
For example,<br>
$$
y^{(i)} =
\begin{bmatrix}
1\\
0\\
0\\
1\\
0
\end{bmatrix}
$$

means the image contains a stop sign and a red traffic light. Because this is a multi-task learning problem, you need to have all your $y^{(i)}$ vectors fully labeled. If one example is equal to<br>
$$
\begin{bmatrix}
0\\
?\\
1\\
1\\
?
\end{bmatrix}
$$
then the learning algorithm will not be able to use that example. True/False?<br>
A. True<br>
B. False<br>
<b>Answer</b>: B.<br><br>

<b>5.</b><br>
The distribution of data you care about contains images from your car’s front-facing camera; which comes from a different distribution than the images you were able to find and download off the internet. How should you split the dataset into train/dev/test sets?<br>
A. Choose the training set to be the 900,000 images from the internet along with 20,000 images from your car’s front-facing camera. The 80,000 remaining images will be split equally in dev and test sets.<br>
B. Choose the training set to be the 900,000 images from the internet along with 80,000 images from your car’s front-facing camera. The 20,000 remaining images will be split equally in dev and test sets.<br>
C. Mix all the 100,000 images with the 900,000 images you found online. Shuffle everything. Split the 1,000,000 images dataset into 980,000 for the training set, 10,000 for the dev set and 10,000 for the test set.<br>
D. Mix all the 100,000 images with the 900,000 images you found online. Shuffle everything. Split the 1,000,000 images dataset into 600,000 for the training set, 200,000 for the dev set and 200,000 for the test set.<br>
<b>Answer</b>: B.<br><br>

<b>6.</b><br>
Assume you’ve finally chosen the following split between of the data:<br>
<table class="c">
  <tr><th>Dataset:</th><th>Contains:</th><th>Error of the algorithm:</th></tr>
  <tr><td>Training</td><td>940,000 images randomly picked from (900,000 internet images + 60,000 car’s front-facing camera images)</td><td>8.8%</td></tr>
  <tr><td>Training-Dev</td><td>20,000 images randomly picked from (900,000 internet images + 60,000 car’s front-facing camera images)</td><td>9.1%</td></tr>
  <tr><td>Dev</td><td>20,000 images from your car’s front-facing camera</td><td>14.3%</td></tr>
  <tr><td>Test</td><td>20,000 images from the car’s front-facing camera</td><td>14.8%</td></tr>
</table><br>
</p>
<p align="justify">
You also know that human-level error on the road sign and traffic signals classification task is around 0.5%. Which of the following are True? (Check all that apply).<br>
A. You have a large avoidable-bias problem because your training error is quite a bit higher than the human-level error.<br>
B. Your algorithm overfits the dev set because the error of the dev and test sets are very close.<br>
C. You have a large variance problem because your model is not generalizing well to data from the same training distribution but that it has never seen before.<br>
D. You have a large variance problem because your training error is quite higher than the human-level error.<br>
E. You have a large data-mismatch problem because your model does a lot better on the training-dev set than on the dev set<br>
<b>Answer</b>: A, E.<br><br>

<b>7.</b><br>
Based on table from the previous question, a friend thinks that the training data distribution is much easier than the dev/test distribution. What do you think?<br>
A. Your friend is right. (I.e., Bayes error for the training data distribution is probably lower than for the dev/test distribution.)<br>
B. Your friend is wrong. (I.e., Bayes error for the training data distribution is probably higher than for the dev/test distribution.)<br>
C. There’s insufficient information to tell if your friend is right or wrong.<br>
<b>Answer</b>: C.<br><br>

<b>8.</b><br>
You decide to focus on the dev set and check by hand what are the errors due to. Here is a table summarizing your discoveries:<br>
<table class="c">
  <tr><td>Overall dev set error</td><td>15.3%</td></tr>
  <tr><td>Errors due to incorrectly labeled data</td><td>4.1%</td></tr>
  <tr><td>Errors due to foggy pictures</td><td>8.0%</td></tr>
  <tr><td>Errors due to rain drops stuck on your car’s front-facing camera</td><td>2.2%</td></tr>
  <tr><td>Errors due to other causes</td><td>1.0%</td></tr>
</table><br>
</p>
<p align="justify">
In this table, 4.1%, 8.0%, etc. are a fraction of the total dev set (not just examples your algorithm mislabeled). For example, about 8.0/15.3 = 52% of your errors are due to foggy pictures.<br><br>

The results from this analysis implies that the team’s highest priority should be to bring more foggy pictures into the training set so as to address the 8.0% of errors in that category. True/False?<br><br>

Additional Note: there are subtle concepts to consider with this question, and you may find arguments for why some answers are also correct or incorrect. We recommend that you spend time reading the feedback for this quiz, to understand what issues that you will want to consider when you are building your own machine learning project.<br>
A. True because it is the largest category of errors. We should always prioritize the largest category of error as this will make the best use of the team's time.<br>
B. True because it is greater than the other error categories added together (8.0 > 4.1+2.2+1.0).<br>
C. False because it depends on how easy it is to add foggy data. If foggy data is very hard and costly to collect, it might not be worth the team’s effort.<br>
D. First start with the sources of error that are least costly to fix.<br>
<b>Answer</b>: C.<br><br>

<b>9.</b><br>
You can buy a specially designed windshield wiper that help wipe off some of the raindrops on the front-facing camera. Based on the table from the previous question, which of the following statements do you agree with?<br>
A. 2.2% would be a reasonable estimate of the maximum amount this windshield wiper could improve performance.<br>
B. 2.2% would be a reasonable estimate of the minimum amount this windshield wiper could improve performance.<br>
C. 2.2% would be a reasonable estimate of how much this windshield wiper will improve performance.<br>
D. 2.2% would be a reasonable estimate of how much this windshield wiper could worsen performance in the worst case.<br>
<b>Answer</b>: A.<br><br>

<b>10.</b><br>
You decide to use data augmentation to address foggy images. You find 1,000 pictures of fog off the internet, and “add” them to clean images to synthesize foggy days, like this:<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/DLB/3_2_2_1.png"/></center>
</p>
<p align="justify">
Which of the following statements do you agree with?<br>
A. There is little risk of overfitting to the 1,000 pictures of fog so long as you are combing it with a much larger (>>1,000) of clean/non-foggy images.<br>
B. So long as the synthesized fog looks realistic to the human eye, you can be confident that the synthesized data is accurately capturing the distribution of real foggy images (or a subset of it), since human vision is very accurate for the problem you’re solving.<br>
C. Adding synthesized images that look like real foggy pictures taken from the front-facing camera of your car to training dataset won’t help the model improve because it will introduce avoidable-bias.<br>
<b>Answer</b>: B.<br><br>

<b>11.</b><br>
After working further on the problem, you’ve decided to correct the incorrectly labeled data on the dev set. Which of these statements do you agree with? (Check all that apply).<br>
A. You should also correct the incorrectly labeled data in the test set, so that the dev and test sets continue to come from the same distribution<br>
B. You should correct incorrectly labeled data in the training set as well so as to avoid your training set now being even more different from your dev set.<br>
C. You should not correct the incorrectly labeled data in the test set, so that the dev and test sets continue to come from the same distribution<br>
D. You do not necessarily need to fix the incorrectly labeled data in the training set, because it's okay for the training set distribution to differ from the dev and test sets. Note that it is important that the dev set and test set have the same distribution.<br>
<b>Answer</b>: A, D.<br><br>

<b>12.</b><br>
So far your algorithm only recognizes red and green traffic lights. One of your colleagues in the startup is starting to work on recognizing a yellow traffic light. (Some countries call it an orange light rather than a yellow light; we’ll use the US convention of calling it yellow.) Images containing yellow lights are quite rare, and she doesn’t have enough data to build a good model. She hopes you can help her out using transfer learning.<br>
What do you tell your colleague?<br>
A. She should try using weights pre-trained on your dataset, and fine-tuning further with the yellow-light dataset.<br>
B. If she has (say) 10,000 images of yellow lights, randomly sample 10,000 images from your dataset and put your and her data together. This prevents your dataset from “swamping” the yellow lights dataset.<br>
C. You cannot help her because the distribution of data you have is different from hers, and is also lacking the yellow label.<br>
D. Recommend that she try multi-task learning instead of transfer learning using all the data.<br>
<b>Answer</b>: A.<br><br>

<b>13.</b><br>
Another colleague wants to use microphones placed outside the car to better hear if there’re other vehicles around you. For example, if there is a police vehicle behind you, you would be able to hear their siren. However, they don’t have much to train this audio system. How can you help?<br>
A. Transfer learning from your vision dataset could help your colleague get going faster. Multi-task learning seems significantly less promising.<br>
B. Multi-task learning from your vision dataset could help your colleague get going faster. Transfer learning seems significantly less promising.<br>
C. Either transfer learning or multi-task learning could help our colleague get going faster.<br>
D. Neither transfer learning nor multi-task learning seems promising.<br>
<b>Answer</b>: D.<br><br>

<b>14.</b><br>
To recognize red and green lights, you have been using this approach:<br>
(A) Input an image (x) to a neural network and have it directly learn a mapping to make a prediction as to whether there’s a red light and/or green light (y).
A teammate proposes a different, two-step approach:<br>
(B) In this two-step approach, you would first (i) detect the traffic light in the image (if any), then (ii) determine the color of the illuminated lamp in the traffic light.<br>
Between these two, Approach B is more of an end-to-end approach because it has distinct steps for the input end and the output end. True/False?<br>
A. True<br>
B. False<br>
<b>Answer</b>: B.<br><br>

<b>15.</b><br>
Approach A (in the question above) tends to be more promising than approach B if you have a ________ (fill in the blank).<br>
A. Large training set<br>
B. Multi-task learning problem.<br>
C. Large bias problem.<br>
D. Problem with a high Bayes error.<br>
<b>Answer</b>: A.<br><br>
</p>