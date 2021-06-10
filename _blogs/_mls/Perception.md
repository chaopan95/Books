## Definition
Perception is a classification model. Input space $X = \{x_{1}, x_{2}, ..., x_{n}\}$, output space $Y = \{+1 , -1\}$. Discriminant function
$$
f(x_{i}) = \text{sign}(w \cdot x_{i} + b), \quad \text{where } \text{sign}(x_{i}) =
\begin{cases}
1, \quad x_{i} \geq 0 \\
-1, \quad x_{i} < 0
\end{cases}
$$
$W \cdot X$ represents an inner product.

## Algorithm
Dataset T = $\{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{n}, y_{n})\}$

If one point ($x_{i}$, $y_{i}$) is misclassified
$$y_{i} \times f(x_{i}) = y_{i} \times \text{sign}(x_{i}) = -1$$

Cost funtion
$$L(w, b) = -\sum_{(x_{i}, y_{i}) \in M} y_{i}(w \cdot x_{i} + b), \quad \text{where } M \text{ is a set of misclassified points}$$

Gradient descent
$$
\begin{aligned}
& \frac{\partial L(w, b)}{\partial w} = -\sum_{(x_{i}, y_{i}) \in M} y_{i} x_{i} \\
& \frac{\partial L(w, b)}{\partial b} = -\sum_{(x_{i}, y_{i}) \in M} y_{i}
\end{aligned}
$$

Parameters update by picking up one misclassified point
$$
\begin{aligned}
& w := w - \eta \frac{\partial L(w, b)}{\partial w} \\
& b := b - \eta \frac{\partial L(w, b)}{\partial b}
\end{aligned} \Rightarrow
\begin{aligned}
& w := w - \eta (-y_{i} x_{i}) \\
& b := b - \eta (-y_{i})
\end{aligned} \Rightarrow
\begin{aligned}
& w := w + \eta y_{i} x_{i} \\
& b := b + \eta y_{i}
\end{aligned}, \quad \text{where } \eta \in (0, 1]
$$

**Algorithm**
01. Input: dataset T = $\{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{n}, y_{n})\}$
02. Output: $w^{*}$, $b^{*}$
03. Initialise w, b with 0
04. While True do:
05.   Calculate misclassified set M
06.   If $M = \varnothing$:
07.     break
08.   Pick up one point form M, update w, b
09. return w, b

```python
import numpy as np

def sign(x):
    res = x.copy()
    idx = x >= 0
    res[:] = -1
    res[idx] = 1
    return res

def Perception(T, eta=0.1):
    X = T[:, :-1]
    Y = T[:, -1]
    n, m = X.shape
    w = np.zeros(m)
    b = 0
    while True:
        y_hat = sign(np.dot(X, w) + b)
        M = np.nonzero(Y != y_hat)[0]
        if len(M) == 0: break
        pickedUp = np.random.choice(M, 1)
        w = w + eta * Y[pickedUp] * X[pickedUp, :][0]
        b = b + eta * Y[pickedUp]
    return w, b

T = [[3, 3, 1],
     [4, 3, 1],
     [1, 1, -1]]
T = np.array(T)
w, b = Perception(T)
```

**Novikoff**
If dataset T is linearly seperable

(i) there exists a hyperplan $w^{*} \cdot x + b^{*} = 0$, which is able to discriminate all positive instances and negative instances, i.e. for all instances $(x_{i}, y_{i}), i = 1, 2, ..., n$, there exists a $\gamma > 0$

$$y_{i} (w^{*} \cdot x_{i} + b^{*}) \geq \gamma$$

(ii) Moreover, set $R = \max_{i = 1, 2, ..., n} \left \| x_{i} \right \|$, the number of misclassification k is

$$k \leq (\frac{R}{\gamma})^{2}$$

**证明**
1) all instances $x_{i}$ can be classified correctly

$$y_{i} (w^{*} \cdot x + b^{*}) > 0$$

So, we can find a $\gamma$ to satisfy (i)

$$\gamma = \min_{i = 1, 2, ..., n} y_{i} (w^{*} \cdot x_{i} + b^{*})$$

2) suppose in $k^{\text{th}}$ iteration, we have parameters

$$
W_{k} =
\begin{bmatrix}
w_{k} \\
b_{k} 
\end{bmatrix}
$$
In last iteration, we update $w_{k-1}$ and $b_{k-1}$
$$
W_{k} =
\begin{bmatrix}
w_{k} \\
b_{k} 
\end{bmatrix} =
\begin{bmatrix}
w_{k-1} + \eta y_{i} x_{i} \\
b_{k-1} + \eta y_{i}
\end{bmatrix} =
\begin{bmatrix}
w_{k-1} \\
b_{k-1}
\end{bmatrix} + \eta y_{i}
\begin{bmatrix}
x_{i} \\
1
\end{bmatrix}
$$

Two inequalities
$$
\begin{aligned}
w_{k} \cdot w^{*} & = (w_{k} + \eta y_{i} x_{i}) \cdot w^{*} \\
& \geq w_{k-1} \cdot w^{*} + \eta \gamma \\
& \geq w_{k-2} \cdot w^{*} + 2 \eta \gamma \\
& \geq k \eta \gamma
\end{aligned}
$$
$$
\begin{aligned}
\left \| w_{k} \right \|^{2} & = (w_{k-1} + \eta y_{i} x_{i})^{2} \\
& = \left \| w_{k-1} \right \|^{2} + 2 w_{k-1} \eta y_{i} x_{i} + \eta^{2} \left \| x_{i} \right \|^{2} \\
& \leq \left \| w_{k-1} \right \|^{2} + \eta^{2} \left \| x_{i} \right \|^{2} \\
& \leq \left \| w_{k-1} \right \|^{2} + \eta^{2} R^{2} \\
& \leq k \eta^{2} \left \| x_{i} \right \|^{2} \\
& \leq k \eta^{2} R^{2}
\end{aligned}
$$
So,
$$
\begin{aligned}
& k \eta \gamma \leq w^{k} \cdot w^{k} \leq \left \| w_{k} \right \| \left \| w^{*} \right \| \leq \sqrt{k} \eta R \\
& k \leq (\frac{R}{\gamma})^{2}
\end{aligned}
$$


## 评价
* `优点`
  - 容易实现
* `缺点`
  - 不能处理非线性分裂问题，如异或XOR
