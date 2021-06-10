---
layout: post
title:  "SVM"
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
SVM: support vector machine<br>
Consider a training set
$$T = \{ (x_{1}, y_{1}), (x_{2}, y_{2}), \cdots, (x_{N}, y_{N}) \}, \quad \text{where } x_{i} \in R^{n}, y_{i} \in \{ +1, -1 \}, i = 1, 2, ..., N$$<br>
</p>

## 1. Linear SVM
<p align="justify">
If we can find a hyperplane to distinguish all positive instances and negative instances
$$w^{*}x + b^{*} = 0$$
This is called linear SVM with Decision function
$$f(x) = \text{sign}(w^{*}x + b^{*})$$
<b>Functional margin</b>: the minimum distance between all sample points and the hyperplane
$$
\begin{aligned}
& \hat{\gamma_{}} = \min_{i=1, 2, \cdots, N} \hat{\gamma_{i}} \\
& \hat{\gamma_{i}} = y_{i}(wx_{i}+b)
\end{aligned}
$$
<b>Geometric margin</b>: the functional margin when $\left \| w \right \| = 1$. Geometric margin is a signed distance from training data to the hyperplane.
$$
\begin{aligned}
& \gamma = \min_{i=1, 2, \cdots, N} \gamma_{i} \\
& \gamma_{i} = y_{i}(\frac{w}{\left \| w \right \|}x_{i} + \frac{b}{\left \| w \right \|})
\end{aligned}
$$
A relationship between functional margin and geometric margin
$$
\begin{aligned}
& \gamma_{i} = \frac{\hat{\gamma_{i}}}{\left \| w \right \|} \\
& \gamma = \frac{\hat{\gamma_{}}}{\left \| w \right \|}
\end{aligned}
$$
For example, in a 2-D plane, a hyperplane $x_{1} + x_{2} = 0$ is actually a line. Suppose a negative instance (2, 1). Then, the margin between this hyperplane and this instance is
$$
\begin{aligned}
& w =
\begin{bmatrix}
1 \\
1
\end{bmatrix} \\
& \frac{w}{\left \| w \right \|} =
\begin{bmatrix}
\frac{\sqrt{2}}{2} \\
\frac{\sqrt{2}}{2}
\end{bmatrix}, \quad
b = 0 \\
& x =
\begin{bmatrix}
2 \\
1
\end{bmatrix} \\
& \text{Functional margin is } y(w^{T}x + b) = -3 \\
& \text{Geometric margin is } y(\frac{w^{T} x + b}{\left \| w \right \|}) = -\frac{3\sqrt{2}}{2}
\end{aligned}
$$
<b>Maximize geometric margin</b>
$$
\begin{cases}
\max_{w, b} \gamma \\
\text{s.t.} \quad y_{i}(\frac{w}{\left \| w \right \|} \cdot x_{i} + \frac{b}{\left \| w \right \|}) \geq \gamma, \quad i = 1, 2, \cdots, N
\end{cases}
$$
Introduce the relationship between geometric margin and functional margin
$$
\begin{cases}
\max_{w, b}\frac{\hat{\gamma_{}}}{\left \| w \right \|} \\
\text{s.t.} \quad y_{i}(w \cdot x_{i}+b) \geq \hat{\gamma_{}}, \quad i = 1, 2, \cdots, N 
\end{cases}
$$
We can set $\hat{\gamma_{}} = 1$ for sake of simplicity because it doesn't affect our final solution w, b
$$y_{i}(w \cdot x_{i}+b) \geq \hat{\gamma_{}} \Leftrightarrow y_{i}(\lambda w \cdot x_{i} + \lambda b) \geq \lambda \hat{\gamma_{}}$$
Besides, $\max_{w, b}\frac{1}{\left \| w \right \|}$ is equivalent to $\min_{w, b}\frac{1}{2}\left \| w \right \|^{2}$
$$
\begin{cases}
\min_{w, b} \frac{1}{2} \left \| w \right \|^{2} & \quad \text{(1)}\\
\text{s.t.} \quad y_{i} (w \cdot x_{i} + b) -1 \geq 0, \quad i = 1, 2, \cdots, N & \quad \text{(2)}
\end{cases}
$$
This is a <b>convex quadratic programming</b>. If we olve it, we can get the optimal solution $w^{*}$, $b^{*}$, which represente a hyperplane. Besides, <b>if our data is linearly seperable, this hyperplane is unique</b>. Prove it<br>
<b>Existence</b>: we want to get a minimum of $\left \| w \right \|^{2}$ which has a low bound, so there must be a feasible solution for it.<br>
<b>Uniquness</b>: suppose we have two optimal solution ($w_{1}^{*}$, $b_{1}^{*}$) and ($w_{2}^{*}$, $b_{2}^{*}$)
$$
\begin{aligned}
& \frac{1}{2} \left \| w_{1}^{*} \right \|^{2} = \frac{1}{2} \left \| w_{2}^{*} \right \|^{2} \rightarrow \left \| w_{1} \right \| = \left \| w_{2} \right \| = c, \quad c \text{ is a some scalar value} \\
\because \quad &  (w = \frac{w_{1}^{*} + w_{2}^{*}}{2}, \quad b = \frac{b_{1}^{*} + b_{2}^{*}}{2}) \text{ is also a feasible solution} \\
\therefore \quad & c \leq \left \| w \right \| = \left \| \frac{w_{1}^{*} + w_{2}^{*}}{2} \right \| \leq \frac{1}{2} \left \| w_{1}^{*} \right \| + \frac{1}{2} \left \| w_{2}^{*} \right \| = c \\
\therefore \quad & \left \| \frac{w_{1}^{*} + w_{2}^{*}}{2} \right \| \leq \frac{1}{2} \left \| w_{1}^{*} \right \| + \frac{1}{2} \left \| w_{2}^{*} \right \| \\
\therefore \quad & w_{1}^{*} = \lambda w_{2}^{*} \text{ (parallel)} \\
\because \quad & \left \| w_{1}^{*} \right \| = \left \| w_{2}^{*} \right \| \\
\therefore \quad & \left \| \lambda \right \| = 1
\end{aligned}
$$
If $\lambda$ = -1, $w = \overrightarrow{\mathbf{0}}$, this is impossible, becaise $\left \| w \right \| = 0$ is not a feasible solution to $(1) \sim (2)$. Therefore $\lambda$ = 1, then $w_{1}^{*} = w_{2}^{*}$.<br>
To prove $b_{1}^{*} = b_{2}^{*}$:
$$
\begin{aligned}
& \quad \text{consider a positve and a negative instance } x_{1}^{+}, x_{1}^{-} \in w^{*} + b_{1}^{*} = \overrightarrow{\mathbf{0}} \\
& \quad \text{another positve negative instance } x_{2}^{+}, x_{2}^{-} \in w^{*} + b_{2}^{*} = \overrightarrow{\mathbf{0}} \\
\therefore & \quad
\begin{cases}
w^{*} \cdot x_{1}^{+} + b_{1}^{*} = 1 \\
w^{*} \cdot x_{1}^{-} + b_{1}^{*}  = -1 \\
\end{cases} \rightarrow
b_{1}^{*} = -\frac{1}{2} (w^{*} \cdot x_{1}^{+} + w^{*} \cdot x_{1}^{-}) \\
& \quad
\begin{cases}
w^{*} \cdot x_{2}^{+} + b_{2}^{*} = 1 \\
w^{*} \cdot x_{2}^{-} + b_{2}^{*}  = -1 \\
\end{cases} \rightarrow
b_{2}^{*} = -\frac{1}{2} (w^{*} \cdot x_{2}^{+} + w^{*} \cdot x_{2}^{-}) \\
\therefore & \quad b_{1}^{*} - b_{2}^{*} = -\frac{1}{2} [w^{*} (x_{1}^{+} - x_{2}^{+}) + w^{*} (x_{1}^{-} - x_{2}^{-})] \\
\because & \quad \text{instances } x_{1}^{+}, x_{1}^{-} \text{ also satisfies hyperplane } w^{*} + b_{2}^{*} = \overrightarrow{\mathbf{0}} \text{ and} \\
& \quad \text{instance } x_{2}^{+}, x_{2}^{-} \text{ also satisfies hyperplane } w^{*} + b_{1}^{*} = \overrightarrow{\mathbf{0}} \\
\therefore & \quad
\begin{cases}
(+1) (w^{*} \cdot x_{2}^{+} + b_{1}^{*}) \geq 1 = w^{*} \cdot x_{1}^{+} + b_{1}^{*} \\
(+1) (w^{*} \cdot x_{1}^{+} + b_{2}^{*}) \geq 1 = w^{*} \cdot x_{2}^{+} + b_{2}^{*}
\end{cases} \\
& \quad \rightarrow w^{*} \cdot x_{2}^{+} + b_{1}^{*} \geq w^{*} \cdot x_{1}^{+} + b_{1}^{*} \geq w^{*} \cdot x_{2}^{+} + b_{2}^{*} \\
& \quad
\begin{cases}
(-1) (w^{*} \cdot x_{2}^{-} + b_{1}^{*}) \geq 1 = (-1) (w^{*} \cdot x_{1}^{-} + b_{1}^{*}) \\
(-1) (w^{*} \cdot x_{1}^{-} + b_{2}^{*}) \geq 1 = (-1) (w^{*} \cdot x_{2}^{-} + b_{2}^{*})
\end{cases} \\
& \quad \rightarrow w^{*} \cdot x_{2}^{-} + b_{1}^{*} \leq w^{*} \cdot x_{1}^{-} + b_{1}^{*} \leq w^{*} \cdot x_{2}^{-} + b_{2}^{*} \\
\therefore & \quad
\begin{cases}
x_{1}^{+} = x_{2}^{+} \\
x_{1}^{-} = x_{2}^{-}
\end{cases} \\
\therefore & \quad
b_{1}^{*} - b_{2}^{*} = -\frac{1}{2} [w^{*} (x_{1}^{+} - x_{2}^{+}) + w^{*} (x_{1}^{-} - x_{2}^{-})] = 0 \\
\end{aligned}
$$
Above all, there is a unique optimal solution to $(1) \sim (2)$.<br><br> 

<b>Support vector</b>: the instances (including positive and negative) in the hyperplane.
$$(y_{i}) (w^{*} \cdot x_{i} + b^{*}) = 1$$
So, we have 2 hyperplane $H^{+}$ and $H^{-}$, which are parallel.
$$
\begin{cases}
H^{+}: \quad w^{*} \cdot x^{+} + b^{*} = 1 \\
H^{-}: \quad w^{*} \cdot x^{-} + b^{*} = -1
\end{cases}
$$
<b>Margin</b>: the distance between $H^{+}$ and $H^{-}$.
$$d_{H^{+}, H^{-}} = \frac{\left | (b^{*} - 1) - (b^{*} + 1) \right |}{\left | w^{*} \right |} = \frac{2}{\left | w^{*} \right |}$$


</p>

## 2. Non-linear SVM
<p align="justify">

</p>

## 3. SMO
<p align="justify">

</p>

