---
layout: post
title:  "Basis"
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


## 1. Probability
### 1.1 Joint probability
<p align="justify">
For 2 random variables
$$
\begin{aligned}
P(X = x \text{ and } Y = y) &= P(X = x \mid Y = y) P(y = y) \\
&= P(Y = y \mid X = x) P(X = x)
\end{aligned}
$$

It must satisfy
$$
\begin{aligned}
& \sum_{x} \sum_{y} P(X = x \text{ and } Y = y) = 1 \\
& \int_{x} \int_{y} f_{X, Y} (x, y) dy dx = 1
\end{aligned}
$$
</p>

### 1.2 Marginal Probability
<p align="justify">
$$
\begin{aligned}
& P(X = x) = \sum_{Y} P(X = a, Y = b) \\
& P(Y = b) = \sum_{X} P(X = a, Y = b)
\end{aligned}
$$
</p>

### 1.3 Conditional probability
<p align="justify">
$$
\begin{aligned}
& \sum_{x} P(X = a \mid Y = b) = 1 \\
& P(X = a \mid Y = b) = \frac{P(X = a, Y = b)}{P(Y = b)}
\end{aligned}
$$
</p>

### 1.4 Expect
<p align="justify">
For ramdom variable X
$$\text{E}(X, Y) = \sum_{i} x_{i} p(x_{i}), \quad i = 1, 2, \cdots, n$$

For random variable function f(X, Y)
$$\text{E}[f] = \sum_{i} f(x_{i}) p(x_{i}, y_{i}), \quad i = 1, 2, \cdots n$$
</p>

### 1.5 Variance
<p align="justify">
$$\text{Var}(f(x)) = \text{E}[(f(x) - \text{E}[f(x)])^{2}]$$
</p>

### 1.6 Covariance
<p align="justify">
$$\text{Cov}(f(x), g(y)) = \text{E}[(f(x) - \text{E}[f(x)])(g(y) - \text{E}[g(y)])]$$
</p>

## 2. Statistics
<p align="justify">

</p>