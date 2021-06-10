---
layout: post
title:  "Why n-1 in sample variance"
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
<a href="https://blog.csdn.net/hearthougan/article/details/77859173"> Thanks to this blog</a><br><br>
</p>


## 1. Sample Variance
<p align="justify">
Suppose sample mean $\bar{X}$, sample variance $S^{2}$, ensemble average $\mu$, ensemble variance $\sigma^{2}$
$$S^{2} = \frac{1}{n-1} \sum_{i=1}^{n}(x_{i} - \bar{X})^{2}$$
</p>


## 2. Unbiased Estimate
<p align="justify">
<b>Under many repeat trials, our expection is close to its real value</b><br><br>

For example, if we want to know an average students' heght at a university. It's hard to investigate all of them (even though this is accurate), so we pick un 100 students randomly and get an average heght $\bar{X}_{1}$, which is not accurate if we take it to estimate ensemble average. In order to acquire a more accurate data, we can repeat our trial and calculate each sample mean $\bar{X}_{2}$, $\bar{X}_{3}$, $\cdots$, $\bar{X}_{k}$. Then we get an average among these sample mean $E(\bar{X})$. Here $\bar{X}$ is a random variable and $\bar{X}_{i}$ is one possible value.<br><br>

Unbiased estimate hopes our sample variance is an unbiased. Suppose our sample variance is like
$$S^{2} = \frac{1}{n} \sum_{i=1}^{n}(x_{i} - \bar{X})^{2}$$

Then
$$
\begin{aligned}
E(S^{2}) & = E(\frac{1}{n} \sum_{i=1}^{n}(x_{i} - \bar{X})^{2}) \\
&  = E(\frac{1}{n} \sum_{i=1}^{n}((x_{i} - \mu) - (\bar{X} - \mu))^{2}) \\
& = E(\frac{1}{n} \sum_{i=1}^{n}((x_{i} - \mu)^{2} - 2(x_{i} - \mu)(\bar{X} - \mu) + (\bar{X} - \mu)^{2})) \\
& = E(\frac{1}{n}\sum_{i=1}^{n}(x_{i} - \mu)^{2} - \frac{1}{n}\sum_{i=1}^{n}2(x_{i} - \mu)(\bar{X} - \mu) + \frac{1}{n}\sum_{i=1}^{n}(\bar{X} - \mu)^{2}) \\
& = E(\frac{1}{n}\sum_{i=1}^{n}(x_{i} - \mu)^{2} - 2(\bar{X} - \mu)(\bar{X}- \mu) + \frac{1}{n}\sum_{i=1}^{n}(\bar{X} - \mu)^{2}) \\
& = E(\frac{1}{n}\sum_{i=1}^{n}(x_{i} - \mu)^{2} - 2(\bar{X} - \mu)(\bar{X}- \mu) + (\bar{X} - \mu)^{2}) \\
& = E(\frac{1}{n}\sum_{i=1}^{n}(x_{i} - \mu)^{2} - (\bar{X} - \mu)^{2}) \\
& = E(\frac{1}{n}\sum_{i=1}^{n}(x_{i} - \mu)^{2}) - E((\bar{X} - \mu)^{2}) \\
& = Var(X) - Var(\bar{X}) \\
& = \sigma^{2} - \frac{1}{n}\sigma^{2} \\
& = \frac{n-1}{n}\sigma^{2} \leq \sigma^{2}
\end{aligned}
$$
so,
$$E(S^{2}) = \frac{n-1}{n}\sigma^{2}$$

We can observe if we divide by n, sample variance is smaller than ensemble variance. If we mutiply $S^{2}$ by $\frac{n}{n-1}$, we can get an unbiased sample variance
$$S^{2} = \frac{n}{n-1}(\frac{1}{n}\sum_{i=1}^{n}(x_{i} - \bar{X})^{2}) = \frac{1}{n-1}\sum_{i=1}^{n}(x_{i} - \bar{X})^{2}$$
We confirm it
$$
\begin{aligned}
E(S^{2}) & = E(\frac{1}{n-1} \sum_{i=1}^{n}(x_{i} - \bar{X})^{2}) = E(\frac{1}{n-1} \sum_{i=1}^{n}((x_{i} - \mu) - (\bar{X} - \mu))^{2}) \\
& = E(\frac{1}{n-1} \sum_{i=1}^{n}((x_{i} - \mu)^{2} - 2(x_{i} - \mu)(\bar{X} - \mu) + (\bar{X} - \mu)^{2})) \\
& = E(\frac{1}{n-1}\sum_{i=1}^{n}(x_{i} - \mu)^{2} - \frac{1}{n-1}\sum_{i=1}^{n}2(x_{i} - \mu)(\bar{X} - \mu) + \frac{1}{n-1}\sum_{i=1}^{n}(\bar{X} - \mu)^{2}) \\
& = E(\frac{1}{n-1}\sum_{i=1}^{n}(x_{i} - \mu)^{2} - \frac{2n}{n-1}(\bar{X} - \mu)(\bar{X}- \mu) + \frac{1}{n-1}\sum_{i=1}^{n}(\bar{X} - \mu)^{2}) \\
& = E(\frac{1}{n-1}\sum_{i=1}^{n}(x_{i} - \mu)^{2} - \frac{2n}{n-1}(\bar{X} - \mu)(\bar{X}- \mu) + \frac{n}{n-1}(\bar{X} - \mu)^{2}) \\
& = E(\frac{1}{n-1}\sum_{i=1}^{n}(x_{i} - \mu)^{2} - \frac{n}{n-1}(\bar{X} - \mu)^{2}) \\
& = E(\frac{1}{n-1}\sum_{i=1}^{n}(x_{i} - \mu)^{2}) - E(\frac{n}{n-1}(\bar{X} - \mu)^{2}) \\
& = \frac{n}{n-1}E(\frac{1}{n}\sum_{i=1}^{n}(x_{i} - \mu)^{2}) - \frac{n}{n-1}E((\bar{X} - \mu)^{2}) \\
& = \frac{n}{n-1}\sigma^{2} - \frac{n}{n-1}\frac{\sigma^{2}}{n} \\
& = \sigma^{2}
\end{aligned}
$$
<br>
</p>