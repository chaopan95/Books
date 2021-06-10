---
layout: page
title:  "How many people with the same birthday?"
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
Consider a group of k people. Assume that each person's birthday is drawn uniformly at random from the 365 possibilities. (And ignore leap years.) What is the smallest value of k such that the expected number of pairs of distinct people with the same birthday is at least one?<br>
[Hint: define an indicator random variable for each ordered pair of people. Use linearity of expectation.]<br><br>

For each pair of k people in a group, we set an indicator $X_{ij}$, which is equal to 1 when i and j have a same birthday.
$$
X_{ij} =
\begin{cases}
	1, \quad B_{i}=B_{j} \\
	0, \quad \text{otherwise}
\end{cases}
$$

The probability for pair (i, j) having a same birthday
$$Pr[ B_{i}=B_{j} ]= \frac{1}{365}*\frac{1}{365}*365 = \frac{1}{365}$$

The expectation for $X_{ij}$
$$
\begin{aligned}
E[ X_{ij} ] & = \sum_{i=1}^{k-1}\sum_{j=i+1}^{k}X_{ij}Pr[ B_{i}=B_{j} ] \\
& = \frac{1}{365}{}\sum_{i=1}^{k}\sum_{j=i+1}^{k}X_{ij} \\
& = \frac{1}{365}*\sum_{i=1}^{k-1}[ k-(i+1)+1 ] \\
& = \frac{1}{365}*(\sum_{i=1}^{k-1}k-\sum_{i=1}^{k-1}i) \\
& = \frac{1}{365}*[ k(k-1)-\frac{k(k-1)}{2} ] \\
& = \frac{1}{365}*\frac{k(k-1)}{2}=1
\end{aligned}
$$
$$
\begin{aligned}
& k=1 \pm 27.02 \\
& k = 28
\end{aligned}
$$
</p>
