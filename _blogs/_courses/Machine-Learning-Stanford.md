---
layout: page
title:  "Machine Learning Stanford"
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


## 1. Week 1
### 1.1 Introduction
#### 1.1.1 Introduction
<p align="justify">
<b>Machine Learning</b><br>
-- grew out of work in AI<br>
New capacity for computers<br>
Examples<br>
-- Database mining: Web click data, medical records, biology<br>
-- Allplication can't program by hand: autonomous helicopter, NLP, CV<br>
-- Self-customizing programs: recommend system<br>
-- Understanding human learning<br><br>

<b>Machine Learning Definition</b><br>
$\bigstar$ Arthur Samuel (1959). Machine Learning: field of study that gives computers the ability to learn without being exolicitly programmed.<br>
$\bigstar$ Tom Mitchell (1998) Well-posed learning Problem: A computer program is said to <b>learn</b> from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.<br><br>

<b>Q</b>: Suppose your email program watches which emails you do or do not mark as spam, and based on that learns how to better filter spam. <b>What is the task T in this setting?</b><br>
A. Classify emails as spam or not spam.<br>
B. Watching you label emails as spam or not spam.<br>
C. The number (or fraction) of emails correctly classified as spam/not spam.<br>
D. None of the above, this is not a machine learning algorithm.<br>
<b>Answer:</b> A.<br><br>

$\bigstar$ Machine learning algorithms<br>
-- Supervised learnings<br>
-- Unsupervised learning<br>
$\bigstar$ Others: Reinforcement learning, recommender systems<br><br>

<b>Supervised Learning</b><br>
Supervised learning has labels, for example, Housing price prediction.<br>
$\bigstar$ Supervised Learning: "right answer" given.<br>
$\bigstar$ Regression: predict continuous valued output (e.g. price)<br>
$\bigstar$ Classification: discrete valued output (e.g. 0 or 1)<br><br>

<b>Q</b>: You’re running a company, and you want to develop learning algorithms to address each of two problems.<br>
Problem 1:You have a large inventory of identical items. You want to predict how many of these items will sell over the next 3 months.<br>
Problem 2: You’d like software to examine individual customer accounts, and for each account decide if it has been hacked/compromised.<br>
<b>Should you treat these as classification or as regression problems?</b><br>
A. Treat both as classification problems.<br>
B. Treat problem 1 as a classification problem, problem 2 as a regression problem.<br>
C. Treat problem 1 as a regression problem, problem 2 as a classification problem.<br>
D. Treat both as regression problems.<br>
<b>Answer:</b> C.<br><br>

<b>Unsupervised Learning</b><br>
Unsupervised learning has no labels.<br>
Some example<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/1_1_1_1.png"/></center>
</p>
<p align="justify">
<b>Cocktail party problem</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/1_1_1_2.png"/></center>
</p>
<p align="justify">
We want to seperate different audios with some unsupervised algorithms.<br>
SVD is a feasible choice<br>
[W, s, v] = svd((repmat(sum(x.*x, 1), size(x, 1), 1).* x) * x');<br><br>

<b>Q</b>: Of the following examples, <b>which would you address using an unsupervised learning algorithm?</b> (Check all that apply.)<br>
A. Given email labeled as spam/not spam, learn a spam filter.<br>
B. Given a set of news articles found on the web, group them into sets of articles about the same stories.<br>
C. Given a database of customer data, automatically discover market segments and group customers into different market segments.<br>
D. Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.<br>
<b>Answer:</b> B, C.<br><br>
</p>

#### 1.1.2 Quiz
<p align="justify">
<b>1.</b><br>
A computer program is said to learn from experience E with respect to some task T and some performance measure P if its performance on T, as measured by P, improves with experience E. Suppose we feed a learning algorithm a lot of historical weather data, and have it learn to predict weather. In this setting, <b>what is T?</b><br>
A. None of these.<br>
B. The weather prediction task.<br>
C. The probability of it correctly predicting a future date's weather.<br>
D. The process of the algorithm examining a large amount of historical weather data.<br>
<b>Answer:</b> B.<br><br>

<b>2.</b><br>
Suppose you are working on weather prediction, and use a learning algorithm to predict tomorrow's temperature (in degrees Centigrade/Fahrenheit). <b>Would you treat this as a classification or a regression problem?</b><br>
A. Regression<br>
B. Classification<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
Suppose you are working on stock market prediction, Typically tens of millions of shares of Microsoft stock are traded (i.e., bought/sold) each day. You would like to predict the number of Microsoft shares that will be traded tomorrow. <b>Would you treat this as a classification or a regression problem?</b><br>
A. Regression<br>
B. Classification<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
Some of the problems below are best addressed using a supervised learning algorithm, and the others with an unsupervised learning algorithm. <b>Which of the following would you apply supervised learning to?</b> (Select all that apply.) In each case, assume some appropriate dataset is available for your algorithm to learn from.<br>
A. Given a large dataset of medical records from patients suffering from heart disease, try to learn whether there might be different clusters of such patients for which we might tailor separate treatments.<br>
B. Examine a web page, and classify whether the content on the web page should be considered "child friendly" (e.g., non-pornographic, etc.) or "adult."<br>
C. In farming, given data on crop yields over the last 50 years, learn to predict next year's crop yields.<br>
D. Given data on how 1000 medical patients respond to an experimental drug (such as effectiveness of the treatment, side effects, etc.), discover whether there are different categories or "types" of patients in terms of how they respond to the drug, and if so what these categories are.<br>
<b>Answer:</b> B, C.<br><br>

<b>5.</b><br>
<b>Which of these is a reasonable definition of machine learning?</b><br>
A. Machine learning is the field of allowing robots to act intelligently.<br>
B. Machine learning is the science of programming computers.<br>
C. Machine learning learns from labeled data.<br>
D. Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.<br>
<b>Answer:</b> D.<br><br>
</p>

### 1.2 Linear Regression with One Variable
#### 1.2.1 Model and Cost Function
<p align="justify">
<b>Model Representation</b><br>
$\bigstar$ Notation:<br>
-- m = Numer of training examples<br>
-- x's = 'input' variable / features<br>
-- y's = 'output' variable / 'target' variable<br>
-- (x, y) is one training example<br>
-- $(x^{i}, y^{i})$ is $i^{th}$ training example<br>
For example, we have a dataset about house price<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/1_2_1_1.png"/></center>
</p>
<p align="justify">
$x^{1}$ = 2104, $y^{3}$ = 315<br>
Here is a model diagram
$$
\begin{matrix}
& & & & \text{Training Set} & & & & \\
\\
& & & & \Downarrow & & & & \\
\\
& & & & \text{Learning Algorithm} & & & & \\
\\
& & & & \Downarrow & & & & \\
\\
\text{Size of house} & & \Rightarrow & & h & & \Rightarrow & & \text{Estimated price} \\
\end{matrix}
$$

h is our hypothesis for this model. We can use a linear regression to represent h<br>
$$h_{\theta}(x) = \theta_{0} + \theta_{1}x$$
In this case, this is an univariate linear regression, in other word, only one variable in linear regression<br><br>

<b>Cost Function</b><br>
$\theta$'s are called parameters<br>
$\bigstar$ How to choose parameters?<br>
Idea is to choose $\theta_{0}$, $\theta_{1}$ so that $h_{\theta}(x)$ is close to y for our training examples (x, y)<br>
$$\theta_{0}, \theta_{1} = \arg\min_{\theta_{0}, \theta_{1}} \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})^{2}$$
Putting the 2 at the constant one half in front may just sound the math probably easier.<br>
So, we define our cost function<br>
$$J(\theta_{0}, \theta_{1}) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})^{2}$$
Our goal is to minimize the cost function by changing $\theta_{0}$, $\theta_{1}$.<br>
In a word, cost function describes a distance or difference between a true value or our predicted value based on our hypothesis.<br><br>
</p>

#### 1.2.2 Parameter Learning
<p align="justify">
<b>Gradient Descent</b><br>
$\bigstar$ Have some function $J(\theta_{0}, \theta_{1})$<br>
$\bigstar$ Want $\min_{\theta_{0}, \theta_{1}} J(\theta_{0}, \theta_{1})$<br>
$\bigstar$ Outline:<br>
-- Start with some $\theta_{0}$, $\theta_{1}$<br>
-- Keep changing $\theta_{0}$, $\theta_{1}$ to reduce $J(\theta_{0}, \theta_{1})$ until we end up at a minimum<br>
$\bigstar$ Gradient descent algorithm<br>
$$\theta_{j} := \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}} J(\theta_{0}, \theta_{1}), \quad j =0, 1$$
If $\alpha$ is too small, gradient descent can be slow. If $\alpha$ is too large, gradient descent can overshoot the minimum, so that it may fail to converge, or even diverge.<br>
$\theta$ is hard to get off a local minimum.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/1_2_2_1.png"/></center>
</p>
<p align="justify">
Gradient descent can converge to a local minimum, even with the learning rate $\alpha$ fixed. As we approach a local minimum, gradient descent will automatically take smaller steps. So, no need to descrease $\alpha$ over time.<br><br>

<b>Gradient Descent For Linear Regression</b><br>
We apply gradient descent to minimize the cost function of a linear regression.<br>
$$h_{\theta} = \theta_{0} + \theta_{1}x$$
$$J(\theta_{0}, \theta_{1}) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})^{2}$$
Gradient<br>
$$\frac{\partial}{\partial \theta_{0}}J(\theta_{0}, \theta_{1}) = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})$$
$$\frac{\partial}{\partial \theta_{1}}J(\theta_{0}, \theta_{1}) = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})x^{i}$$
Update $\theta_{j}$<br>
$$\theta_{0} := \theta_{0} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})$$
$$\theta_{1} := \theta_{1} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i}) \cdot x^{i}$$
$\bigstar$ Batch Gradient Descent<br>
Batch: Each step of gradient descent uses all the training examples.<br><br>

<b>Q</b>: <b>Which of the following are true statements?</b><br> Select all that apply.<br>
A. To make gradient descent converge, we must slowly decrease $\alpha$ over time.<br>
B. Gradient descent is guaranteed to find the global minimum for any function $J(\theta_{0}, \theta_{1})$<br>
C. Gradient descent can converge even if $\alpha$ is kept fixed. (But $\alpha$ cannot be too large, or else it may fail to converge.)<br>
D. For the specific choice of cost function $J(\theta_{0}, \theta_{1})$ used in linear regression, there are no local optima (other than the global optimum).<br>
<b>Answer:</b> C, D.<br><br>
</p>

#### 1.2.3 Quiz
<p align="justify">
<b>1.</b><br>
Consider the problem of predicting how well a student does in her second year of college/university, given how well she did in her first year.<br>
Specifically, let x be equal to the number of "A" grades (including A-. A and A+ grades) that a student receives in their first year of college (freshmen year). We would like to predict the value of y, which we define as the number of "A" grades they get in their second year (sophomore year).<br>
Here each row is one training example. Recall that in linear regression, our hypothesis is $h_{\theta} = \theta_{0} + \theta_{1}x$ and we use m to denote the number of training examples.<br>
<table class="a">
  <tr><th>X</th><th>Y</th></tr>
  <tr><td>3</td><td>2</td></tr>
  <tr><td>1</td><td>2</td></tr>
  <tr><td>0</td><td>1</td></tr>
  <tr><td>4</td><td>3</td></tr>
</table><br>
</p>
<p align="justify">
For the training set given above (note that this training set may also be referenced in other questions in this quiz), what is the value of m? In the box below, please enter your answer (which should be a number between 0 and 10).<br>
<b>Answer:</b> 4.<br><br>

<b>2.</b><br>
What is the cost function for $L(0, 1)$ according to $J(\theta_{0}, \theta_{1}) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})^{2}$?<br>
<b>Answer:</b> 0.5.<br><br>

<b>3.</b><br>
Suppose we set $\theta_{0}$ = -1, $\theta_{1}$ = 0.5. What id $h_{\theta}(4)$?<br>
<b>Answer:</b> 1.<br><br>

<b>4.</b><br>
Let f be some function so that $f(\theta_{0}, \theta_{1})$ outputs a number. For this problem, f is some arbitrary/unknown smooth function (not necessarily the cost function of linear regression, so f may have local optima). Suppose we use gradient descent to try to minimize $f(\theta_{0}, \theta_{1})$ as a function of $\theta_{0}$ and $\theta_{1}$. <b>Which of the following statements are true?</b> (Check all that apply.)<br>
A. If $\theta_{0}$ and $\theta_{1}$ are initialized at a local minimum, then one iteration will not change their value<br>
B. If $\theta_{0}$ and $\theta_{1}$ are initialized so that $\theta_{0}$ = $\theta_{1}$, then by symmetry (because we do simultaneous updates to the two parameters), after one iteration of gradient descent, we will still have $\theta_{0}$ = $\theta_{1}$<br>
C. If the learning rate is too small, then gradient descent may take a very long time to converge<br>
D. Even if the learning rate $\alpha$ is very large, every iteration of gradient descent will decrease the value of $f(\theta_{0}, \theta_{1})$<br>
<b>Answer:</b> A, C.<br><br>

<b>5.</b><br>
Suppose that for some linear regression problem (say, predicting housing prices as in the lecture), we have some training set, and for our training set we managed to find some $\theta_{0}$, $\theta_{1}$ such that $J(\theta_{0}, \theta_{1}) = 0$. <b>Which of the statements below must then be true?</b> (Check all that apply.)<br>
A. Gradient descent is likely to get stuck at a local minimum and fail to find the global minimum<br>
B. Our training set can be fit perfectly by a straight line, i.e., all of our training examples lie perfectly on some straight line<br>
C. For this to be true, we must have $\theta_{0} = 0$ and $\theta_{1} = 0$ so that $h_{\theta}(x) = 0$<br>
D. For this to be true, we must have $y^{i} = 0$ for every value of i = 1, 2, ..., m<br>
<b>Answer:</b> B.<br><br>
</p>

### 1.3 Linear Algebra Review 
#### 1.3.1 Linear Algebra Review
<p align="justify">
<b>Matrices and Vectors</b><br>
$\bigstar$ Matrix: rectangular array of numbers<br>
$\bigstar$ Dimension of matrix: number of rows $\times$ number of columns<br><br>
$\bigstar$ Matrix elements (entries of matrix)<br>
$$
A =
\begin{bmatrix}
1402 & 2\\
1371 & 821\\
949 & 1437\\
147 & 1448
\end{bmatrix}
$$
$A_{ij}$ = i, j entry in the $i^{th}$ row, $j^{th}$ column.<br>
$\bigstar$ Vector: An n $\times$ 1 matrix.<br>
$$
y =
\begin{bmatrix}
460\\
232\\
315\\
178
\end{bmatrix}
$$<br>

<b>Addition and Scalar Multiplication</b><br>
$\bigstar$ Matrix Addition<br>
$$
\begin{bmatrix}
1 & 0\\
2 & 5\\
3 & 1\\
\end{bmatrix} +
\begin{bmatrix}
4 & 0.5\\
2 & 5\\
0 & 1\\
\end{bmatrix} = 
\begin{bmatrix}
5 & 0.5\\
4 & 10\\
3 & 2\\
\end{bmatrix}
$$
$\bigstar$ Scalar Multiplication<br>
$$
3 \times
\begin{bmatrix}
1 & 0\\
2 & 5\\
3 & 1\\
\end{bmatrix} =
\begin{bmatrix}
3 & 0\\
6 & 15\\
9 & 3\\
\end{bmatrix}
$$<br>

<b>Matrix Vector Multiplication</b><br>
$$
\begin{bmatrix}
1 & 3\\
4 & 0\\
2 & 1\\
\end{bmatrix} \times
\begin{bmatrix}
1\\
5
\end{bmatrix} = 
\begin{bmatrix}
16\\
4\\
7
\end{bmatrix} 
$$<br>

<b>Matrix Matrix Multiplication</b><br>
$$
\begin{bmatrix}
1 & 3 & 2\\
4 & 0 & 1
\end{bmatrix} \times
\begin{bmatrix}
1 & 3\\
0 & 1\\
5 & 2
\end{bmatrix} = 
\begin{bmatrix}
11 & 10\\
9 & 14
\end{bmatrix} 
$$
Consider A and B are matrix, in general matrix multiplication is not commutative.<br>
$$A \times B \neq B \times A$$
$\bigstar$ Identity Matrix: $I_{n \times n}$<br>
For any matrix A<br>
$$A \cdot I = I \cdot A$$<br><br>

<b>Inverse and Transpose</b><br>
Not all numbers have an inverse.<br>
$\bigstar$ Matrix Inverse<br>
If A is an $m \times m$ matrix, and if it has an inverse<br>
$$AA^{-1} = A^{-1}A = I$$
Matrices that don't have an inverse are singular or degenerate<br>
$\bigstar$ Matrix Transpose<br>
Let A be an $m \times n$ matrix, and let $B = A^{T}$, then B is an $n \times m$ matrix and $B_{ij} = A_{ji}$.<br><br>
</p>

#### 1.3.2 Quiz
<p align="justify">
<b>1.</b><br>
Let two matrices A, B<br>
$$
A =
\begin{bmatrix}
4 & 3\\
6 & 9
\end{bmatrix}, \quad
B =
\begin{bmatrix}
-2 & 9\\
-5 & 2
\end{bmatrix}
$$
<b>What is A-B?</b><br>
<b>Answer:</b>
$$
\begin{bmatrix}
6 & -6\\
11 & 7
\end{bmatrix}
$$

<b>2.</b><br>
Let x<br>
$$
x =
\begin{bmatrix}
5\\
5\\
2\\
7
\end{bmatrix}
$$
<b>What is 2*x?</b><br>
<b>Answer:</b>
\begin{bmatrix}
10\\
10\\
4\\
14
\end{bmatrix}

<b>3.</b><br>
Let u be a 3-dimensional vector, where specifically<br>
$$
u =
\begin{bmatrix}
2\\
1\\
8
\end{bmatrix}
$$
<b>What is $u^{T}$?</b><br>
<b>Answer:</b>
\begin{bmatrix}
2 & 1 & 8
\end{bmatrix}

<b>4.</b><br>
Let u and v be 3-dimensional vectors, where specifically<br>
$$
u =
\begin{bmatrix}
1\\
3\\
-1
\end{bmatrix}, \quad
v =
\begin{bmatrix}
2\\
2\\
4
\end{bmatrix}
$$
<b>What is $u^{T}v$?</b><br>
<b>Answer:</b> 4.<br><br>

<b>5.</b><br>
Let A and B be 3x3 (square) matrices. <b>Which of the following must necessarily hold true?</b> Check all that apply.<br>
A. A*B = B*A<br>
B. If A is the $3 \times 3$ identity matrix, then A*B = B*A<br>
C. If C = A*B, then C is a $6 \times 6$ matrix<br>
D. A + B = B + A<br>
<b>Answer:</b> B, D.<br><br>
</p>


## 2. Week 2
### 2.1 Linear Regression with Multiple Variables
#### 2.1.1 Multivariate Linear Regression
<p align="justify">
<b>Multiple Features</b><br>
$\bigstar$ Notation:<br>
-- n = number of features<br>
-- $x^{i}$: input (features) of i-th training example<br>
-- $x_{j}^{i}$: value of feature j in i-th training example<br>
$$h_{\theta}(x) = \theta_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{n}x_{n}$$<br>

<b>Gradient Descent for Multiple Variables</b><br>
Hypothesis:<br>
$$h_{\theta}(x) = \theta^{T}x = \theta_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{n}x_{n}$$
Parameters:<br>
$$\theta = \{\theta_{0}, \theta_{1}, ..., \theta_{n}\}$$
Cost function<br>
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})^{2} = \frac{1}{2m} \sum_{i=1}^{m} (\theta^{T} x^{i} - y^{i})^{2} = \frac{1}{2m} \sum_{i=1}^{m} ((\sum_{j=0}^{n} \theta_{j} x_{j}^{i})^{2} - y^{i})^{2}$$
Gradient descent<br>
$$\theta_{j} = \theta_{j} - \alpha \frac{\partial J(\theta)}{\partial \theta_{j}}$$<br>

<b>Feature Scaling</b><br>
Make sure features are on a similar scale e.g. [-1, 1]<br>
$\bigstar$ Mean normalization: replace $x_{i}$ except $x_{0}$ = 1 to make features have approximately 2 mean.<br><br>

<b>Learning Rate</b><br>
Making sure gradient descent is working correctly.<br>
For sufficiently small $\alpha$, $J(\theta)$ should decrease on every iteration.<br>
but if $\alpha$ is too small , gradient descent can be slow to converge.<br><br>
</p>

#### 2.1.2 Computing Parameters Analytically
<p align="justify">
<b>Normal Equation</b><br>
If 1D ($\theta \in \mathbb{R}$)<br>
$$J(\theta) = a \theta^{2} + b \theta + c$$
solve<br>
$$\frac{d J(\theta)}{d \theta} = 2 \theta + b = 0$$
If $\theta \in \mathbb{R}^{n+1}$<br>
$$J(\theta_{0}, \theta_{1}, ..., \theta_{n}) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})^{2}$$
For each j = 0, 1, ..., n, solve<br>
$$\frac{\partial J(\theta)}{\partial \theta_{j}} = 0$$
In a format of matrix<br>
$$
\begin{aligned}
J(\theta) & = (X\theta - Y)^{T} (X\theta - Y) \\
& = (\theta^{T} X^{T} - Y^{T}) (X\theta - Y) \\
& = \theta^{T} X^{T} X \theta - \theta^{T} X^{T} Y - Y^{T} X \theta + Y^{T}Y
\end{aligned}
$$
Derivatives of Matrices
$$\frac{ \partial x^{T}a}{\partial x} = \frac{\partial a^{T} x}{\partial x} = a$$
$$\frac{\partial x^{T} B x}{\partial x} = (B + B^{T}) x$$
$$
\begin{aligned}
\frac{\partial J(\theta)}{\partial \theta} & = \frac{\theta^{T} X^{T} X \theta}{\partial \theta} - \frac{Y^{T} X \theta}{\partial \theta} - \frac{\partial \theta^{T} X^{T} Y}{\partial \theta} \\
& = (X^{T}X + (X^{T}X)^{T}) \theta -X^{T}Y  - X^{T}Y \\
& = 2 X^{T} X \theta - 2 X^{T} Y = 0
\end{aligned}
$$
$$\theta = (X^{T}X)^{-1}X^{T}y$$
Normal equation don't need $\alpha$ and don't need itertaion. If we have many feature (n >> 1), computation is slow.<br><br>

<b>Normal Equation Noninvertibility</b><br>
$X^{T}X$ is non-invertible (singular / degenerate)<br>
$\bigstar$ reason<br>
-- redundant features (linearly dependent)<br>
-- too many features (m < n)<br>
$\bigstar$ remedy: remove some features or use regularization<br><br>
</p>

#### 2.1.3 Quiz
<p align="justify">
<b>1.</b><br>
Suppose m=4 students have taken some class, and the class had a midterm exam and a final exam. You have collected a dataset of their scores on the two exams, which is as follows:<br>
<table class="c">
  <tr><th>midterm exam</th><th>$\text{(midterm exam)}^{2}$</th><th>final exam</th></tr>
  <tr><td>89</td><td>7921</td><td>96</td></tr>
  <tr><td>72</td><td>5184</td><td>74</td></tr>
  <tr><td>94</td><td>8836</td><td>87</td></tr>
  <tr><td>69</td><td>4761</td><td>78</td></tr>
</table><br>
</p>
<p align="justify">
You'd like to use polynomial regression to predict a student's final exam score from their midterm exam score. Concretely, suppose you want to fit a model of the form $h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2} x_{2}$, where $x_{1}$ is the midterm score and $x_{2}$ is $\text{(midterm score)}^{2}$. Further, you plan to use both feature scaling (dividing by the "max-min", or range, of a feature) and mean normalization. What is the normalized feature $x_{1}^{3}$? (Hint: midterm = 94, final = 87 is training example 3.) Please round off your answer to two decimal places and enter in the text box below.<br>
<b>Answer</b>: 0.52.<br><br>

<b>2.</b><br>
You run gradient descent for 15 iterations with α=0.3 and compute J(θ) after each iteration. You find that the value of J(θ) decreases slowly and is still decreasing after 15 iterations. Based on this, which of the following conclusions seems most plausible?<br>
A. α=0.3 is an effective choice of learning rate.<br>
B. Rather than use the current value of α, it'd be more promising to try a smaller value of α (say α=0.1).<br>
C. Rather than use the current value of α, it'd be more promising to try a larger value of α (say α=1.0).<br>
<b>Answer</b>: C.<br><br>

<b>3.</b><br>
Suppose you have m=28 training examples with n=4 features (excluding the additional all-ones feature for the intercept term, which you should add). The normal equation is $\theta = (X^{T}X)^{-1}X^{T}y$. For the given values of m and n, what are the dimensions of θ, X, and y in this equation?<br>
A. X is 28 $\times$ 5, y is 28 $\times$ 1, $\theta$ is 5 $\times$ 1<br>
B. X is 28 $\times$ 4, y is 28 $\times$ 1, $\theta$ is 4 $\times$ 4<br>
C. X is 28 $\times$ 5, y is 28 $\times$ 5, $\theta$ is 5 $\times$ 5<br>
D. X is 28 $\times$ 4, y is 28 $\times$ 1, $\theta$ is 4 $\times$ 1<br>
<b>Answer</b>: A.<br><br>

<b>4.</b><br>
Suppose you have a dataset with m=1000000 examples and n=200000 features for each example. You want to use multivariate linear regression to fit the parameters θ to our data. Should you prefer gradient descent or the normal equation?<br>
A. The normal equation, since gradient descent might be unable to find the optimal θ.<br>
B. Gradient descent, since it will always converge to the optimal θ.<br>
C. The normal equation, since it provides an efficient way to directly find the solution.<br>
D. Gradient descent, since $(X^{T}X)^{-1}$ will be very slow to compute in the normal equation.<br>
<b>Answer</b>: D.<br><br>

<b>5.</b><br>
Which of the following are reasons for using feature scaling?<br>
A. It speeds up gradient descent by making it require fewer iterations to get to a good solution.<br>
B. It is necessary to prevent gradient descent from getting stuck in local optima.<br>
C. It speeds up solving for θ using the normal equation.<br>
D. It prevents the matrix $X^{T}X$ (used in the normal equation) from being non-invertable (singular/degenerate).<br>
<b>Answer</b>: A.<br><br>
</p>

### 2.2 Octave/Matlab Tutorial
#### 2.2.1 Octave/Matlab Tutorial
{% highlight Matlab %}
a = pi;
disp(a);
disp(sprintf('2 decimals: %0.2f', a));

I = eye(6);
floor(a);

A = magic(3);
pinv(A);
{% endhighlight %}

#### 2.2.2 Quiz
<p align="justify">
<b>1.</b><br>
Suppose I first execute the following in Octave/Matlab:<br>
</p>
{% highlight Matlab %}
A = [1 2; 3 4; 5 6];
B = [1 2 3; 4 5 6];
{% endhighlight %}
<p align="justify">
Which of the following are then valid commands? Check all that apply. (Hint: A' denotes the transpose of A.)<br>
A. C = A * B;<br>
B. C = B' + A;<br>
C. C = A' * B;<br>
D. C = B + A;<br>
<b>Answer</b>: A, B.<br><br>

<b>2.</b><br>
Let<br>
$$
A =
\begin{bmatrix}
16 & 2 & 3 & 13 \\
5 & 11 & 10 & 8 \\
9 & 7 & 6 & 12 \\
4 & 14 & 15 & 1
\end{bmatrix}
$$
Which of the following indexing expressions gives B?<br>
$$
B =
\begin{bmatrix}
16 & 2\\
5 & 11\\
9 & 7\\
4 & 14
\end{bmatrix}
$$
Check all that apply.<br>
A. B = A(:, 1:2);<br>
B. B = A(1:4, 1:2);<br>
C. B = A(0:2, 0:4);<br>
D. B = A(1:2, 1:4);<br>
<b>Answer</b>: A, B.<br><br>

<b>3.</b><br>
Let A be a 10x10 matrix andx be a 10-element vector. Your friend wants to compute the product Ax and writes the following code:<br>
</p>
{% highlight Matlab %}
v = zeros(10, 1);
for i = 1:10
  for j = 1:10
    v(i) = v(i) + A(i, j) * x(j);
  end
end
{% endhighlight %}
<p align="justify">
How would you vectorize this code to run without any for loops? Check all that apply.<br>
A. v = A * x;<br>
B. v = Ax;<br>
C. v = x' * A;<br>
D. v = sum (A * x);<br>
<b>Answer</b>: A.<br><br>

<b>4.</b><br>
Say you have two column vectors v and w, each with 7 elements (i.e., they have dimensions 7x1). Consider the following code:<br>
</p>
{% highlight Matlab %}
z = 0;
for i = 1:7
  z = z + v(i) * w(i)
end
{% endhighlight %}
<p align="justify">
Which of the following vectorizations correctly compute z? Check all that apply.<br>
A. z = sum (v .* w);<br>
B. z = w' * v;<br>
C. z = v * w;<br>
D. z = w * v;<br>
<b>Answer</b>: A, B.<br><br>

<b>5.</b><br>
In Octave/Matlab, many functions work on single numbers, vectors, and matrices. For example, the sin function when applied to a matrix will return a new matrix with the sin of each element. But you have to be careful, as certain functions have different behavior. Suppose you have an 7x7 matrix X. You want to compute the log of every element, the square of every element, add 1 to every element, and divide every element by 4. You will store the results in four matrices, A,B,C,D. One way to do so is the following code:<br>
</p>
{% highlight Matlab %}
for i = 1:7
  for j = 1:7
    A(i, j) = log(X(i, j));
    B(i, j) = X(i, j) ^ 2;
    C(i, j) = X(i, j) + 1;
    D(i, j) = X(i, j) / 4;
  end
end
{% endhighlight %}
<p align="justify">
Which of the following correctly compute A,B,C, or D? Check all that apply.<br>
A. C = X + 1;<br>
B. D = X / 4;<br>
C. A = log (X);<br>
D. B = X ^ 2;<br>
<b>Answer</b>: A, B, C.<br><br>
</p>

#### 2.2.3 Programming Assignment: Linear Regression
{% highlight Matlab %}
function A = warmUpExercise()
%WARMUPEXERCISE Example function in octave
%   A = WARMUPEXERCISE() is an example function that returns the 5x5
%   identity matrix
A = [];

% ============= YOUR CODE HERE ==============
% Instructions: Return the 5x5 identity matrix 
%               In octave, we return values by defining which variables
%               represent the return values (at the top of the file)
%               and then set them accordingly. 
A = eye(5);
% ===========================================
end


function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
diff = X * theta - y;
J = (diff' * diff) / (2 * m);
% =========================================================================
end


function [theta, J_history] = gradientDescent(X, y, theta, alpha,...
    num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    diff = X * theta - y;
    theta = theta - X'*diff*(alpha/m);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end
end


function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
mu = mean(X);
sigma = std(X);
X_norm = (X - repmat(mu, size(X, 1), 1))./repmat(sigma, size(X, 1), 1);

% ============================================================
end


function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
diff = X * theta - y;
J = (diff' * diff) / (2 * m);
% =========================================================================
end


function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha,...
    num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates
%   theta by taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    diff = X * theta - y;
    theta = theta - X'*diff*(alpha/m);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end
end


function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%
theta = pinv((X'*X))*X'*y;
% ============================================================
end
{% endhighlight %}
<p align="justify">
<br>
</p>


## 3. Week 3
### 3.1. Logistic Regression
#### 3.1.1 Classification and Representation
<p align="justify">
<b>Classification</b><br>
Logistic Regression: $0 \leq h_{\theta}(x) \leq 1$<br><br>

<b>Hypothesis Representation</b><br>
Sigmoid function<br>
$$h_{\theta}(x) = g(\theta^{T} x) = \frac{1}{1 + e^{-\theta^{T}x}}$$
$\bigstar$ Interpretation of Hypothesis Output<br>
$h_{\theta}(x)$ = estimation probability that y = 1 on input x<br><br>

<b>Decision Boundary</b><br>
Predict y = 1 if $h_{\theta}(x) \geq$ 0.5; y = 0 if $h_{\theta}(x)$ < 0.5<br><br>
</p>

#### 3.1.2 Logistic Regression Model
<p align="justify">
<b>Cost Function</b><br>
Using square cost function leads to a non-convex function<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/3_1_2_1.png"/></center>
</p>
<p align="justify">
Logistic regression cost function<br>
$$
\begin{aligned}
\text{Loss}(h_{\theta}(x), y) & =
\begin{cases}
-\log(h_{\theta}(x)), & \quad y = 1\\
-\log(1 - h_{\theta}(x)), & \quad y = 0
\end{cases} \\
& = - y\log(h_{\theta}(x)) - (1 - y)\log(1 - h_{\theta}(x))
\end{aligned}
$$
$$
\begin{aligned}
J(\theta) & = \frac{1}{m} \sum_{i=1}^{m} \text{Loss}(h_{\theta}(x^{i}), y^{i}) \\
& = -\frac{1}{m} [\sum_{i=1}^{m} y^{i} \log(h_{\theta}(x^{i})) + (1 - y^{i}) \log(1 - h_{\theta}(x^{i}))]
\end{aligned}
$$
$\bigstar$ To fit parameters $\theta$:<br>
$$\min_{\theta} J(\theta)$$
$\bigstar$ To make a prediction given new x<br>
$$h_{\theta}(x) = \frac{1}{1 + e^{-\theta^{T}x}}$$
$\bigstar$ Gradient descent<br>
$$\frac{\partial J(\theta)}{\partial \theta} = -\frac{1}{m} \sum_{i}^{m} x^{i}(y^{i} - h_{\theta}(x^{i}))$$

<b>Advanced Optimization</b><br>
$\bigstar$ optimization algorithms:<br>
Gradient descent, Conjugate gradient, BFGS, L-BFGS<br><br>
</p>

#### 3.1.3 Multiclass Classification
<p align="justify">
<b>One-vs-all</b><br>
Train a logistic regression classifier $h_{\theta}^i(x)$ for each class i to predict the probability that y = i<br>
On a new input x, to make a prediction, pick the class i that maximizes<br>
$$\max_{i} h_{\theta}^{i}(x)$$
Suppose you have a multi-class classification problem with k classes (so y∈{1,2,…,k}). Using the 1-vs.-all method, we will need k different logistic regression classifiers.<br><br>
</p>

#### 3.1.4 Quiz
<p align="justify">
<b>1.</b><br>
Suppose that you have trained a logistic regression classifier, and it outputs on a new example x a prediction $h_{\theta}$(x) = 0.4. This means (check all that apply):<br>
A. Our estimate for P(y = 1 | x; $\theta$) is 0.6<br>
B. Our estimate for P(y = 0 | x; $\theta$) is 0.6<br>
C. Our estimate for P(y = 0 | x; $\theta$) is 0.4<br>
D. Our estimate for P(y = 1 | x; $\theta$) is 0.4<br><br>
<b>Answer</b>: B, D.<br><br>

<b>2.</b><br>
Suppose you have the following training set, and fit a logistic regression classifier $h_{\theta}(x) = g(\theta_{0} + \theta_{1}x_{1} + \theta_{2} x_{2})$.<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/3_1_4_1.png"/></center>
</p>
<p align="justify">
Which of the following are true? Check all that apply.<br>
A. Adding polynomial features e.g. instead using<br>
$$h_{\theta}(x) = g(\theta_{0} + \theta_{1}x_{1} + \theta_{2} x_{2} + \theta_{3} x_{1}^{2} + \theta_{1}x_{1}x_{2} + \theta_{5} x_{2}^{2})$$
could increase how well we can fit the training data<br>
B. At the optimal value of $\theta$ (e.g. found by fminnunc), we will have $J(\theta)$ > 0<br>
C. Adding polynomial features e.g. instead using<br>
$$h_{\theta}(x) = g(\theta_{0} + \theta_{1}x_{1} + \theta_{2} x_{2} + \theta_{3} x_{1}^{2} + \theta_{1}x_{1}x_{2} + \theta_{5} x_{2}^{2})$$
could increase J($\theta$) how well we can fit the training data<br>
D. If we train gradient descent for enough iterations, for some examples $x^{i}$ in the training set it is possible to obtain $h_{\theta}(x^{i})$ > 1<br>
<b>Answer</b>: A, B.<br><br>

<b>3.</b><br>
For logistic regression, the gradient is given by<br>
$$\frac{\partial J(\theta)}{\partial \theta_{j}} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i}) x_{j}^{i}$$
Which of these is a correct gradient descent update for logistic regression with a learning rate of α? Check all that apply.<br>
$$
\begin{aligned}
& \text{A.} \quad \theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m} (\theta^{T}x - y^{i}) x_{j}^{i} \quad \text{simultaneously update for all j} \\
\\
& \text{B.} \quad \theta = \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} (\frac{1}{1 + e^{-\theta^{T}x^{i}}} - y^{i}) x^{i} \\
\\
& \text{C.} \quad \theta = \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i}) x^{i} \\
\\
& \text{D.} \quad \theta = \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} (\theta^{T}x - y^{i}) x^{i} 
\end{aligned}
$$
<b>Answer</b>: B, C.<br><br>

<b>4.</b><br>
Which of the following statements are true? Check all that apply.<br>
A. For logistic regression, sometimes gradient descent will converge to a local minimum (and fail to find the global minimum). This is the reason we prefer more advanced optimization algorithms such as fminunc (conjugate gradient/BFGS/L-BFGS/etc).<br>
B. The sigmoid function $g(z) = \frac{1}{1 + e^{-z}}$ is never gretaer than 1<br>
C. The cost function J(θ) for logistic regression trained with m≥1 examples is always greater than or equal to zero.<br>
D. Linear regression always works well for classification if you classify by using a threshold on the prediction made by linear regression.<br>
<b>Answer</b>: B, C.<br><br>

<b>5.</b><br>
Suppose you train a logistic classifier $h_{\theta}(x) = g(\theta_{0} + \theta_{1}x_{1} + \theta_{2} x_{2})$. Suppose $\theta_{0}$ = -6, $\theta_{1}$ = 1, $\theta_{2}$ = 0. Which of the following figures represents the decision boundary found by your classifier?<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/3_1_4_2.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: B.<br><br>
</p>

### 3.2. Regularization
#### 3.2.1 Solving the Problem of Overfitting
<p align="justify">
<b>The Problem of Overfitting</b><br>
If we have too many features, the learned hypothesis may fit the training set very well, but fail to generalize to new examples.<br>
$\bigstar$ Addressing overfitting:<br>
-- reduce number of features: manually select, model selection algorithm<br>
-- regularization: keep all features, but reduce magnitude / values of parameters<br><br>

<b>Cost Function</b><br>
$$J(\theta) = \frac{1}{2m} [\sum_{i=1}^{m} (h_{\theta} (x^{i}) - y^{i})^{2} + \lambda \sum_{j=1}^{n} \theta_{j}^{2}]$$<br>

<b>Regularized Linear Regression</b><br>
$\bigstar$ gradient descent
$$\frac{\partial J(\theta)}{\partial \theta_{j}} = \frac{1}{m}\sum_{i=1}^{m} (h_{\theta} (x^{i}) - y^{i}) x_{j}^{i} + \frac{\lambda}{m} \theta_{j}, \quad i = 1, ..., m, \quad j = 0, 1, .., n, \quad x_{0}^{i} = 1$$
$\bigstar$ Normal equation<br>
$$
\begin{aligned}
J(\theta) & = (X\theta - Y)^{T} (X\theta - Y) + \lambda \theta^{T} \theta \\
& = ( \theta^{T} X^{T} - Y^{T}) (X\theta - Y) + \lambda \theta^{T} \theta \\
& = \theta^{T} X^{T} X \theta - \theta^{T} X^{T} Y - Y^{T} X \theta + Y^{T}Y + \lambda \theta^{T} \theta
\end{aligned}
$$
$$
\begin{aligned}
\frac{\partial J(\theta)}{\partial \theta} & = \frac{\theta^{T} X^{T} X \theta}{\partial \theta} - \frac{Y^{T} X \theta}{\partial \theta} - \frac{\partial \theta^{T} X^{T} Y}{\partial \theta} + \lambda \frac{\partial \theta^{T} \theta}{\partial \theta} \\
& = (X^{T}X + (X^{T}X)^{T}) \theta - X^{T}Y - X^{T}Y + 2 \lambda \theta \\
& = 2 X^{T} X \theta - 2 X^{T} Y + 2\lambda \theta = 0
\end{aligned}
$$
We don't penalize in the bias term
$$
\theta = (X^{T}X + \lambda E)^{-1} X^{T} Y, \quad \text{where } E =
\begin{bmatrix}
0 &  &  &  &  & \\
   & 1 &  & &  & \\
   &  & \ddots &  &  & \\
   &  & & &  1  & \\
   &  &  &  &  & 1
\end{bmatrix}_{n \times n}
$$<br>

<b>Regularized Logistic Regression</b><br>
Cost function
$$J(\theta) = -\frac{1}{m} [\sum_{i=1}^{m} y^{i} \log(h_{\theta}(x^{i})) + (1 - y^{i}) \log(1 - h_{\theta}(x^{i}))] + \frac{\lambda}{2m} \sum_{j=1}^{m} \theta_{j}^{2}$$
gradient descent
$$\frac{\partial J(\theta)}{\partial \theta_{j}} = -\frac{1}{m} \sum_{i}^{m} x^{i}(y^{i} - h_{\theta}(x^{i})) + \frac{\lambda}{m} \theta_{j}, \quad i = 1, ..., m, \quad j = 0, 1, .., n, \quad x_{0}^{i} = 1$$<br>
</p>

#### 3.2.2 Quiz
<p align="justify">
<b>1.</b><br>
You are training a classification model with logistic regression. Which of the following statements are true? Check all that apply.<br>
A. Introducing regularization to the model always results in equal or better performance on the training set.<br>
B. Adding many new features to the model makes it more likely to overfit the training set.<br>
C. Adding a new feature to the model always results in equal or better performance on examples not in the training set.<br>
D. Introducing regularization to the model always results in equal or better performance on examples not in the training set.<br>
<b>Answer</b>: B.<br><br>

<b>2.</b><br>
Suppose you ran logistic regression twice, once with λ=0, and once with λ=1. One of the times, you got parameters
$$
\theta =
\begin{bmatrix}
23.4 \\
37.9
\end{bmatrix}
$$
and the other time you got
$$
\theta =
\begin{bmatrix}
1.03 \\
0.28
\end{bmatrix}
$$
However, you forgot which value of λ corresponds to which value of θ. Which one do you think corresponds to λ=1?
$$
\begin{aligned}
& \text{A.} \quad
\theta =
\begin{bmatrix}
23.4 \\
37.9
\end{bmatrix} \\
& \text{B.} \quad
\theta =
\begin{bmatrix}
1.03 \\
0.28
\end{bmatrix}
\end{aligned}
$$
<b>Answer</b>: B.<br><br>

<b>3.</b><br>
Which of the following statements about regularization are true? Check all that apply.<br>
A. Consider a classification problem. Adding regularization may cause your classifier to incorrectly classify some training examples (which it had correctly classified when not using regularization, i.e. when λ=0).<br>
B. Using a very large value of λ cannot hurt the performance of your hypothesis; the only reason we do not set λ to be too large is to avoid numerical problems.<br>
C. Because logistic regression outputs values $0 \leq h_{\theta}(x) \leq 1$ (x)≤1, its range of output values can only be "shrunk" slightly by regularization anyway, so regularization is generally not helpful for it.<br>
D. Using too large a value of λ can cause your hypothesis to overfit the data; this can be avoided by reducing λ.<br>
<b>Answer</b>: A.<br><br>

<b>4.</b><br>
In which one of the following figures do you think the hypothesis has overfit the training set?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/3_2_2_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: A.<br><br>

<b>5.</b><br>
In which one of the following figures do you think the hypothesis has underfit the training set?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/3_2_2_2.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: A.<br><br>
</p>

#### 3.2.3 Programming Assignment: Logistic Regression
{% highlight Matlab %}
function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
g = 1./ (1 + exp(-z));
% =============================================================
end


function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h = sigmoid(X*theta);
J = -(y'*log(h) + (1-y)'*log(1-h))/m;
grad = (h-y)'*X / m;
% =============================================================
end


function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
yhat = sigmoid(X*theta);
p = yhat>0.5;
% =========================================================================
end


function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with
%regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X*theta);
J = -(y'*log(h) + (1-y)'*log(1-h))/m;
J = J + (theta'*theta - theta(1)*theta(1))*(lambda/2/m);

grad = ((h-y)'*X)/m;
grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end)';
% =============================================================
end
{% endhighlight %}
<p align="justify">
<br>
</p>

## 4. Week 4
### 4.1 Neural Networks: Representation
#### 4.1.1 Motivations
<p align="justify">
<b>Non-linear Hypotheses</b><br>
Non-linear classification<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/4_1_1_1.png"/></center>
</p>
<p align="justify">
<br>
<b>Neurons and the Brain</b><br>
$\bigstar$ Neural networks<br>
Origins: algorithms that try to mimic the brain<br>
Was very widely used in 80s and early 90s; popularity diminished in late 90s<br>
Recent ressurgence: state-of-the-art technique for many applications<br><br>
</p>

#### 4.1.2 Neural Networks
<p align="justify">
<b>Model Representation</b><br>
$a_{j}^{(i)}$ denotes the activated value in the j-th neuron (generally from top to down) in i-th layer. Conventionally, we count from the first hidden layer.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/4_1_2_1.png"/></center>
</p>
<p align="justify">
Forward propagation<br>
$$
\begin{aligned}
& Z^{(i)} = A^{(i-1)} (\Theta^{(i)})^{T} \\
& A^{(i)} = \text{sigmoid}(Z^{(i)})
\end{aligned}
$$
$\bigstar$ Don't forget the bias term.<br>
If $i^{\text{th}}$ layer has $n^{(i)}$ neurons and $(i-1)^{\text{th}}$ has $n^{(i-1)}$ neurons, $\Theta^{(i)}$ is a matrix of ($n^{(i)} \times n^{(i-1)}$).<br><br>
</p>

#### 4.1.3 Applications
<p align="justify">
<b>Multiclass Classification</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/4_1_3_1.png"/></center>
<br>
</p>

#### 4.1.4 Quiz
<p align="justify">
<b>1.</b><br>
Which of the following statements are true? Check all that apply.<br>
A. Any logical function over binary-valued (0 or 1) inputs $x_{1}$ and $x_{2}$ can be (approximately) represented using some neural network.<br>
B. The activation values of the hidden units in a neural network, with the sigmoid activation function applied at every layer, are always in the range (0, 1).<br>
C. Suppose you have a multi-class classification problem with three classes, trained with a 3 layer network. Let $a_{1}^{(3)} = (h_{\theta}(x))_{1}$ be the activation of the first output unit, and similarly $a_{2}^{(3)} = (h_{\theta}(x))_{2}$ and $a_{3}^{(3)} = (h_{\theta}(x))_{3}$. Then for any input x, it must be the case that $a_{1}^{(3)} + a_{2}^{(3)} + a_{3}^{(3)} = 1$<br>
D. A two layer (one input layer, one output layer; no hidden layer) neural network can represent the XOR function.<br>
<b>Answer</b>: A, B.<br><br>

<b>2.</b><br>
Consider the following neural network which takes two binary-valued inputs $x_{1}$, $x_{2}$ ∈{0,1} and outputs $h_{\theta}$(x). Which of the following logical functions does it (approximately) compute?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/4_1_4_1.png"/></center>
</p>
<p align="justify">
A. OR<br>
B. AND<br>
C. NAND (means NOT AND)<br>
D. XOR (exclusive OR)<br>
<b>Answer</b>: A.<br><br>

<b>3.</b><br>
Consider the neural network given below. Which of the following equations correctly computes the activation $a_{1}^{(3)}$? Note: g(z) is the sigmoid activation function.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/4_1_4_2.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: A.<br><br>

<b>4.</b><br>
You have the following neural network. You'd like to compute the activations of the hidden layer $a^{(2)} \in \mathbb{R}^{3}$. One way to do so is the following Octave code. You want to have a vectorized implementation of this (i.e., one that does not use for loops). Which of the following implementations correctly compute $a^{(2)}$? Check all that apply.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/4_1_4_3.png"/></center>
</p>
<p align="justify">
A. z = Theta1 * x; a2 = sigmoid(z)<br>
B. a2 = sigmoid(x * Theta1)<br>
C. a2 = sigmoid(Theta2 * x)<br>
D. z = sigmoid(x); a2 = sigmoid(Theta1 * z)<br>
<b>Answer</b>: A.<br><br>

<b>5.</b><br>
You are using the neural network pictured below and have learned the parameters
$$
\Theta^{(1)} =
\begin{bmatrix}
1 & 1 & 2.4 \\
1 & 1.7 & 3.2
\end{bmatrix}
$$
(used to compute $a^{(2)}$) and
$$
\Theta{(2)} =
\begin{bmatrix}
1 & 0.3 & -1.2
\end{bmatrix}
$$
(used to compute $a^{(3)}$) as a function of $a^{(2)}$. Suppose you swap the parameters for the first hidden layer between its two units so
$$
\Theta{(1)} =
\begin{bmatrix}
1 & 1.7 & 3.2 \\
1 & 1 & 2.4
\end{bmatrix}
$$
and also swap the output layer so
$$
\Theta{(2)} =
\begin{bmatrix}
1 & -1.2 & 0.3
\end{bmatrix}
$$
How will this change the value of the output $h_{\theta}(x)$?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/4_1_4_4.png"/></center>
</p>
<p align="justify">
A. it will stay the same<br>
B. it will increase<br>
C. it will decrease<br>
D. insufficient information to tell: it may increase or decrease<br>
<b>Answer</b>: A.<br><br>
</p>

#### 4.1.5 Programming Assignment: Multi-class Classification and Neural Networks
{% highlight Matlab %}
function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
h = sigmoid(X*theta);
J = -(y'*log(h) + (1-y)'*log(1-h))/m;
J = J + (theta'*theta - theta(1)*theta(1))*(lambda/2/m);

grad = ((h-y)'*X)/m;
grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end)';
% =============================================================
grad = grad(:);
end


function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
for c = 1: num_labels
    initial_theta = zeros(n+1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [theta] = ...
        fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
        initial_theta, options);
    all_theta(c, :) = theta';
end;
% =========================================================================
end


function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
H = sigmoid(X*all_theta');
[~, p] = max(H, [], 2);
% =========================================================================
end


function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1), X];
A1 = sigmoid(X*Theta1');
A1 = [ones(m, 1), A1];
A2 = sigmoid(A1*Theta2');
[~, p] = max(A2, [], 2);
% =========================================================================
end
{% endhighlight %}
<p align="justify">
<br>
</p>


## 5. Week 5
### 5.1 Neural Networks: Learning
#### 5.1.1 Cost Function and Backpropagation
<p align="justify">
<b>Cost Function</b><br>
Binary classification: 1 output unit<br>
Multi-class classification (K classes): K output units<br>
Logistic regression cost function
$$J(\theta) = -\frac{1}{m} [\sum_{i=1}^{m} y^{i} \log(h_{\theta}(x^{i})) + (1 - y^{i}) \log(1 - h_{\theta}(x^{i}))] + \frac{\lambda}{2m} \sum_{j=1}^{m} \theta_{j}^{2}$$
Neural network
$$h_{\Theta}(x) \in \mathbb{R}^{K}, \quad (h_{\Theta}(x))_{i} = i^{th} \text{ output}$$
$$
J(\Theta) = -\frac{1}{m} [\sum_{i=1}^{m} \sum_{k=1}^{K} y_{k}^{(i)} \log(h_{\Theta}(x^{(i)}))_{k} + (1 - y_{k}^{(i)}) \log(1 - h_{\Theta}(x^{(i)})_{k})] + \frac{\lambda}{2m} \sum_{l=1}^{L}\sum_{i=1}^{n^{(l-1)}}\sum_{j=1}^{n^{(l)}} (\Theta_{ji}^{(l)})^{2}
$$<br>

<b>Backpropagation Algorithm</b><br>
Forward propagation<br>
$$a^{(l)} = g^{(l)}(z^{(l)}), \quad z^{(l)} = a^{(l-1)} (\theta^{(l)})^{T}$$
For each output unit (last layer / output layer)<br>
$$\delta_{j}^{(L)} = a_{j}^{(L)} - y_{j}, \quad \text{where } j \text{ denotes } j^{\text{th}} \text{ neuron in this layer}$$
For other layers
$$
\begin{aligned}
& \partial a^{(l-1)} = \partial z^{(l)} \theta^{(l)} \\
& \partial z^{(l)} = \partial a^{l} * g^{(l)'}(z^{(l)}), \quad \text{where} * \text{ is pair-wise multiplication} \\
& g^{(l)'}(z^{(l)}) = g^{(l)}(z^{(l)}) (1- g^{(l)}(z^{(l)})), \quad \text{if } g \text{ is a sigmoid function}
\end{aligned}
$$
So,
$$\delta^{(l-1)} = [\delta^{(l)} * g^{(l)'}(z^{(l)})] \theta^{(l)}$$<br>
</p>

#### 5.1.2 Backpropagation in Practice
<p align="justify">
<b>Parameter vector</b><br>
We can stack all parameters into a vector called $\theta$<br>
$$\theta = \{\theta_{0}, \theta_{1},..., \theta_{n}\}$$<br>

<b>Gradient Checking</b><br>
$$\frac{d J(\theta)}{d\theta} \approx \frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2 \epsilon}, \quad \text{where } \epsilon = 10^{-4}$$
Be sure to disable your gradient checking code before training your classifiler for sake of efficiency.<br><br>

<b>Random Initialization</b><br>
If zero initialization, after each update, parameters corresponding to inputs going into each of two hidden units are identical.<br>
Random initialization: symmetry breaking.<br><br>

<b>Putting It Together</b><br>
$\bigstar$ Training a neural network:<br>
1) randomly initializa weights<br>
2) implement forward propagation to get $h_{\theta}(x^{(i)})$ for any $x^{(i)}$<br>
3) implement code to compute cost function J($\Theta$)<br>
4) implement backprop to compute partial derivatives $\frac{\partial J(\Theta)}{\partial \Theta_{jk}^{(l)}}$<br>
5) use gradient check to compare $\frac{\partial J(\Theta)}{\partial \Theta_{jk}^{(l)}}$ computed using backpropagation and using numerical estimate of gradient J($\Theta$)<br>
6) use gradient descent or advanced optimization method with backpropagagtion to try to minimize J$\Theta$ as a function of parameter $\theta$.<br><br>
</p>

#### 5.1.3 Quiz
<p align="justify">
<b>1.</b><br>
You are training a three layer neural network and would like to use backpropagation to compute the gradient of the cost function. In the backpropagation algorithm, one of the steps is to update
$$\Delta_{ij}^{(2)} = \Delta_{ij}^{(2)} + \delta_{i}^{(3)} * (a^{(2)})_{j}$$
for every i,j. Which of the following is a correct vectorization of this step?
$$
\begin{aligned}
& \text{A.} \quad \Delta^{(2)} = \Delta^{(2)} + \delta^{(3)} * (a^{(3)})^{T} \\
& \text{B.} \quad \Delta^{(2)} = \Delta^{(2)} + \delta^{(3)} * (a^{(2)})^{T} \\
& \text{C.} \quad \Delta^{(2)} = \Delta^{(2)} + (a^{(3)})^{T} * \delta^{(2)} \\
& \text{D.} \quad \Delta^{(2)} = \Delta^{(2)} + (a^{(2)})^{T} * \delta^{(3)} 
\end{aligned}
$$
<b>Answer</b>: B.<br><br>

<b>2.</b><br>
Suppose Theta1 is a 5x3 matrix, and Theta2 is a 4x6 matrix. You set thetavec = [Theta1(:); Theta2(:)], Which of the following correctly recovers Theta2?<br>
A. reshape(thetaVec(16:39), 4, 6)<br>
B. reshape(thetaVec(15:38), 4, 6)<br>
C. reshape(thetaVec(16:24), 4, 6)<br>
D. reshape(thetaVec(15:39), 4, 6)<br>
E. reshape(thetaVec(16:39), 6, 4)<br>
<b>Answer</b>: A.<br><br>

<b>3.</b><br>
Let $J(\theta) = 2 \theta^{4} + 2$. Let $\theta$ = 1, $\epsilon$ = 0.01. Use the formula
$$\frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2 \epsilon}$$
to numerically compute an approximation to the derivative at $\theta$ = 1. What value do you get? (When θ=1, the true/exact derivative is $\frac{d J(\theta)}{d \theta} = 8$)<br>
A. 8.0008<br>
B. 10<br>
C. 8<br>
D. 7.9992<br>
<b>Answer</b>: A.<br><br>

<b>4.</b><br>
Which of the following statements are true? Check all that apply.<br>
A. Using a large value of λ cannot hurt the performance of your neural network; the only reason we do not set λ to be too large is to avoid numerical problems.<br>
B. If our neural network overfits the training set, one reasonable step to take is to increase the regularization parameter λ.<br>
C. Using gradient checking can help verify if one's implementation of backpropagation is bug-free.<br>
D. Gradient checking is useful if we are using gradient descent as our optimization algorithm. However, it serves little purpose if we are using one of the advanced optimization methods (such as in fminunc).<br>
<b>Answer</b>: B, C.<br><br>

<b>5.</b><br>
Which of the following statements are true? Check all that apply.<br>
A. Suppose we are using gradient descent with learning rate α. For logistic regression and linear regression, J(θ) was a convex optimization problem and thus we did not want to choose a learning rate α that is too large. For a neural network however, J(Θ) may not be convex, and thus choosing a very large value of α can only speed up convergence.<br>
B. Suppose we have a correct implementation of backpropagation, and are training a neural network using gradient descent. Suppose we plot J(Θ) as a function of the number of iterations, and find that it is increasing rather than decreasing. One possible cause of this is that the learning rate α is too large.<br>
C. If we are training a neural network using gradient descent, one reasonable "debugging" step to make sure it is working is to plot J(Θ) as a function of the number of iterations, and make sure it is decreasing (or at least non-increasing) after each iteration.<br>
D. Suppose that the parameter $\Theta^{(1)}$ is a square matrix (meaning the number of rows equals the number of columns). If we replace $\Theta^{(1)}$ with its transpose $(\Theta^{(1)})^{T}$, then we have not changed the function that the network is computing.<br>
<b>Answer</b>: B, C.<br><br>
</p>

#### 5.1.4 Programming Assignment: Neural Network Learning
{% highlight Matlab %}
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight
% matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size *...
    (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size *...
    (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial
%         derivatives of the cost function with respect to Theta1 and
%         Theta2 in Theta1_grad and Theta2_grad, respectively. After
%         implementing Part 2, you can check that your implementation is
%         correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector
%               into a binary vector of 1's and 0's to be used with the
%               neural network cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for
%               the first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to
%               Theta1_grad and Theta2_grad from Part 2.
%
A1 = [ones(m, 1), X];
Z2 = A1*Theta1';
A2 = [ones(m, 1), sigmoid(Z2)];
Z3 = A2*Theta2';
A3 = sigmoid(Z3);

I = eye(max(y));
yhot = I(y, :);

J = -sum(sum(yhot.*log(A3) + (1-yhot).*log(1-A3)))/m +...
    (sum(sum(Theta1(:, 2:end).*Theta1(:, 2:end))) +...
    sum(sum(Theta2(:, 2:end).*Theta2(:, 2:end))))*(lambda/(2*m));

delta3 = A3-yhot;
delta2 = (delta3*Theta2).*[ones(m, 1), sigmoidGradient(Z2)];
delta2 = delta2(:, 2:end);
Theta2_grad = Theta2_grad + delta3'*A2;
Theta2_grad = Theta2_grad/m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) +...
    (lambda/m)*Theta2(:, 2:end);
Theta1_grad = Theta1_grad + delta2'*A1;
Theta1_grad = Theta1_grad/m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) +...
    (lambda/m)*Theta1(:, 2:end);
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end


function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%
W = rand(L_out, 1+L_in);
% =========================================================================
end


function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).
g = sigmoid(z).*(1-sigmoid(z));
% =============================================================
end
{% endhighlight %}
<p align="justify">
<br>
</p>


## 6. Week 6
### 6.1 Advice for Applying Machine Learning
#### 6.1.1 Evaluating a Learning Algorithm
<p align="justify">
<b>Machine learning diagnostic</b><br>
A test you can run to gain insight what is / isn't working with a learning algorithm, and gain guidance as to know how best to improve its performance. Diagnostic can take time to implement, but doing so can be a very good use of your time.<br><br>

<b>Evaluating a Hypothesis</b><br>
For a classification task, do a misclassification error as a performance on test data.<br><br>

<b>Model Selection and Train/Validation/Test Sets</b><br>
Overfitting: the error of the parameters on some data is likely to be lower than the actual generalization error.<br><br>
Model selection: estimate generalization on test set.<br><br>
</p>

#### 6.1.2 Bias vs. Variance
<p align="justify">
<b>Diagnosing Bias vs. Variance</b><br>
Bias (underfit): high training error and high validation error.<br>
Variance(overfit): low training error but high validation error.<br>
If a learning algorithm is suffering high bias, getting more training data will not help much; if a learning algorithm is suffering high variance, getting more training data will help.<br><br>

<b>Learning Curves</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/6_1_2_1.png"/></center>
</p>
<p align="justify">
<b>Deciding What to Do Next Revisited</b><br>
$\bigstar$ High variance:<br>
-- get more training data<br>
-- try smaller sets of features<br>
-- increase $\lambda$<br>
-- try smalller neural network<br>
$\bigstar$ High variance<br>
-- get additional features<br>
-- reduce $\lambda$<br>
-- try larger neural network<br><br>
</p>

#### 6.1.3 Quiz
<p align="justify">
<b>1.</b><br>
You train a learning algorithm, and find that it has unacceptably high error on the test set. You plot the learning curve, and obtain the figure below. Is the algorithm suffering from high bias, high variance, or neither?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/6_1_3_1.png"/></center>
</p>
<p align="justify">
A. High bias<br>
B. High variance<br>
C. Neither<br>
<b>Answer</b>: A.<br><br>

<b>2.</b><br>
Suppose you have implemented regularized logistic regression to classify what object is in an image (i.e., to do object recognition). However, when you test your hypothesis on a new set of images, you find that it makes unacceptably large errors with its predictions on the new images. However, your hypothesis performs well (has low error) on the training set. Which of the following are promising steps to take? Check all that apply.<br>
A. Use fewer training examples.<br>
B. Get more training examples.<br>
C. Try adding polynomial features.<br>
D. Try using a smaller set of features.<br>
<b>Answer</b>: B, D.<br><br>

<b>3.</b><br>
Suppose you have implemented regularized logistic regressionto predict what items customers will purchase on a webshopping site. However, when you test your hypothesis on a newset of customers, you find that it makes unacceptably largeerrors in its predictions. Furthermore, the hypothesisperforms poorly on the training set. Which of thefollowing might be promising steps to take? Check all thatapply.<br>
A. Try increasing the regularization parameter λ.<br>
B. Try using a smaller set of features.<br>
C. Try to obtain and use additional features.<br>
D. Try adding polynomial features.<br>
<b>Answer</b>: C, D.<br><br>

<b>4.</b><br>
Which of the following statements are true? Check all that apply.<br>
A. Suppose you are using linear regression to predict housing prices, and your dataset comes sorted in order of increasing sizes of houses. It is then important to randomly shuffle the dataset before splitting it into training, validation and test sets, so that we don’t have all the smallest houses going into the training set, and all the largest houses going into the test set.<br>
B. Suppose you are training a logistic regression classifier using polynomial features and want to select what degree polynomial (denoted d in the lecture videos) to use. After training the classifier on the entire training set, you decide to use a subset of the training examples as a validation set. This will work just as well as having a validation set that is separate (disjoint) from the training set.<br>
C. A typical split of a dataset into training, validation and test sets might be 60% training set, 20% validation set, and 20% test set.<br>
D. It is okay to use data from the test set to choose the regularization parameter λ, but not the model parameters (θ).<br>
<b>Answer</b>: A, C.<br><br>

<b>5.</b><br>
Which of the following statements are true? Check all that apply.<br>
A. If a learning algorithm is suffering from high variance, adding more training examples is likely to improve the test error.<br>
B. When debugging learning algorithms, it is useful to plot a learning curve to understand if there is a high bias or high variance problem.<br>
C. We always prefer models with high variance (over those with high bias) as they will able to better fit the training set.<br>
D. If a learning algorithm is suffering from high bias, only adding more training examples may not improve the test error significantly.<br>
<b>Answer</b>: A, B, D.<br><br>
</p>

#### 6.1.4 Programming Assignment: Regularized Linear Regression and Bias/Variance
{% highlight Matlab %}
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
diff = X*theta - y;
J = (diff'*diff + lambda*(theta'*theta - theta(1)*theta(1)))/(2*m);
grad = (X'*diff)/m;
grad(2:end) = grad(2:end) + theta(2:end)*(lambda/m);
% =========================================================================

grad = grad(:);
end


function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%
for i = 1:m
    [theta] = trainLinearReg(X(1:i, :), y(1:i), lambda);
    [error_train(i), ~] =...
        linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
    [error_val(i), ~] =...
        linearRegCostFunction(Xval, yval, theta, 0);
end;
% =========================================================================
end


function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 
for i = 1:p
    X_poly(:, i) = X.^i;
end;
% =========================================================================

end


function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%
for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    [theta] = trainLinearReg(X, y, lambda);
    [error_train(i), ~] =...
        linearRegCostFunction(X, y, theta, 0);
    [error_val(i), ~] =...
        linearRegCostFunction(Xval, yval, theta, 0);
end;
% =========================================================================

end
{% endhighlight %}
<p align="justify">
learning curve
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/6_1_4_1.png"/></center>
<br>
</p>

### 6.2 Machine Learning System Design
#### 6.2.1 Building a Spam Classifier
<p align="justify">
<b>Prioritizing What to Work On</b><br>
Supervised learning. x = features of email. y = spam (1) or not spam (0).<br>
Features x: choose 100 words indicative of spam / not spam.<br><br>

<b>Error Analysis</b><br>
$\bigstar$ Recommended approach<br>
-- Start with a simple algorithm that you can implement quickly. Implement it and test it on your cross-validation data.<br>
-- Plot learning curves to decide if more data, more features, etc. are likely to help.<br>
-- Error analysis: manually examine the examples (in cross-validation set) that your algorithm made errors on. See if you splt any systematic trend in what type of examples it is making errors on.<br><br>
</p>

#### 6.2.2 Handling Skewed Data
<p align="justify">
<b>Error Metrics for Skewed Classes</b><br>
Take cancer classification for example, we got 1% error on test set. Only 0.5% (skewed classes) of patients have cancer.<br>
Precision: of all patients where we predict y = 1
$$\frac{\text{True positive}}{\text{number of predicted patients}} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
Recall: of all pateints that actually have cancer
$$\frac{\text{True positive}}{\text{number of actual positive pateints}} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
T and F denote actual class, while P and N represent predicted class
<table class="c">
  <tr><th></th><th>T</th><th>F</th></tr>
  <tr><td>P</td><td>True Positive</td><td>False Positive</td></tr>
  <tr><td>N</td><td>False Negative</td><td>True Negative</td></tr>
</table><br>
</p>
<p align="justify">
<b>Trading Off Precision and Recall</b><br>
Suppose we want to predict y = 1 (cancer) only if very confident, high precision, lower recall<br>
Suppose we want to avoid missing to many cases of cancer (avoid false negative), high recall, lower precision.<br>
F-score
$$\text{F}_{1}\text{ score} = \frac{2PR}{P + R}, \quad \text{P is precision and R is recall}$$<br>
</p>

#### 6.2.3 Using Large Data Sets
<p align="justify">
<b>Data For Machine Learning</b><br>
Use a learning algorithm with many parameters (e.g. logistic regression / linear regression with many features; neural network with many hidden units). Use a large training set is unlikely to overfit.<br><br>
</p>

#### 6.2.4 Quiz
<p align="justify">
<b>1.</b><br>
You are working on a spam classification system using regularized logistic regression. "Spam" is a positive class (y = 1) and "not spam" is the negative class (y = 0). You have trained your classifier and there are m = 1000 examples in the cross-validation set. The chart of predicted class vs. actual class is:
<table class="c">
  <tr><th></th><th>Actual Class: 1</th><th>Actual Class: 0</th></tr>
  <tr><td>Predicted Class: 1</td><td>85</td><td>890</td></tr>
  <tr><td>Predicted Class: 0</td><td>15</td><td>10</td></tr>
</table><br>
</p>
<p align="justify">
For reference:<br>
Accuracy = (true positives + true negatives) / (total examples)<br>
Precision = (true positives) / (true positives + false positives)<br>
Recall = (true positives) / (true positives + false negatives)<br>
F1 score = (2 * precision * recall) / (precision + recall)<br>
What is the classifier's precision (as a value from 0 to 1)? Enter your answer in the box below. If necessary, provide at least two values after the decimal point.<br>
<b>Answer</b>: 0.0872.<br><br>

<b>2.</b><br>
Suppose a massive dataset is available for training a learning algorithm. Training on a lot of data is likely to give good performance when two of the following conditions hold true. Which are the two?<br>
A. We train a learning algorithm with a large number of parameters (that is able to learn/represent fairly complex functions).<br>
B. We train a learning algorithm with a small number of parameters (that is thus unlikely to overfit).<br>
C. The features x contain sufficient information to predict y accurately. (For example, one way to verify this is if a human expert on the domain can confidently predict y when given only x).<br>
D. We train a model that does not use regularization.<br>
<b>Answer</b>: A, C.<br><br>

<b>3.</b><br>
Suppose you have trained a logistic regression classifier which is outputing $h_{\theta}$(x) Currently, you predict 1 if $h_{\theta}$(x) ≥threshold, and predict 0 if $h_{\theta}$(x) < threshold, where currently the threshold is set to 0.5. Suppose you increase the threshold to 0.7. Which of the following are true? Check all that apply.<br>
A. The classifier is likely to now have higher precision.<br>
B. The classifier is likely to have unchanged precision and recall, and thus the same F1 score.<br>
C. The classifier is likely to have unchanged precision and recall, but higher accuracy.<br>
D. The classifier is likely to now have higher recall.<br>
<b>Answer</b>: A.<br><br>

<b>4.</b><br>
Suppose you are working on a spam classifier, where spam emails are positive examples (y=1) and non-spam emails are negative examples (y=0). You have a training set of emails in which 99% of the emails are non-spam and the other 1% is spam. Which of the following statements are true? Check all that apply.<br>
A. If you always predict spam (output y=1), your classifier will have a recall of 100% and precisionof 1%.<br>
B. If you always predict non-spam (outputy=0), your classifier will have an accuracy of 99%.<br>
C. If you always predict non-spam (outputy=0), your classifier will have a recall of 0%.<br>
D. If you always predict spam (output y=1), your classifier will have a recall of 0% and precision of 99%.<br>
<b>Answer</b>: A, B, C.<br><br>

<b>5.</b><br>
Which of the following statements are true? Check all that apply.<br>
A. On skewed datasets (e.g., when there are more positive examples than negative examples), accuracyis not a good measure of performance and you should instead use F1 score based on the precision and recall.<br>
B. After training a logistic regression classifier, you must use 0.5 as your threshold for predicting whether an example is positive or negative.<br>
C. It is a good idea to spend a lot of time collecting a large amount of data before building your first version of a learning algorithm.<br>
D. If your model is underfitting the training set, then obtaining more data is likely to help.<br>
E. Using a very large training set makes it unlikely for model to overfit the training data.<br>
<b>Answer</b>: A, E.<br><br>
</p>

## 7. Week 7
### 7.1 Support Vector Machines
#### 7.1.1 Large Margin Classification
<p align="justify">
<b>Optimization Objective</b><br>
Logistic regression
$$\min_{\theta} -\frac{1}{m} [\sum_{i=1}^{m} y^{(i)} \log(h_{\theta}(x^{(i)})) + (1- y^{(i)}) \log(1-h_{\theta}(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^{n} \theta_{j}^{2}$$
Support vector machine
$$
\begin{aligned}
& \min_{\theta} \frac{1}{m} [\sum_{i=1}^{m} y^{(i)} \text{Cost}_{1}(\theta^{T}x^{(i)}) + (1 - y^{(i)}) \text{Cost}_{0}(\theta^{T}x^{(i)}) ]+ \frac{\lambda}{m} \sum_{j=1}^{n} \theta_{j}^{2} = \\
& \min_{\theta} C [\sum_{i=1}^{m} y^{(i)} \text{Cost}_{1}(\theta^{T}x^{(i)}) + (1 - y^{(i)}) \text{Cost}_{0}(\theta^{T}x^{(i)}) ] + \frac{1}{2} \sum_{j=1}^{n} \theta_{j}^{2}
\end{aligned}
$$
Hypothesis
$$
h_{\theta}(x) =
\begin{cases}
1 , \quad & \theta^{T}x \geq 0 \\
0, \quad & \text{otherwise}
\end{cases}
$$

<b>Large Margin Intuition</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/7_1_1_1.png"/></center>
</p>
<p align="justify">
If y = 1, we want $\theta^{T}x \geq 1$; if y = 0, we want $\theta^{T}x \leq -1$.<br><br>

<b>Mathematics Behind Large Margin Classification</b><br>
SVM decision boundary
$$
\begin{aligned}
& \min_{\theta} \frac{1}{2} \sum_{j=1}^{n} \theta_{j}^{2} \\
& \text{s.t.}
\begin{cases}
\theta^{T} x^{(i)} \geq 1, & \quad y^{(i)} = 1 \\
\theta^{T} x^{(i)} \leq -1, & \quad y^{(i)} = 0
\end{cases}
\end{aligned}
$$<br>
</p>

#### 7.1.2 Kernels
<p align="justify">
<b>Kernels and Similarity</b><br>
$l^{(1)}$ is a landmark
$$
\begin{aligned}
f_{1} & = \text{similarity} (x, l^{(1)}) \\
& = \exp(-\frac{\left \| x - l^{(1)} \right \|^{2}}{2 \sigma^{2}}) \\
& = \exp(-\frac{\sum_{j=1}^{n} (x_{j} - l_{j}^{(l)})^{2}}{2 \sigma^{2}})
\end{aligned}
$$
If $x \approx l^{(1)}$, $f_{1} \approx 1$; if x is far from $l^{(1)}$, $f_{1} \approx 0$<br><br>

<b>SVM with Kernels</b><br>
Given ($x^{(1)}$, y^{(1)}), ($x^{(2)}$, y^{(2)}), ..., ($x^{(m)}$, y^{(m)})<br>
Choose $l^{(1)}$ = x^{(1)}, $l^{(2)}$ = x^{(2)}, ..., $l^{(m)}$ = x^{(m)}<br>
Given example x:<br>
$$
\begin{aligned}
& f_{1} = \text{similarity}(x, l^{(1)}) \\
& f_{2} = \text{similarity}(x, l^{(2)}) \\
& \cdots
\end{aligned}
$$
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/7_1_2_1.png"/></center>
</p>
<p align="justify">
Hypothesis: given x, compute features $f \in \mathbb{R}^{m+1}$, predict "y = 1" if $\theta^{T} f \geq 0$.<br>
training:
$$\min_{\theta} C [\sum_{i=1}^{m} y^{(i)} \text{Cost}_{1}(\theta^{T} f^{(i)}) + (1 - y^{(i)}) \text{Cost}_{0}(\theta^{T}f^{(i)}) ] + \frac{1}{2} \sum_{j=1}^{n} \theta_{j}^{2}$$
SVM parameters C (= $\frac{1}{\lambda}$), $\sigma^{2}$<br>
-- large C: lower bias, high variance<br>
-- small C: higher bias, low variance<br>
-- large $\sigma^{2}$: features f vary more smoothly, higher bias, lower variance<br>
-- small $\sigma^{2}$: features f vary less smoothly, lower bias, higher variance<br><br>
</p>

#### 7.1.3 SVMs in Practice
<p align="justify">
<b>Using An SVM</b><br>
Use SVM software package (e.g. liblinear, libsvm, ...) to solve for parameters $\theta$.<br>
Need to specify C and choice of kernel (similarity function)<br>
E.g. No kernel ("linear kernel")<br>
Gaussian kernel
$$f_{i} = \exp(-\frac{\left \| x - l^{(i)} \right \|^{2}}{2 \sigma^{2}}), \quad \text{where } l^{(i)} = x^{(i)}$$
need to choose $\sigma^{2}$.<br>
Note: Do perform feature scaling before using the Gaussian kernel.<br>
Other choices of kernel<br>
Not all similarity function $\text{similarity}(x, l)$ make valid kernels. Need to satisfy technical condition called "Mercer's Theorem" to make sure SVM packages' optimizations run correctly, and do not diverge<br>
For example<br>
-- Polynomial kernel $k(x, l) = (x^{T} l)^{2}$<br>
-- More estoric: string kernel, chi-square kernel, histogram intersection kernel,...<br><br>

<b>Multi-class classification</b><br>
Many SVM packages already have built-in multi-class classification functionality. Otherwise, use one-vs-all method: i class and the others.<br><br>

<b>Logistic regression vs. SVM</b><br>
n is the number of features, m is the number if training examples<br>
If n is small and m is intermediate, use SVM with Gaussian kernel<br>
if n is small and m is large, create / add more features, then use logistic regression or SVM without a kernel.<br><br>
</p>

#### 7.1.4 Quiz
<p align="justify">
<b>1.</b><br>
Suppose you have trained an SVM classifier with a Gaussian kernel, and it learned the following decision boundary on the training set:
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/7_1_4_1.png"/></center>
</p>
<p align="justify">
You suspect that the SVM is underfitting your dataset. Should you try increasing or decreasing C? Increasing or decreasing $\sigma^{2}$?<br>
A. It would be reasonable to try decreasing C. It would also be reasonable to try decreasing $\sigma^{2}$.<br>
B. It would be reasonable to try decreasing C. It would also be reasonable to try increasing $\sigma^{2}$.<br>
C. It would be reasonable to try increasing C. It would also be reasonable to try decreasing $\sigma^{2}$.<br>
D. It would be reasonable to try increasing C. It would also be reasonable to try increasing $\sigma^{2}$.<br>
<b>Answer</b>: C.<br>
Underfit means high bias<br><br>

<b>2.</b><br>
The formula for the Gaussian kernel is given by
$$\text{similarity} (x, l^{(1)}) = \exp(-\frac{\left \| x - l^{(1)} \right \|^{2}}{2 \sigma^{2}})$$
The figure below shows a plot of $f_{1} = \text{similarity}(x, l^{(1)})$ when $\sigma^{2}$ = 1. Which of the following is a plot of $f_{1}$ when $\sigma^{2}$ = 0.25?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/7_1_4_2.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: A.<br><br>

<b>3.</b><br>
The SVM solves
$$\min_{\theta} C [\sum_{i=1}^{m} y^{(i)} \text{Cost}_{1}(\theta^{T}x^{(i)}) + (1 - y^{(i)}) \text{Cost}_{0}(\theta^{T}x^{(i)}) ] + \frac{1}{2} \sum_{j=1}^{n} \theta_{j}^{2}$$
where the functions $\text{Cost}_{0}(z)$ and $\text{Cost}_{1}(z)$ look like this:
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/7_1_1_1.png"/></center>
</p>
<p align="justify">
The first term in the objective is:
$$C [\sum_{i=1}^{m} y^{(i)} \text{Cost}_{1}(\theta^{T}x^{(i)}) + (1 - y^{(i)}) \text{Cost}_{0}(\theta^{T}x^{(i)}) ]$$
This first term will be zero if two of the following four conditions hold true. Which are the two conditions that would guarantee that this term equals zero?<br>
A. For every example with $y^{(i)}$ = 1, we have that $\theta^{T} x^{(i)} \geq 1$<br>
B. For every example with $y^{(i)}$ = 0, we have that $\theta^{T} x^{(i)} \leq 0$<br>
C. For every example with $y^{(i)}$ = 1, we have that $\theta^{T} x^{(i)} \geq 0$<br>
D. For every example with $y^{(i)}$ = 0, we have that $\theta^{T} x^{(i)} \leq -1$<br>
<b>Answer</b>: A, D.<br><br>

<b>4.</b><br>
Suppose you have a dataset with n = 10 features and m = 5000 examples. After training your logistic regression classifier with gradient descent, you find that it has underfit the training set and does not achieve the desired performance on the training or cross validation sets. Which of the following might be promising steps to take? Check all that apply.<br>
A. Create / add new polynomial features.<br>
B. Increase the regularization parameter λ.<br>
C. Use an SVM with a linear kernel, without introducing new features.<br>
D. Use an SVM with a Gaussian Kernel.<br>
<b>Answer</b>: A, D.<br><br>

<b>5.</b><br>
Which of the following statements are true? Check all that apply.<br>
A. If you are training multi-class SVMs with the one-vs-all method, it is not possible to use a kernel.<br>
B. The maximum value of the Gaussian kernel (i.e., $\text{sim}(x, l^{(1)})$) is 1<br>
C. Suppose you have 2D input examples (ie, $x^{(i)} \in \mathbb{R}^{2}$). The decision boundary of the SVM (with the linear kernel) is a straight line.<br>
D. If the data are linearly separable, an SVM using a linear kernel will return the same parameters θ regardless of the chosen value ofC (i.e., the resulting value of θ does not depend on C).<br>
<b>Answer</b>: B, C.<br><br>
</p>

#### 7.1.5 Programming Assignment: Support Vector Machines
{% highlight Matlab %}
function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
diff = x1 - x2;
sim = exp((diff'*diff)/(-2*sigma^2));
% =============================================================
end


function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the
%exercise where you select the optimal (C, sigma) learning parameters to
%use for SVM with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of
%   C and sigma. You should complete this function to return the optimal
%   C and sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
candidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
minError = 1;

for i = 1:8
    for j = 1:8
        C = candidates(i);
        sigma = candidates(j);
        model= svmTrain(X, y, C,...
            @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if error < minError
            minError = error;
            bestC = C;
            bestSigma = sigma;
        end;
    end;
end;
C = bestC;
sigma = bestSigma;
% =========================================================================
end


function word_indices = processEmail(email_contents)
%PROCESSEMAIL preprocesses a the body of an email and
%returns a list of word_indices 
%   word_indices = PROCESSEMAIL(email_contents) preprocesses 
%   the body of an email and returns a list of indices of the 
%   words contained in the email. 
%

% Load Vocabulary
vocabList = getVocabList();

% Init return value
word_indices = [];

% ========================== Preprocess Email ===========================

% Find the Headers ( \n\n and remove )
% Uncomment the following lines if you are working with raw emails with the
% full headers

% hdrstart = strfind(email_contents, ([char(10) char(10)]));
% email_contents = email_contents(hdrstart(1):end);

% Lower case
email_contents = lower(email_contents);

% Strip all HTML
% Looks for any expression that starts with < and ends with > and replace
% and does not have any < or > in the tag it with a space
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

% Handle Numbers
% Look for one or more characters between 0-9
email_contents = regexprep(email_contents, '[0-9]+', 'number');

% Handle URLS
% Look for strings starting with http:// or https://
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');

% Handle Email Addresses
% Look for strings with @ in the middle
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

% Handle $ sign
email_contents = regexprep(email_contents, '[$]+', 'dollar');


% ========================== Tokenize Email ===========================

% Output the email to screen as well
fprintf('\n==== Processed Email ====\n\n');

% Process file
l = 0;
count = 1;
while ~isempty(email_contents)

    % Tokenize and also get rid of any punctuation
    [str, email_contents] = ...
       strtok(email_contents, ...
              [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
   
    % Remove any non alphanumeric characters
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word 
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    try str = porterStemmer(strtrim(str)); 
    catch str = ''; continue;
    end;

    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end

    % Look up the word in the dictionary and add to word_indices if
    % found
    % ====================== YOUR CODE HERE ======================
    % Instructions: Fill in this function to add the index of str to
    %               word_indices if it is in the vocabulary. At this point
    %               of the code, you have a stemmed word from the email in
    %               the variable str. You should look up str in the
    %               vocabulary list (vocabList). If a match exists, you
    %               should add the index of the word to the word_indices
    %               vector. Concretely, if str = 'action', then you should
    %               look up the vocabulary list to find where in vocabList
    %               'action' appears. For example, if vocabList{18} =
    %               'action', then, you should add 18 to the word_indices 
    %               vector (e.g., word_indices = [word_indices ; 18]; ).
    % 
    % Note: vocabList{idx} returns a the word with index idx in the
    %       vocabulary list.
    % 
    % Note: You can use strcmp(str1, str2) to compare two strings (str1 and
    %       str2). It will return 1 only if the two strings are equivalent.
    %
    for i = 1:length(vocabList)
        if strcmp(str, vocabList(i))
            word_indices(count) = i;
            count = count + 1;
            break;
        end;
    end;
    % =============================================================

    % Print to screen, ensuring that the output lines are not too long
    if (l + length(str) + 1) > 78
        fprintf('\n');
        l = 0;
    end
    fprintf('%s ', str);
    l = l + length(str) + 1;

end

% Print footer
fprintf('\n\n=========================\n');

end


function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

% You need to return the following variables correctly.
x = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return a feature vector for the
%               given email (word_indices). To help make it easier to 
%               process the emails, we have have already pre-processed each
%               email and converted each word in the email into an index in
%               a fixed dictionary (of 1899 words). The variable
%               word_indices contains the list of indices of the words
%               which occur in one email.
% 
%               Concretely, if an email has the text:
%
%                  The quick brown fox jumped over the lazy dog.
%
%               Then, the word_indices vector for this text might look 
%               like:
%               
%                   60  100   33   44   10     53  60  58   5
%
%               where, we have mapped each word onto a number, for example:
%
%                   the   -- 60
%                   quick -- 100
%                   ...
%
%              (note: the above numbers are just an example and are not the
%               actual mappings).
%
%              Your task is take one such word_indices vector and construct
%              a binary feature vector that indicates whether a particular
%              word occurs in the email. That is, x(i) = 1 when word i
%              is present in the email. Concretely, if the word 'the' (say,
%              index 60) appears in the email, then x(60) = 1. The feature
%              vector should look like:
%
%              x = [ 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 ... 0 0 0 1 0 ..];
%
%
x(word_indices) = 1;
% =========================================================================

end
{% endhighlight %}
<p align="justify">
<br>
</p>


## 8. Week 8
### 8.1 Unsupervised Learning
#### 8.1.1 Clustering
<p align="justify">
<b>Unsupervised Learning</b><br>
No labels<br>
Training set: $\{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$<br>
Application of clustering: market segmentation, social network analysis, organize computing clusters, astronomical data analysis<br><br>

<b>K-Means Algorithm</b><br>
Randomly initialize K cluster centroids $\mu_{1}$, $\mu_{2}$, ..., $\mu_{K}$ $\in \mathbb{R}^{n}$<br>
Repeat<br>
&emsp;for i = 1 to m<br>
&emsp;&emsp;$c^{(i)}$ = index (from 1 to K) of cluster centroids closest to $x^{(i)}$<br>
&emsp;for k = 1 to K<br>
&emsp;&emsp;$\mu_{k}$ = average of points assigned to cluster k<br><br>

<b>Optimization Objective</b><br>
$c^{(i)}$ = index of cluster (1, 2, ..., K) to which example $x^{(i)}$ is currently assigned<br>
$\mu_{k}$ = cluster centroid k (\mu_{k} \in \mathbb{R}^{n})<br>
$\mu_{c^{(i)}}$ = cluster centroid of cluster to which example $x^{(i)}$ has been assigned<br>
Optimization objective
$$
\begin{aligned}
& \min_{c^{(1)}, ..., c^{(m)}, \mu_{1}, ..., \mu_{K}} J(c^{(1)}, ..., c^{(m)}, \mu_{1}, ..., \mu_{K}) \\
& = \min_{c^{(1)}, ..., c^{(m)}, \mu_{1}, ..., \mu_{K}} \frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} - \mu_{c^{(i)}} \right \|^{2}
\end{aligned}
$$<br>

<b>Random Initialization</b><br>
Should have K < m<br>
Randomly pick K training examples<br>
Set $\mu_{1}, ..., \mu_{K}$ equal to these K examples<br>
For i = 1 to 100 (50 - 1000)<br>
&emsp;Randomly initialize K-means<br>
&emsp;Run K-mean. get $c^{(1)}, ..., c^{(m)}, \mu_{1}, ..., \mu_{K}$<br>
&emsp;Compute cost function $J(c^{(1)}, ..., c^{(m)}, \mu_{1}, ..., \mu_{K})$<br>
Pick clustering that gave the lowest cost $J(c^{(1)}, ..., c^{(m)}, \mu_{1}, ..., \mu_{K})$<br><br>

<b>Choosing the Number of Clusters</b><br>
Elbow method
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/8_1_1_1.png"/></center>
</p>
<p align="justify">
Sometimes, you are running K-means to get clusters to use for some later/downstream purpose. Evaluate K-means based on a metric for how well it performs for that later purpose.<br><br>
</p>

#### 8.1.2 Quiz
<p align="justify">
<b>1.</b><br>
For which of the following tasks might K-means clustering be a suitable algorithm? Select all that apply.<br>
A. Given a database of information about your users, automatically group them into different market segments.<br>
B. Given sales data from a large number of products in a supermarket, figure out which products tend to form coherent groups (say are frequently purchased together) and thus should be put on the same shelf.<br>
C. Given historical weather records, predict the amount of rainfall tomorrow (this would be a real-valued output)<br>
D. Given sales data from a large number of products in a supermarket, estimate future sales for each of these products.<br>
<b>Answer</b>: A, B.<br><br>

<b>2.</b><br>
Suppose we have three cluster centroids
$$
\mu_{1} =
\begin{bmatrix}
1 \\
2
\end{bmatrix} \quad
\mu_{2} =
\begin{bmatrix}
-3 \\
0
\end{bmatrix} \quad
\mu_{3} =
\begin{bmatrix}
4 \\
2
\end{bmatrix} 
$$
Furthermore, we have a training example
$$
x^{(i)} =
\begin{bmatrix}
-2 \\
1
\end{bmatrix} 
$$
After a cluster assignment step, what will $c^{(i)}$ be?<br>
A. $c^{(i)}$ is not assigned<br>
B. $c^{(i)}$ = 3<br>
C. $c^{(i)}$ = 2<br>
D. $c^{(i)}$ = 1<br>
<b>Answer</b>: C.<br>
$x^{(i)}$ is closest to $\mu_{2}$, class = 2<br><br>

<b>3.</b><br>
K-means is an iterative algorithm, and two of the following steps are repeatedly carried out in its inner-loop. Which two?<br>
A. The cluster assignment step, where the parameters $c^{(i)}$ are updated.<br>
B. Move the cluster centroids, where the centroids $\mu_{k}$ are updated.<br>
C. Using the elbow method to choose K.<br>
D. Feature scaling, to ensure each feature is on a comparable scale to the others.<br>
<b>Answer</b>: A, B.<br><br>

<b>4.</b><br>
Suppose you have an unlabeled dataset $\{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$. You run K-means with 50 different random initializations, and obtain 50 different clusterings of the data. What is the recommended way for choosing which one of these 50 clusterings to use?<br>
A. The only way to do so is if we also have labels $y^{(i)}$ for our data.<br>
B. Always pick the final (50th) clustering found, since by that time it is more likely to have converged to a good solution.<br>
C. The answer is ambiguous, and there is no good way of choosing.<br>
D. For each of the clusterings, compute $\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} - \mu_{c^{(i)}} \right \|^{2}$, and pick the one that minimizes this.<br>
<b>Answer</b>: D.<br><br>

<b>5.</b><br>
Which of the following statements are true? Select all that apply.<br>
A. For some datasets, the "right" or "correct" value of K (the number of clusters) can be ambiguous, and hard even for a human expert looking carefully at the data to decide.<br>
B. The standard way of initializing K-means is setting $\mu_{1} = ... = \mu_{k}$ to be equal to a vector of zeros.<br>
C. Since K-Means is an unsupervised learning algorithm, it cannot overfit the data, and thus it is always better to have as large a number of clusters as is computationally feasible.<br>
D. If we are worried about K-means getting stuck in bad local optima, one way to ameliorate (reduce) this problem is if we try using multiple random initializations.<br>
<b>Answer</b>: A, D.<br><br>
</p>

### 8.2 Dimensionality Reduction
#### 8.2.1 Motivation
<p align="justify">
<b>Data Compression</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/8_2_1_1.png"/></center>
<br>
</p>

#### 8.2.2 Principal Component Analysis
<p align="justify">
<b>Principal Component Analysis Problem Formulation</b><br>
Reduce from 2-dimension to 1-dimension: find a direction (a vector $u^{(1)} \in \mathbb{R}^{n}$) onto which to project the data so as to minimize the projection error.<br>
Reduce from n-dimension to k-dimension: find k vectors $u^{(1)}, u^{(2)}, ..., u^{(k)}$ onto which to project the data, so as to minimize the projection error.<br><br>

<b>Principal Component Analysis Algorithm</b><br>
Training set: $x^{(1)}$, $x^{(2)}$, ..., $x^{(m)}$<br>
preprocessing (feature scaling / mean normalization)
$$\mu_{j} = \frac{1}{m} \sum_{i=1}^{m} x_{j}^{(i)}$$
Replace each $x_{j}^{(i)}$ with $x_{j} - \mu_{j}$<br>
$$x_{j}^{(i)} \leftarrow \frac{x_{j}^{(i)} - \mu_{j}}{s_{j}} , \quad \text{where } s_{j} = \max(x_{j}) - \min(x_{j})$$
If different features on different scales, scale features to have comparable range of values.<br><br>
Reduce data from n-dimension to k-dimension. Compute "covariance matrix"
$$\Sigma = \frac{1}{m} \sum_{i=1}^{n} (x^{(i)})(x^{(i)})^{T}$$
The $\Sigma$ is a n-by-n matrix<br>
In matlab:
</p>
{% highlight Matlab %}
[U, S, V] = svd(Sigma);
Ureduce = U(:, 1:k);
z = Ureduce'*x;
{% endhighlight %}
<p align="justify">
We can get
$$
U =
\begin{bmatrix}
| & | &   & | \\
u^{(1)} & u^{(2)} & \cdots & u^{(n)} \\
| & | &   & | 
\end{bmatrix} \in \mathbb{R}^{n \times n}
$$
After getting U matrix, we can project our data X onto U. Note X is $m \times n$, U is $n \times n$ and Z is $m \times n$.
</p>
{% highlight Matlab %}
Z = X*U(:, 1:K);
{% endhighlight %}
<p align="justify">
By contrast, we can recover the original data
</p>
{% highlight Matlab %}
X_recover = Z*U(:, 1:K)';
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 8.2.3 Applying PCA
<p align="justify">
<b>Reconstruction from Compressed Representation</b><br>
$$z = U_{\text{reduce}}^{T} x$$<br>

<b>Choosing the Number of Principal Components</b><br>
Average squared projection error
$$\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} - x_{\text{approx}}^{(i)} \right \|^{2}$$
Total variation in the data
$$\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} \right \|^{2}$$
typically, choose k to be smallest value so that
$$\frac{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} - x_{\text{approx}}^{(i)} \right \|^{2}}{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} \right \|^{2}} \leq 0.01$$
Which means 99% of variance is retained.<br>
Lucily, we can use matrix S to calculate the fraction above.
$$\frac{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} - x_{\text{approx}}^{(i)} \right \|^{2}}{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} \right \|^{2}} = 1 - \frac{\sum_{i=1}^{k} S_{ii}}{\sum_{i=1}^{n} S_{ii}} \leq 0.01$$
We pick the smallest k to satisfy the condition above.<br><br>

<b>Applying PCA</b><br>
$\bigstar$ Compression<br>
-- reduce memory / disk needed to store data<br>
-- spped up learning algorithm<br>
$\bigstar$ Visualization<br>
$\bigstar$ Bad use of PCA: to prevent overfitting. Use regularization instead.<br><br>
</p>

#### 8.2.4 Quiz
<p align="justify">
<b>1.</b><br>
Consider the following 2D dataset. Which of the following figures correspond to possible values that PCA may return for $u^{(1)}$ (the first eigenvector / first principal component)? Check all that apply (you may have to check more than one figure).
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/8_2_4_1.png"/></center>
</p>
<p align="justify">
<b>Answer</b>: A, B.<br><br>

<b>2.</b><br>
Which of the following is a reasonable way to select the number of principal components k? (Recall that n is the dimensionality of the input data and m is the number of input examples.)<br>
A. Choose k to be the smallest value so that at least 1% of the variance is retained.<br>
B. Choose k to be 99% of n (i.e., k=0.99∗n, rounded to the nearest integer).<br>
C. Choose the value of k that minimizes the approximation error $\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} - x_{\text{approx}}^{(i)} \right \|^{2}$.<br>
D. Choose k to be the smallest value so that at least 99% of the variance is retained.<br>
<b>Answer</b>: D.<br><br>

<b>3.</b><br>
Suppose someone tells you that they ran PCA in such a way that "95% of the variance was retained." What is an equivalent statement to this?
$$
\begin{aligned}
& \text{A.} \quad \frac{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} - x_{\text{approx}}^{(i)} \right \|^{2}}{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} \right \|^{2}} \geq 0.05 \\
& \text{B.} \quad \frac{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} - x_{\text{approx}}^{(i)} \right \|^{2}}{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} \right \|^{2}} \geq 0.95 \\
& \text{C.} \quad \frac{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} - x_{\text{approx}}^{(i)} \right \|^{2}}{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} \right \|^{2}} \leq 0.95 \\
& \text{D.} \quad \frac{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} - x_{\text{approx}}^{(i)} \right \|^{2}}{\frac{1}{m} \sum_{i=1}^{m} \left \| x^{(i)} \right \|^{2}} \leq 0.05 
\end{aligned}
$$
<b>Answer</b>: D.<br><br>

<b>4.</b><br>
Which of the following statements are true? Check all that apply.<br>
A. PCA can be used only to reduce the dimensionality of data by 1 (such as 3D to 2D, or 2D to 1D).<br>
B. If the input features are on very different scales, it is a good idea to perform feature scaling before applying PCA.<br>
C. Feature scaling is not useful for PCA, since the eigenvector calculation (such as using Octave's svd(Sigma) routine) takes care of this automatically.<br>
D. Given an input $x \in \mathbb{R}^{n}$, PCA compresses it to a lower-dimensional vector x \in \mathbb{R}^{k}.<br>
<b>Answer</b>: B, D.<br><br>

<b>5.</b><br>
Which of the following are recommended applications of PCA? Select all that apply.<br>
A. To get more features to feed into a learning algorithm.<br>
B. Data compression: Reduce the dimension of your data, so that it takes up less memory / disk space.<br>
C. Preventing overfitting: Reduce the number of features (in a supervised learning problem), so that there are fewer parameters to learn.<br>
D. Data visualization: Reduce data to 2D (or 3D) so that it can be plotted.<br>
<b>Answer</b>: B, D.<br><br>
</p>

#### 8.2.5 Programming Assignment: K-Means Clustering and PCA
{% highlight Matlab %}
function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X, 1);
for i = 1:m
    dist = repmat(X(i, :), K, 1) - centroids;
    [~, id] = min(sum(dist.^2, 2));
    idx(i) = id;
end;
% =============================================================

end


function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
for k = 1:K
    centroids(k, :) = mean(X(idx==k, :));
end;
% =============================================================

end


function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%
[U, S, V] = svd((X'*X)/m);
% =========================================================================

end


function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%
Z = X*U(:, 1:K);
% =============================================================

end


function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%
X_rec = Z*U(:, 1:K)';
% =============================================================

end
{% endhighlight %}
<p align="justify">
<br>
</p>


## 9. Week 9
### 9.1 Anomaly Detection
#### 9.1.1 Density Estimation
<p align="justify">
<b>Problem Motivation</b><br>
Fraud detection:<br>
$x^{(i)}$ = features of user i's activities<br>
model p(x) from data<br>
identify unusual users by checking which have p(x) < $\epsilon$.<br><br>

<b>Gaussian Distribution</b><br>
x $\in \mathbb{R}$ if x is a distributed Gaussian with mean $\mu$, variance $\sigma^{2}$.<br><br>

<b>Algorithm</b><br>
1) choose features $x_{i}$ that you think might be indicative of anomalous examples.<br>
2) fit parameters $\mu_{1}, ..., \mu_{n}, \sigma_{1}^{2}, ..., \sigma_{n}^{2}$
$$
\begin{aligned}
& \mu_{j} = \frac{1}{m} \sum_{i=1}^{m} x_{j}^{(i)} \\
& \sigma_{j}^{2} = \frac{1}{m} \sum_{i=1}^{m} (x_{j}^{(i)} - \mu_{j})^{2}
\end{aligned}
$$
3) given new example x, compute p(x)
$$p(x) = \prod_{j=1}^{n} p(x_{j}; \mu_{j}, \sigma_{j}^{2}) = \prod_{j=1}^{n} \frac{1}{\sqrt{2 \pi} \sigma_{j}} \exp(-\frac{(x_{j} - \mu_{j})^{2}}{2 \sigma_{j}^{2}})$$
Anomaly if p(x) < $\epsilon$<br><br>
</p>

#### 9.1.2 Building an Anomaly Detection System
<p align="justify">
<b>Developing and Evaluating an Anomaly Detection System</b><br>
$\bigstar$ The importance of real-number evaluation<br>
When developing a learning algorithm (choosing features etc.) making decision is much easier if we have a way of evaluating our learning algorithm. Assuming we have some labeled data, of anomalous and non-anomalous examples (0 or 1).<br><br>

<b>Anomaly Detection vs. Supervised Learning</b><br>
$\bigstar$ Anomaly detection<br>
-- very small number of positive examples; large number of negative examples<br>
-- many different types of anomaliesl hard for any algorithm to learn from positive examples what the anomalies look like<br>
-- future anomalies may look nothing like any of the anomalous examples we've seen so far<br>
$\bigstar$ Supervised learning<br>
-- large number of positive and negative examples<br>
-- enough positive examples for algorithm to get a sense of what positive examples are like<br>
-- future positive examples are likely to be similar to ones in training set.<br><br>
</p>

#### 9.1.3 Multivariate Gaussian Distribution (Optional)
<p align="justify">
<b>Multivariate Gaussian Distribution</b><br>
$x \in \mathbb{R}^{n}$. Model p(x) all in one go. Parameters $\mu \in \mathbb{R}^{n}, \quad \Sigma \in \mathbb{R}^{n \times n}$ (covariance matrix).<br><br>

<b>Anomaly Detection using the Multivariate Gaussian Distribution</b><br>
1) fit model p(x) by setting Parameter fitting given training set $\{x^{(1)}, ..., x^{(m)}\}$
$$\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}, \quad \Sigma = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)(x^{(i)} - \mu)^{T}$$
2) given a new example x, compute
$$p(x; \mu, \Sigma).= \frac{1}{(2 \pi)^{\frac{n}{2}} \left | \Sigma \right |^{\frac{1}{2}}} \exp(-\frac{1}{2} (x - \mu)^{T} \Sigma^{-1}(x - \mu))$$
Flag an anomaly if p(x) < $\epsilon$
$\bigstar$ Multivariate gaussian<br>
-- automatically captures correlations between features<br>
-- computationally more expensive<br>
-- must have m > n, or else $\Sigma$ is non-invertible<br>
$\bigstar$ original model<br>
$$p(x_{1}; \mu_{1}, \sigma_{1}^{2}) \times \cdots \times p(x_{n}; \mu_{n}, \sigma_{n}^{2})$$
-- manually create features to capture anomalies where $x_{1}$, $x_{2}$ take unusual combinations of values.<br>
-- computationally cheaper<br>
-- ok even if m is small<br><br>
</p>

#### 9.1.4 Quiz
<p align="justify">
<b>1.</b><br>
For which of the following problems would anomaly detection be a suitable algorithm?<br>
A. Given data from credit card transactions, classify each transaction according to type of purchase (for example: food, transportation, clothing).<br>
B. Given an image of a face, determine whether or not it is the face of a particular famous individual.<br>
C. From a large set of primary care patient records, identify individuals who might have unusual health conditions.<br>
D. Given a dataset of credit card transactions, identify unusual transactions to flag them as possibly fraudulent.<br>
<b>Answer</b>: C, D.<br><br>

<b>2.</b><br>
Suppose you have trained an anomaly detection system for fraud detection, and your system that flags anomalies when p(x) is less than ε, and you find on the cross-validation set that it mis-flagging far too many good transactions as fradulent. What should you do?<br>
A. Increase ε<br>
B. Decrease ε<br>
<b>Answer</b>: B.<br><br>

<b>3.</b><br>
Suppose you are developing an anomaly detection system to catch manufacturing defects in airplane engines. You model uses
$$p(x) = \prod_{j=1}^{n} p(x_{j}; \mu_{j}, \sigma_{j}^{2})$$
You have two features $x_{1}$ = vibration intensity, and $x_{2}$ = heat generated. Both $x_{1}$ and $x_{2}$ take on values between 0 and 1 (and are strictly greater than 0), and for most "normal" engines you expect that $x_{1} \approx x_{2}$ One of the suspected anomalies is that a flawed engine may vibrate very intensely even without generating much heat (large $x_{1}$, small $x_{2}$), even though the particular values of $x_{1}$ and $x_{2}$may not fall outside their typical ranges of values. What additional feature $x_{3}$ should you create to capture these types of anomalies:
$$
\begin{aligned}
& \text{A.} \quad x_{3} = x_{1}^{2} \times x_{2} \\
& \text{B.} \quad x_{3} = \frac{x_{1}}{x_{2}} \\
& \text{C.} \quad x_{3} = x_{1} \times x_{2} \\
& \text{D.} \quad x_{3} = x_{1} + x_{2}
\end{aligned}
$$
<b>Answer</b>: B.<br><br>

<b>4.</b><br>
Which of the following are true? Check all that apply.<br>
A. When developing an anomaly detection system, it is often useful to select an appropriate numerical performance metric to evaluate the effectiveness of the learning algorithm.<br>
B. In anomaly detection, we fit a model p(x) to a set of negative (y=0) examples, without using any positive examples we may have collected of previously observed anomalies.<br>
C. When evaluating an anomaly detection algorithm on the cross validation set (containing some positive and some negative examples), classification accuracy is usually a good evaluation metric to use.<br>
D. In a typical anomaly detection setting, we have a large number of anomalous examples, and a relatively small number of normal/non-anomalous examples.<br>
<b>Answer</b>: A, B.<br><br>

<b>5.</b><br>
You have a 1-D dataset $\{x^{(1)}, ..., x^{(m)}\}$ and you want to detect outliers in the dataset. You first plot the dataset and it looks like this:
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/9_1_4_1.png"/></center>
</p>
<p align="justify">
Suppose you fit the gaussian distribution parameters $\mu_{1}$ and $\sigma_{1}^{2}$ to this dataset. Which of the following values for $\mu_{1}$ and $\sigma_{1}^{2}$ might you get?
$$
\begin{aligned}
& \text{A.} \quad \mu_{1} = -3, \quad \sigma_{1}^{2} = 4 \\
& \text{B.} \quad \mu_{1} = -6, \quad \sigma_{1}^{2} = 4 \\
& \text{C.} \quad \mu_{1} = -3, \quad \sigma_{1}^{2} = 2 \\
& \text{D.} \quad \mu_{1} = -6, \quad \sigma_{1}^{2} = 2
\end{aligned}
$$
<b>Answer</b>: A.<br><br>
</p>

### 9.2 Recommender Systems
#### 9.2.1 Predicting Movie Ratings
<p align="justify">
<b>Problem Formulation</b><br>
r(i, j) = 1 if user j has rated movie i (0 otherwise)<br>
$y^{(i, j)}$ = rate by user j on movie i (if defined)<br>
$\theta^{(j)}$ = parameter vector for user j<br>
$x^{(i)}$ = feature vector movie i<br>
For user j, movie i, predict rating $(\theta^{(j)})^{T}(x^{(i)})$<br>
$m^{(j)}$ = no. of movies rated by user j<br>
To learn $\theta^{(j)}$
$$\min_{\theta^{(j)}} \frac{1}{2} \sum_{i: r(i, j) = 1} ((\theta^{(j)})^{T}(x^{(i)}) - y^{(i, j)})^{2} + \frac{\lambda}{2} \sum_{k=1}^{n} (\theta_{k}^{(j)})^{2}$$
To learn $\theta^{(1)}$, $\theta^{(2)}$, ..., $\theta^{(ij)}$
$$\min_{\theta^{(1)}, ..., \theta^{(n_{u})}} \frac{1}{2} \sum_{j=1}^{n_{u}} \sum_{i: r(i, j) = 1} ((\theta^{(j)})^{T}(x^{(i)}) - y^{(i, j)})^{2} + \frac{\lambda}{2} \sum_{j=1}^{n_{u}} \sum_{k=1}^{n} (\theta_{k}^{(j)})^{2}$$
Gradient descent update
$$
\begin{aligned}
& \theta_{k}^{(j)} = \theta_{k}^{(j)} - \alpha \sum_{i: r(i, j) = 1} ((\theta^{(j)})^{T}(x^{(i)}) - y^{(i, j)}) x_{k}^{(i)}, & \quad k = 0 \\
& \theta_{k}^{(j)} = \theta_{k}^{(j)} - \alpha (\sum_{i: r(i, j) = 1} ((\theta^{(j)})^{T}(x^{(i)}) - y^{(i, j)}) x_{k}^{(i)} + \lambda \theta_{k}^{(j)}), & \quad k \neq 0 
\end{aligned}
$$<br>
</p>

#### 9.2.2 Collaborative Filtering
<p align="justify">
<b>Collaborative Filtering</b><br>
$\bigstar$ given $x^{(1)}$, ..., $x^{(n_{m})}$ (and movie rating) can estimate $\theta^{(1)}$, ..., $\theta^{(n_{ij})}$
$$\min_{\theta^{(1)}, ..., \theta^{(n_{u})}} \frac{1}{2} \sum_{j=1}^{n_{u}} \sum_{i: r(i, j) = 1} ((\theta^{(j)})^{T}(x^{(i)}) - y^{(i, j)})^{2}  + \frac{\lambda}{2} \sum_{j=1}^{n_{u}} \sum_{k=1}^{n} (\theta_{k}^{(j)})^{2}$$
$\bigstar$ given $\theta^{(1)}$, ..., $\theta^{(n_{ij})}$, can estimate $x^{(1)}$, ..., $x^{(n_{m})}$
$$\min_{x^{(1)}, ..., x^{(n_{m})}} \frac{1}{2} \sum_{i=1}^{n_{m}} \sum_{j: r(i, j) = 1} ((\theta^{(j)})^{T}(x^{(i)}) - y^{(i, j)})^{2}  + \frac{\lambda}{2} \sum_{i=1}^{n_{m}} \sum_{k=1}^{n} (x_{k}^{(i)})^{2}$$

<b>Collaborative Filtering Algorithm</b><br>
minimizing $x^{(1)}$, ..., $x^{(n_{m})}$ and $\theta^{(1)}$, ..., $\theta^{(n_{ij})}$ simultaneously
$$J(x^{(1)}, ..., x^{(n_{m})}, \theta^{(1)}, ..., \theta^{(n_{u})}) = \frac{1}{2} \sum_{(i, j): r(i, j) = 1} ((\theta^{(j)})^{T}(x^{(i)}) - y^{(i, j)})^{2}  + \frac{\lambda}{2} \sum_{i=1}^{n_{m}} \sum_{k=1}^{n} (x_{k}^{(i)})^{2} + \frac{\lambda}{2} \sum_{j=1}^{n_{u}} \sum_{k=1}^{n} (\theta_{k}^{(j)})^{2}$$
Algorithm<br>
1) initialize $x^{(1)}$, ..., $x^{(n_{m})}$ and $\theta^{(1)}$, ..., $\theta^{(n_{ij})}$ to small random values<br>
2) minimize $J(x^{(1)}, ..., x^{(n_{m})}, \theta^{(1)}, ..., \theta^{(n_{ij})})$ using gradient descent for every j = 1, ..., $n_{u}$, i = 1, ..., $n_{m}$:
$$
\begin{aligned}
& x_{k}^{(i)} = x_{k}^{(i)} - \alpha (\sum_{j:r(i, j) = 1} ((\theta^{(j)})^{T}(x^{(i)}) - y^{(i, j)}) \theta_{k}^{(j)} + \lambda x_{k}^{(i)}) \\
& \theta_{k}^{(j)} = \theta_{k}^{(j)} - \alpha (\sum_{i:r(i, j) = 1} ((\theta^{(j)})^{T}(x^{(i)}) - y^{(i, j)}) x_{k}^{(i)} + \lambda \theta_{k}^{(j)})
\end{aligned}
$$
3) for a user with parameters $\theta$ and a movie with leanred features x, predict a star rating of $\theta^{T}x$ 
</p>

#### 9.2.3 Low Rank Matrix Factorization
<p align="justify">
<b>Vectorization: Low Rank Matrix Factorization</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/9_2_3_1.png"/></center>
<br>
</p>

#### 9.2.4 Quiz
<p align="justify">
<b>1.</b><br>
Suppose you run a bookstore, and have ratings (1 to 5 stars) of books. Your collaborative filtering algorithm has learned a parameter vector $\theta^{(j)}$ for user j, and a feature vector $x^{(i)}$ for each book. You would like to compute the "training error", meaning the average squared error of your system's predictions on all the ratings that you have gotten from your users. Which of these are correct ways of doing so (check all that apply)? For this problem, let m be the total number of ratings you have gotten from your users. (Another way of saying this is that
$$m = \sum_{i=1}^{n_{m}} \sum_{j=1}^{n_{u}} r(i, j)$$
[Hint: Two of the four options below are correct.]
$$
\begin{aligned}
& \text{A.} \quad \frac{1}{m} \sum_{(i, j): r(i, j) = 1} ((\theta^{(j)})^{T}x^{(i)} - y^{(i, j)})^{2} \\
& \text{B.} \quad \frac{1}{m} \sum_{i=1}^{n_{m}} \sum_{j: r(i, j) = 1} (\sum_{k=1}^{n} (\theta^{(j)})_{k} x_{k}^{(i)} - y^{(i, j)})^{2} \\
& \text{C.} \quad \frac{1}{m} \sum_{(i, j): r(i, j) = 1} ((\theta^{(j)})^{T}x^{(i)} - r(i, j))^{2} \\
& \text{D.} \quad \frac{1}{m} \sum_{j=1}^{n_{u}} \sum_{i: r(i, j) = 1} (\sum_{k=1}^{n} (\theta^{(k)})_{j} x_{i}^{(k)} - y^{(i, j)})^{2}
\end{aligned}
$$
<b>Answer</b>: A, B.<br><br>

<b>2.</b><br>
In which of the following situations will a collaborative filtering system be the most appropriate learning algorithm (compared to linear or logistic regression)?<br>
A. You manage an online bookstore and you have the book ratings from many users. You want to learn to predict the expected sales volume (number of books sold) as a function of the average rating of a book.<br>
B. You've written a piece of software that has downloaded news articles from many news websites. In your system, you also keep track of which articles you personally like vs. dislike, and the system also stores away features of these articles (e.g., word counts, name of author). Using this information, you want to build a system to try to find additional new articles that you personally will like.<br>
C. You run an online news aggregator, and for every user, you know some subset of articles that the user likes and some different subset that the user dislikes. You'd want to use this to find other articles that the user likes.<br>
D. You manage an online bookstore and you have the book ratings from many users. For each user, you want to recommend other books she will enjoy, based on her own ratings and the ratings of other users.<br>
<b>Answer</b>: C, D.<br><br>

<b>3.</b><br>
You run a movie empire, and want to build a movie recommendation system based on collaborative filtering. There were three popular review websites (which we'll call A, B and C) which users to go to rate movies, and you have just acquired all three companies that run these websites. You'd like to merge the three companies' datasets together to build a single/unified system. On website A, users rank a movie as having 1 through 5 stars. On website B, users rank on a scale of 1 - 10, and decimal values (e.g., 7.5) are allowed. On website C, the ratings are from 1 to 100. You also have enough information to identify users/movies on one website with users/movies on a different website. Which of the following statements is true?<br>
A. You can combine all three training sets into one without any modification and expect high performance from a recommendation system.<br>
B. You can combine all three training sets into one as long as your perform mean normalization and feature scaling after you merge the data.<br>
C. You can merge the three datasets into one, but you should first normalize each dataset separately by subtracting the mean and then dividing by (max - min) where the max and min (5-1) or (10-1) or (100-1) for the three websites respectively.<br>
D. It is not possible to combine these websites' data. You must build three separate recommendation systems.<br>
<b>Answer</b>: C.<br><br>

<b>4.</b><br>
Which of the following are true of collaborative filtering systems? Check all that apply.<br>
A. Even if each user has rated only a small fraction of all of your products (so r(i,j)=0 for the vast majority of (i,j) pairs), you can still build a recommender system by using collaborative filtering.<br>
B. Suppose you are writing a recommender system to predict a user's book preferences. In order to build such a system, you need that user to rate all the other books in your training set.<br>
C. For collaborative filtering, it is possible to use one of the advanced optimization algoirthms (L-BFGS/conjugate gradient/etc.) to solve for both the $x^{(i)}$'s and $\theta^{(j)}$'s simultaneously.<br>
D. For collaborative filtering, the optimization algorithm you should use is gradient descent. In particular, you cannot use more advanced optimization algorithms (L-BFGS/conjugate gradient/etc.) for collaborative filtering, since you have to solve for both the $x^{(i)}$'s and $\theta^{(j)}$'s simultaneously.<br>
<b>Answer</b>: A, C.<br><br>

<b>5.</b><br>
Suppose you have two matrices A and B, where A is 5x3 and B is 3x5. Their product is C=AB, a 5x5 matrix. Furthermore, you have a 5x5 matrix R where every entry is 0 or 1. You want to find the sum of all elements C(i,j) for which the corresponding R(i,j) is 1, and ignore all elements C(i,j) where R(i,j)=0. One way to do so is the following code:
</p>
{% highlight Matlab %}
C = A*B;
total = 0;
for i = 1:5
  for j  = 1:5
    if (R(i, j) == 1)
      total = total + C(i, j);
    end;
  end;
end;
{% endhighlight %}
<p align="justify">
Which of the following pieces of Octave code will also correctly compute this total? Check all that apply. Assume all options are in code.<br>
A. total = sum(sum((A * B) .* R))<br>
B. C = A * B; total = sum(sum(C(R == 1)));<br>
C. C = (A * B) * R; total = sum(C(:));<br>
D. total = sum(sum(A(R == 1) * B(R == 1));<br>
<b>Answer</b>: A, B.<br><br>
</p>

#### 9.2.5 Programming Assignment: Anomaly Detection and Recommender Systems
{% highlight Matlab %}
function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%
mu = mean(X);
diff = X - repmat(mu, m, 1);
sigma2 = sum(diff.^2)/m;
% =============================================================
end


function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    predictions = (pval < epsilon);
    true = (yval == 1);
    false = (yval == 0);
    positive = (predictions == 1);
    negative = (predictions == 0);
    TP = sum(positive.*true);
    FP = sum(positive.*false);
    FN = sum(negative.*true);
    P = TP/(TP + FP);
    R = TP/(TP + FN);
    F1 = (2*P*R)/(P+R);
    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end
end


function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
ThetaTX = X*Theta';
diff = ThetaTX - Y;
J = sum(sum((diff.^2).*R))/2 + sum(sum(Theta.^2))*(lambda/2) +...
    sum(sum(X.^2))*(lambda/2);
X_grad = (diff.*R)*Theta + lambda*X;
Theta_grad = (diff.*R)'*X + lambda*Theta;
% =============================================================

grad = [X_grad(:); Theta_grad(:)];
end
{% endhighlight %}
<p align="justify">
<br>
</p>


## 10. Week 10
### 10.1 Large Scale Machine Learning
#### 10.1.1 Gradient Descent with Large Datasets
<p align="justify">
<b>Stochastic Gradient Descent</b><br>
$\bigstar$ Batch gradient descent
$$J_{\text{train}}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^{2}$$
$$\theta_{j} = \theta_{j} - \frac{\alpha}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) x_{j}^{(i)}, \quad j = 0, 1, ..., n$$
$\bigstar$ Stochastic gradient descent<br>
1) randomly shuffle training example<br>
2) gradient descent with one example at each iteration
$$\theta_{j} = \theta_{j} - \alpha (h_{\theta}(x^{(i)}) - y^{(i)}) x_{j}^{(i)}, \quad j = 0, 1, ..., n$$<br>

<b>Mini-Batch Gradient Descent</b><br>
Batch gradient descent: use all m examples in each iteration<br>
Stochastic gradient descent: use 1 example in each iteration<br>
Mini-batch gradient descent: use b ($1 \leq b \leq m$) examples in each iteration<br><br>
</p>

#### 10.1.2 Advanced Topics
<p align="justify">
<b>Online Learning</b><br>
Shipping service website where user comes, specifies origin and destination, you offer to ship their packages for some asking price, and users sometimes choose to use your shipping service (y = 1), sometimes not (y = 0).<br>
Features x capture properties of user, of origin / destination and asking price. We want to learn p(y = 1 | x; $\theta$) to optimize price.<br><br>

<b>Map Reduce and Data Parallelism</b><br>
Parallel computation<br>
Map-reduce and summation over the training set<br><br>
</p>

#### 10.1.3 Quiz
<p align="justify">
<b>1.</b><br>
Suppose you are training a logistic regression classifier using stochastic gradient descent. You find that the cost (say $\text{cost}(\theta, (x^{(i)}, y^{(i)}))$, averaged over the last 500 examples), plotted as a function of the number of iterations, is slowly increasing over time. Which of the following changes are likely to help?<br>
A. Try halving (decreasing) the learning rate α, and see if that causes the cost to now consistently go down; and if not, keep halving it until it does.<br>
B. This is not possible with stochastic gradient descent, as it is guaranteed to converge to the optimal parameters θ.<br>
C. Try averaging the cost over a smaller number of examples (say 250 examples instead of 500) in the plot.<br>
D. Use fewer examples from your training set.<br>
<b>Answer</b>: A.<br><br>

<b>2.</b><br>
Which of the following statements about stochastic gradient descent are true? Check all that apply.<br>
A. One of the advantages of stochastic gradient descent is that it uses parallelization and thus runs much faster than batch gradient descent.<br>
B. Before running stochastic gradient descent, you should randomly shuffle (reorder) the training set.<br>
C. If you have a huge training set, then stochastic gradient descent may be much faster than batch gradient descent.<br>
D. In order to make sure stochastic gradient descent is converging, we typically compute $J_{\text{train}}(\theta)$ after each iteration (and plot it) in order to make sure that the cost function is generally decreasing.<br>
<b>Answer</b>: B, C.<br><br>

<b>3.</b><br>
Which of the following statements about online learning are true? Check all that apply.<br>
A. One of the advantages of online learning is that there is no need to pick a learning rate α.<br>
B. One of the disadvantages of online learning is that it requires a large amount of computer memory/disk space to store all the training examples we have seen.<br>
C. In the approach to online learning discussed in the lecture video, we repeatedly get a single training example, take one step of stochastic gradient descent using that example, and then move on to the next example.<br>
D. When using online learning, in each step we get a new example (x,y), perform one step of (essentially stochastic gradient descent) learning on that example, and then discard that example and move on to the next.<br>
E. One of the advantages of online learning is that if the function we're modeling changes over time (such as if we are modeling the probability of users clicking on different URLs, and user tastes/preferences are changing over time), the online learning algorithm will automatically adapt to these changes.<br>
F. Online learning algorithms are usually best suited to problems were we have a continuous/non-stop stream of data that we want to learn from.<br>
G. When using online learning, you must save every new training example you get, as you will need to reuse past examples to re-train the model even after you get new training examples in the future.<br>
H. Online learning algorithms are most appropriate when we have a fixed training set of size m that we want to train on.<br>
<b>Answer</b>: C, D, E, F.<br><br>

<b>4.</b><br>
Assuming that you have a very large training set, which of the following algorithms do you think can be parallelized using map-reduce and splitting the training set across different machines? Check all that apply.<br>
A. Linear regression trained using batch gradient descent.<br>
B. A neural network trained using batch gradient descent.<br>
C. An online learning setting, where you repeatedly get a single example (x,y), and want to learn from that single example before moving on.<br>
D. Logistic regression trained using stochastic gradient descent.<br>
E. Computing the average of all the features in your training set $\mu = \sum_{i=1}^{m} x^{(i)}$ (say in order to perform mean normalization).<br>
F. Logistic regression trained using batch gradient descent.<br>
G. Linear regression trained using stochastic gradient descent.<br>
<b>Answer</b>: A, B, E, F.<br><br>

<b>5.</b><br>
Which of the following statements about map-reduce are true? Check all that apply.<br>
A. Running map-reduce over N computers requires that we split the training set into $N^{2}$ pieces.<br>
B. When using map-reduce with gradient descent, we usually use a single machine that accumulates the gradients from each of the map-reduce machines, in order to compute the parameter update for that iteration.<br>
C. If you have just 1 computer, but your computer has multiple CPUs or multiple cores, then map-reduce might be a viable way to parallelize your learning algorithm.<br>
D. In order to parallelize a learning algorithm using map-reduce, the first step is to figure out how to express the main work done by the algorithm as computing sums of functions of training examples.<br>
E. Linear regression and logistic regression can be parallelized using map-reduce, but not neural network training.<br>
F. If you have only 1 computer with 1 computing core, then map-reduce is unlikely to help.<br>
G. Because of network latency and other overhead associated with map-reduce, if we run map-reduce using N computers, we might get less than an N-fold speedup compared to using 1 computer.<br>
<b>Answer</b>: B, C, D, F, G.<br><br>
</p>


## 11. Week 11
### 11.1 Photo OCR
#### 11.1.1 Application Example: Photo OCR
<p align="justify">
<b>Problem Description and Pipeline</b><br>
1) text detection<br>
2) character segmentation<br>
3) character classification
$$\text{Image} \rightarrow \text{Text detection} \rightarrow \text{Character segmentation} \rightarrow \text{Character recognition}$$<br>

<b>Getting Lots of Data and Artificial Data</b><br>
Make sure you have a low bias classifier before expending the effort. keep increasing the number of features / number of hidden units in neural network until you have a low bias classifier<br><br>
</p>

#### 11.1.2 Quiz
<p align="justify">
<b>1.</b><br>
Suppose you are running a sliding window detector to find text in images. Your input images are 1000x1000 pixels. You will run your sliding windows detector at two scales, 10x10 and 20x20 (i.e., you will run your classifier on lots of 10x10 patches to decide if they contain text or not; and also on lots of 20x20 patches), and you will "step" your detector by 2 pixels each time. About how many times will you end up running your classifier on a single 1000x1000 test set image?<br>
A. 100,000<br>
B. 1,000,000<br>
C. 500,000<br>
D. 250,000<br>
<b>Answer</b>: C.<br><br>

<b>2.</b><br>
Suppose that you just joined a product team that has been developing a machine learning application, using m=1,000 training examples. You discover that you have the option of hiring additional personnel to help collect and label data. You estimate that you would have to pay each of the labellers 10 per hour, and that each labeller can label 4 examples per minute. About how much will it cost to hire labellers to label 10,000 new training examples?<br>
A. 10,000<br>
B. 250<br>
C. 600<br>
D. 400<br>
<b>Answer</b>: D.<br><br>

<b>3.</b><br>
What are the benefits of performing a ceiling analysis? Check all that apply.<br>
A. It helps us decide on allocation of resources in terms of which component in a machine learning pipeline to spend more effort on.<br>
B. It is a way of providing additional training data to the algorithm.<br>
C. If we have a low-performing component, the ceiling analysis can tell us if that component has a high bias problem or a high variance problem.<br>
D. It can help indicate that certain components of a system might not be worth a significant amount of work improving, because even if it had perfect performance its impact on the overall system may be small.<br>
<b>Answer</b>: A, D.<br>
The ceiling analysis reveals which parts of the pipeline have the most room to improve the performance of the overall system.<br>
An unpromising component will have little effect on overall performance when it is replaced with ground truth.<br><br>

<b>4.</b><br>
Suppose you are building an object classifier, that takes as input an image, and recognizes that image as either containing a car (y=1) or not (y=0). For example, here are a positive example and a negative example:
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/COURSES/ML/11_1_2_1.png"/></center>
</p>
<p align="justify">
After carefully analyzing the performance of your algorithm, you conclude that you need more positive (y=1) training examples. Which of the following might be a good way to get additional positive examples?<br>
A. Apply translations, distortions, and rotations to the images already in your training set.<br>
B. Select two car images and average them to make a third example.<br>
C. Take a few images from your training set, and add random, gaussian noise to every pixel.<br>
D. Make two copies of each image in the training set; this immediately doubles your training set size.<br>
<b>Answer</b>: A.<br><br>

<b>5.</b><br>
Suppose you have a PhotoOCR system, where you have the following pipeline:
$$\text{Image} \rightarrow \text{Text detection} \rightarrow \text{Character segmentation} \rightarrow \text{Character recognition}$$
You have decided to perform a ceiling analysis on this system, and find the following:
<table class="a">
  <tr><th>Component</th><th>Accuracy</th></tr>
  <tr><td>Overall System</td><td>70%</td></tr>
  <tr><td>Text Detection</td><td>72%</td></tr>
  <tr><td>Character segmentation</td><td>82%</td></tr>
  <tr><td>Character recognition</td><td>100%</td></tr>
</table><br>
</p>
<p align="justify">
Which of the following statements are true?<br>
A. If the text detection system was trained using gradient descent, running gradient descent for more iterations is unlikely to help much.<br>
B. If we conclude that the character recognition's errors are mostly due to the character recognition system having high variance, then it may be worth significant effort obtaining additional training data for character recognition.<br>
C. We should dedicate significant effort to collecting additional training data for the text detection system.<br>
D. The least promising component to work on is the character recognition system, since it is already obtaining 100% accuracy.<br>
<b>Answer</b>: A, B.<br><br>
</p>
