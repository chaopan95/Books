---
layout: post
title:  "Linear Models and Optimizations"
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
The complete code is available <a href="https://github.com/chaopan1995/PROJECTS/tree/master/Linear-Models-and-Optimizations">here</a>.
</p>


## 1. Task
<p align="justify">
2-dimensional classification
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/LinearModelsandOptimizations/1.png"/></center>
</p>
<p align="justify">
We notice the data above isn't linearly separable. Since that we should add features (or use non-linear model). Note that decision line between two classes have form of circle, since that we can add quadratic features to make the problem linearly separable. The idea under this displayed on image below
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/LinearModelsandOptimizations/2.png"/></center>
</p>
<p align="justify">
So, we want to expand our data by adding quadratic features. For each sample (row in matrix), compute an expanded row:<br>
[feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
</p>
{% highlight python %}
def expand(X):
    """
    Adds quadratic features. 
    This expansion allows your linear model to make non-linear separation.
    
    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
    
    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    # TODO:<your code here>
    X_expanded = np.zeros((X.shape[0], 6))
    X_expanded[:, :2] = X
    X_expanded[:, 2] = X[:, 0]**2
    X_expanded[:, 3] = X[:, 1]**2
    X_expanded[:, 4] = X[:, 0]*X[:, 1]
    X_expanded[:, 5] = 1
    return X_expanded
{% endhighlight %}


## 2. Logistic Regression
<p align="justify">
To classify objects we will obtain probability of object belongs to class '1'. To predict probability we will use output of linear model and logistic function:
$$a(x; w) = \langle w, x \rangle$$

$$P( y=1 \; \big| \; x, \, w) = \dfrac{1}{1 + \exp(- \langle w, x \rangle)} = \sigma(\langle w, x \rangle)$$
</p>
{% highlight python %}
def probability(X, w):
    """
    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x),
    see description above
        
    Don't forget to use expand(X) function (where necessary)
    in this and subsequent functions.
    
    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """
    # TODO:<your code here>
    return 1./(1+np.exp(-np.matmul(w, X.T)))
{% endhighlight %}

## 3. Loss Function
<p align="justify">
In logistic regression the optimal parameters $w$ are found by cross-entropy minimization:<br>
Loss for one sample:
$$l(x_i, y_i, w) = - \left[ {y_i \cdot log P(y_i = 1 \, | \, x_i,w) + (1-y_i) \cdot log (1-P(y_i = 1\, | \, x_i,w))}\right]$$

Loss for many samples:
$$L(X, \vec{y}, w) =  {1 \over \ell} \sum_{i=1}^\ell l(x_i, y_i, w)$$
</p>
{% highlight python %}
def compute_loss(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute scalar loss function L using formula
    above.
    Keep in mind that our loss is averaged over all samples (rows) in X.
    """

    # TODO:<your code here>
    n, _ = X.shape
    p = probability(X, w)
    return -sum(y*np.log(p)+(1-y)*np.log(1-p))/n
{% endhighlight %}


## 4. Gradient Descent
<p align="justify">
Since we train our model with gradient descent, we should compute gradients. To be specific, we need a derivative of loss function over each weight.
$$\nabla_w L = {1 \over \ell} \sum_{i=1}^\ell \nabla_w l(x_i, y_i, w)$$

For each sample $X_{i}, i \in (1, 2, \cdots, n)$
$$\nabla_w l(X_{i}, y_{i}, w) = X_{i}(y_{i} - P_{i}), \quad P_{i} = \frac{1}{1+e^{X_{i}w^{T}}}$$

Where $X_{i}$ is a vector of $1 \times 6$ and w is a vector of $1 \times 6$<br><br>

For total dataset
$$\nabla_{w} = \frac{1}{n} \sum_{i=1}^{n}X_{i}(y_{i} - P_{i}) = \frac{1}{n} (y-p)^{T}X$$

Finally, we will obtain a vector with a same shape as w, $1 \times 6$
</p>
{% highlight python %}
def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each
    weights.
    Keep in mind that our loss is averaged over all samples (rows) in X.
    """
    
    # TODO<your code here>
    n, _ = X.shape
    return np.matmul((y-probability(X, w)).T, X)/n
{% endhighlight %}


## 5. Training
### 5.1 Mini-batch SGD
<p align="justify">
Stochastic gradient descent just takes a random batch of $m$ samples on each iteration, calculates a gradient of the loss on it and makes a step:
$$w_t = w_{t-1} - \eta \dfrac{1}{m} \sum_{j=1}^m \nabla_w l(x_{i_j}, y_{i_j}, w_t)$$
</p>
{% highlight python %}
w = np.array([0, 0, 0, 0, 0, 1])

eta= 0.1 # learning rate

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    # Keep in mind that compute_grad already does averaging over batch for you!
    # TODO:<your code here>
    w = w + eta*compute_grad(X_expanded[ind, :], y[ind], w)

visualize(X, y, w, loss)
{% endhighlight %}
<p align="justify">
Here is the training result
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/LinearModelsandOptimizations/3.png"/></center>
</p>

### 5.2 SGD with Momentum
<p align="justify">
Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations as can be seen in image below.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/LinearModelsandOptimizations/4.png"/></center>
</p>
<p align="justify">
It does this by adding a fraction $\alpha$ of the update vector of the past time step to the current update vector.
$$\nu_t = \alpha \nu_{t-1} + \eta\dfrac{1}{m} \sum_{j=1}^m \nabla_w l(x_{i_j}, y_{i_j}, w_t)$$
$$w_t = w_{t-1} - \nu_t$$
</p>
{% highlight python %}
w = np.array([0, 0, 0, 0, 0, 1])

eta = 0.05 # learning rate
alpha = 0.9 # momentum
nu = np.zeros_like(w)

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    nu = alpha*nu + eta*compute_grad(X_expanded[ind, :], y[ind], w)
    w = w + nu

visualize(X, y, w, loss)
{% endhighlight %}
<p align="justify">
Here is the training result
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/LinearModelsandOptimizations/5.png"/></center>
</p>

### 5.3 RMSprop
<p align="justify">
Implement RMSPROP algorithm, which use squared gradients to adjust learning rate:

$$G_j^t = \alpha G_j^{t-1} + (1 - \alpha) g_{tj}^2$$
$$w_j^t = w_j^{t-1} - \dfrac{\eta}{\sqrt{G_j^t + \varepsilon}} g_{tj}$$
</p>
{% highlight python %}
w = np.array([0, 0, 0, 0, 0, 1.])
# learning rate
eta = 0.1
# moving average of gradient norm squared
alpha = 0.9
# we start with None so that you can update this value correctly on
# the first iteration
g2 = np.zeros_like(w)
eps = 1e-8

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12,5))
for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    # TODO:<your code here>
    grad = compute_grad(X_expanded[ind, :], y[ind], w)
    g2 = alpha*g2 + (1-alpha)*grad**2
    w = w + eta/(g2+eps)**0.5*grad

visualize(X, y, w, loss)
{% endhighlight %}
<p align="justify">
Here is the training result
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/LinearModelsandOptimizations/6.png"/></center>
</p>
