---
layout: post
title:  "Natural Language Processing"
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


## 1. Classical Structure
### 1.1 Recurrent Neural Network
#### 1.1.1 Notations
<p align="justify">
$\bigstar$ Text:<br>
-- A sequence of token (words)<br>
$\bigstar$ Token/Word:<br>
-- A sequence of characters<br>
-- For example, a sentence "I am a student" is tokenized to ['I', 'am', 'a', 'student']<br>
$\bigstar$ Characters<br>
-- An atomic elements of text<br>
$\bigstar$ Input $x_{t}, t \in [1, T_{x}]$<br>
-- $t^{th}$ token in some sequence<br>
-- $T_{x}$ denotes the length of input<br>
$\bigstar$ Output (prediction) $\hat{y}_{t}, t \in [1, T_{y}]$<br>
-- $t^{th}$ prediction in the output sequence<br>
-- $T_{y}$ denotes the length of output<br>
$\bigstar$ Intermediate output (after a MLP) $h_{t}, t \in [1, T_{x}]$<br>
$\bigstar$ True label $y_{t}$<br>
$\bigstar$ Loss function $L_{t}(y_{t}, \hat{y}_{t})$<br>
$\bigstar$ EOS: end of sentence; UNK: unknow words<br>
</p>

#### 1.1.2 Model
<p align="justify">
<b>Rucurrent Architecture</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_1_1_1.png"/></center>
</p>
<p align="justify">
$$h_{t} = f_{h}(V \cdot x_{t} + W \cdot h_{t-1} + b_{h})$$
$$\hat{y_{}}_{t} = f_{y}(U \cdot h_{t} + b_{y})$$
$$L = \sum_{t} L_{t}(y_{t}, \hat{y_{t}})$$

$\bigstar$ RNN advantages<br>
-- Arbitrary sequence length<br>
Fixed number of inputs at each time step. At the first step we use some initial vector as an input from previous time step<br>
-- Small number of parameters<br>
All the parameters of a MLP are shared across the different time steps so we need a much smaller number of parameters<br><br>

<b>Why MLP doesn't work?</b><br>
The main problem is arbitrary length of sequence, because MLP requires a fixed shape of input. Actually, we can use a window of a fixed size as an input. This is a heuristic and it is not clear how to choose the width of the window. In some tasks, we need very wide window therefore there is a problem with the large number of parameters.<br><br>

<b>Bidirectional RNN (BRNN)</b><br>
We have two nodes at each layer
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_1_1_2.png"/></center>
</p>
<p align="justify">
$$\hat{y}_{t} = f_{h}(W_{\overrightarrow{h}} \overrightarrow{h}_{t-1} + W_{\overleftarrow{h}} \overleftarrow{h}_{t-1} + V x_{t} + b)$$
Both of two progatations are forward: from head to tail and from tail to head.
</p>

#### 1.1.3 Backpropagation Through Time (BPTT)
<p align="justify">
<b>Forward pass</b><br>
$$h_{t}, \hat{y_{t}}, L_{t}, L$$

<b>Backward pass</b><br>
We backpropagate through layers and time. <b>All weights are shared across time steps</b>. That means, if we want to calculate a gradient for some parameters, we should sum it up for all time steps.
$$\frac{\partial L}{\partial U}, \frac{\partial L}{\partial V}, \frac{\partial L}{\partial W}, \frac{\partial L}{\partial b_{x}}, \frac{\partial L}{\partial b_{h}}$$

$\bigstar$ Calculate $\frac{\partial L}{\partial U}$
$$\frac{\partial L}{\partial U} = \sum_{t=0}^{T} \frac{\partial L_{t}}{\partial U} =  \sum_{t=0}^{T} \frac{\partial L_{t}}{\partial \hat{y_{t}}}\frac{\partial \hat{y_{t}}}{\partial U}$$

$\hat{y_{t}}$ depends only on U, because
$$\hat{y_{}}_{t} = f_{y}(U \cdot h_{t} + b_{y})$$

$\bigstar$ Calculate $\frac{\partial L}{\partial W}$
$$
\begin{aligned}
\frac{\partial L}{\partial W} & = \sum_{t=0}^{T} \frac{\partial L_{t}}{\partial W} \\
& = \sum_{t=0}^{T} \frac{\partial L_{t}}{\partial \hat{y_{t}}} \frac{\partial \hat{y_{t}}}{\partial h_{t}} (\frac{\partial h_{t}}{\partial W} + \frac{\partial h_{t}}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial W} + \frac{\partial h_{t}}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial h_{t-2}} \frac{\partial h_{t-2}}{\partial W}+ ...) \\
& = \sum_{t=0}^{T} \frac{\partial L_{t}}{\partial \hat{y_{t}}} \frac{\partial \hat{y_{t}}}{\partial h_{t}} \sum_{k=0}^{t} \frac{\partial h_{t}}{\partial h_{t-1}} ... \frac{\partial h_{k+1}}{\partial h_{k}} \frac{\partial h_{k}}{\partial W} \\
& = \sum_{t=0}^{T} \frac{\partial L_{t}}{\partial \hat{y_{t}}} \frac{\partial \hat{y_{t}}}{\partial h_{t}} \sum_{k=0}^{t} (\prod_{i=k+1}^{t} \frac{\partial h_{i}}{\partial h_{i-1}})\frac{\partial h_{k}}{\partial W}
\end{aligned}
$$

$h_{t}$ is dependent of its previous hidden layers<br><br>

$\bigstar$ Calculate $\frac{\partial L}{\partial V}$<br>
We have a same situation as calculating W, $h_{t}$ is dependent of its previous hidden layers.
$$
\begin{aligned}
\frac{\partial L}{\partial V} &= \sum_{t=0}^{T} \frac{\partial L_{t}}{\partial V} \\
&= \sum_{t=0}^{T} \frac{\partial L_{t}}{\partial \hat{y_{t}}}\frac{\partial \hat{y_{t}}}{\partial h_{t}} \sum_{k=0}^{t} (\prod_{i=k+1}^{t}\frac{\partial h_{i}}{\partial h_{i-1}}) \frac{\partial h_{k}}{\partial V}
\end{aligned}
$$

$\bigstar$ Calculate $\frac{\partial L}{\partial b_{y}}$ and $\frac{\partial L}{\partial b_{h}}$
$$
\begin{aligned}
& \frac{\partial L}{\partial b_{y}} = \sum_{t=0}^{T} \frac{\partial L_{t}}{\partial \hat{y_{t}}}\frac{\partial \hat{y_{t}}}{\partial b_{y}} \\
& \frac{\partial L}{\partial b_{h}} = \sum_{t=0}^{T} \frac{\partial L_{t}}{\partial \hat{y_{t}}}\frac{\partial \hat{y_{t}}}{\partial h_{t}} \sum_{k=0}^{t} (\prod_{i=k+1}^{t} \frac{\partial h_{i}}{\partial h_{i-1}}) \frac{\partial h_{k}}{\partial b_{h}}
\end{aligned}
$$

<b>Quiz:</b> Consider an RNN for a language generation task. $\hat{y_{t}}$ is an output of this RNN at each time step, L is a length of the input sequence, N is a number of words in the vocabulary. Choose correct statements about $\hat{y_{t}}$<br>
A. $\hat{y_{t}}$ is a vector of length N<br>
B. $\hat{y_{t}}$ is a vector of length (L-t)<br>
C. $\hat{y_{t}}$ is a vector of length $L \times N$<br>
D. Each element of $\hat{y_{t}}$ is either 0 or 1<br>
E. Each element of $\hat{y_{t}}$ is a number from 0 to 1<br>
F. Each element of $\hat{y_{t}}$ is a number from 0 to N<br>
<b>Answer:</b> A, E.<br>
The output at each time step is a distribution over a vocabulary, therefore the length of $\hat{y_{t}}$ is equal to the vocabulary size.<br>
Each element of $\hat{y_{t}}$ is a probability from 0 to 1 and sum up all elements is equal to 1<br>
</p>

#### 1.1.4 RNN types
<p align="justify">
There are 5 types of RNN<br>
$\bigstar$ One to One<br>
$\bigstar$ One to Many<br>
-- Character-based language model<br>
-- Word-based language model<br>
-- Music generation<br>
-- Speech generation<br>
-- Handwritting generation<br>
$\bigstar$ Many to One<br>
-- sentiment analysis<br>
-- emotion classification<br>
-- comment preference analysis<br>
$\bigstar$ Many to Many (equivalent or inequivalent length)<br>
-- Handwritting to text / text to handwritting<br>
-- Speech to text / text to speech<br>
-- Machine translation
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_1_4_1.png"/></center>
</p>


### 1.2 Word Embeddings
#### 1.2.1 Word2Vec
<p align="justify">
$$
\begin{aligned}
& \text{one hot vector for a word (e.g. id=1337)} \quad X_{1337} =
\begin{bmatrix}
0 \\
\vdots \\
0 \\
1 \\
0 \\
\vdots \\
0
\end{bmatrix} \\
& \text{parameters between input layer and one hidden layer} \quad W =
\begin{bmatrix}
w_{1,1} & \cdots & w_{1,n} \\
\vdots & \cdots & \vdots \\
w_{1336, 1} & \cdots & w_{1336, n} \\
w_{1337, 1} & \cdots & w_{1337, n} \\
w_{1338, 1} & \cdots & w_{1338, n} \\
\vdots & \cdots & \vdots \\
w_{m,1} & \cdots & w_{m,n} \\
\end{bmatrix} \\
& X_{1337}^{T} W = W_{1337,1:n} \quad \text{the 1337th row in W}
\end{aligned}
$$

$\bigstar$ CBOW Skip-Gram<br>
$\bigstar$ Hoffman tree, hierarchical softmax, negative sampling<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_2_1_1.png"/></center>
</p>
<p align="justify">
<b>Quiz:</b> Which part of word2vec model would take most time to compute? For simplicity, assume that you compute it on modern CPU as of year 2017, your vocabulary contains 100 000 words and hidden vector size is 1000<br>
A. Building one-hot vector from word id<br>
B. Multiplying word vector by the right matrix<br>
C. All steps are equally computationally heavy<br>
D. Multiplying one-hot encoded word by the left matrix<br>
E. Computing softmax given the predicted logits<br>
<b>Answer:</b> B.<br><br>

<b>Quiz:</b> How can you train word2vec model?<br>
A. By minimizing crossentropy (aka maximizing likelihood).<br>
B. By learning to predict context (neighboring words) given one word.<br>
C. By applying stochastic gradient descent.<br>
D. By minimizing distance between human-defined synonyms and maximizing distance between antonyms.<br>
E. By learning to predict omitted word by it's context.<br>
F. By changing order of words in the corpora.<br>
<b>Answer:</b> A, B, C, E.<br><br>

<b>Quiz:</b> Which of the following is an appropriate way to measure similarity between word vectors v1 and v2? (more = better)<br>
A. ||v1 - v2||<br>
B. cos(v1,v2)<br>
C. sin(v1,v2)<br>
D. -||v1 - v2||<br>
<b>Answer:</b> A, D.
</p>

#### 1.2.2 GloVe
<p align="justify">

</p>

### 1.3 GRU and LSTM
#### 1.3.1 Vanishing and Exploding Gradients
<p align="justify">
<b>Let's look at the gradient</b><br>
$$\frac{\partial L_{t}}{\partial W} = \frac{\partial L_{t}}{\partial \hat{y_{t}}} \frac{\partial \hat{y_{t}}}{\partial h_{t}} \sum_{k=0}^{t} (\prod_{i=k+1}^{t} \frac{\partial h_{i}}{\partial h_{i-1}})\frac{\partial h_{k}}{\partial W}$$

This term $\prod_{i=k+1}^{t} \frac{\partial h_{i}}{\partial h_{i-1}}$ denotes a contribution of a state at time step k to the gradient of the loss at time step t.<br><br>

Obviously, the more steps between the k and t, the more elements are in this product<br><br>

Values of these Jacobian matrices have particularly severe impact on the contributions from faraway layers.<br><br>

Suppose for a moment that $h_{i}$ is a scalar and consequently $\frac{\partial h_{i}}{\partial h_{i-1}}$ is also a scalar.<br>
-- if $\left | \frac{\partial h_{i}}{\partial h_{i-1}} \right | < 1$, the product goes to 0 exponentially fast<br>
-- if $\left | \frac{\partial h_{i}}{\partial h_{i-1}} \right | > 1$, the product goes to infinity exponentially fast<br><br>

<b>Vanishing gradients</b><br>
$$\left | \frac{\partial h_{i}}{\partial h_{i-1}} \right | < 1$$

$\bigstar$ contributions from faraway steps vanish and don't affect the training<br>
$\bigstar$ difficult to learn long-range dependencies<br><br>

<b>Exploding gradients</b><br>
$$\left | \frac{\partial h_{i}}{\partial h_{i-1}} \right | > 1$$

$\bigstar$ make the learning process unstable<br>
$\bigstar$ gradient could enevn become NaN<br><br>

<b>The same problem is for matrices but with the spectral matrix norm instead of the absolute value</b><br>
$\bigstar$ Vanishing gradients
$$\left \| \frac{\partial h_{i}}{\partial h_{i-1}} \right \|_{2} < 1$$

$\bigstar$ Exploding gradients
$$\left \| \frac{\partial h_{i}}{\partial h_{i-1}} \right \|_{2} > 1$$

<b>This is really a problem in practice</b>
$$h_{t} = f_{h}(V \cdot x_{t} + W \cdot h_{t-1} + b_{h}) = f_{h}(pr_{t})$$
$$\frac{\partial h_{t}}{\partial h_{t-1}} = \frac{\partial h_{t}}{\partial pr_{t}}\frac{\partial pr_{t}}{\partial h_{t-1}} = diag(f'_{h}(pr_{t})) \cdot W$$

Vanishing gradients are very likely especially with sigmoid and tanh
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_3_1_1.png"/></center>
</p>
<p align="justify">
Besides, $\left \| W \right \|$ could be either small or large: small $\left \| W \right \|$ could aggravate the vanishing gradient problem while large $\left \| W \right \|$ could cause exploding gradients (especially with ReLU)<br><br>

<b>Summary</b><br>
$\bigstar$ In practice vanishing and exploding gradients are common for RNNs. These problems also occur in deep Feedforward NNs<br>
$\bigstar$ Vanishing gradients make the learning of long-range dependencies very difficult<br>
$\bigstar$ Exploding gradients make the learning process very unstable and even crash it<br>
</p>

<p align="justify">
<b>Exploding gradients: detection</b><br>
Exploding gradients are easy to detect
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_3_1_2.png"/></center>
</p>
<p align="justify">
If the gradients contain NaNs, you end up with NaNs in the weights<br><br>

<b>Gradient clipping</b><br>
Gradient
$$g = \frac{\partial L}{\partial \theta}, \quad \theta \sim \text{all the network parapemters}$$

If $\left \| g \right \| > \text{thereshold}$
$$g \leftarrow \frac{\text{threshold}}{\left \| g \right \|} g$$

Simple but very effective. It is enough to clip $\frac{\partial h_{t}}{\partial h_{t-1}}$<br><br>

Choose the highest threshold which helps to overcome the exploding gradient problem
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_3_1_3.png"/></center>
</p>
<p align="justify">
<b>BPTT</b><br>
Forward pass through the entire sequence to compute the loss while backward pass through the entire sequence to compute the gradient.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_3_1_4.png"/></center>
</p>
<p align="justify">
What if we have a very long training sequences? Way too expensive + exploding gradients<br><br>

<b>Truncated BPTT</b><br>
Let's run forward and backwward passes through the chunks of the sequence instead of the whole sequence
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_3_1_5.png"/></center>
</p>
<p align="justify">
Carry hidden states forward in time forvever, but only backpropagation for some smaller numbers of steps.<br><br>

Truncated BPTT is much faster but it doesn't come without a price. <b>Dependencies longer than the chunk size don't affect the training but at least they still work at forward pass.</b><br><br>

<b>Vanishing gradients: detection</b><br>
It is not clear how to detect vanishing gradients
Gradient norm
$$\left \| \frac{\partial L_{t}}{\partial_{t-100}} \right \|_{2} \text{is small}$$

$\bigstar$ How to deal with vanishing gradients?<br>
-- LSTM, GRU<br>
-- ReLU activation function<br>
-- Initialization of the recurrent weight matrix<br>
-- Skip connections<br><br>

<b>ReLU activation function</b><br>
$$\frac{\partial h_{t}}{\partial h_{t-1}} = \frac{\partial h_{t}}{\partial pr_{t}}\frac{\partial pr_{t}}{\partial h_{t-1}} = diag(f'_{h}(pr_{t})) \cdot W$$

ReLU is much resistant to the vanishing gradient problem with regard to $diag(f'_{h}(pr_{t}))$<br><br>

<b>Initialization of the recurrent weight matrix</b><br>
$$\text{Q is orthogonal if } Q^{T} = Q^{-1} \Rightarrow \prod_{i}Q_{i} \text{ doesn't explore or vanish}$$

$\bigstar$ Initialize W with an orthogonal matrix<br>
$\bigstar$ Use orthogonal W through the whole training<br><br>

<b>Skip connections</b><br>
Add structure $\Rightarrow$ shorter ways for the gradients $\Rightarrow$ learn longer dependencies
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_3_1_6.png"/></center>
</p>
<p align="justify">
The idea is similar to the residual connections in the ResNet<br><br>

<b>Summary</b><br>
$\bigstar$ Exploding gradients are easy to detect but it is not clear how to detect vanishing gradients<br>
$\bigstar$ Exploding gradients: gradients clipping and truncated BPTT<br>
$\bigstar$ Vanishing gradients: ReLU nonlinearity, orthogonal initialization of the recurrent weights, skip connections<br><br>

<b>Quiz: </b>Choose correct statements about the exploding gradient problem:<br>
A. Exploding gradient problem is easy to detect.<br>
B. ReLU nonlinearity helps with the exploding gradient problem.<br>
C. The reason of the exploding gradient problem in the simple RNN is the recurrent weight matrix W. Nonlinearities sigmoid, tanh, and ReLU does not cause the problem.<br>
D. The threshold for gradient clipping should be as low as possible to make the training more efficient.<br>
<b>Answer:</b> A, C.<br><br>

<b>Quiz: </b>Choose correct statements about the vanishing gradient problem:<br>
A. Vanishing gradient problem is easy to detect.<br>
B. Both nonlinearity and the recurrent weight matrix W cause the vanishing gradient problem.<br>
C. Orthogonal initialization of the recurrent weight matrix helps with the vanishing gradient problem.<br>
D. Truncated BPTT helps with the vanishing gradient problem.<br>
<b>Answer:</b> B, C.<br><br>
</p>

#### 1.3.2 GRU
<p align="justify">
GRU: Gated Recurrent Unit<br>
r: reset gate<br>
u: update gate<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_3_2_1.png"/></center>
</p>
<p align="justify">
$$
\begin{aligned}
& r_{t} = \sigma (V_{r} x_{t} + W_{r} h_{t-1} + b_{r}) \\
& u_{t} = \sigma (V_{u} x_{t} + W_{u} h_{t-1} + b_{u})
\end{aligned} \Rightarrow
\begin{pmatrix}
r_{t} \\
u_{t}
\end{pmatrix} =
\begin{pmatrix}
\sigma \\
\sigma
\end{pmatrix}
(V x_{t} + W h_{t-1} + b)
$$
$$
\begin{aligned}
& g_{t} = f_{h} (V_{g} x_{t} + W_{g} (h_{t-1} \cdot r_{t}) + b_{g}) \\
& h_{t} = (1 - u_{t}) \cdot g_{t} + u_{t} \cdot h_{t-1}
\end{aligned}
$$
The GRU layer doesn't contain an additional internal memory, but it also contains two gates that are called <b>reset gate</b> and <b>update gate</b>. These gates are computed in the same manner as the ones from LSTM so they're equal to the sigmoid function or the linear combination of the inputs. As a result, they can take queries from zero to one.<br><br>

We use reset gate as an input to the information vector g. It acts quite similar to an input gate in the LSTM. The update gate controls the balance between the storing the previous values of the hidden units.<br><br>

<b>GRU avoids vanishing gradients</b><br>
$$
\begin{aligned}
& u_{t} = \sigma(V_{u}x_{t} + W_{u}h_{t-1} + b_{u}) \\
& h_{t} = (1-u_{t}) \cdot g_{t} + u_{t} \cdot h_{t-1} \\
& \frac{\partial h_{t}}{\partial h_{t-1}} = diag(1 - u_{t}) \cdot \frac{\partial g_{h}}{\partial h_{h-1}} + diag(u_{h}) \Rightarrow \text{ High initial } b_{u}
\end{aligned}
$$

<b>Quiz: </b>Consider the GRU architecture: Which combination of the gate values makes this model equivalent to the simple RNN? Here value zero corresponds to a closed gate and value one corresponds to an open gate.<br>
A. Both reset and update gates are open.<br>
B. Both reset and update gates are closed.<br>
C. Reset gate is open and update gate is closed.<br>
D. Update gate is open and reset gate is closed.<br>
<b>Answer:</b> C.<br><br>

<b>LSTM or GRU</b><br>
$\bigstar$ LSTM is more flexible, while GRU has less parameters<br>
$\bigstar$ First LSTM first $\rightarrow$ second train GRU $\rightarrow$ compare and choose<br><br>

<b>Summary</b><br>
$\bigstar$ Gated recurrent architectures: LSTM and GRU<br>
$\bigstar$ They don't suffer from vanishing gradients much because there is an additional short way for the gradients through them<br><br>
</p>

#### 1.3.3 LSTM
<p align="justify">
LSTM: Long Short Term Memory<br>
c: internal memory<br>
h: hidden unit with a same diemsion as c (vector)<br>
i: input gate<br>
0: output gate<br>
f: forget gate<br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_3_3_1.png"/></center>
</p>
<p align="justify">
$$
\begin{aligned}
&
\begin{aligned}
& g_{t} = f_{h} (V_{g} x_{t} + W_{g} h_{t-1} + b_{g}) \\
& i_{t} = \sigma(V_{i} x_{t} + W_{i} h_{t-1} + b_{i}) \\
& o_{t} = \sigma(V_{o} x_{t} + W_{o} h_{t-1} + b_{o}) \\
& f_{t} = \sigma(V_{f} x_{t} + W_{f} h_{t-1} + b_{f})
\end{aligned} \quad \Rightarrow \quad
\begin{pmatrix}
g_{t} \\
i_{t} \\
o_{t} \\
f_{t}
\end{pmatrix} = 
\begin{pmatrix}
f\\
\sigma \\
\sigma \\
\sigma
\end{pmatrix}
(V x_{t} + W h_{t-1} + b)
\end{aligned}
$$
$$
\begin{aligned}
& c_{t} = f_{t} \cdot c_{t-1} + i_{t} \cdot g_{t}, \quad \text{where } \cdot \text{ means element-wise multiplication} \\
& h_{t} = o_{t} \cdot f_{h}(c_{t}) \\
& \frac{\partial c_{t}}{\partial c_{t-1}} = diag(f_{t}) \Rightarrow \text{ High initial } b_{f}, \quad \text{to avoid vanishing gradients}
\end{aligned}
$$

$\bigstar$ <b>What size do the matrices V and W and the vector b have if $x \in R^{n}$ and $h \in R^{m}$?</b>
$$V \in R^{4m \times n}, \quad W \in R^{4m \times m}, \quad b \in R^{4m}$$

$\bigstar$ For the gates, we use only the sigmoid non-linearity.<br>
It is important because you want the elements on the gates to take values from zero to one. In this case, the value one can be interpreted as an open gate and the value zero as a closed gate. If we multiply some information by the gate vector, we either get the same information or zero or something in between.<br><br>

$\bigstar$ The input gate controls what to store in the memory.<br>
The information vector g is multiplied by the input gate and then added to the memory. Multiplication here is element wise.<br><br>

$\bigstar$ The output gate controls what to read from the memory and return to the outer world.<br>
Memory cell C are multiplied by the output gate and then returned as a new hidden units.<br><br>

$\bigstar$ LSTM forgets somethings<br>
We need to be able to erase the information from the memory sometimes. A <b>forget gate</b> will help us with this. We compute in the same manner as the previous two gates and use it on the input memory cells before doing something else with them. If the forget gate is closed, we erase the information from the memory.<br><br>

$f_{t}$ takes values from zero to one, so it is usually less than one and may cause vanishing gradient problem. In order do correct this, a proper initialization can be used. If the base of the forget gate is initialized with high positive numbers, for example, five, then the forget gate at first iteration with training is almost equal to one.<br><br>

At the beginning, LSTM doesn't forget and can't find long range dependencies in the data. Later, it learns to forget if it's necessary.<br><br>

<b>LSTM: extreme regimes</b><br>
By controling three gates, namely input gate, output gate and forget gate, we can realize sevevral functions. Particularly, if forget gate is closed, input gate and output gate are open, LSTM is reduced to a simple RNN.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ML/NLP/1_3_3_2.png"/></center>
</p>
<p align="justify">
<b>Quiz: </b>Consider the LSTM architecture: Choose correct statements about this architecture:<br>
A. The LSTM needs four times more parameters than the simple RNN.<br>
B. Gradients do not vanish on the way through memory cells c in the LSTM with forget gate.<br>
C. There is a combination of the gates values which makes the LSTM completely equivalent to the simple RNN.<br>
D. The exploding gradient problem is still possible in LSTM on the way between $h_{t-1}$ and $h_{t}$<br>
<b>Answer:</b> A, D.<br>
Very large norm of W may cause the exploding gradient problem. Therefore gradient clipping is useful for LSTM and GRU architectures too.<br><br>

<b>3. Suppose Nick has a trained sequence-to-sequence machine translation model. He wants to generate the translation for a new sentence in the way that this translation has the highest probability in the model. To do this at each time step of the decoder he chooses the most probable next word instead of the generating from the distribution. Does this scheme guaranty that the resulting output sentence is the most probable one?</b><br>
<b>Answer:</b> No.<br>
That is why beam search is usually used to generate more probable sequences.<br><br>
</p>



## 2. Image captioning
