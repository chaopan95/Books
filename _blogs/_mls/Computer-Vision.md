---
layout: post
title:  "Computer Vision"
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
### 1.1 LeNet5 (1998)
<p align="justify">
$\bigstar$ First deep convolutional neural network for ImageNet<br>
$\bigstar$ Significant reduced top 5 error from 26% to 15%<br>
$\bigstar$ $11\times11$, $5\times5$, $3\times3$ convolutions, max pooling, dropout, data augmentation, ReLU activation, SGD with momentum<br>
$\bigstar$ 60 million parameters<br>
$\bigstar$ Trains on 2 GPUs for 6 days
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/ML/CV/1_1_1.png"/></center>
<table class="c">
  <tr><th></th><th>Activation shape</th><th>Activation size</th><th># parameters</th></tr>
  <tr><td>Input</td><td>(32, 32, 3)</td><td>3072</td><td>0</td></tr>
  <tr><td>CONV1 (f=5, s=1)</td><td>(28, 28, 8)</td><td>6272</td><td>(5*5*3 + 1) * 8 = 608</td></tr>
  <tr><td>POOL1</td><td>(14, 14, 8)</td><td>1568</td><td>0</td></tr>
  <tr><td>CONV2 (f=5, s=1)</td><td>(10, 10, 16)</td><td>1600</td><td>(5*5*8 + 1) * 16 = 3216</td></tr>
  <tr><td>POOL2</td><td>(5, 5, 16)</td><td>400</td><td>0</td></tr>
  <tr><td>FC3</td><td>(120, 1)</td><td>120</td><td>400*120 + 120 = 48120</td></tr>
  <tr><td>FC4</td><td>(84, 1)</td><td>84</td><td>120*84 + 84 (not 1) = 10164</td></tr>
  <tr><td>Softmax</td><td>(10, 1)</td><td>10</td><td>84*10 + 10 = 850</td></tr>
</table>
</p>

### 1.2 AlexNet (2012)
<p align="justify">
$\bigstar$ First deep convolutional neural network for ImageNet<br>
$\bigstar$ Significant reduced top 5 error from 26% to 15%<br>
$\bigstar$ $11\times11$, $5\times5$, $3\times3$ convolutions, max pooling, dropout, data augmentation, ReLU activation, SGD with momentum<br>
$\bigstar$ 60 million parameters<br>
$\bigstar$ Trains on 2 GPUs for 6 days
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/ML/CV/1_1_2.png"/></center>
</p>

### 1.3 VGG (2015)
<p align="justify">
$\bigstar$ Similar to AlexNet, only $3\times3$ convolutions but lots of filters<br>
$\bigstar$ ImageNet top 5 error: 8% (single model)<br>
$\bigstar$ Training similar to AlexNet with additional multi-scale cropping<br>
$\bigstar$ 138 million parameters<br>
$\bigstar$ Trains on 4 GPUs for 2-3 weeks
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/ML/CV/1_1_3.png"/></center>
</p>

### 1.4 InceptionV3 (2015)
<p align="justify">
$\bigstar$ Use Inception block introduced in GoogleNet (InceptionV1)<br>
$\bigstar$ ImageNet top 5 error 5.6% (single model), 3.6% (ensemble)<br>
$\bigstar$ Batch normalization, image distortion, RMSprop<br>
$\bigstar$ 25 million parameters<br>
$\bigstar$ Trains on 8 GPUs for 2 weeks<br><br>
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/ML/CV/1_4_1.png"/></center>
</p>
<p align="justify">
<b>$1 \times 1$ convolutions</b><br>
$\bigstar$ Such convolutions capture interactions of input channels in one <b>pixel</b> of feature map<br>
$\bigstar$ They can reduce number of channels not hurting the quality of model, because different channels can correlate.<br>
$\bigstar$ Dimensionality reduction with added ReLU activation
$$
\begin{bmatrix}
1 & 2 & 3 & 6 & 5 & 8 \\
3 & 5 & 5 & 1 & 3 & 4 \\
2 & 1 & 3 & 4 & 9 & 3 \\
4 & 7 & 8 & 5 & 7 & 9 \\
1 & 5 & 3 & 7 & 4 & 8 \\
5 & 4 & 9 & 8 & 3 & 5
\end{bmatrix} * 
\begin{bmatrix} 2 \end{bmatrix} =
\begin{bmatrix}
2 & 4 & 6 & 12 & 10 & 16 \\
6 & 10 & 10 & 2 & 6 & 8 \\
4 & 2 & 6 & 8 & 18 & 6 \\
8 & 14 & 16 & 10 & 14 & 18 \\
2 & 10 & 6 & 14 & 8 & 16 \\
10 & 8 & 18 & 16 & 6 & 10
\end{bmatrix}
$$
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/ML/CV/1_4_2.png"/></center>
</p>
<p align="justify">
<b>Basic Inception Block</b><br>
$\bigstar$ All operations inside a block use stride 1 ans enough padding to output the same spatial dimensions $(W\times H)$ of feature map<br>
$\bigstar$ 4 different feature maps are concatenated on depth at the end
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/ML/CV/1_4_3.png"/></center>
</p>

### 1.5 ResNet (2015)
<p align="justify">
$\bigstar$ Introduce residual connections<br>
$\bigstar$ ImageNet top 5 error: 4.5% (single model), 3.5% (ensemble)<br>
$\bigstar$ 152 layers, few $7\times7$ convolutional layers, the rest are $3\times3$, batch normalization, max and average pooling<br>
$\bigstar$ 60 million parameters<br>
$\bigstar$ Trains on 8GPUs for 2-3 weeks<br>
$\bigstar$ Residual connection<br>
-- We creat output channels adding a small delta F(x) to original input channels x<br>
-- Thousands of layers and gradients do not vanish in this way
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/ML/CV/1_5_1.png"/></center>
</p>

### 1.6 Summary
<p align="justify">
$\bigstar$ By stacking more comvolution and pooling layers you can reduce the error like AlexNet or VGG<br>
$\bigstar$ But you cannot do that forever, you need to utilize new kind of layers like Inception block or residual connections<br>
$\bigstar$ You've probably notice that one needs a lot of time to train a neural network<br>
</p>

## 2. Simple classification
### 2.1 MNIST digits classification
<p align="justify">
We use TensorFlow 2.x
</p>
{% highlight python %}
from matplotlib import pyplot as plt
%matplotlib inline
import tensorflow as tf
print("We're using TF", tf.__version__)
# building a model with keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Load data and preprocess
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000, 28*28))/255.
X_test = X_test.reshape((10000, 28*28))/255.
y_train_oh = to_categorical(y_train, 10)
y_test_oh = to_categorical(y_test, 10)

# we still need to clear a graph though
tf.keras.backend.clear_session()

# it is a feed-forward network without loops like in RNN
model = Sequential()
# the first layer must specify the input shape (replacing placeholders)
model.add(Dense(256, input_shape=(784,)))
model.add(Activation('sigmoid'))
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))
# you can look at all layers and parameter count
model.summary()
# now we "compile" the model specifying the loss and optimizer
model.compile(
    loss='categorical_crossentropy', # this is our cross-entropy
    optimizer='adam',
    metrics=['accuracy']  # report accuracy during training
)

# and now we can fit the model with model.fit()
# and we don't have to write loops and batching manually as in TensorFlow
history = model.fit(
    X_train, 
    y_train_oh,
    batch_size=512, 
    epochs=40,
    validation_data=(X_test, y_test_oh),
    verbose=0
)

def plot(loss_train, loss_test, accuracy_train, accuracy_test, EPOCHS):
    x = np.arange(EPOCHS)
    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(20, 5))
    ax1.plot(x, loss_train, color='red', linewidth=1.0, linestyle='solid',
             label='train loss')
    ax1.plot(x, loss_test, color='green', linewidth=1.0, linestyle='solid',
             label='test loss')
    ax1.legend()
    ax1.set_title('Cross entropy loss')

    ax2.plot(x, accuracy_train, color='red', linewidth=1.0, linestyle='solid',
             label='accuracy train')
    ax2.plot(x, accuracy_test, color='green', linewidth=1.0, linestyle='solid',
             label='accuracy test')
    ax2.legend()
    ax2.set_title('Accuracy')

    plt.show()

plot(history.history['loss'], history.history['val_loss'],
     history.history['accuracy'], history.history['val_accuracy'],
     len(history.epoch))
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/ML/CV/2_1_1.png"/></center>
</p>

## 3. Semantic segmentation

## 4. Object localisation

## 5. Autoencoders
<p align="justify">
<b>Main idea</b><br>
$\bigstar$ Take data in some original (high-dimensional) space<br>
$\bigstar$ Project data into a new sapce from which it can be accurately restored<br>
$\bigstar$ Encode = data to hidden<br>
$\bigstar$ Decoder = hidden to data<br>
$\bigstar$ Decoder(Encoder(x)) ~ x<br><br>

<b>Why do we need that</b><br>
$\bigstar$ Compress data<br>
-- |code| << |data|<br>
$\bigstar$ Dimensionality reduction<br>
-- Before feeding data to your XGBoost<br><br>

<b>Linear case: Matrix factorization</b><br>
We need to minimize reconstruction error
$$\left \| X - UV^{T} \right \| \rightarrow \min_{U, V}$$

<b>Why do we need that</b><br>
$\bigstar$ Compress data<br>
-- |code| << |data|<br>
$\bigstar$ Dimensionality reduction<br>
-- Before feeding data to your XGBoost<br>
$\bigstar$ Learn some great features<br>
$\bigstar$ Unsupervised pretraining<br>
$\bigstar$ Generate new data<br><br>

<b>Autoencoder types</b><br>
$\bigstar$ Expanding autoencoder<br>
-- In this case, autoencoder can learn in such a way that doesn't produce good features. So H shoudl be smaller than D.
$$L = \left \| X - Dec(Enc(X)) \right \|$$
$\bigstar$ Sparse autoencoder<br>
-- $\bigstar$ L1 on activation, sparse code
$$L = \left \| X - Dec(Enc(X)) \right \| + \sum_{i} \left | Enc_{i}(X) \right |$$
$\bigstar$ Redundant autoencoder<br>
-- noize/dropout, redundant code
$$L = \left \| X - Enc(Dec(Noize(X))) \right \|$$
$\bigstar$ Denoizing autoencoder<br>
-- distort input, learn to fix distorsion
$$L = \left \| X - Enc(Dec(Noize(X))) \right \|$$
</p>


## 6. Generative Adversarial Networks
<p align="justify">
<b>Image generation</b><br>
<b>Mean Square Error: Pixelwise MSE</b><br>
$\bigstar$ A <b>cat on the left</b> is closer to <b>dog on the left</b> than to <b>cat on the right</b><br>
$\bigstar$ We may want to avoid that effect<br>
$\bigstar$ Can we obtain image representation that is less sensitive to small shifts?<br><br>

<b>Which of those image representations is least sensitive to small shifts of objects on an image?</b><br>
A. Image pixels in CMYK format<br>
B. Image pixels in RGB format<br>
C. Image pixels in RGB after x2 super-resolution (linear interpolation)<br>
D. Activation of pre-final layer of a convolutional network trained on imagenet<br>
E. Activation of first layer of a convolutional network trained on imagenet<br>
<b>Answer:</b> D.<br><br>

<b>Generative Adversarial Networks</b><br>
For Generator
$$L_{G} = -\log[1 - Disc(Gen(Seed))]$$

For Discriminator
$$L_{D} = -\log[1 - Disc(RealData)] - \log[ Disc(Gen(Seed)) ]$$

$\bigstar$ Training process<br>
(1) initialize generator and discriminator weights at random<br>
(2) train discriminator on to classify actual images against images generated by <b>untrained</b> generator<br>
(3) train generator to generate images that fool discriminator into believing they're real<br>
(4) train discriminator again on images generated by updated generator<br><br>

$\bigstar$ Algorithm<br>
Sample noize z and images x<br>
for k in 1...K<br>
&emsp;Train discriminator(x), discriminator(generator(z))<br>
for m in 1...M<br>
&emsp;Train generator(z)<br><br>

<b>Adversarial domain adaptation</b><br>
$\bigstar$ Two domain<br>
-- e.g. mnist digits vs actual digits on photo<br>
$\bigstar$ First domain is labeled, while second is not<br>
$\bigstar$ Wanna learn for second domian<br><br>

<b>Domain adaptation</b>
$$-\log P(real \mid h(x_{real})) - \log[1-P(real \mid h(x_{mc}))] \rightarrow \min_{discriminator}$$
$$L_{classifier}(y_{mc}, y(h(x_{mc}))) - \log P(real \mid h(x_{mc})) \rightarrow \min_{classifier}$$

<b>Art style transfer</b>
$$L = \left \| \text{Texture}(x_{ref}) - \text{Texture}(x_{cand}) \right \| + \left \| \text{Content}(x_{orig}) - \text{Content}(x_{cand}) \right \|$$
</p>