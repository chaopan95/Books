---
layout: page
title:  "Deep Learning"
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
<a href="https://www.coursera.org/account/accomplishments/specialization/certificate/KHD6QJD3VPLT"> My certificate.</a><br><br>
</p>


## 4. Convolutional Neural Networks
### 4.2 Case studies
#### 4.2.3 Quiz: Deep convolutional models
<p align="justify">
<b>1.</b><br>
Which of the following do you typically see as you move to deeper layers in a ConvNet?<br>
A. $n_{H}$ and $n_{W}$ decrease, while $n_{C}$ increases<br>
B. $n_{H}$ and $n_{W}$ increases, while $n_{C}$ also increases<br>
C. $n_{H}$ and $n_{W}$ increases, while $n_{C}$ decreases<br>
D. $n_{H}$ and $n_{W}$ decreases, while $n_{C}$ also decreases<br>
<b>Answer</b>: A.<br><br>

<b>2.</b><br>
Which of the following do you typically see in a ConvNet? (Check all that apply.)<br>
A. Multiple CONV layers followed by a POOL layer<br>
B. Multiple POOL layers followed by a CONV layer<br>
C. FC layers in the last few layers<br>
D. FC layers in the first few layers<br>
<b>Answer</b>: A, C.<br><br>

<b>3.</b><br>
In order to be able to build very deep networks, we usually only use pooling layers to downsize the height/width of the activation volumes while convolutions are used with “valid” padding. Otherwise, we would downsize the input of the model too quickly.<br>
A. True<br>
B. False<br>
<b>Answer</b>: B.<br><br>

<b>4.</b><br>
Training a deeper network (for example, adding additional layers to the network) allows the network to fit more complex functions and thus almost always results in lower training error. For this question, assume we’re referring to “plain” networks.<br>
A. True<br>
B. False<br>
<b>Answer</b>: B.<br><br>

<b>5.</b><br>
The following equation captures the computation in a ResNet block. What goes into the two blanks above?
$$a^{[l+2]} = g(W^{[l+2]} g(W^{[l+1]} a^{[l]} + b^{[l+1]}) + b^{[l+2]} + \text{ ... }) + \text{ ... } $$
A. 0 and $z^{[l+1]}$<br>
B. $a^{[l]}$ and 0<br>
C. $z^{[l]}$ and $a^{[l]}$<br>
D. 0 and $a^{[l]}$<br>
<b>Answer</b>: B.<br><br>

<b>6.</b><br>
Which ones of the following statements on Residual Networks are true? (Check all that apply.)<br>
A ResNet with L layers would have on the order of $L^{2}$ skip connections in total.<br>
B. The skip-connections compute a complex non-linear function of the input to pass to a deeper layer in the network.<br>
C. Using a skip-connection helps the gradient to backpropagate and thus helps you to train deeper networks<br>
D. The skip-connection makes it easy for the network to learn an identity mapping between the input and the output within the ResNet block.<br>
<b>Answer</b>: C, D.<br><br>

<b>7.</b><br>
Suppose you have an input volume of dimension 64x64x16. How many parameters would a single 1x1 convolutional filter have (including the bias)?<br>
A. 4097<br>
B. 1<br>
C. 2<br>
D. 17<br>
<b>Answer</b>: D.<br><br>

<b>8.</b><br>
Suppose you have an input volume of dimension $n_{H}$ x $n_{W}$ x $n_{C}$. Which of the following statements you agree with? (Assume that “1x1 convolutional layer” below always uses a stride of 1 and no padding.)<br>
A. You can use a 1x1 convolutional layer to reduce $n_{H}$, $n_{W}$, and $n_{C}$.<br>
B. You can use a pooling layer to reduce $n_{H}$, $n_{W}$, but not $n_{C}$.<br>
C. You can use a pooling layer to reduce $n_{H}$, $n_{W}$, and $n_{C}$.<br>
D. You can use a 1x1 convolutional layer to reduce $n_{C}$ but not $n_{H}$, $n_{W}$.<br>
<b>Answer</b>: B, D.<br><br>

<b>9.</b><br>
Which ones of the following statements on Inception Networks are true? (Check all that apply.)<br>
A. A single inception block allows the network to use a combination of 1x1, 3x3, 5x5 convolutions and pooling.<br>
B. Inception blocks usually use 1x1 convolutions to reduce the input data volume’s size before applying 3x3 and 5x5 convolutions.<br>
C. Inception networks incorporates a variety of network architectures (similar to dropout, which randomly chooses a network architecture on each step) and thus has a similar regularizing effect as dropout.<br>
D. Making an inception network deeper (by stacking more inception blocks together) should not hurt training set performance.<br>
<b>Answer</b>: A, B.<br><br>

<b>10.</b><br>
Which of the following are common reasons for using open-source implementations of ConvNets (both the model and/or weights)? Check all that apply.<br>
A. It is a convenient way to get working an implementation of a complex ConvNet architecture.<br>
B. The same techniques for winning computer vision competitions, such as using multiple crops at test time, are widely used in practical deployments (or production system deployments) of ConvNets.<br>
C. Parameters trained for one computer vision task are often useful as pretraining for other computer vision tasks.<br>
D. A model trained for one computer vision task can usually be used to perform data augmentation even for a different computer vision task.<br>
<b>Answer</b>: A, C.<br><br>
</p>

#### 4.2.4 Programming Assignment: Residual Networks
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>

### 4.3 Detection
#### 4.3.1 Algorithms
<p align="justify">
<b>Sliding Windows</b><br>

<b>Bounding Box Predictions</b><br>

<b>Intersection Over Union</b><br>
More generally, IoU is a measure of the overlap between 2 bounding boxes
$$\text{IoU} = \frac{\text{Intersection area}}{\text{Union area}} \geq 0.5$$

<b>Non-max Suppression</b><br>

<b>Anchor Boxes</b><br>

<b>YOLO Algorithm</b><br>

<b>Region Proposals</b><br>

</p>

#### 4.3.2 Quiz: Detection algorithms
<p align="justify">
<b>1.</b><br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
<b>Answer</b>: .<br><br>

<b>9.</b><br>
<b>Answer</b>: .<br><br>

<b>10.</b><br>
<b>Answer</b>: .<br><br>
</p>

#### 4.3.3 Programming Assignment: Car detection with YOLO
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>

### 4.4 Face Recognition
#### 4.4.1 What is face recognition?
<p align="justify">

<b></b><br>

<b></b><br>
</p>

#### 4.5.2 One Shot Learning
<p align="justify">

</p>

#### 4.5.3 Siamese Network
<p align="justify">

</p>

#### 4.5.4 Triplet Loss
<p align="justify">

</p>

#### 4.5.5 Face Verification and Binary Classification
<p align="justify">

</p>

#### 4.4.2 Neural Style Transfer
#### 4.6.1 What is neural style transfer?
<p align="justify">

</p>

#### 4.6.2 What are deep ConvNets learning?
<p align="justify">

</p>

#### 4.6.3 Cost Function
<p align="justify">

</p>

#### 4.6.4 Content Cost Function
<p align="justify">

</p>

#### 4.6.5 Style Cost Function
<p align="justify">

</p>

#### 4.6.6 1D and 3D Generalizations
<p align="justify">

</p>

#### 4.4.3 Quiz: Special applications: Face recognition & Neural style transfer
<p align="justify">
<b>1.</b><br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
<b>Answer</b>: .<br><br>

<b>9.</b><br>
<b>Answer</b>: .<br><br>

<b>10.</b><br>
<b>Answer</b>: .<br><br>
</p>

#### 4.4.4 Programming Assignment: Art generation with Neural Style Transfer
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>

#### 4.4.5 Programming Assignment: Face Recognition
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>


## 5. Sequence Models
### 5.1 Recurrent Neural Networks
#### 5.1.13 Quiz: Recurrent Neural Networks
<p align="justify">
<b>1.</b><br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
<b>Answer</b>: .<br><br>

<b>9.</b><br>
<b>Answer</b>: .<br><br>

<b>10.</b><br>
<b>Answer</b>: .<br><br>
</p>

#### 5.1.14 Programming Assignment: Building a recurrent neural network - step by step
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>

#### 5.1.15 Programming Assignment: Dinosaur Island - Character-Level Language Modeling
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>

#### 5.1.16 Programming Assignment: Jazz improvisation with LSTM
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>

### 5.2 Introduction to Word Embeddings
#### 5.2.1 Word Representation
<p align="justify">

</p>

#### 5.2.2 sing word embeddings
<p align="justify">

</p>

#### 5.2.3 Properties of word embeddings
<p align="justify">

</p>

#### 5.2.4 Embedding matrix
<p align="justify">

</p>

### 5.3 Learning Word Embeddings: Word2vec & GloVe
#### 5.3.1 Learning word embeddings
<p align="justify">

</p>

#### 5.3.2 Word2Vec
<p align="justify">

</p>

#### 5.3.3 Negative Sampling
<p align="justify">

</p>

#### 5.3.4 GloVe word vectors
<p align="justify">

</p>

### 5.4 Applications using Word Embeddings
#### 5.4.1 Sentiment Classification
<p align="justify">

</p>

#### 5.4.2 Debiasing word embeddings
<p align="justify">

</p>

#### 5.4.3 Quiz: Natural Language Processing & Word Embeddings
<p align="justify">
<b>1.</b><br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
<b>Answer</b>: .<br><br>

<b>9.</b><br>
<b>Answer</b>: .<br><br>

<b>10.</b><br>
<b>Answer</b>: .<br><br>
</p>

#### 5.4.4 Programming Assignment: Operations on word vectors - Debiasing
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>

#### 5.4.5 Programming Assignment: Emojify
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>

### 5.5 Various sequence to sequence architectures
#### 5.5.1 Basic Models
<p align="justify">

</p>

#### 5.5.2 Picking the most likely sentence
<p align="justify">

</p>

#### 5.5.3 Beam Search
<p align="justify">

</p>

#### 5.5.4 Refinements to Beam Search
<p align="justify">

</p>

#### 5.5.5 Error analysis in beam search
<p align="justify">

</p>

#### 5.5.6 Bleu Score (optional)
<p align="justify">

</p>

#### 5.5.7 Attention Model Intuition
<p align="justify">

</p>

#### 5.5.8 Attention Model
<p align="justify">

</p>

#### 5.5.9 Quiz: Sequence models & Attention mechanism
<p align="justify">
<b>1.</b><br>
<b>Answer</b>: .<br><br>

<b>2.</b><br>
<b>Answer</b>: .<br><br>

<b>3.</b><br>
<b>Answer</b>: .<br><br>

<b>4.</b><br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
<b>Answer</b>: .<br><br>

<b>6.</b><br>
<b>Answer</b>: .<br><br>

<b>7.</b><br>
<b>Answer</b>: .<br><br>

<b>8.</b><br>
<b>Answer</b>: .<br><br>

<b>9.</b><br>
<b>Answer</b>: .<br><br>

<b>10.</b><br>
<b>Answer</b>: .<br><br>
</p>

#### 5.5.10 Programming Assignment: Neural Machine Translation with Attention
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>

#### 5.5.11 Programming Assignment: Trigger word detection
{% highlight Python %}

{% endhighlight %}
<p align="justify">
<br>
</p>

### 5.6 Speech recognition - Audio data
#### 5.6.1 Speech recognition
<p align="justify">

</p>

#### 5.6.2 Trigger Word Detection
<p align="justify">

</p>
