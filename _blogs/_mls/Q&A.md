---
layout: post
title:  "Q & A"
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


## 1. Deep Learning
<p align="justify">
<b>1. Why is it necessary to introduce non-linearities in a neural network?</b><br>
Solution: otherwise, we would have a composition of linear functions, which is also a linear function, giving a linear model. A linear model has a much smaller number of parameters, and is therefore limited in the complexity it can model.<br><br>

<b>2. Describe two ways of dealing with the vanishing gradient problem in a neural network.</b><br>
Using ReLU activation instead of sigmoid.<br>
Using Xavier initialization.<br><br>

<b>3. What are some advantages in using a CNN (convolutional neural network) rather than a DNN (dense neural network) in an image classification task?</b><br>
Solution: while both models can capture the relationship between close pixels, CNNs have the following properties:
It is translation invariant — the exact location of the pixel is irrelevant for the filter.<br>
It is less likely to overfit — the typical number of parameters in a CNN is much smaller than that of a DNN.<br>
Gives us a better understanding of the model — we can look at the filters’ weights and visualize what the network “learned”.<br>
Hierarchical nature — learns patterns in by describing complex patterns using simpler ones.<br><br>

<b>4. Describe two ways to visualize features of a CNN in an image classification task.</b><br>
输入遮挡：遮挡输入图像的一部分，看看哪部分对分类的影响最大。 例如，针对某个训练好的图像分类模型，将下列图像作为输入。如果我们看到第三幅图像被分类为狗狗的概率为98%，而第二幅图像的准确率仅为65%，则说明眼睛对于对分类的影响更大<br>
激活最大化：创建一个人造的输入图像，以最大化目标响应（梯度上升)<br><br>

<b>5. Is trying the following learning rates: 0.1,0.2,…,0.5 a good strategy to optimize the learning rate?</b><br>
Solution: No, it is recommended to try a logarithmic scale to optimize the learning rate.<br><br>

<b>6. Suppose you have a NN with 3 layers and ReLU activations. What will happen if we initialize all the weights with the same value? what if we only had 1 layer (i.e linear/logistic regression?)</b><br>
Solution: If we initialize all the weights to be the same we would not be able to break the symmetry; i.e, all gradients will be updated the same and the network will not be able to learn. In the 1-layers scenario, however, the cost function is convex (linear/sigmoid) and thus the weights will always converge to the optimal point, regardless of the initial value (convergence may be slower).<br><br>

<b>7. Explain the idea behind the Adam optimizer.</b><br>
Solution: Adam, or adaptive momentum, combines two ideas to improve convergence: per-parameter updates which give faster convergence, and momentum which helps to avoid getting stuck in saddle point.<br><br>

<b>8. Compare batch, mini-batch and stochastic gradient descent.</b><br>
8.比较批处理，小批处理和随机梯度下降。
Solution: batch refers to estimating the data by taking the entire data, mini-batch by sampling a few datapoints, and SGD refers to update the gradient one datapoint at each epoch. The tradeoff here is between how precise the calculation of the gradient is versus what size of batch we can keep in memory. Moreover, taking mini-batch rather than the entire batch has a regularizing effect by adding random noise at each epoch.<br><br>

<b>9. What is data augmentation? Give examples.</b><br>
Solution: Data augmentation is a technique to increase the input data by performing manipulations on the original data. For instance in images, one can: rotate the image, reflect (flip) the image, add Gaussian blur<br><br>

<b>10. What is the idea behind GANs?</b><br>
Solution: GANs, or generative adversarial networks, consist of two networks (D,G) where D is the “discriminator” network and G is the “generative” network. The goal is to create data — images, for instance, which are undistinguishable from real images. Suppose we want to create an adversarial example of a cat. The network G will generate images. The network D will classify images according to whether they are a cat or not. The cost function of G will be constructed such that it tries to “fool” D — to classify its output always as cat.<br><br>

<b>11. What are the advantages of using Batchnorm?</b><br>
Solution: Batchnorm accelerates the training process. It also (as a byproduct of including some noise) has a regularizing effect.<br><br>

<b>12. What is multi-take learning? When should it be used?</b><br>
Solution: Multi-tasking is useful when we have a small amount of data for some task, and we would benefit from training a model on a large dataset of another task. Parameters of the models are shared — either in a “hard” way (i.e the same parameters) or a “soft” way (i.e regularization/penalty to the cost function).<br><br>

<b>13. What is end-to-end learning? Give a few of its advantages.</b><br>
Solution: End-to-end learning is usually a model which gets the raw data and outputs directly the desired outcome, with no intermediate tasks or feature engineering. It has several advantages, among which: there is no need to handcraft features, and it generally leads to lower bias.<br><br>

<b>14. What happens if we use a ReLU activation and then a sigmoid as the final layer?</b><br>
Solution: Since ReLU always outputs a non-negative result, the network will constantly predict one class for all the inputs.<br><br>

<b>15. How to solve the exploding gradient problem?</b><br>
Solution: A simple solution to the exploding gradient problem is gradient clipping — taking the gradient to be ±M when its absolute value is bigger than M, where M is some large number.<br><br>

<b>16. Is it necessary to shuffle the training data when using batch gradient descent?</b><br>
Solution: No, because the gradient is calculated at each epoch using the entire training data, so shuffling does not make a difference.<br><br>

<b>17. When using mini batch gradient descent, why is it important to shuffle the data?</b><br>
答：如果不打乱数据的顺序，那么假设我们训练一个神经网络分类器，且有两个类别：A和B，那么各个epoch中的所有小批量都会完全相同，这会导致收敛速度变慢，甚至导致神经网络对数据的顺序产生倾向性。<br><br>

<b>18. Describe some hyperparameters for transfer learning.</b><br>
Solution: How many layers to keep, how many layers to add, how many to freeze.<br><br>

<b>19. Is dropout used on the test set?</b><br>
Solution: No! only in the train set. Dropout is a regularization technique that is applied in the training process.<br><br>

<b>20. Explain why dropout in a neural network acts as a regularizer.</b><br>
Solution: There are several (related) explanations to why dropout works. It can be seen as a form of model averaging — at each step we “turn off” a part of the model and average the models we get. It also adds noise, which naturally has a regularizing effect. It also leads to more sparsity of the weights and essentially prevents co-adaptation of neurons in the network.<br><br>

<b>21. Give examples in which a many-to-one RNN architecture is appropriate.</b><br>
Solution: A few examples are: sentiment analysis, gender recognition from speech<br><br>

<b>22. When can’t we use BiLSTM? Explain what assumption has to be made.</b><br>
Solution: in any bi-directional model, we assume that we have access to the next elements of the sequence in a given “time”. This is the case for text data (i.e sentiment analysis, translation etc.), but not the case for time-series data.<br><br>

<b>23. True/false: adding L2 regularization to a RNN can help with the vanishing gradient problem.</b><br>
Solution: false! Adding L2 regularization will shrink the weights towards zero, which can actually make the vanishing gradients worse in some cases.<br><br>

<b>24. Suppose the training error/cost is high and that the validation cost/error is almost equal to it. What does it mean? What should be done?</b><br>
Solution: this indicates underfitting. One can add more parameters, increase the complexity of the model, or lower the regularization.<br><br>

<b>25. Describe how L2 regularization can be explained as a sort of a weight decay.</b><br>
Solution: Suppose our cost function is C(w), and that we add a penalization $\alpha \left \| w \right \|^{2}$. When using gradient descent, the iterations will look like
$$w = w - \triangledown C(w) - 2\alpha w = (1-2\alpha)w -\triangledown C(w)$$

In this equation, the weight is multiplied by a factor < 1<br><br>
</p>
