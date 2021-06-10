---
layout: post
title:  "CIFAR10 Classification"
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
The complete code is available <a href="https://github.com/chaopan1995/PROJECTS/tree/master/CIFAR10-Classification">here</a>.
</p>


## 1. Object
<p align="justify">
$\bigstar$ Your first CNN on CIFAR-10<br><br>

$\bigstar$ In this task you will:<br>
-- define your first CNN architecture for CIFAR-10 dataset<br>
-- train it from scratch<br><br>

CIFAR-10 dataset contains 32x32 color images from 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck:
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/CIFAR10Classification/1.png"/></center>
</p>


## 2. Packages
{% highlight python %}
import numpy as np
from tensorflow import __version__
import matplotlib.pyplot as plt
%matplotlib inline
# import necessary building blocks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,\
Activation, Dropout, LeakyReLU
from tensorflow.keras.backend import clear_session, get_value
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix, accuracy_score
{% endhighlight %}
<p align="justify">
We use Tensorflow 2.x
</p>


## 3. Dataset
<p align="justify">
Dataset cifar10 has been already implemented in keras. Training set is 50000 and test set is 10000. Besides, each image has a shape of (32, 32, 3) and it beongs to one of ten classes.
</p>
{% highlight python %}
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)
NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]
# show random images from train
def show_dataset():
    cols = 8
    rows = 2
    fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            random_index = np.random.randint(0, len(y_train))
            ax = fig.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid('off')
            ax.axis('off')
            ax.imshow(x_train[random_index, :])
            ax.set_title(cifar10_classes[y_train[random_index, 0]])
    plt.show()

show_dataset()
{% endhighlight %}


## 4. Preprocess
<p align="justify">
We need to normalize inputs like this:
$$x_{norm} = \frac{x}{255} - 0.5$$

We need to convert class labels to one-hot encoded vectors.
</p>
{% highlight python %}
# normalize inputs
x_train2 = x_train/255.- 0.5
x_test2 = x_test/255. - 0.5
# convert class labels to one-hot encoded, should have shape (?, NUM_CLASSES)
y_train2 = to_categorical(y_train)
y_test2 = to_categorical(y_test)
{% endhighlight %}


## 5. Define CNN Architecture
<p align="justify">
<b>Convolutional networks are built from several types of layers:</b><br>
$\bigstar$ Conv2D<br>
-- filters: number of output channels<br>
-- kernel_size: an integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution window<br>
-- padding: padding="same" adds zero padding to the input, so that the output has the same width and height, padding='valid' performs convolution only in locations where kernel and the input fully overlap<br>
-- activation: "relu", "tanh", etc<br>
-- input_shape: shape of input.<br>
$\bigstar$ MaxPooling2D<br>
$\bigstar$ Flatten: flattens the input, does not affect the batch size<br>
$\bigstar$ Dense: fully-connected layer<br>
$\bigstar$ Activation: applies an activation function<br>
$\bigstar$ LeakyReLU: applies leaky relu activation<br>
$\bigstar$ Dropout: applies dropout.
</p>
{% highlight python %}
tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)

tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs
)

tf.keras.layers.Flatten(
    data_format=None, **kwargs
)

tf.keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)

tf.keras.layers.Activation(activation, **kwargs)

tf.keras.layers.LeakyReLU(alpha=0.3, **kwargs)

tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
{% endhighlight %}
<p align="justify">
You need to define a model which takes (None, 32, 32, 3) input and predicts (None, 10) output with probabilities for all classes. None in shapes stands for batch dimension.<br><br>

Stack 4 convolutional layers with kernel size (3, 3) with growing number of filters (16, 32, 32, 64), use "same" padding.<br><br>

Add 2x2 pooling layer after every 2 convolutional layers (conv-conv-pool scheme).<br><br>

Use LeakyReLU activation with recommended parameter 0.1 for all layers that need it (after convolutional and dense layers)<br><br>

Add a dense layer with 256 neurons and a second dense layer with 10 neurons for classes. Remember to use Flatten layer before first dense layer to reshape input volume into a flat vector!<br><br>

Add Dropout after every pooling layer (0.25) and between dense layers (0.5).
</p>
{% highlight python %}
def make_model():
    """
    Define your model architecture here.
    Returns `Sequential` model.
    """
    # clear default graph
    clear_session()
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding="same",
                     activation="relu", input_shape=(32, 32, 3)))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(LeakyReLU(0.1))
    model.add(Activation("softmax"))
    return model
{% endhighlight %}


## 6. Train
{% highlight python %}
# initial learning rate
INIT_LR = 5e-3
BATCH_SIZE = 32
EPOCHS = 10
model_filename = 'cifar.{0:03d}.hdf5'

# define our model
model = make_model()

# prepare model for fitting (loss, optimizer, etc)
model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=Adamax(lr=INIT_LR),  # for SGD
    metrics=['accuracy']  # report accuracy during training
)

# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch

# fit model
model.fit(
    x_train2, y_train2,  # prepared data
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[callbacks.LearningRateScheduler(lr_scheduler)],
    validation_data=(x_test2, y_test2),
    shuffle=True,
    verbose=1
)
{% endhighlight %}


## 7. Evaluate Model
{% highlight python %}
# make test predictions
y_pred_test = model.predict(x_test2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)

# confusion matrix and accuracy
plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(y_test, y_pred_test_classes))
plt.xticks(np.arange(10), cifar10_classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), cifar10_classes, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(y_test, y_pred_test_classes))

# inspect preditions
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_test))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_test[random_index, :])
        pred_label = cifar10_classes[y_pred_test_classes[random_index]]
        pred_proba = y_pred_test_max_probas[random_index]
        true_label = cifar10_classes[y_test[random_index, 0]]
        ax.set_title("pred: {}\nscore: {:.3}\ntrue: {}".format(
               pred_label, pred_proba, true_label
        ))
plt.show()
{% endhighlight %}
<p align="justify">
Confusion matrix and accuracy
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/CIFAR10Classification/2.png"/></center>
</p>
<p align="justify">
Inspect preditions
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/CIFAR10Classification/3.png"/></center>
</p>