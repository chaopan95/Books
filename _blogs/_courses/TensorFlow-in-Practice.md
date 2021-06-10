---
layout: page
title:  "TensorFlow in Practice"
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
<a href="https://www.coursera.org/account/accomplishments/specialization/certificate/FQY37DGM28HU"> My certificate.</a><br>
</p>


## 1. Introduction to TensorFlow
### 1.1 A new programming paradigm
#### 1.1.1 What is TensorFlow
<p align="justify">
<b>TensorFlow DL Framework</b><br>
$\bigstar$ We will use it in Jupyter Notebook with Python 3 kernel<br>
</p>
{% highlight python %}
import numpy as np
import tensorflow as tf
{% endhighlight %}
<p align="justify">
$\bigstar$ We will overview Python API for EnsorFlow 1.2+<br>
$\bigstar$ APIs in other langauges exist: Java, C++, Go<br>
$\bigstar$ Python API is at present the most complete and essiest to use<br><br>

<b>What is TensorFlow?</b><br>
$\bigstar$ A tool to describe computational graphs<br>
-- The foundation of computation in TnesorFlow is the <b>Graph</b> object. This holds a network of nodes, each representing one <b>operation</b>, connected to each other as inputs and outputs<br>
$\bigstar$ A runtime for execution of these graphs<br>
-- On CPU, GPU, TPU<br>
-- On one node or in distributed node<br><br>

<b>Why this name</b><br>
$\bigstar$ Input to any operation will be a collection of tensors<br>
$\bigstar$ Output will be a collection of tensors as well<br>
$\bigstar$ We will have a graph of operations, each of which transforms tensors into another tensors, so it's a kind of flow of tensors<br><br>

<b>How the input looks like</b><br>
$\bigstar$ Placeholder<br>
-- This is placeholder for tensor, which will be fed during graph execution (e.g. input features)
</p>
{% highlight python %}
x = tf.placeholder(tf.float32, (None, 10))
{% endhighlight %}
<p align="justify">
$\bigstar$ Variable<br>
-- This is a tensor with some value that is updated during execution (e.g. weights matrix in MLP)
</p>
{% highlight python %}
w = tf.get_variable('w', shape=(10, 20), dtype=tf.float32)
w = tf.Variable(tf.random_uniform((10, 20)), name='w')
{% endhighlight %}
<p align="justify">
$\bigstar$ Constant<br>
-- This is a tensor with a value, that cannot be changed
</p>
{% highlight python %}
c = tf.constant(np.ones((4, 4)))
{% endhighlight %}
<p align="justify">
<b>Operation example</b><br>
$\bigstar$ Matrix product<br>
</p>
{% highlight python %}
x = tf.placeholder(tf.float32, (None, 10))
w = tf.Variable(tf.random_uniform((10, 20)), name='w')
z = x @ w  # z = tf.matmul(x, w)
print(z)
{% endhighlight %}
<p align="justify">
$\bigstar$ Output<br>
</p>
{% highlight python %}
Tensor('matmul:0', shape=(?, 20), dtype=float32)
{% endhighlight %}
<p align="justify">
$\bigstar$ We don't do any computations here, we just define the graph<br><br>

<b>Computational graph</b><br>
$\bigstar$ Tensorflow creates a default graph after importing<br>
-- All the operations will go there by default<br>
-- You can get it with tf.get_default_graph(), which returns an instance of tf.Graph<br>
$\bigstar$ You can creat your own graph variable and define operations
</p>
{% highlight python %}
g = tf.Graph()
with g.as_default():
    pass
{% endhighlight %}
<p align="justify">
$\bigstar$ You can clear the default graph
</p>
{% highlight python %}
tf.reset_default_graph()
{% endhighlight %}
<p align="justify">
<b>Jupyter Notebook cells</b><br>
$\bigstar$ If you run this 3 times
</p>
{% highlight python %}
x = tf.placeholder(tf.float32, (None, 10))
{% endhighlight %}
<p align="justify">
$\bigstar$ This is what you get in your default graph<br>
-- using tf.get_default_graph().get_operations()<br>
[&lt;tf.Operation 'PLaceholder' type=Placeholder&gt;, &lt;tf.Operation 'PLaceholder_1' type=Placeholder&gt;, &lt;tf.Operation 'PLaceholder_2' type=Placeholder&gt;]<br>
$\bigstar$ Your graph is cluttered<br>
-- Clean your graph with tf.reset_default_graph()<br><br>

<b>Operations and tensors</b><br>
$\bigstar$ Every node in our graph is an operation
</p>
{% highlight python %}
x = tf.placeholder(tf.float32, (None, 10), name='x')
{% endhighlight %}
<p align="justify">
$\bigstar$ Listing nodes with tf.get_default_graph().get_operations()<br>
[&lt;tf.Operation 'x' type=PLaceholder&gt;]<br>
$\bigstar$ How to get outputs tensors of operations<br>
-- tf.get_default_graph().get_operations()[0].outputs()<br>
-- Output: [&lt;tf.Tensor x:0' shape=(?, 10) dtype=float32&gt;]<br><br>

<b>Running a graph</b><br>
$\bigstar$ A tf.Session object encapsulates the environment in which tf.Operation objects are executed and tf.Tnesor objects are evaluated<br>
$\bigstar$ Create a session
</p>
{% highlight python %}
s = tf.InteractiveSession()
{% endhighlight %}
<p align="justify">
$\bigstar$ Define a graph
</p>
{% highlight python %}
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b
{% endhighlight %}
<p align="justify">
$\bigstar$ Run a graph
</p>
{% highlight python %}
print(c)
print(s.run(c))
{% endhighlight %}
<p align="justify">
$\bigstar$ Output<br>
Tensor('mul:0', shape=(), dtype=float32)<br>
30.0<br><br>

$\bigstar$ Operations are written in C++ and executed on CPU or GPU<br>
$\bigstar$ tf.Session owns necessary resources to execute your graph, such as tf.Variable that occupy RAM<br>
$\bigstar$ It is important to release these resources when they are no longer required with tf.Session.close()<br><br>

<b>Initialization of variables</b><br>
$\bigstar$ A variable has an initial value<br>
-- Tensor: tf.Variable(tf.random_uniform((10, 20)), name='w')<br>
-- Initializer: tf.get_variable('w', shape=(10, 20), dtype=tf.float32)<br>
$\bigstar$ You need to run some code to compute that initial value in graph execution environment<br>
$\bigstar$ This is done with a call in your session s<br>
-- s.run(tf.global_variables_initializer())<br>
$\bigstar$ Without it you will get 'Attempting to use uninitialized value' error<br><br>

<b>Example</b><br>
$\bigstar$ Definition
</p>
{% highlight python %}
tf.reset_default_graph()
a = tf.constant(np.ones((2, 2), dtype=np.float32))
b = tf.Variable(tf.ones((2, 2)))
c = a @ b
{% endhighlight %}
<p align="justify">
$\bigstar$ Running attempt
</p>
{% highlight python %}
s = tf.InterativeSession()
s.run(c)
{% endhighlight %}
<p align="justify">
-- Output<br>
'Attempting to use uninitialized value' error<br>
$\bigstar$ Running properly
</p>
{% highlight python %}
s.run(tf.global_variables_initializer())
s.run(c)
{% endhighlight %}
<p align="justify">
-- Output<br>
array([[2, 2], [2, 2]], dtype=float32)<br><br>

<b>Feeding placeholder values</b><br>
$\bigstar$ Definition
</p>
{% highlight python %}
tf.reset_default_graph()
a = tf.placeholder(np.float32, (2, 2))
b = tf.Variable(tf.ones((2, 2)))
c = a @ b
{% endhighlight %}
<p align="justify">
$\bigstar$ Running attempt
</p>
{% highlight python %}
s = tf.InterativeSession()
s.run(tf.global_variables_initializer())
s.run(c)
{% endhighlight %}
<p align="justify">
-- Output<br>
'You must feed a value for placeholder tensor' error<br>
$\bigstar$ Running properly
</p>
{% highlight python %}
s.run(tf.global_variables_initializer())
s.run(c, feed_dict={a: np.ones((2, 2))})
{% endhighlight %}
<p align="justify">
-- Output<br>
array([[2, 2], [2, 2]], dtype=float32)<br><br>

<b>Summary</b><br>
$\bigstar$ TensorFlow: defining and running computational graphs<br>
$\bigstar$ Nodes of a graph are operations, that convert a collection of tensors into another collection of tensors<br>
$\bigstar$ In Python API you define the graph, you don't execute it along the way<br>
-- In 1.5+ the latter mode is supported: eager execution<br>
$\bigstar$ You create a session to execute your graph (fast C++ code on CPU or GPU)<br>
$\bigstar$ Session owns all the resources (tensors eat RAM)<br>
</p>

#### 1.6.2 Our First Model in TensorFlow
<p align="justify">
<b>Optimizes in TensorFlow</b><br>
$\bigstar$ Let's define f as a suqare of variable x
</p>
{% highlight python %}
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
x = tf.get_variable('x', shape=(), dtype=tf.float32)
f = x ** 2
{% endhighlight %}
<p align="justify">
$\bigstar$ Let's say we want to optimize the value of f
</p>
{% highlight python %}
optimizer = tf.tran.GradientDescentOptimizer(0.1)
setp = optimizer.minimize(f, var_list=[x])
{% endhighlight %}
<p align="justify">
$\bigstar$ You don't have to specify all the optimized variables
</p>
{% highlight python %}
setp = optimizer.minimize(f, var_list=[x])
setp = optimizer.minimize(f)
{% endhighlight %}
<p align="justify">
$\bigstar$ Because all variables are trainable by default
</p>
{% highlight python %}
x = tf.get_variable('x', shape=(), dtype=tf.float32)
x = tf.get_variable('x', shape=(), dtype=tf.float32, trainable=True)
{% endhighlight %}
<p align="justify">
$\bigstar$ You can get all of them
</p>
{% highlight python %}
tf.trainable_variables()
{% endhighlight %}
<p align="justify">
$\bigstar$ Output<br>
-- [&lt;tf.Variable 'x:0' shape=() dtype=float32_ref&gt;]<br><br>

<b>Making gradient descent steps</b><br>
$\bigstar$ Now we need to create a session and initialize variables
</p>
{% highlight python %}
s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())
{% endhighlight %}
<p align="justify">
$\bigstar$ We are ready to make 10 gradient descent steps
</p>
{% highlight python %}
for i in range(10):
    _, cur_x, cur_f = s.run([step, x, f])
    print(cur_x, cur_f)
{% endhighlight %}
<p align="justify">
$\bigstar$ Output<br>
0.448929 0.314901<br>
0.359143 0.201537<br>
...<br>
0.0753177 0.00886368<br>
0.0602542 0.00567276<br><br>

<b>Logging with Tnesorboard</b><br>
$\bigstar$ We can add so-called summaries
</p>
{% highlight python %}
tf.summary.sclar('cur_x', x)
tf.summary.scalr('cur_f', f)
summaries = tf.summary.merge_all()
{% endhighlight %}
<p align="justify">
$\bigstar$ This is how we log these summaries
</p>
{% highlight python %}
s = tf.InteractiveSession()
summary_writer = tf.summary.FileWriter('logs/1', s.graph)
s.run(tf.global_variables_initializer())
for i in range(10):
    _, cur_summaries = s.run([step, summaries])
    summary_writer.add_summary(cur_summaries)
    summary_writer.flush()
{% endhighlight %}
<p align="justify">
<b>Solving a linear regression</b><br>
$\bigstar$ Let's generate a model dataset<br>
-- N = 1000<br>
-- D = 3<br>
-- x = np.random.random((N, D))<br>
-- y = x @ w + np.random.randn(N, 1)*0.2<br><br>

$\bigstar$ We will need placeholder for input data
</p>
{% highlight python %}
tf.reset_default_graph()
features = tf.placeholder(tf.float32, shape=(None, D))
target = tf.placeholder(tf.float32, shape=(None, 1))
{% endhighlight %}
<p align="justify">
$\bigstar$ This is how we make predictions
</p>
{% highlight python %}
weights = tf.get_variable('w', shape=(D, 1), stype=tf.float32)
predictions = features @ weights
{% endhighlight %}
<p align="justify">
$\bigstar$ Define our loss
</p>
{% highlight python %}
loss = tf.reduce_mean((target - predictions)**2)
{% endhighlight %}
<p align="justify">
$\bigstar$ Optimize
</p>
{% highlight python %}
optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(loss)
{% endhighlight %}
<p align="justify">
$\bigstar$ Gradient descent
</p>
{% highlight python %}
s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())
for i in range(300):
    _, cur_loss, cur_weights = s.run([step, loss, weights],
                                      feed_dict={features: x,
                                                 target: y})
    if i%50 == 0:
        print(cur_loss)
{% endhighlight %}
<p align="justify">
$\bigstar$ Ground truth weights<br>
-- [0.11649134, 0.82753164, 0.46924019]<br>
$\bigstar$ Found weights<br>
-- [0.13715988, 0.79555332, 0.47024861]<br><br>

<b>Model checkpoints</b><br>
$\bigstar$ We can save variables' state with tf.train.Saver
</p>
{% highlight python %}
s = tf.InteractiveSession()
saver = tf.train.Saver(tf.trainable_variables())
s.run(tf.global_variables_initializer())
for i in range(300):
    _, cur_loss, cur_weights = s.run([step, loss, weights],
                                     feed_dict={features: x,
                                                target: y})
    if i%50 == 0:
        saver.save(s, 'log/2/model.ckpt', global_step=i)
        print(cur_loss)
{% endhighlight %}
<p align="justify">
$\bigstar$ We can list last checkpoint
</p>
{% highlight python %}
saver.last_checkppoints
{% endhighlight %}
<p align="justify">
['log/2/model.ckpt-50', 'log/2/model.ckpt-100', 'log/2/model.ckpt-150', 'log/2/model.ckpt-200', 'log/2/model.ckpt-250']<br><br>

$\bigstar$ We can restore a previous checkpoint
</p>
{% highlight python %}
saver.restore(s, 'log/2/model.ckpt-50')
{% endhighlight %}
<p align="justify">
$\bigstar$ Only variables' value are restored, which means that you need to define a graph in the same way before restoring a checkpoint<br><br>

<b>Summary</b><br>
$\bigstar$ TensorFlow has built-in optimizers that do back-propagation automatically<br>
$\bigstar$ TensorBoard provides tools for visualizing your training progress<br>
$\bigstar$ TensorFlow allows you to checkpoint your graph to restore its state later (you need to define it in exactly same way though)<br>
</p>

#### 1.1.1 Quiz
<p align="justify">
<b>1.</b><br>
The diagram for traditional programming had Rules and Data In, but what came out?<br>
A. Machine Learning<br>
B. Binary<br>
C. Answers<br>
D. Bugs<br>
<b>Answer:</b> C.<br><br>

<b>2.</b><br>
The diagram for Machine Learning had Answers and Data In, but what came out?<br>
A. Rules<br>
B. Binary<br>
C. Bugs<br>
D. Models<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
When I tell a computer what the data represents (i.e. this data is for walking, this data is for running), what is that process called?<br>
A. Labelling the Data<br>
B. Learning the Data<br>
C. Programming the Data<br>
D. Categorizing the Data<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
What is a Dense?<br>
A. A single neuron<br>
B. A layer of disconnected neurons<br>
C. A layer of connected neurons<br>
D. Mass over Volume<br>
<b>Answer:</b> C.<br><br>

<b>5.</b><br>
What does a Loss function do?<br>
A. Measures how good the current ‘guess’ is<br>
B. Figures out if you win or lose<br>
C. Decides to stop training a neural network<br>
D. Generates a guess<br>
<b>Answer:</b> A.<br><br>

<b>6.</b><br>
What does the optimizer do?<br>
A. Generates a new and improved guess<br>
B. Figures out how to efficiently compile your code<br>
C. Measures how good the current guess is<br>
D. Decides to stop training a neural network<br>
<b>Answer:</b> A.<br><br>

<b>7.</b><br>
What is Convergence?<br>
A. The process of getting very close to the correct answer<br>
B. The bad guys in the next ‘Star Wars’ movie<br>
C. A dramatic increase in loss<br>
D. A programming API for AI<br>
<b>Answer:</b> A.<br><br>

<b>8.</b><br>
What does model.fit do?<br>
A. It optimizes an existing model<br>
B. It makes a model fit available memory<br>
C. It trains the neural network to fit one set of values to another<br>
D. It determines if your activity is good for your body<br>
<b>Answer:</b> C.<br><br>
</p>

#### 1.1.2 Programming Assignment: Housing Prices
{% highlight Python %}
import tensorflow as tf
import numpy as np
from tensorflow import keras

# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4], dtype=float)
    ys = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(ys, xs, epochs=500)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)
{% endhighlight %}

### 1.2 Introduction to Computer Vision
#### 1.2.1 Quiz
<p align="justify">
<b>1.</b><br>
What’s the name of the dataset of Fashion images used in this week’s code?<br>
A. Fashion MN<br>
B. Fashion Data<br>
C. Fashion MNIST<br>
D. Fashion Tensors<br>
<b>Answer:</b> C.<br><br>

<b>2.</b><br>
What do the above mentioned Images look like?<br>
A. 28x28 Color<br>
B.82x82 Greyscale<br>
C. 100x100 Color<br>
D. 28x28 Greyscale<br>
<b>Answer:</b> D.<br><br>

<b>3.</b><br>
How many images are in the Fashion MNIST dataset?<br>
A. 42<br>
B. 60,000<br>
C. 10,000<br>
D. 70,000<br>
<b>Answer:</b> D.<br><br>

<b>4.</b><br>
Why are there 10 output neurons?<br>
A. Purely arbitrary<br>
B. To make it train 10x faster<br>
C. There are 10 different labels<br>
D. To make it classify 10x faster<br>
<b>Answer:</b> C.<br><br>

<b>5.</b><br>
What does Relu do?<br>
A. For a value x, it returns 1/x<br>
B. It only returns x if x is greater than zero<br>
C. It returns the negative of x<br>
D. It only returns x if x is less than zero<br>
<b>Answer:</b> B.<br><br>

<b>6.</b><br>
Why do you split data into training and test sets?<br>
A. To make training quicker<br>
B. To test a network with previously unseen data<br>
C. To make testing quicker<br>
D. To train a network with previously unseen data<br>
<b>Answer:</b> B.<br><br>

<b>7.</b><br>
What method gets called when an epoch finishes?<br>
A. on_epoch_finished<br>
B. on_epoch_end<br>
C. On_training_complete<br>
D. on_end<br>
<b>Answer:</b> B.<br><br>

<b>8.</b><br>
What parameter to you set in your fit function to tell it to use callbacks?<br>
A. callback=<br>
B. oncallback=<br>
C. callbacks=<br>
D. oncallbacks=<br>
<b>Answer:</b> C.<br><br>
</p>

#### 1.2.2 Programming Assignment: Handwriting Recognition
{% highlight Python %}
import tensorflow as tf

# GRADED FUNCTION: train_mnist
def train_mnist():
    tf.keras.backend.clear_session()
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("loss") < 0.001:
                print("\nLoss is low enough\n")
                self.model.step_training=True
    callbacks = myCallback()
    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    
    # YOUR CODE SHOULD START HERE
    x_train, x_test = x_train/255.0, x_test/255.0
    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(# YOUR CODE SHOULD START HERE
        x_train, y_train, epochs=7, callbacks=[callbacks]
              # YOUR CODE SHOULD END HERE
    )
    # model fitting
    return history.epoch, history.history['accuracy'][-1]

train_mnist()
{% endhighlight %}

### 1.3 Enhancing Vision with Convolutional Neural Networks
#### 1.3.1 Quiz
<p align="justify">
<b>1.</b><br>
What is a Convolution?<br>
A. A technique to isolate features in images<br>
B. A technique to make images smaller<br>
C. A technique to filter out unwanted images<br>
D. A technique to make images bigger<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
What is a Pooling?<br>
A. A technique to isolate features in images<br>
B. A technique to make images sharper<br>
C. A technique to combine pictures<br>
D. A technique to reduce the information in an image while maintaining features<br>
<b>Answer:</b> D.<br><br>

<b>3.</b><br>
How do Convolutions improve image recognition?<br>
A. They make the image smaller<br>
B. They make processing of images faster<br>
C. They make the image clearer<br>
D. They isolate features in images<br>
<b>Answer:</b> D.<br><br>

<b>4.</b><br>
After passing a 3x3 filter over a 28x28 image, how big will the output be?<br>
A. 26x26<br>
B. 28x28<br>
C. 31x31<br>
D. 25x25<br>
<b>Answer:</b> A.<br><br>

<b>5.</b><br>
After max pooling a 26x26 image with a 2x2 filter, how big will the output be?<br>
A. 56x56<br>
B. 26x26<br>
C. 28x28<br>
D. 13x13<br>
<b>Answer:</b> D.<br><br>

<b>6.</b><br>
Applying Convolutions on top of our Deep neural network will make training:<br>
A. It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN!<br>
B. Stay the same<br>
C. Slower<br>
D. Faster<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.3.2 Programming Assignment: Improve MNIST with convolutions
<p align="justify">
In the class you learned how to enhance the Fashion MNIST neural network with Convolutions to make it more accurate. Now it’s time to revisit the handwriting MNIST dataset from last week, and see if you can enhance it with Convolutions.<br>
</p>
{% highlight Python %}
import tensorflow as tf

# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    tf.keras.backend.clear_session()
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("accuracy") >= 0.998:
                print("Reached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) =\
        mnist.load_data()
    # YOUR CODE STARTS HERE
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images/255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images/255.0
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
            # YOUR CODE STARTS HERE
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu",
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
            # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
        training_images, training_labels, epochs=100, callbacks=[myCallback()]
        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['accuracy'][-1]


_, _ = train_mnist_conv()
{% endhighlight %}

### 1.4 Using Real-world Images
#### 1.4.1 Quiz
<p align="justify">
<b>1.</b><br>
Using Image Generator, how do you label images?<br>
A. You have to manually do it<br>
B. It’s based on the file name<br>
C. TensorFlow figures it out from the contents<br>
D. It’s based on the directory the image is contained in<br>
<b>Answer:</b> D.<br><br>

<b>2.</b><br>
What method on the Image Generator is used to normalize the image?<br>
A. normalize_image<br>
B. Rescale_image<br>
C. rescale<br>
D. normalize<br>
<b>Answer:</b> C.<br><br>

<b>3.</b><br>
How did we specify the training size for the images?<br>
A. The training_size parameter on the validation generator<br>
B. The training_size parameter on the training generator<br>
C. The target_size parameter on the training generator<br>
D. The target_size parameter on the validation generator<br>
<b>Answer:</b> C.<br><br>

<b>4.</b><br>
When we specify the input_shape to be (300, 300, 3), what does that mean?<br>
A. There will be 300 images, each size 300, loaded in batches of 3<br>
B. There will be 300 horses and 300 humans, loaded in batches of 3<br>
C. Every Image will be 300x300 pixels, and there should be 3 Convolutional Layers<br>
D. Every Image will be 300x300 pixels, with 3 bytes to define color<br>
<b>Answer:</b> D.<br><br>

<b>5.</b><br>
If your training data is close to 1.000 accuracy, but your validation data isn’t, what’s the risk here?<br>
A. You’re overfitting on your validation data<br>
B. No risk, that’s a great result<br>
C. You’re underfitting on your validation data<br>
D. You’re overfitting on your training data<br>
<b>Answer:</b> D.<br><br>

<b>6.</b><br>
Convolutional Neural Networks are better for classifying images like horses and humans because:<br>
A. In these images, the features may be in different parts of the frame<br>
B. There’s a wide variety of horses<br>
C. There’s a wide variety of humans<br>
D. All of the above<br>
<b>Answer:</b> D.<br><br>

<b>7.</b><br>
After reducing the size of the images, the training results were different. Why?<br>
A. We removed some convolutions to handle the smaller images<br>
B. There was less information in the images<br>
C. The training was faster<br>
D. There was more condensed information in the images<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.4.2 Programming Assignment: Handling complex images
<p align="justify">
Let’s now create your own image classifier for complex images. See if you can create a classifier for a set of happy or sad images that I’ve provided. Use a callback to cancel training once accuracy is greater than .999.<br><br>

Dataset should be downloaded at first<a href="https://github.com/chaopan1995/PROJECTS/blob/master/TensorFlow-in-Practice-Programming-Assignment/happy-or-sad.zip"> here.</a><br>
</p>
{% highlight Python %}
import tensorflow as tf

# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    tf.keras.backend.clear_session()
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("accuracy") >= DESIRED_ACCURACY:
                print("Accuracy is reached")
                self.model.stop_training = True

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model.
    # Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                               input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
        # Your Code Here
    ])

    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(lr=0.001),
        metrics=["accuracy"]
        # Your Code Here #
    )
    # This code block should create an instance of an
    # ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory
    # Your Code Here
    train_datagen = ImageDataGenerator(rescale=1.0/255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        "happy-or-sad/",
        target_size=(150, 150),
        batch_size=5,
        class_mode="binary"
        # Your Code Here
    )
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        callbacks=[callbacks]
        # Your Code Here
    )
    # model fitting
    return history.history['accuracy'][-1]


# The Expected output: "Reached 99.9% accuracy so cancelling training!""
train_happy_sad_model()
{% endhighlight %}


## 2. Convolutional Neural Networks in TensorFlow
### 2.1 Larger Dataset
#### 2.1.1 Quiz
<p align="justify">
<b>1.</b><br>
What does flow_from_directory give you on the ImageGenerator?<br>
A. The ability to easily load images for training<br>
B. The ability to pick the size of training images<br>
C. The ability to automatically label images based on their directory name<br>
D. All of the above<br>
<b>Answer:</b> D.<br><br>

<b>2.</b><br>
If my Image is sized 150x150, and I pass a 3x3 Convolution over it, what size is the resulting image?<br>
A. 450x450<br>
B. 153x153<br>
C. 150x150<br>
D. 148x148<br>
<b>Answer:</b> D.<br><br>

<b>3.</b><br>
If my data is sized 150x150, and I use Pooling of size 2x2, what size will the resulting image be?<br>
A. 75x75<br>
B. 300x300<br>
C. 149x149<br>
D. 148x148<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
If I want to view the history of my training, how can I access it?<br>
A. Use a model.fit_generator<br>
B. Pass the parameter ‘history=true’ to the model.fit<br>
C. Download the model and inspect it<br>
D. Create a variable ‘history’ and assign it to the return of model.fit or model.fit_generator<br>
<b>Answer:</b> D.<br><br>

<b>5.</b><br>
What’s the name of the API that allows you to inspect the impact of convolutions on the images?<br>
A. The model.images API<br>
B. The model.layers API<br>
C. The model.convolutions API<br>
D. The model.pools API<br>
<b>Answer:</b> B.<br><br>

<b>6.</b><br>
When exploring the graphs, the loss levelled out at about .75 after 2 epochs, but the accuracy climbed close to 1.0 after 15 epochs. What's the significance of this?<br>
A. There was no point training after 2 epochs, as we overfit to the validation data<br>
B. There was no point training after 2 epochs, as we overfit to the training data<br>
C. A bigger training set would give us better validation accuracy<br>
D. A bigger validation set would give us better training accuracy<br>
<b>Answer:</b> B.<br><br>

<b>7.</b><br>
Why is the validation accuracy a better indicator of model performance than training accuracy?<br>
A. It isn't, they're equally valuable<br>
B. There's no relationship between them<br>
C. The validation accuracy is based on images that the model hasn't been trained with, and thus a better indicator of how the model will perform with new images.<br>
D. The validation dataset is smaller, and thus less accurate at measuring accuracy, so its performance isn't as important<br>
<b>Answer:</b> C.<br><br>

<b>8.</b><br>
Why is overfitting more likely to occur on smaller datasets?<br>
A. Because in a smaller dataset, your validation data is more likely to look like your training data<br>
B. Because there isn't enough data to activate all the convolutions or neurons<br>
C. Because with less data, the training will take place more quickly, and some features may be missed<br>
D. Because there's less likelihood of all possible features being encountered in the training process.<br>
<b>Answer:</b> D.<br><br>
</p>

#### 2.1.2 Programming Assignment: Cats vs. Dogs
{% highlight Python %}
import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# If the URL doesn't work, visit
# https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# And right click on the 'Download Manually' link to get a new URL to the
# dataset
# Note: This is a very large dataset and will take time to download

!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
    -O "/tmp/cats-and-dogs.zip"

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# Expected Output:
# 12501
# 12501
print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))

try:
    os.mkdir('/tmp/cats-v-dogs')
    os.mkdir('/tmp/cats-v-dogs/training')
    os.mkdir('/tmp/cats-v-dogs/testing')
    os.mkdir('/tmp/cats-v-dogs/training/cats')
    os.mkdir('/tmp/cats-v-dogs/training/dogs')
    os.mkdir('/tmp/cats-v-dogs/testing/cats')
    os.mkdir('/tmp/cats-v-dogs/testing/dogs')
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

# Expected output
# 666.jpg is zero length, so ignoring
# 11702.jpg is zero length, so ignoring
split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# Expected output:
# 11250
# 11250
# 1250
# 1250
print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/')))


tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy',
              metrics=['accuracy'])

# Expected Output:
# Found 22498 images belonging to 2 classes.
# Found 2500 images belonging to 2 classes.
TRAINING_DIR = "/tmp/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR, batch_size=100, class_mode='binary',
        target_size=(150, 150))

# Note that this may take some time.
history = model.fit(train_generator,
                              epochs=50,
                              verbose=1,
                              validation_data=validation_generator)

%matplotlib inline
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()
{% endhighlight %}

### 2.2 Augmentation
#### 2.2.1 Quiz
<p align="justify">
<b>1.</b><br>
How do you use Image Augmentation in TensorFLow<br>
A. Using parameters to the ImageDataGenerator<br>
B. You have to write a plugin to extend tf.layers<br>
C. With the keras.augment API<br>
D. With the tf.augment API<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
If my training data only has people facing left, but I want to classify people facing right, how would I avoid overfitting?<br>
A. Use the ‘horizontal_flip’ parameter<br>
B. Use the ‘flip’ parameter<br>
C. Use the ‘flip’ parameter and set ‘horizontal’<br>
D. Use the ‘flip_vertical’ parameter around the Y axis<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
When training with augmentation, you noticed that the training is a little slower. Why?<br>
A. Because the augmented data is bigger<br>
B. Because the image processing takes cycles<br>
C. Because there is more data to train on<br>
D. Because the training is making more mistakes<br>
<b>Answer:</b> B.<br><br>

<b>4.</b><br>
What does the fill_mode parameter do?<br>
A. There is no fill_mode parameter<br>
B. It creates random noise in the image<br>
C. It attempts to recreate lost information after a transformation like a shear<br>
D. It masks the background of an image<br>
<b>Answer:</b> C.<br><br>

<b>5.</b><br>
When using Image Augmentation with the ImageDataGenerator, what happens to your raw image data on-disk.<br>
A. It gets overwritten, so be sure to make a backup<br>
B. A copy is made and the augmentation is done on the copy<br>
C. Nothing, all augmentation is done in-memory<br>
D. It gets deleted<br>
<b>Answer:</b> C.<br><br>

<b>6.</b><br>
How does Image Augmentation help solve overfitting?<br>
A. It slows down the training process<br>
B. It manipulates the training set to generate more scenarios for features in the images<br>
C. It manipulates the validation set to generate more scenarios for features in the images<br>
D. It automatically fits features to images by finding them through image processing techniques<br>
<b>Answer:</b> B.<br><br>

<b>7.</b><br>
When using Image Augmentation my training gets...<br>
A. Slower<br>
B. Faster<br>
C. Stays the Same<br>
D. Much Faster<br>
<b>Answer:</b> A.<br><br>

<b>8.</b><br>
Using Image Augmentation effectively simulates having a larger data set for training.<br>
A. False<br>
B. True<br>
<b>Answer:</b> B.<br><br>
</p>

#### 2.2.2 Programming Assignment: Cats vs. Dogs using augmentation
{% highlight Python %}
import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile


# If the URL doesn't work, visit
# https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# And right click on the 'Download Manually' link to get a new URL to the
# dataset

# Note: This is a very large dataset and will take time to download

!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
    -O "/tmp/cats-and-dogs.zip"

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()


print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))

# Expected Output:
# 12501
# 12501


try:
    os.mkdir('/tmp/cats-v-dogs')
    os.mkdir('/tmp/cats-v-dogs/training')
    os.mkdir('/tmp/cats-v-dogs/testing')
    os.mkdir('/tmp/cats-v-dogs/training/cats')
    os.mkdir('/tmp/cats-v-dogs/training/dogs')
    os.mkdir('/tmp/cats-v-dogs/testing/cats')
    os.mkdir('/tmp/cats-v-dogs/testing/dogs')
except OSError:
    pass


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[:testing_length]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# Expected output
# 666.jpg is zero length, so ignoring
# 11702.jpg is zero length, so ignoring


print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/')))

# Expected output:
# 11250
# 11250
# 1250
# 1250


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy',
              metrics=['accuracy'])


TRAINING_DIR = "/tmp/cats-v-dogs/training/"
# Experiment with your own parameters here to really try to drive it to
# 99.9% accuracy or better
train_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"
# Experiment with your own parameters here to really try to drive it to 99.9%
# accuracy or better
validation_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR, batch_size=100, class_mode='binary',
        target_size=(150, 150))

# Expected Output:
# Found 22498 images belonging to 2 classes.
# Found 2500 images belonging to 2 classes.


# Note that this may take some time.
history = model.fit(train_generator,
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator)


%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()
{% endhighlight %}

### 2.3 Transfer Learning
#### 2.3.1 Quiz
<p align="justify">
<b>1.</b><br>
If I put a dropout parameter of 0.2, how many nodes will I lose?<br>
A. 20% of them<br>
B. 2% of them<br>
C. 20% of the untrained ones<br>
D. 2% of the untrained ones<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
Why is transfer learning useful?<br>
A. Because I can use all of the data from the original training set<br>
B. Because I can use all of the data from the original validation set<br>
C. Because I can use the features that were learned from large datasets that I may not have access to<br>
D. Because I can use the validation metadata from large datasets that I may not have access to<br>
<b>Answer:</b> C.<br><br>

<b>3.</b><br>
How did you lock or freeze a layer from retraining?<br>
A. tf.freeze(layer)<br>
B. tf.layer.frozen = true<br>
C. tf.layer.locked = true<br>
D. layer.trainable = false<br>
<b>Answer:</b> D.<br><br>

<b>4.</b><br>
How do you change the number of classes the model can classify when using transfer learning? (i.e. the original model handled 1000 classes, but yours handles just 2)<br>
A. Ignore all the classes above yours (i.e. Numbers 2 onwards if I'm just classing 2)<br>
B. Use all classes but set their weights to 0<br>
C. When you add your DNN at the bottom of the network, you specify your output layer with the number of classes you want<br>
D. Use dropouts to eliminate the unwanted classes<br>
<b>Answer:</b> C.<br><br>

<b>5.</b><br>
Can you use Image Augmentation with Transfer Learning Models?<br>
A. No, because you are using pre-set features<br>
B. Yes, because you are adding new layers at the bottom of the network, and you can use image augmentation when training these<br>
<b>Answer:</b> B.<br><br>

<b>6.</b><br>
Why do dropouts help avoid overfitting?<br>
A. Because neighbor neurons can have similar weights, and thus can skew the final training<br>
B. Having less neurons speeds up training<br>
<b>Answer:</b> A.<br><br>

<b>7.</b><br>
What would the symptom of a Dropout rate being set too high?<br>
A. The network would lose specialization to the effect that it would be inefficient or ineffective at learning, driving accuracy down<br>
B. Training time would increase due to the extra calculations being required for higher dropout<br>
<b>Answer:</b> A.<br><br>

<b>8.</b><br>
Which is the correct line of code for adding Dropout of 20% of neurons using TensorFlow<br>
A. tf.keras.layers.Dropout(20)<br>
B. tf.keras.layers.DropoutNeurons(20),<br>
C. tf.keras.layers.Dropout(0.2),<br>
D. tf.keras.layers.DropoutNeurons(0.2),<br>
<b>Answer:</b> C.<br><br>
</p>

#### 2.3.2 Programming Assignment: Horses vs. humans using Transfer Learning
{% highlight Python %}
# Import all the necessary files!
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


# Download the inception v3 weights
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  layer.trainable = False
  
# Print the model summary
pre_trained_model.summary()

# Expected Output is extremely large, but should end with:

#batch_normalization_v1_281 (Bat (None, 3, 3, 192)    576         conv2d_281[0][0]                 
#__________________________________________________________________________________________________
#activation_273 (Activation)     (None, 3, 3, 320)    0           batch_normalization_v1_273[0][0] 
#__________________________________________________________________________________________________
#mixed9_1 (Concatenate)          (None, 3, 3, 768)    0           activation_275[0][0]             
#                                                                 activation_276[0][0]             
#__________________________________________________________________________________________________
#concatenate_5 (Concatenate)     (None, 3, 3, 768)    0           activation_279[0][0]             
#                                                                 activation_280[0][0]             
#__________________________________________________________________________________________________
#activation_281 (Activation)     (None, 3, 3, 192)    0           batch_normalization_v1_281[0][0] 
#__________________________________________________________________________________________________
#mixed10 (Concatenate)           (None, 3, 3, 2048)   0           activation_273[0][0]             
#                                                                 mixed9_1[0][0]                   
#                                                                 concatenate_5[0][0]              
#                                                                 activation_281[0][0]             
#==================================================================================================
#Total params: 21,802,784
#Trainable params: 0
#Non-trainable params: 21,802,784


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Expected Output:
# ('last layer output shape: ', (None, 7, 7, 768))

# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True


from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

model.summary()

# Expected output will be large. Last few lines should be:

# mixed7 (Concatenate)            (None, 7, 7, 768)    0           activation_248[0][0]             
#                                                                  activation_251[0][0]             
#                                                                  activation_256[0][0]             
#                                                                  activation_257[0][0]             
# __________________________________________________________________________________________________
# flatten_4 (Flatten)             (None, 37632)        0           mixed7[0][0]                     
# __________________________________________________________________________________________________
# dense_8 (Dense)                 (None, 1024)         38536192    flatten_4[0][0]                  
# __________________________________________________________________________________________________
# dropout_4 (Dropout)             (None, 1024)         0           dense_8[0][0]                    
# __________________________________________________________________________________________________
# dense_9 (Dense)                 (None, 1)            1025        dropout_4[0][0]                  
# ==================================================================================================
# Total params: 47,512,481
# Trainable params: 38,537,217
# Non-trainable params: 8,975,264


# Get the Horse or Human dataset
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip -O /tmp/horse-or-human.zip

# Get the Horse or Human Validation dataset
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip -O /tmp/validation-horse-or-human.zip 
  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile

local_zip = '//tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/training')
zip_ref.close()

local_zip = '//tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation')
zip_ref.close()


# Define our example directories and files
train_dir = '/tmp/training'
validation_dir = '/tmp/validation'

train_horses_dir = os.path.join(train_dir, 'horses') # Directory with our training horse pictures
train_humans_dir = os.path.join(train_dir, 'humans') # Directory with our training humans pictures
validation_horses_dir = os.path.join(validation_dir, 'horses') # Directory with our validation horse pictures
validation_humans_dir = os.path.join(validation_dir, 'humans')# Directory with our validation humanas pictures

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))

# Expected Output:
# 500
# 527
# 128
# 128


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary', 
                                                    target_size = (150, 150))     

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = 20,
                                                          class_mode  = 'binary', 
                                                          target_size = (150, 150))

# Expected Output:
# Found 1027 images belonging to 2 classes.
# Found 256 images belonging to 2 classes.


# Run this and see how many epochs it should take before the callback
# fires, and stops training at 99.9% accuracy
# (It should take less than 100 epochs)
callbacks = myCallback()
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 100,
            validation_steps = 50,
            verbose = 2,
            callbacks=[callbacks])

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
{% endhighlight %}

### 2.4 Multiclass Classifications
#### 2.4.1 Quiz
<p align="justify">
<b>1.</b><br>
The diagram for traditional programming had Rules and Data In, but what came out?<br>
A. Answers<br>
B. Binary<br>
C. Machine Learning<br>
D. Bugs<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
Why does the DNN for Fashion MNIST have 10 output neurons?<br>
A. To make it train 10x faster<br>
B. To make it classify 10x faster<br>
C. Purely Arbitrary<br>
D. The dataset has 10 classes<br>
<b>Answer:</b> D.<br><br>

<b>3.</b><br>
What is a Convolution?<br>
A. A technique to make images smaller<br>
B. A technique to make images larger<br>
C. A technique to extract features from an image<br>
D. A technique to remove unwanted images<br>
<b>Answer:</b> C.<br><br>

<b>4.</b><br>
Applying Convolutions on top of a DNN will have what impact on training?<br>
A. It will be slower<br>
B. It will be faster<br>
C. There will be no impact<br>
D. It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN!<br>
<b>Answer:</b> D.<br><br>

<b>5.</b><br>
What method on an ImageGenerator is used to normalize the image?<br>
A. normalize<br>
B. flatten<br>
C. rezize<br>
D. rescale<br>
<b>Answer:</b> D.<br><br>

<b>6.</b><br>
When using Image Augmentation with the ImageDataGenerator, what happens to your raw image data on-disk.<br>
A. A copy will be made, and the copies are augmented<br>
B. A copy will be made, and the originals will be augmented<br>
C. Nothing<br>
D. The images will be edited on disk, so be sure to have a backup<br>
<b>Answer:</b> C.<br><br>

<b>7.</b><br>
Can you use Image augmentation with Transfer Learning?<br>
A. No - because the layers are frozen so they can't be augmented<br>
B. Yes. It's pre-trained layers that are frozen. So you can augment your images as you train the bottom layers of the DNN with them<br>
<b>Answer:</b> B.<br><br>

<b>8.</b><br>
When training for multiple classes what is the Class Mode for Image Augmentation?<br>
A. class_mode='multiple'<br>
B. class_mode='non_binary'<br>
C. class_mode='categorical'<br>
D. class_mode='all'<br>
<b>Answer:</b> C.<br><br>
</p>

#### 2.4.2 Programming Assignment: Multi-class classifier
{% highlight Python %}
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files

uploaded=files.upload()

def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=',')
        first_line = True
        temp_images = []
        temp_labels = []
        for row in csv_reader:
            if first_line:
                # print("Ignoring first line")
                first_line = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_data_as_array = np.array_split(image_data, 28)
                temp_images.append(image_data_as_array)
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
    return images, labels


training_images, training_labels = get_data('sign_mnist_train.csv')
testing_images, testing_labels = get_data('sign_mnist_test.csv')

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)


training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    rescale=1. / 255)

print(training_images.shape)
print(testing_images.shape)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(26, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_datagen.flow(training_images, training_labels,
                                       batch_size=32),
                              steps_per_epoch=len(training_images) / 32,
                              epochs=15,
                              validation_data=validation_datagen.flow(
                                      testing_images, testing_labels,
                                      batch_size=32),
                              validation_steps=len(testing_images) / 32)

model.evaluate(testing_images, testing_labels)


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
{% endhighlight %}


## 3. Natural Language Processing in TensorFlow
### 3.1 Sentiment in text
#### 3.1.1 Quiz
<p align="justify">
<b>1.</b><br>
What is the name of the object used to tokenize sentences?<br>
A. CharacterTokenizer<br>
B. TextTokenizer<br>
C. WordTokenizer<br>
D. Tokenizer<br>
<b>Answer:</b> D.<br><br>

<b>2.</b><br>
What is the name of the method used to tokenize a list of sentences?<br>
A. tokenize(sentences)<br>
B. fit_on_texts(sentences)<br>
C. fit_to_text(sentences)<br>
D. tokenize_on_text(sentences)<br>
<b>Answer:</b> B.<br><br>

<b>3.</b><br>
Once you have the corpus tokenized, what’s the method used to encode a list of sentences to use those tokens?<br>
A. text_to_sequences(sentences)<br>
B. texts_to_sequences(sentences)<br>
C. texts_to_tokens(sentences)<br>
D. text_to_tokens(sentences)<br>
<b>Answer:</b> B.<br><br>

<b>4.</b><br>
When initializing the tokenizer, how to you specify a token to use for unknown words?<br>
A. oov_token=Token<br>
B. unknown_word=Token<br>
C. unknown_token=Token<br>
D. out_of_vocab=Token<br>
<b>Answer:</b> A.<br><br>

<b>5.</b><br>
If you don’t use a token for out of vocabulary words, what happens at encoding?<br>
A. The word isn’t encoded, and is replaced by a zero in the sequence<br>
B. The word is replaced by the most common token<br>
C. The word isn’t encoded, and is skipped in the sequence<br>
D. The word isn’t encoded, and the sequencing ends<br>
<b>Answer:</b> C.<br><br>

<b>6.</b><br>
If you have a number of sequences of different lengths, how do you ensure that they are understood when fed into a neural network?<br>
A. Make sure that they are all the same length using the pad_sequences method of the tokenizer<br>
B. Use the pad_sequences object from the tensorflow.keras.preprocessing.sequence namespace<br>
C. Specify the input layer of the Neural Network to expect different sizes with dynamic_length<br>
D. Process them on the input layer of the Neural Netword using the pad_sequences property<br>
<b>Answer:</b> B.<br><br>

<b>7.</b><br>
If you have a number of sequences of different length, and call pad_sequences on them, what’s the default result?<br>
A. Nothing, they’ll remain unchanged<br>
B. They’ll get padded to the length of the longest sequence by adding zeros to the beginning of shorter ones<br>
C. They’ll get cropped to the length of the shortest sequence<br>
D. They’ll get padded to the length of the longest sequence by adding zeros to the end of shorter ones<br>
<b>Answer:</b> B.<br><br>

<b>8.</b><br>
When padding sequences, if you want the padding to be at the end of the sequence, how do you do it?<br>
A. Call the padding method of the pad_sequences object, passing it ‘post’<br>
B. Call the padding method of the pad_sequences object, passing it ‘after’<br>
C. Pass padding=’after’ to pad_sequences when initializing it<br>
D. Pass padding=’post’ to pad_sequences when initializing it<br>
<b>Answer:</b> D.<br><br>
</p>

#### 3.1.2 Programming Assignment: Explore the BBC news archive
{% highlight Python %}
!wget --no-check-certificate \
https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv\
    -O /tmp/bbc-text.csv

  
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Stopwords list from
# https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stopwords = [
              "a", "about", "above", "after", "again", "against", "all", "am",
              "an", "and", "any", "are", "as", "at", "be", "because", "been",
              "before", "being", "below", "between", "both", "but", "by",
              "could", "did", "do", "does", "doing", "down", "during", "each",
              "few", "for", "from", "further", "had", "has", "have", "having",
              "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
              "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
              "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
              "its", "itself", "let's", "me", "more", "most", "my", "myself",
              "nor", "of", "on", "once", "only", "or", "other", "ought", "our",
              "ours", "ourselves", "out", "over", "own", "same", "she",
              "she'd", "she'll", "she's", "should", "so", "some", "such",
              "than", "that", "that's", "the", "their", "theirs", "them",
              "themselves", "then", "there", "there's", "these", "they",
              "they'd", "they'll", "they're", "they've", "this", "those",
              "through", "to", "too", "under", "until", "up", "very", "was",
              "we", "we'd", "we'll", "we're", "we've", "were", "what",
              "what's", "when", "when's", "where", "where's", "which", "while",
              "who", "who's", "whom", "why", "why's", "with", "would", "you",
              "you'd", "you'll", "you're", "you've", "your", "yours",
              "yourself", "yourselves"
]
print(len(stopwords))


sentences = []
labels = []
with open("/tmp/bbc-text.csv", 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter=",")
  next(reader)
  for lab, sen in reader:
    labels.append(lab)
    for w in stopwords:
      token = " " + w + " "
      sen = sen.replace(token, " ")
      sen = sen.replace("  ", " ")
    sentences.append(sen)
print(len(labels))
print(len(sentences))
print(sentences[0])
#Expected output
# 2225
# tv future hands viewers home theatre systems plasma high-definition tvs
# digital video recorders moving living room way people watch tv will
# radically different five years time. according expert panel gathered annual
# consumer electronics show las vegas discuss new technologies will impact
# one favourite pastimes. us leading trend programmes content will delivered
# viewers via home networks cable satellite telecoms companies broadband
# service providers front rooms portable devices. one talked-about technologies
# ces digital personal video recorders (dvr pvr). set-top boxes like us s tivo
# uk s sky+ system allow people record store play pause forward wind tv
# programmes want. essentially technology allows much personalised tv. also
# built-in high-definition tv sets big business japan us slower take off europe
# lack high-definition programming. not can people forward wind adverts can
# also forget abiding network channel schedules putting together a-la-carte
# entertainment. us networks cable satellite companies worried means terms
# advertising revenues well brand identity viewer loyalty channels. although
# us leads technology moment also concern raised europe particularly growing
# uptake services like sky+. happens today will see nine months years time uk
# adam hume bbc broadcast s futurologist told bbc news website. likes bbc no
# issues lost advertising revenue yet. pressing issue moment commercial uk
# broadcasters brand loyalty important everyone. will talking content brands
# rather network brands said tim hanlon brand communications firm starcom
# mediavest. reality broadband connections anybody can producer content. added:
# challenge now hard promote programme much choice. means said stacey jolna
# senior vice president tv guide tv group way people find content want watch
# simplified tv viewers. means networks us terms channels take leaf google s
# book search engine future instead scheduler help people find want watch.
# kind channel model might work younger ipod generation used taking control
# gadgets play them. might not suit everyone panel recognised. older
# generations comfortable familiar schedules channel brands know getting.
# perhaps not want much choice put hands mr hanlon suggested. end kids just
# diapers pushing buttons already - everything possible available said mr
# hanlon. ultimately consumer will tell market want. 50 000 new gadgets
# technologies showcased ces many enhancing tv-watching experience.
# high-definition tv sets everywhere many new models lcd (liquid crystal
# display) tvs launched dvr capability built instead external boxes. one
# example launched show humax s 26-inch lcd tv 80-hour tivo dvr dvd recorder.
# one us s biggest satellite tv companies directtv even launched branded dvr
# show 100-hours recording capability instant replay search function. set can
# pause rewind tv 90 hours. microsoft chief bill gates announced pre-show
# keynote speech partnership tivo called tivotogo means people can play
# recorded programmes windows pcs mobile devices. reflect increasing trend
# freeing multimedia people can watch want want.


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))
# Expected output
# 29714


sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)
# Expected output
# [  96  176 1158 ...    0    0    0]
# (2225, 2442)


label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)
print(label_seq)
print(label_word_index)
# Expected Output
# [[4], [2], [1], [1], ,,,, [3], [1]]
# {'sport': 1, 'business': 2, 'politics': 3, 'tech': 4, 'entertainment': 5}
{% endhighlight %}

### 3.2 Word Embeddings
#### 3.2.1 Quiz
<p align="justify">
<b>1.</b><br>
What is the name of the TensorFlow library containing common data that you can use to train and test neural networks?<br>
A. There is no library of common data sets, you have to use your own<br>
B. TensorFlow Datasets<br>
C. TensorFlow Data<br>
D. TensorFlow Data Libraries<br>
<b>Answer:</b> B.<br><br>

<b>2.</b><br>
How many reviews are there in the IMDB dataset and how are they split?<br>
A. 50,000 records, 50/50 train/test split<br>
B. 60,000 records, 50/50 train/test split<br>
C. 50,000 records, 80/20 train/test split<br>
D. 60,000 records, 80/20 train/test split<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
How are the labels for the IMDB dataset encoded?<br>
A. Reviews encoded as a boolean true/false<br>
B. Reviews encoded as a number 0-1<br>
C. Reviews encoded as a number 1-5<br>
D. Reviews encoded as a number 1-10<br>
<b>Answer:</b> B.<br><br>

<b>4.</b><br>
What is the purpose of the embedding dimension?<br>
A. It is the number of dimensions for the vector representing the word encoding<br>
B. It is the number of letters in the word, denoting the size of the encoding<br>
C. It is the number of dimensions required to encode every word in the corpus<br>
D. It is the number of words to encode in the embedding<br>
<b>Answer:</b> A.<br><br>

<b>5.</b><br>
When tokenizing a corpus, what does the num_words=n parameter do?<br>
A. It specifies the maximum number of words to be tokenized, and picks the first ‘n’ words that were tokenized<br>
B. It specifies the maximum number of words to be tokenized, and picks the most common ‘n’ words<br>
C. It errors out if there are more than n distinct words in the corpus<br>
D. It specifies the maximum number of words to be tokenized, and stops tokenizing when it reaches n<br>
<b>Answer:</b> B.<br><br>

<b>6.</b><br>
To use word embeddings in TensorFlow, in a sequential layer, what is the name of the class?<br>
A. tf.keras.layers.Embed<br>
B. tf.keras.layers.Embedding<br>
C. tf.keras.layers.Word2Vector<br>
D. tf.keras.layers.WordEmbedding<br>
<b>Answer:</b> B.<br><br>

<b>7.</b><br>
IMDB Reviews are either positive or negative. What type of loss function should be used in this scenario?<br>
A. Categorical crossentropy<br>
B. Binary crossentropy<br>
C. Adam<br>
D. Binary Gradient descent<br>
<b>Answer:</b> B.<br><br>

<b>8.</b><br>
When using IMDB Sub Words dataset, our results in classification were poor. Why?<br>
A. The sub words make no sense, so can’t be classified<br>
B. Our neural network didn’t have enough layers<br>
C. We didn’t train long enough<br>
D. Sequence becomes much more important when dealing with subwords, but we’re ignoring word positions<br>
<b>Answer:</b> D.<br><br>
</p>

#### 3.2.2 Programming Assignment: BBC news archive
{% highlight Python %}
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


!wget --no-check-certificate\
  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv\
    -O /tmp/bbc-text.csv


vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_portion = .8

sentences = []
labels = []
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am",
             "an", "and", "any", "are", "as", "at", "be", "because", "been",
             "before", "being", "below", "between", "both", "but", "by",
             "could", "did", "do", "does", "doing", "down", "during", "each",
             "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
             "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
             "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
             "its", "itself", "let's", "me", "more", "most", "my", "myself",
             "nor", "of", "on", "once", "only", "or", "other", "ought", "our",
             "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
             "she'll", "she's", "should", "so", "some", "such", "than", "that",
             "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll",
             "they're", "they've", "this", "those", "through", "to", "too",
             "under", "until", "up", "very", "was", "we", "we'd", "we'll",
             "we're", "we've", "were", "what", "what's", "when", "when's",
             "where", "where's", "which", "while", "who", "who's", "whom",
             "why", "why's", "with", "would", "you", "you'd", "you'll",
             "you're", "you've", "your", "yours", "yourself", "yourselves" ]
print(len(stopwords))
# Expected Output
# 153

with open("/tmp/bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence)

print(len(labels))
print(len(sentences))
print(sentences[0])
# Expected Output
# 2225
# 2225
# tv future hands viewers home theatre systems  plasma high-definition tvs 
# digital video recorders moving living room  way people watch tv will
# radically different five years  time.  according expert panel gathered
# annual consumer electronics show las vegas discuss new technologies will
# impact one favourite pastimes. us leading trend  programmes content will
# delivered viewers via home networks  cable  satellite  telecoms companies 
# broadband service providers front rooms portable devices.  one talked-about
# technologies ces digital personal video recorders (dvr pvr). set-top boxes 
# like us s tivo uk s sky+ system  allow people record  store  play  pause
# forward wind tv programmes want.  essentially  technology allows much
# personalised tv. also built-in high-definition tv sets  big business japan
# us  slower take off europe lack high-definition programming. not can people
# forward wind adverts  can also forget abiding network channel schedules 
# putting together a-la-carte entertainment. us networks cable satellite
# companies worried means terms advertising revenues well  brand identity 
# viewer loyalty channels. although us leads technology moment  also concern
# raised europe  particularly growing uptake services like sky+.  happens
# today  will see nine months years  time uk   adam hume  bbc broadcast s
# futurologist told bbc news website. likes bbc  no issues lost advertising
# revenue yet. pressing issue moment commercial uk broadcasters  brand loyalty
# important everyone.  will talking content brands rather network brands  
# said tim hanlon  brand communications firm starcom mediavest.  reality
# broadband connections  anybody can producer content.  added:  challenge now
# hard promote programme much choice.   means  said stacey jolna  senior vice
# president tv guide tv group  way people find content want watch simplified
# tv viewers. means networks  us terms  channels take leaf google s book
# search engine future  instead scheduler help people find want watch. kind
# channel model might work younger ipod generation used taking control gadgets
# play them. might not suit everyone  panel recognised. older generations
# comfortable familiar schedules channel brands know getting. perhaps not want
# much choice put hands  mr hanlon suggested.  end  kids just diapers pushing
# buttons already - everything possible available   said mr hanlon.  ultimately
#  consumer will tell market want.   50 000 new gadgets technologies showcased
# ces  many enhancing tv-watching experience. high-definition tv sets
# everywhere many new models lcd (liquid crystal display) tvs launched dvr
# capability built  instead external boxes. one example launched show humax s
# 26-inch lcd tv 80-hour tivo dvr dvd recorder. one us s biggest satellite tv
# companies  directtv  even launched branded dvr show 100-hours recording
# capability  instant replay  search function. set can pause rewind tv 90
# hours. microsoft chief bill gates announced pre-show keynote speech
# partnership tivo  called tivotogo  means people can play recorded programmes
# windows pcs mobile devices. reflect increasing trend freeing multimedia
# people can watch want  want.


train_size = int(len(sentences) * training_portion)

train_sentences = sentences[:train_size]
train_labels = labels[:train_size]

validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))
# Expected output (if training_portion=.8)
# 1780
# 1780
# 1780
# 445
# 445


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type,
                             maxlen=max_length)

print(len(train_sequences[0]))
print(len(train_padded[0]))
print(len(train_sequences[1]))
print(len(train_padded[1]))
print(len(train_sequences[10]))
print(len(train_padded[10]))
# Expected Ouput
# 449
# 120
# 200
# 120
# 192
# 120


validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type,
                                  maxlen=max_length)

print(len(validation_sequences))
print(validation_padded.shape)
# Expected output
# 445
# (445, 120)


label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(
        validation_labels))

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)
print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)
# Expected output
# [4]
# [2]
# [1]
# (1780, 1)
# [5]
# [4]
# [3]
# (445, 1)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
# Expected Output
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, 120, 16)           16000     
# _________________________________________________________________
# global_average_pooling1d (Gl (None, 16)                0         
# _________________________________________________________________
# dense (Dense)                (None, 24)                408       
# _________________________________________________________________
# dense_1 (Dense)              (None, 6)                 150       
# =================================================================
# Total params: 16,558
# Trainable params: 16,558
# Non-trainable params: 0


num_epochs = 30
history = model.fit(train_padded, training_label_seq, epochs=num_epochs,
                    validation_data=(validation_padded, validation_label_seq),
                    verbose=2)


import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
{% endhighlight %}

### 3.3 Sequence models
#### 3.3.1 Quiz
<p align="justify">
<b>1.</b><br>
Why does sequence make a large difference when determining semantics of language?<br>
A. Because the order in which words appear dictate their impact on the meaning of the sentence<br>
B. Because the order of words doesn’t matter<br>
C. It doesn’t<br>
D. Because the order in which words appear dictate their meaning<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
How do Recurrent Neural Networks help you understand the impact of sequence on meaning?<br>
A. They don’t<br>
B. They shuffle the words evenly<br>
C. They look at the whole sentence at a time<br>
D. They carry meaning from one cell to the next<br>
<b>Answer:</b> D.<br><br>

<b>3.</b><br>
How does an LSTM help understand meaning when words that qualify each other aren’t necessarily beside each other in a sentence?<br>
A. They load all words into a cell state<br>
B. They don’t<br>
C. Values from earlier words can be carried to later ones via a cell state<br>
D. They shuffle the words randomly<br>
<b>Answer:</b> C.<br><br>

<b>4.</b><br>
What keras layer type allows LSTMs to look forward and backward in a sentence?<br>
A. Bothdirection<br>
B. Bidirectional<br>
C. Bilateral<br>
D. Unilateral<br>
<b>Answer:</b> B.<br><br>

<b>5.</b><br>
What’s the output shape of a bidirectional LSTM layer with 64 units?<br>
A. (128,1)<br>
B. (None, 64)<br>
C. (128,None)<br>
D. (None, 128)<br>
<b>Answer:</b> D.<br><br>

<b>6.</b><br>
When stacking LSTMs, how do you instruct an LSTM to feed the next one in the sequence?<br>
A. Ensure that they have the same number of units<br>
B. Do nothing, TensorFlow handles this automatically<br>
C. Ensure that return_sequences is set to True only on units that feed to another LSTM<br>
D. Ensure that return_sequences is set to True on all units<br>
<b>Answer:</b> C.<br><br>

<b>7.</b><br>
If a sentence has 120 tokens in it, and a Conv1D with 128 filters with a Kernal size of 5 is passed over it, what’s the output shape?<br>
A. (None, 120, 124)<br>
B. (None, 120, 128)<br>
C. (None, 116, 128)<br>
D. (None, 116, 124)<br>
<b>Answer:</b> C.<br><br>

<b>8.</b><br>
What’s the best way to avoid overfitting in NLP datasets?<br>
A. Use LSTMs<br>
B. Use GRUs<br>
C. Use Conv1D<br>
D. None of the above<br>
<b>Answer:</b> D.<br><br>
</p>

#### 3.3.2 Programming Assignment: Exploring overfitting in NLP
{% highlight Python %}
import json
import tensorflow as tf
import csv
import random
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers


embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=160000
test_portion=.1

corpus = []


# Note that I cleaned the Stanford dataset to remove LATIN1 encoding to make it
# easier for Python CSV reader
# You can do that yourself with:
# iconv -f LATIN1 -t UTF8 training.1600000.processed.noemoticon.csv -o
# training_cleaned.csv
# I then hosted it on my site to make it easier to use in this notebook

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv \
    -O /tmp/training_cleaned.csv

num_sentences = 0

with open("/tmp/training_cleaned.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        list_item=[]
        list_item.append(row[5])
        this_label=row[0]
        if this_label=='0':
            list_item.append(0)
        else:
            list_item.append(1)
        num_sentences = num_sentences + 1
        corpus.append(list_item)


print(num_sentences)
print(len(corpus))
print(corpus[1])
# Expected Output:
# 1600000
# 1600000
# ["is upset that he can't update his Facebook by texting it... and might cry
# as a result  School today also. Blah!", 0]


sentences=[]
labels=[]
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])


tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size=len(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type,
                       truncating=trunc_type)

split = int(test_portion * training_size)

test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = labels[0:split]
training_labels = labels[split:training_size]


print(vocab_size)
print(word_index['i'])
# Expected Output
# 138858
# 1


# Note this is the 100 dimension version of GloVe from Stanford
# I unzipped and hosted it on my site to make this notebook easier
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt \
    -O /tmp/glove.6B.100d.txt
embeddings_index = {};
with open('/tmp/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;



print(len(embeddings_matrix))
# Expected Output
# 138859


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim,
                              input_length=max_length,
                              weights=[embeddings_matrix],
                              trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 50

training_padded = np.array(training_sequences)
training_labels = np.array(training_labels)
testing_padded = np.array(test_sequences)
testing_labels = np.array(test_labels)

history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)
print("Training Complete")



import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()
{% endhighlight %}

### 3.4 Sequence models and literature
#### 3.4.1 Quiz
<p align="justify">
<b>1.</b><br>
What is the name of the method used to tokenize a list of sentences?<br>
A. fit_on_texts(sentences)<br>
B. fit_to_text(sentences)<br>
C. tokenize(sentences)<br>
D. tokenize_on_text(sentences)<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
If a sentence has 120 tokens in it, and a Conv1D with 128 filters with a Kernal size of 5 is passed over it, what’s the output shape?<br>
A. (None, 120, 128)<br>
B. (None, 120, 124)<br>
C. (None, 116, 128)<br>
D. (None, 116, 124)<br>
<b>Answer:</b> C.<br><br>

<b>3.</b><br>
What is the purpose of the embedding dimension?<br>
A. It is the number of words to encode in the embedding<br>
B. It is the number of dimensions for the vector representing the word encoding<br>
C. It is the number of letters in the word, denoting the size of the encoding<br>
D. It is the number of dimensions required to encode every word in the corpus<br>
<b>Answer:</b> B.<br><br>

<b>4.</b><br>
IMDB Reviews are either positive or negative. What type of loss function should be used in this scenario?<br>
A. Categorical crossentropy<br>
B. Adam<br>
C. Binary crossentropy<br>
D. Binary Gradient descent<br>
<b>Answer:</b> C.<br><br>

<b>5.</b><br>
If you have a number of sequences of different lengths, how do you ensure that they are understood when fed into a neural network?<br>
A. Process them on the input layer of the Neural Network using the pad_sequences property<br>
B. Use the pad_sequences object from the tensorflow.keras.preprocessing.sequence namespace<br>
C. Make sure that they are all the same length using the pad_sequences method of the tokenizer<br>
D. Specify the input layer of the Neural Network to expect different sizes with dynamic_length<br>
<b>Answer:</b> B.<br><br>

<b>6.</b><br>
When predicting words to generate poetry, the more words predicted the more likely it will end up gibberish. Why?<br>
A. Because you are more likely to hit words not in the training set<br>
B. Because the probability of prediction compounds, and thus increases overall<br>
C. It doesn’t, the likelihood of gibberish doesn’t change<br>
D. Because the probability that each word matches an existing phrase goes down the more words you create<br>
<b>Answer:</b> D.<br><br>

<b>7.</b><br>
What is a major drawback of word-based training for text generation instead of character-based generation?<br>
A. Word based generation is more accurate because there is a larger body of words to draw from<br>
B. Because there are far more words in a typical corpus than characters, it is much more memory intensive<br>
C. Character based generation is more accurate because there are less characters to predict<br>
D. There is no major drawback, it’s always better to do word-based training<br>
<b>Answer:</b> B.<br><br>

<b>8.</b><br>
How does an LSTM help understand meaning when words that qualify each other aren’t necessarily beside each other in a sentence?<br>
A. They shuffle the words randomly<br>
B. They don’t<br>
C. Values from earlier words can be carried to later ones via a cell state<br>
D. They load all words into a cell state<br>
<b>Answer:</b> C.<br><br>
</p>

#### 3.4.2 Using LSTMs, see if you can write Shakespeare!
{% highlight Python %}
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,\
Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np 


tokenizer = Tokenizer()
!wget --no-check-certificate \
  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt\
    -O /tmp/sonnets.txt
data = open('/tmp/sonnets.txt').read()

corpus = data.lower().split("\n")


tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# create input sequences using list of tokens
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)


# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,
                                         maxlen=max_sequence_len,
                                         padding='pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words)


model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150, return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2, activation='relu',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(predictors, label, epochs=100, verbose=1)


import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()
{% endhighlight %}


## 4. Sequences, Time Series and Prediction
### 4.1 Sequences and Prediction
#### 4.1.1 Quiz
<p align="justify">
<b>1.</b><br>
What is an example of a Univariate time series?<br>
A. Baseball scores<br>
B. Hour by hour weather<br>
C. Hour by hour temperature<br>
D. Fashion items<br>
<b>Answer:</b> C.<br><br>

<b>2.</b><br>
What is an example of a Multivariate time series?<br>
A. Fashion items<br>
B. Hour by hour weather<br>
C. Hour by hour temperature<br>
D. Baseball scores<br>
<b>Answer:</b> B.<br><br>

<b>3.</b><br>
What is imputed data?<br>
A. A good prediction of future data<br>
B. Data that has been withheld for various reasons<br>
C. A bad prediction of future data<br>
D. A projection of unknown (usually past or missing) data<br>
<b>Answer:</b> D.<br><br>

<b>4.</b><br>
A sound wave is a good example of time series data<br>
A. True<br>
B. False<br>
<b>Answer:</b> A.<br><br>

<b>5.</b><br>
What is Seasonality?<br>
A. Data aligning to the 4 seasons of the calendar<br>
B. Data that is only available at certain times of the year<br>
C. Weather data<br>
D. A regular change in shape of the data<br>
<b>Answer:</b> D.<br><br>

<b>6.</b><br>
What is a trend?<br>
A. An overall consistent flat direction for data<br>
B. An overall consistent upward direction for data<br>
C. An overall consistent downward direction for data<br>
D. An overall direction for data regardless of direction<br>
<b>Answer:</b> D.<br><br>

<b>7.</b><br>
In the context of time series, what is noise?<br>
A. Data that doesn’t have a trend<br>
B. Data that doesn’t have seasonality<br>
C. Unpredictable changes in time series data<br>
D. Sound waves forming a time series<br>
<b>Answer:</b> C.<br><br>

<b>8.</b><br>
What is autocorrelation?<br>
A. Data that automatically lines up in trends<br>
B. Data that doesn’t have noise<br>
C. Data that follows a predictable shape, even if the scale is different<br>
D. Data that automatically lines up seasonally<br>
<b>Answer:</b> C.<br><br>

<b>9.</b><br>
What is a non-stationary time series?<br>
A. One that has a disruptive event breaking trend and seasonality<br>
B. One that moves seasonally<br>
C. One that is consistent across all seasons<br>
D. One that has a constructive event forming trend and seasonality<br>
<b>Answer:</b> A.<br><br>
</p>

#### 4.1.2 Create and predict synthetic data
{% highlight Python %}
import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 7 * np.pi),
                    1 / np.exp(5 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.01
noise_level = 2

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365,
                         amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


split_time = 1100
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.show()


naive_forecast = series[split_time - 1:-1]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)

print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())


def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)


moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)


print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())


diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()


diff_moving_avg =\
moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()


diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()


print(keras.metrics.mean_squared_error(x_valid,
                                       diff_moving_avg_plus_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid,
                                        diff_moving_avg_plus_past).numpy())


diff_moving_avg_plus_smooth_past =\
moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()


print(keras.metrics.mean_squared_error(
        x_valid, diff_moving_avg_plus_smooth_past).numpy())
print(keras.metrics.mean_absolute_error(
        x_valid, diff_moving_avg_plus_smooth_past).numpy())
{% endhighlight %}

### 4.2 Deep Neural Networks for Time Series
#### 4.2.1 Quiz
<p align="justify">
<b>1.</b><br>
What is a windowed dataset?<br>
A. A fixed-size subset of a time series<br>
B. There’s no such thing<br>
C. The time series aligned to a fixed shape<br>
D. A consistent set of subsets of a time series<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
What does ‘drop_remainder=true’ do?<br>
A. It ensures that all data is used<br>
B. It ensures that all rows in the data window are the same length by cropping data<br>
C. It ensures that the data is all the same shape<br>
D. It ensures that all rows in the data window are the same length by adding data<br>
<b>Answer:</b> B.<br><br>

<b>3.</b><br>
What’s the correct line of code to split an n column window into n-1 columns for features and 1 column for a label<br>
A. dataset = dataset.map(lambda window: (window[n-1], window[1]))<br>
B. dataset = dataset.map(lambda window: (window[:-1], window[-1:]))<br>
C. dataset = dataset.map(lambda window: (window[-1:], window[:-1]))<br>
D. dataset = dataset.map(lambda window: (window[n], window[1]))<br>
<b>Answer:</b> B.<br><br>

<b>4.</b><br>
What does MSE stand for?<br>
A. Mean Slight error<br>
B. Mean Squared error<br>
C. Mean Series error<br>
D. Mean Second error<br>
<b>Answer:</b> B.<br><br>

<b>5.</b><br>
What does MAE stand for?<br>
A. Mean Average Error<br>
B. Mean Advanced Error<br>
C. Mean Absolute Error<br>
D. Mean Active Error<br>
<b>Answer:</b> C.<br><br>

<b>6.</b><br>
If time values are in time[], series values are in series[] and we want to split the series into training and validation at time 1000, what is the correct code?<br><br>
A.<br>
time_train = time[split_time]<br>
x_train = series[split_time]<br>
time_valid = time[split_time]<br>
x_valid = series[split_time]<br><br>

B.<br>
time_train = time[:split_time]<br>
x_train = series[:split_time]<br>
time_valid = time[split_time]<br>
x_valid = series[split_time]<br><br>

C.<br>
time_train = time[split_time]<br>
x_train = series[split_time]<br>
time_valid = time[split_time:]<br>
x_valid = series[split_time:]<br><br>

D.<br>
time_train = time[:split_time]<br>
x_train = series[:split_time]<br>
time_valid = time[split_time:]<br>
x_valid = series[split_time:]<br>
<b>Answer:</b> D.<br><br>

<b>7.</b><br>
If you want to inspect the learned parameters in a layer after training, what’s a good technique to use?<br>
A. Decompile the model and inspect the parameter set for that layer<br>
B. Run the model with unit data and inspect the output for that layer<br>
C. Iterate through the layers dataset of the model to find the layer you want<br>
D. Assign a variable to the layer and add it to the model using that variable. Inspect its properties after training<br>
<b>Answer:</b> D.<br><br>

<b>8.</b><br>
How do you set the learning rate of the SGD optimizer?<br>
A. Use the Rate property <br>
B. Use the RateOfLearning property<br>
C. Use the lr property<br>
D. You can’t set it<br>
<b>Answer:</b> C.<br><br>

<b>9.</b><br>
If you want to amend the learning rate of the optimizer on the fly, after each epoch, what do you do?<br>
A. Use a LearningRateScheduler and pass it as a parameter to a callback<br>
B. Callback to a custom function and change the SGD property<br>
C. Use a LearningRateScheduler object in the callbacks namespace and assign that to the callback<br>
D. You can’t set it<br>
<b>Answer:</b> C.<br><br>
</p>

#### 4.2.2 Predict with a DNN
{% highlight Python %}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(10 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.005
noise_level = 3

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365,
                         amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=51)

split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

plot_series(time, series)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1],
                           window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset


dataset = windowed_dataset(x_train, window_size, batch_size,
                           shuffle_buffer_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6,
                                                            momentum=0.9))
model.fit(dataset,epochs=100,verbose=0)


forecast = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)


tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
{% endhighlight %}

### 4.3 Recurrent Neural Networks for time series
#### 4.3.1 Quiz
<p align="justify">
<b>1.</b><br>
If X is the standard notation for the input to an RNN, what are the standard notations for the outputs?<br>
A. Y<br>
B. H<br>
C. Y(hat) and H<br>
D. H(hat) and Y<br>
<b>Answer:</b> C.<br><br>

<b>2.</b><br>
What is a sequence to vector if an RNN has 30 cells numbered 0 to 29<br>
A. The total Y(hat) for all cells<br>
B. The average Y(hat) for all 30 cells<br>
C. The Y(hat) for the first cell<br>
D. The Y(hat) for the last cell<br>
<b>Answer:</b> D.<br><br>

<b>3.</b><br>
What does a Lambda layer in a neural network do?<br>
A. Changes the shape of the input or output data<br>
B. There are no Lambda layers in a neural network<br>
C. Allows you to execute arbitrary code while training<br>
D. Pauses training without a callback<br>
<b>Answer:</b> C.<br><br>

<b>4.</b><br>
What does the axis parameter of tf.expand_dims do?<br>
A. Defines the dimension index to remove when you expand the tensor<br>
B. Defines the dimension index at which you will expand the shape of the tensor<br>
C. Defines the axis around which to expand the dimensions<br>
D. Defines if the tensor is X or Y<br>
<b>Answer:</b> B.<br><br>

<b>5.</b><br>
A new loss function was introduced in this module, named after a famous statistician. What is it called?<br>
A. Hubble loss<br>
B. Hawking loss<br>
C. Hyatt loss<br>
D. Huber loss<br>
<b>Answer:</b> D.<br><br>

<b>6.</b><br>
What’s the primary difference between a simple RNN and an LSTM<br>
A. In addition to the H output, LSTMs have a cell state that runs across all cells<br>
B. In addition to the H output, RNNs have a cell state that runs across all cells<br>
C. LSTMs have a single output, RNNs have multiple<br>
D. LSTMs have multiple outputs, RNNs have a single one<br>
<b>Answer:</b> A.<br><br>

<b>7.</b><br>
If you want to clear out all temporary variables that tensorflow might have from previous sessions, what code do you run?<br>
A. tf.keras.clear_session<br>
B. tf.cache.clear_session()<br>
C. tf.keras.backend.clear_session()<br>
D. tf.cache.backend.clear_session()<br>
<b>Answer:</b> C.<br><br>

<b>8.</b><br>
What happens if you define a neural network with these two layers?<br>
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),<br>
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),<br>
tf.keras.layers.Dense(1),<br>
A. Your model will fail because you need return_sequences=True after each LSTM layer<br>
B. Your model will fail because you need return_sequences=True after the first LSTM layer<br>
C. Your model will compile and run correctly<br>
D. Your model will fail because you have the same number of cells in each LSTM<br>
<b>Answer:</b> B.<br><br>
</p>

#### 4.3.2 Mean Absolute Error
{% highlight Python %}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(10 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.005
noise_level = 3

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365,
                         amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=51)

split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

plot_series(time, series)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(
          lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size,
                           shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,
                                                     return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 10.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])


plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size,
                           shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,
                                                      return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])


model.compile(loss="mse",
              optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=["mae"])
history = model.fit(dataset,epochs=500,verbose=1)


forecast = []
results = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)


tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()


import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
mae=history.history['mae']
loss=history.history['loss']

epochs=range(len(loss)) # Get number of epochs

#------------------------------------------------
# Plot MAE and Loss
#------------------------------------------------
plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()

epochs_zoom = epochs[200:]
mae_zoom = mae[200:]
loss_zoom = loss[200:]

#------------------------------------------------
# Plot Zoomed MAE and Loss
#------------------------------------------------
plt.plot(epochs_zoom, mae_zoom, 'r')
plt.plot(epochs_zoom, loss_zoom, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()
{% endhighlight %}

### 4.4 Real-world time series data
#### 4.4.1 Quiz
<p align="justify">
<b>1.</b><br>
How do you add a 1 dimensional convolution to your model for predicting time series data?<br>
A. Use a 1DConv layer type<br>
B. Use a 1DConvolution layer type<br>
C. Use a Convolution1D layer type<br>
D. Use a Conv1D layer type<br>
<b>Answer:</b> D.<br><br>

<b>2.</b><br>
What’s the input shape for a univariate time series to a Conv1D? <br>
A. [None, 1]<br>
B. []<br>
C. [1]<br>
D. [1, None]<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
You used a sunspots dataset that was stored in CSV. What’s the name of the Python library used to read CSVs?<br>
A. CSV<br>
B. CommaSeparatedValues<br>
C. PyCSV<br>
D. PyFiles<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
If your CSV file has a header that you don’t want to read into your dataset, what do you execute before iterating through the file using a ‘reader’ object?<br><br>
A. reader.ignore_header()<br>
B. reader.next<br>
C. next(reader)<br>
D. reader.read(next)<br>
<b>Answer:</b> C.<br><br>

<b>5.</b><br>
When you read a row from a reader and want to cast column 2 to another data type, for example, a float, what’s the correct syntax?<br>
A. Convert.toFloat(row[2])<br>
B. You can’t. It needs to be read into a buffer and a new float instantiated from the buffer<br>
C. float f = row[2].read()<br>
D. float(row[2])<br>
<b>Answer:</b> D.<br><br>

<b>6.</b><br>
What was the sunspot seasonality?<br>
A. 4 times a year<br>
B. 11 years<br>
C. 11 or 22 years depending on who you ask<br>
D. 22 years<br>
<b>Answer:</b> C.<br><br>

<b>7.</b><br>
After studying this course, what neural network type do you think is best for predicting time series like our sunspots dataset?<br>
A. Convolutions<br>
B. DNN<br>
C. RNN / LSTM<br>
D. A combination of all of the above<br>
<b>Answer:</b> D.<br><br>

<b>8.</b><br>
Why is MAE a good analytic for measuring accuracy of predictions for time series?<br>
A. It doesn’t heavily punish larger errors like square errors do<br>
B. It biases towards small errors<br>
C. It punishes larger errors<br>
D. It only counts positive errors<br>
<b>Answer:</b> A.<br><br>
</p>

#### 4.4.2 Sunspots
{% highlight Python %}
import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


!wget --no-check-certificate \
https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv \
    -O /tmp/daily-min-temperatures.csv


import csv
time_step = []
temps = []

with open('/tmp/daily-min-temperatures.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  next(reader)
  step=0
  for row in reader:
    temps.append(float(row[1]))
    time_step.append(step)
    step = step + 1

series = np.array(temps)
time = np.array(time_step)
plt.figure(figsize=(10, 6))
plot_series(time, series)


split_time = 2500
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer_size = 1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
window_size = 64
batch_size = 256
train_set = windowed_dataset(x_train, window_size, batch_size,
                             shuffle_buffer_size)
print(train_set)
print(x_train.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])


plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 60])


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
train_set = windowed_dataset(x_train, window_size=60,
                             batch_size=100,
                             shuffle_buffer=shuffle_buffer_size)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])


optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set,epochs=150)


rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]


plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)


tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
print(rnn_forecast)
{% endhighlight %}
