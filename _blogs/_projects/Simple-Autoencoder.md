---
layout: post
title:  "Simple Autoencoder"
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
The complete code is available <a href="https://github.com/chaopan1995/PROJECTS/tree/master/SimpleAutoencoder">here</a>.
</p>


## 1. Object
<p align="justify">
In this task you will train our own autoencoder for human faces!
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/SimpleAutoencoder/1.png"/></center>
</p>
<p align="justify">
We're going to train deep autoencoders and apply them to faces and similar images search.<br><br>

Our new test subjects are human faces from the <a href="http://vis-www.cs.umass.edu/lfw/">lfw dataset</a>.
</p>


## 2. Packages
{% highlight python %}
import cv2
import os
import tarfile
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Flatten,\
Dense, Reshape, Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.nn import conv2d_transpose
{% endhighlight %}
<p align="justify">
We use Tensorflow 2.x
</p>


## 3. Dataset
<p align="justify">
Dataset should be downloaded at first: <a href="http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt">attributes</a>, <a href="http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz">images</a> and <a href="http://vis-www.cs.umass.edu/lfw/lfw.tgz">raw images</a>.<br><br>

Normalize the images
$$x_{norm} = \frac{x}{255} - 0.5$$
</p>
{% highlight python %}
# http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
ATTRS_NAME = "lfw_attributes.txt"
# http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
IMAGES_NAME = "lfw-deepfunneled.tar"
# http://vis-www.cs.umass.edu/lfw/lfw.tgz
RAW_IMAGES_NAME = "lfw.tar"
# workaround for https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0


class SimpleTqdm():
    def __init__(self, iterable=None, total=None, **kwargs):
        self.iterable = list(iterable) if iterable is not None else None
        self.total = len(self.iterable) if self.iterable is not None else total
        assert self.iterable is not None or self.total is not None
        self.current_step = 0
        self.print_frequency = max(self.total // 50, 1)
        self.desc = ""

    def set_description_str(self, desc):
        self.desc = desc

    def set_description(self, desc):
        self.desc = desc

    def update(self, steps):
        last_print_step = (self.current_step // self.print_frequency) *\
        self.print_frequency
        i = 1
        while last_print_step + i * self.print_frequency <=\
        self.current_step + steps:
            print("*", end='')
            i += 1
        self.current_step += steps

    def close(self):
        print("\n" + self.desc)

    def __iter__(self):
        assert self.iterable is not None
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.total:
            element = self.iterable[self.index]
            self.update(1)
            self.index += 1
            return element
        else:
            self.close()
            raise StopIteration


def tqdm_notebook_failsafe(*args, **kwargs):
    try:
        return tqdm.notebook.tqdm(*args, **kwargs)
    except:
        # tqdm is broken on Google Colab
        print('using simple tqdm')
        return SimpleTqdm(*args, **kwargs)


def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_lfw_dataset(
        use_raw=False,
        dx=80, dy=80,
        dimx=45, dimy=45):

    # read attrs
    df_attrs = pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)
    df_attrs.columns = list(df_attrs.columns)[1:] + ["NaN"]
    df_attrs = df_attrs.drop("NaN", axis=1)
    imgs_with_attrs = set(map(tuple, df_attrs[["person", "imagenum"]].values))

    # read photos
    all_photos = []
    photo_ids = []

    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
        for m in tqdm_notebook_failsafe(f.getmembers()):
            if m.isfile() and m.name.endswith(".jpg"):
                # prepare image
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                img = img[dy:-dy, dx:-dx]
                img = cv2.resize(img, (dimx, dimy))
                # parse person
                fname = os.path.split(m.name)[-1]
                fname_splitted = fname[:-4].replace('_', ' ').split()
                person_id = ' '.join(fname_splitted[:-1])
                photo_number = int(fname_splitted[-1])
                if (person_id, photo_number) in imgs_with_attrs:
                    all_photos.append(img)
                    photo_ids.append({'person': person_id,
                                      'imagenum': photo_number})

    photo_ids = pd.DataFrame(photo_ids)
    all_photos = np.stack(all_photos).astype('uint8')

    # preserve photo_ids order!
    all_attrs = photo_ids.merge(df_attrs,
                                on=('person', 'imagenum')).\
                                drop(["person", "imagenum"],
                                axis=1)

    return all_photos, all_attrs

# load images
X, attr = load_lfw_dataset(use_raw=True, dimx=32, dimy=32)
IMG_SHAPE = X.shape[1:]

# center images
X = X.astype('float32') / 255.0 - 0.5

# split
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
{% endhighlight %}


## 4. Autoencoder Architecture
<p align="justify">
Let's design autoencoder as two sequential keras models: the encoder and decoder respectively.<br><br>

We will then use symbolic API to apply and train these models.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/SimpleAutoencoder/2.png"/></center>
</p>

### 4.1 PCA
<p align="justify">
Principial Component Analysis is a popular dimensionality reduction method.<br><br>

Under the hood, PCA attempts to decompose object-feature matrix $X$ into two smaller matrices: $A$ and $B$:
$$\|(XAB- X\|_2^2  \rightarrow \min_{A, B}$$

-- $X \in \mathbb{R}^{n \times m}$ - object matrix (centered)<br>
-- $A \in \mathbb{R}^{m \times d}$ - matrix of direct transformation<br>
-- $B \in \mathbb{R}^{d \times m}$ - matrix of reverse transformation<br>
-- $n$ samples, $m$ original dimensions and $d$ target dimensions<br><br>

In geometric terms, we want to find d axes along which most of variance occurs. The "natural" axes, if you wish.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/SimpleAutoencoder/3.png"/></center>
</p>
<p align="justify">
PCA can also be seen as a special case of an autoencoder
-- X -> Dense(d units) -> code<br>
-- code -> Dense(m units) -> X<br>
Where Dense is a fully-connected layer with linear activaton:
$$f(X) = W \cdot X + \vec{b}$$

Note: the bias term in those layers is responsible for "centering" the matrix i.e. substracting mean.
</p>
{% highlight python %}
def build_pca_autoencoder(img_shape, code_size):
    """
    Here we define a simple linear autoencoder as described above.
    We also flatten and un-flatten data to be compatible with image shapes
    """
    clear_session()
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    # Flatten image to vector
    encoder.add(Flatten())
    # Actual encoder
    encoder.add(Dense(code_size))

    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    # Actual decoder, height*width*3 units
    decoder.add(Dense(np.prod(img_shape)))
    # Un-flatten
    decoder.add(Reshape(img_shape))
    
    return encoder, decoder

encoder, decoder = build_pca_autoencoder(IMG_SHAPE, code_size=32)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

autoencoder.fit(x=X_train, y=X_train, epochs=15,
                validation_data=[X_test, X_test],
                verbose=1)
{% endhighlight %}
<p align="justify">
Visualize the output
</p>
{% highlight python %}
def visualize(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    # img[None] is the same as img[np.newaxis, :]
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

score = autoencoder.evaluate(X_test,X_test,verbose=0)
print("PCA MSE:", score)

for i in range(5):
    img = X_test[i]
    visualize(img,encoder,decoder)
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/SimpleAutoencoder/4.png"/></center>
</p>

### 4.2 Convolutional Autoencoder
<p align="justify">
$\bigstar$ Encoder<br>
-- The encoder part is pretty standard, we stack convolutional and pooling layers and finish with a dense layer to get the representation of desirable size (code_size).<br>
-- We recommend to use activation='elu' for all convolutional and dense layers.<br>
-- We recommend to repeat (conv, pool) 4 times with kernel size (3, 3), padding='same' and the following numbers of output channels: 32, 64, 128, 256.
-- Remember to flatten (L.Flatten()) output before adding the last dense layer!<br><br>

$\bigstar$ Decoder<br>
-- For decoder we will use so-called "transpose convolution".<br>
-- Traditional convolutional layer takes a patch of an image and produces a number (patch -> number). In "transpose convolution" we want to take a number and produce a patch of an image (number -> patch). We need this layer to "undo" convolutions in encoder.<br>
-- Here's how "transpose convolution" works:
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/SimpleAutoencoder/5.png"/></center>
</p>
<p align="justify">
In this example we use a stride of 2 to produce 4x4 output, this way we "undo" pooling as well. Another way to think about it: we "undo" convolution with stride 2 (which is similar to conv + pool).<br><br>

You can add <b>transpose convolution</b> layer in Keras like this:<br>
</p>
{% highlight python %}
Conv2DTranspose(filters=?, kernel_size=(3, 3), strides=2,
                activation='elu', padding='same')
{% endhighlight %}
<p align="justify">
Our decoder starts with a dense layer to "undo" the last layer of encoder. Remember to reshape its output to "undo" <b>Flatten()</b> in encoder.<br><br>

Now we're ready to undo (conv, pool) pairs. For this we need to stack 4 <b>Conv2DTranspose</b> layers with the following numbers of output channels: 128, 64, 32, 3. Each of these layers will learn to "undo" (conv, pool) pair in encoder. For the last <b>Conv2DTranspose</b> layer use activation=None because that is our final image.
</p>
{% highlight python %}
# Let's play around with transpose convolution on examples first
def test_conv2d_transpose(img_size, filter_size):
    print("Transpose convolution test for img_size={}, filter_size={}:".\
          format(img_size, filter_size))
    
    x = (np.arange(img_size ** 2, dtype=np.float32) + 1).\
    reshape((1, img_size, img_size, 1))
    f = (np.ones(filter_size ** 2, dtype=np.float32)).\
    reshape((filter_size, filter_size, 1, 1))

    clear_session()

    
    conv = conv2d_transpose(x, f, 
                            output_shape=(1, img_size * 2, img_size * 2, 1), 
                            strides=[1, 2, 2, 1], 
                            padding='SAME')

    #result = s.run(conv)
    print("input:")
    print(x[0, :, :, 0])
    print("filter:")
    print(f[:, :, 0, 0])
    print("output:")
    print(conv[0, :, :, 0])
        
test_conv2d_transpose(img_size=2, filter_size=2)
test_conv2d_transpose(img_size=2, filter_size=3)
test_conv2d_transpose(img_size=4, filter_size=2)
test_conv2d_transpose(img_size=4, filter_size=3)
{% endhighlight %}
<p align="justify">
Set up a convolutional autoencoder
</p>
{% highlight python %}
def build_deep_autoencoder(img_shape, code_size):
    """
    PCA's deeper brother. See instructions above. Use `code_size` in
    layer definitions.
    """
    H,W,C = img_shape
    clear_session()
    # encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Conv2D(filters=32, kernel_size=(3,3),
                       padding="same", activation='elu'))
    encoder.add(MaxPooling2D())
    encoder.add(Conv2D(filters=64, kernel_size=(3,3),
                       padding="same", activation='elu'))
    encoder.add(MaxPooling2D())
    encoder.add(Conv2D(filters=128, kernel_size=(3,3),
                       padding="same", activation='elu'))
    encoder.add(MaxPooling2D())
    encoder.add(Conv2D(filters=256, kernel_size=(3,3),
                       padding="same", activation='elu'))
    encoder.add(MaxPooling2D())
    # flatten image to vector
    encoder.add(Flatten())
    # actual encoder
    encoder.add(Dense(code_size, activation='elu'))

    # decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    # actual decoder, height*width*3 units
    decoder.add(Dense(np.prod((2, 2, 256))))
    # un-flatten
    decoder.add(Reshape((2, 2, 256)))
    decoder.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2,
                                activation='elu', padding='same'))
    decoder.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2,
                                activation='elu', padding='same'))
    decoder.add(Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2,
                                activation='elu', padding='same'))
    decoder.add(Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2,
                                activation=None, padding='same'))
    
    return encoder, decoder

encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
{% endhighlight %}
<p align="justify">
We should check our code
</p>
{% highlight python %}
# Check autoencoder shapes along different code_sizes
get_dim = lambda layer: np.prod(layer.output_shape[1:])
for code_size in [1,8,32,128,512]:
    encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=code_size)
    print("Testing code size %i" % code_size)
    assert encoder.output_shape[1:]==(code_size,),\
    "encoder must output a code of required size"
    assert decoder.output_shape[1:]==IMG_SHAPE,\
    "decoder must output an image of valid shape"
    assert len(encoder.trainable_weights)>=6,\
    "encoder must contain at least 3 layers"
    assert len(decoder.trainable_weights)>=6,\
    "decoder must contain at least 3 layers"
    
    for layer in encoder.layers + decoder.layers:
        assert get_dim(layer) >= code_size,\
        "Encoder layer %s is smaller than bottleneck (%i units)"%\
        (layer.name,get_dim(layer))

print("All tests passed!")
{% endhighlight %}
<p align="justify">
Run the model
</p>
{% highlight python %}
# Look at encoder and decoder shapes.
# Total number of trainable parameters of encoder and decoder should be close.
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
encoder.summary()
decoder.summary()

encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adamax", loss='mse')

autoencoder.fit(x=X_train, y=X_train, epochs=25,
                validation_data=[X_test, X_test],
                verbose=1)
{% endhighlight %}
<p align="justify">
Finally, we have a convolutional autoencoder MSE: 0.005864742211997509
</p>

### 4.3 Denoising Autoencoder
<p align="justify">
Let's now turn our model into a denoising autoencoder:
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/SimpleAutoencoder/6.png"/></center>
</p>
<p align="justify">
We'll keep the model architecture, but change the way it is trained. In particular, we'll corrupt its input data randomly with noise before each epoch.<br><br>

There are many strategies to introduce noise: adding gaussian white noise, occluding with random black rectangles, etc. We will add gaussian white noise.
</p>
{% highlight python %}
def apply_gaussian_noise(X,sigma=0.1):
    """
    adds noise from standard normal distribution with standard deviation sigma
    :param X: image tensor of shape [batch,height,width,3]
    Returns X + noise.
    """
    noise = np.random.normal(X.mean(), sigma, X.shape)
    return X + noise

# noise tests
theoretical_std = (X_train[:100].std()**2 + 0.5**2)**.5
our_std = apply_gaussian_noise(X_train[:100],sigma=0.5).std()
assert abs(theoretical_std - our_std) < 0.01,\
"Standard deviation does not match it's required value. "
"Make sure you use sigma as std."
assert abs(apply_gaussian_noise(X_train[:100],sigma=0.5).mean() -
           X_train[:100].mean()) < 0.01,\
           "Mean has changed. Please add zero-mean noise"

# test different noise scales
plt.subplot(1,4,1)
show_image(X_train[0])
plt.subplot(1,4,2)
show_image(apply_gaussian_noise(X_train[:1],sigma=0.01)[0])
plt.subplot(1,4,3)
show_image(apply_gaussian_noise(X_train[:1],sigma=0.1)[0])
plt.subplot(1,4,4)
show_image(apply_gaussian_noise(X_train[:1],sigma=0.5)[0])
{% endhighlight %}
<p align="justify">
Train the model
</p>
{% highlight python %}
# we use bigger code size here for better quality
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=512)
assert encoder.output_shape[1:]==(512,),\
"encoder must output a code of required size"

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile('adamax', 'mse')

for i in range(25):
    print("Epoch %i/25, Generating corrupted samples..."%(i+1))
    X_train_noise = apply_gaussian_noise(X_train)
    X_test_noise = apply_gaussian_noise(X_test)
    
    # we continue to train our model with new noise-augmented data
    autoencoder.fit(x=X_train_noise, y=X_train, epochs=1,
                    validation_data=[X_test_noise, X_test],
                    verbose=0)
{% endhighlight %}
{% highlight python %}
X_test_noise = apply_gaussian_noise(X_test)
denoising_mse = autoencoder.evaluate(X_test_noise, X_test, verbose=0)
print("Denoising MSE:", denoising_mse)
for i in range(5):
    img = X_test_noise[i]
    visualize(img,encoder,decoder)
{% endhighlight %}
<p align="justify">
Finally, we have a Denoising MSE: 0.0029486105777323246
</p>

### 4.4 Image Retrieval with Autoencoders
<p align="justify">
So we've just trained a network that converts image into itself imperfectly. This task is not that useful in and of itself, but it has a number of awesome side-effects. Let's see them in action.<br><br>

First thing we can do is image retrieval aka image search. We will give it an image and find similar images in latent space:
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/SimpleAutoencoder/7.png"/></center>
</p>
<p align="justify">
To speed up retrieval process, one should use Locality Sensitive Hashing on top of encoded vectors. This <a href="https://erikbern.com/2015/07/04/benchmark-of-approximate-nearest-neighbor-libraries.html">technique</a> can narrow down the potential nearest neighbours of our image in latent space (encoder code). We will caclulate nearest neighbours in brute force way for simplicity.
</p>
{% highlight python %}
# restore trained encoder weights
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)

images = X_train
codes = encoder.predict(images)
assert len(codes) == len(images)

from sklearn.neighbors import NearestNeighbors
nei_clf = NearestNeighbors(metric="euclidean")
nei_clf.fit(codes)
{% endhighlight %}
{% highlight python %}
def get_similar(image, n_neighbors=5):
    assert image.ndim==3,"image must be [batch,height,width,3]"

    code = encoder.predict(image[None])
    
    (distances,),(idx,) = nei_clf.kneighbors(code,n_neighbors=n_neighbors)
    
    return distances,images[idx]

def show_similar(image):
    
    distances,neighbors = get_similar(image,n_neighbors=3)
    
    plt.figure(figsize=[8,7])
    plt.subplot(1,4,1)
    show_image(image)
    plt.title("Original image")
    
    for i in range(3):
        plt.subplot(1,4,i+2)
        show_image(neighbors[i])
        plt.title("Dist=%.3f"%distances[i])
    plt.show()
{% endhighlight %}
<p align="justify">
For a smile face
</p>
{% highlight python %}
# smiles
show_similar(X_test[247])
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/SimpleAutoencoder/8.png"/></center>
</p>

### 4.5 Cheap Image Morphing
<p align="justify">
We can take linear combinations of image codes to produce new images with decoder.
</p>
{% highlight python %}
# restore trained encoder weights
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
encoder.load_weights("encoder.h5")
decoder.load_weights("decoder.h5")

for _ in range(5):
    image1,image2 = X_test[np.random.randint(0,len(X_test),size=2)]

    code1, code2 = encoder.predict(np.stack([image1, image2]))

    plt.figure(figsize=[10,4])
    for i,a in enumerate(np.linspace(0,1,num=7)):

        output_code = code1*(1-a) + code2*(a)
        output_image = decoder.predict(output_code[None])[0]

        plt.subplot(1,7,i+1)
        show_image(output_image)
        plt.title("a=%.2f"%a)
        
    plt.show()
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/SimpleAutoencoder/9.png"/></center>
</p>