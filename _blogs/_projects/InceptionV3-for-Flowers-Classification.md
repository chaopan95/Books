---
layout: post
title:  "InceptionV3 for Flowers Classification"
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
The complete code is available <a href="https://github.com/chaopan1995/PROJECTS/tree/master/InceptionV3-for-Flowers-Classification">here</a>.
</p>


## 1. Object
<p align="justify">
In this task you will fine-tune InceptionV3 architecture for flowers classification task.<br><br>

$\bigstar$ InceptionV3 architecture
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/InceptionV3forFlowersClassification/1.png"/></center>
</p>
<p align="justify">
Flowers classification dataset consists of 102 flower categories commonly occurring in the United Kingdom. Each class contains between 40 and 258 images
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/InceptionV3forFlowersClassification/2.png"/></center>
</p>


## 2. Packages
{% highlight python %}
import cv2
import tarfile

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from tensorflow.keras.backend import clear_session
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense,\
BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adamax
# we will crop and resize input images to IMG_SIZE x IMG_SIZE
IMG_SIZE = 250
# batch generator
BATCH_SIZE = 32
{% endhighlight %}
<p align="justify">
We use Tensorflow 2.x
</p>


## 3. Dataset
<p align="justify">
<a href="http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz">Images</a> and <a href="http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat">labels</a> should be downloaded at first. If you are interested in their website, <a href="http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html">click here</a>
</p>


## 4. Preprocess
<p align="justify">
We will take a center crop from each image like this:
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/InceptionV3forFlowersClassification/3.png"/></center>
</p>
{% highlight python %}
def image_center_crop(img):
    """
    Makes a square center crop of an img, which is a [h, w, 3] numpy array.
    Returns [min(h, w), min(h, w), 3] output with same width and height.
    For cropping use numpy slicing.
    """
    h, w, _ = img.shape
    a = min(h, w)
    cropped_img = img[int((h-a)/2): int((h+a)/2), int((w-a)/2): int((w+a)/2),:]
    
    # checks for errors
    h, w, c = img.shape
    assert cropped_img.shape == (min(h, w), min(h, w), c),\
    "error in image_center_crop!"
    
    return cropped_img
{% endhighlight %}
<p align="justify">
Becase images are stored in a .tar file and labels are stored in .mat file, we need code some functions to read them.
</p>
{% highlight python %}
def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def prepare_raw_bytes_for_model(raw_bytes, normalize_for_model=True):
    # decode image raw bytes to matrix
    img = decode_image_from_raw_bytes(raw_bytes)
    # take squared center crop
    img = image_center_crop(img)
    # resize for our model
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if normalize_for_model:
        # prepare for normalization
        img = img.astype("float32")
        # normalize for model
        img = preprocess_input(img)
    return img

# reads bytes directly from tar by filename
def read_raw_from_tar(tar_fn, fn):
    with tarfile.open(tar_fn) as f:
        m = f.getmember(fn)
        return f.extractfile(m).read()

# read all filenames and labels for them
def get_all_filenames(tar_fn):
    with tarfile.open(tar_fn) as f:
        return [m.name for m in f.getmembers() if m.isfile()]

# Yield raw image bytes from tar with corresponding label
def raw_generator_with_label_from_tar(tar_fn, files, labels):
    label_by_fn = dict(zip(files, labels))
    with tarfile.open(tar_fn) as f:
        while True:
            m = f.next()
            if m is None:
                break
            if m.name in label_by_fn:
                yield f.extractfile(m).read(), label_by_fn[m.name]
{% endhighlight %}
<p align="justify">
We look at the first picture
</p>
{% highlight python %}
# test cropping
def test_cropping():
    raw_bytes = read_raw_from_tar("Data/102flowers.tar", "jpg/image_00001.jpg")

    img = decode_image_from_raw_bytes(raw_bytes)
    print(img.shape)
    plt.imshow(img)
    plt.show()

    img = prepare_raw_bytes_for_model(raw_bytes, normalize_for_model=False)
    print(img.shape)
    plt.imshow(img)
    plt.show()

test_cropping()
{% endhighlight %}
<p align="justify">
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/PROJECTS/InceptionV3forFlowersClassification/4.png"/></center>
</p>
<p align="justify">
Load all data dan convert them. Notice that we have 102 classes.
</p>
{% highlight python %}
# list all files in tar sorted by name
all_files = sorted(get_all_filenames("Data/102flowers.tar"))
# read class labels (0, 1, 2, ...)
all_labels = loadmat('Data/imagelabels.mat')['labels'][0] - 1
# all_files and all_labels are aligned now
N_CLASSES = len(np.unique(all_labels))
print(N_CLASSES)

# split into train/test
tr_files, te_files, tr_labels, te_labels = \
    train_test_split(all_files, all_labels, test_size=0.2,
                     random_state=42, stratify=all_labels)
{% endhighlight %}
<p align="justify">
Mini-batch function: we want to pick up randomly 32 images once
</p>
{% highlight python %}
def batch_generator(items, batch_size):
    """
    Implement batch generator that yields items in batches of size batch_size.
    There's no need to shuffle input items, just chop them into batches.
    Remember about the last batch that can be smaller than batch_size!
    Input: any iterable (list, generator, ...). You should do `for item in
        items: ...`.
        In case of generator you can pass through your items only once!
    Output: In output yield each batch as a list of items.
    """
    count = 1
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch):
        yield batch

def train_generator(files, labels):
    while True:  # so that Keras can loop through this as long as it wants
        for batch in batch_generator(raw_generator_with_label_from_tar(
                "Data/102flowers.tar", files, labels), BATCH_SIZE):
            # prepare batch images
            batch_imgs = []
            batch_targets = []
            for raw, label in batch:
                img = prepare_raw_bytes_for_model(raw)
                batch_imgs.append(img)
                batch_targets.append(label)
            # stack images into 4D tensor [batch_size, img_size, img_size, 3]
            batch_imgs = np.stack(batch_imgs, axis=0)
            # convert targets into 2D tensor [batch_size, num_classes]
            batch_targets = to_categorical(batch_targets, N_CLASSES)
            yield batch_imgs, batch_targets
{% endhighlight %}


## 5. Train
<p align="justify">
We cannot train such a huge architecture from scratch with such a small dataset.<br><br>

But using fine-tuning of last layers of pre-trained network you can get a pretty good classifier very quickly.
</p>
{% highlight python %}
def inception(use_imagenet=True):
    clear_session()
    # load pre-trained model graph, don't add final layer
    model = InceptionV3(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                        weights='imagenet' if use_imagenet else None)
    # add global pooling just like in InceptionV3
    new_output = GlobalAveragePooling2D()(model.output)
    # add new dense layer for our labels
    new_output = Dense(N_CLASSES, activation='softmax')(new_output)
    model = Model(model.inputs, new_output)
    return model

model = inception()
model.summary()
{% endhighlight %}
<p align="justify">
InceptionV3 has 313 layers. In order to fine tune, we change some parameters in the last 50 layers
</p>
{% highlight python %}
# set all layers trainable by default
for layer in model.layers:
    layer.trainable = True
    if isinstance(layer, BatchNormalization):
        # we do aggressive exponential smoothing of batch norm
        # parameters to faster adjust to our new dataset
        layer.momentum = 0.9
    
# fix deep layers (fine-tuning only last 50)
for layer in model.layers[:-50]:
    # fix all but batch norm layers, because we neeed to
    # update moving averages for a new dataset!
    if not isinstance(layer, BatchNormalization):
        layer.trainable = False
{% endhighlight %}
<p align="justify">
Train the model
</p>
{% highlight python %}
# compile new model
model.compile(
    # we train 102-way classification
    loss='categorical_crossentropy',
    # we can take big lr here because we fixed first layers
    optimizer=Adamax(lr=1e-2),
    # report accuracy during training
    metrics=['accuracy']
)

# fine tune for 2 epochs (full passes through all training data)
# we make 2*8 epochs, where epoch is 1/8 of our training data to
# see progress more often
model.fit(
    train_generator(tr_files, tr_labels), 
    steps_per_epoch=len(tr_files) // BATCH_SIZE // 8,
    epochs=2 * 8,
    validation_data=train_generator(te_files, te_labels), 
    validation_steps=len(te_files) // BATCH_SIZE // 4,
    verbose=1,
    initial_epoch= 0
)
{% endhighlight %}
<p align="justify">
Finally, we have a well-trained model with 0.9987 accuracy for training set and 0.9425 for testing set
</p>