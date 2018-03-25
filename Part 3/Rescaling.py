
# coding: utf-8

# In[53]:


from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, Flatten, Reshape, Dropout, Conv2DTranspose
from keras.callbacks import ModelCheckpoint
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread, concatenate_images
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import matplotlib.pyplot as plt



# In[60]:


# function for importing images from a folder
def load_images(path, input_size, output_size):
    x_ = []
    y_ = []
    counter, totalnumber = 1, len(os.listdir(path))
    for imgpath in os.listdir(path):
        if counter % 100 == 0:
            print("Importing image %s of %s (%s%%)" %(counter, totalnumber, round(counter/totalnumber*100)))
        y = imread(path + "/" + imgpath)
        y = rgb2gray(resize(y, output_size, mode="constant"))
        x = resize(y, input_size, mode="constant")
        x_.append(x)
        y_.append(y)
        counter += 1
    return concatenate_images(x_), concatenate_images(y_)


# In[61]:


# defining input and output size
input_size = (32, 32)
output_size = (96, 96)

# loading and reshaping train set
x_train, y_train = load_images("C:/Users/carmel/Desktop/biger/", input_size, output_size)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
print(x_train.shape, y_train.shape)

# loading and reshaping validation set
x_test, y_test = load_images('C:/Users/carmel/Desktop/validy/', input_size, output_size)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)
print(x_test.shape, y_test.shape)

# saving the data in arrays
print("Creating a compressed dataset...")
np.savez_compressed("image",
                    x_train = x_train,
                    y_train = y_train,
                    x_test = x_test,
                    y_test = y_test)


# In[62]:


# loading previously created data
input_dim = x_test.shape[1:]
output_dim =y_test.shape[1:]


# In[67]:


# Building the model
model = Sequential()
# convolution layers
model.add(Conv2D(1, (1, 1), data_format="channels_last", input_shape=input_dim))
model.add(Conv2D(2, (3, 3)))
model.add(Conv2D(3, (4, 4)))
model.add(Dropout(.05))

# Transpose convolution layers (Deconvolution)
model.add(Conv2DTranspose(3, (3, 3)))
model.add(Conv2DTranspose(2, (5, 5)))
model.add(Conv2DTranspose(1, (8, 8)))
model.add(Dropout(.1))

# Fully connected layers
model.add(Flatten())
model.add(Dense(np.prod(output_dim)))
model.add(Reshape(output_dim)) # scaling to the output dimension
model.add(Activation("linear")) # using a "soft" activation

model.compile(optimizer = "adam", loss = "mse")
print(model.summary())

# fitting the model
print("Fitting the model...")
checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(x_train,y_train,
          batch_size=5,
          epochs=100,
          validation_data=(x_test,y_test),
          callbacks = callbacks_list)


# In[68]:


# making a prediction on test dataset
y_pred = model.predict(x_test)

# plotting some random images of the dataset
for i in np.random.random_integers(0, 1, size=10):

    # input image
    plt.subplot(221)
    plt.imshow(x_test[i,:,:,0], cmap="Greys_r")
    plt.title("Low res. input image")
    plt.axis("off")

    # target image
    plt.subplot(222)
    plt.imshow(y_test[i,:,:,0], cmap="Greys_r")
    plt.title("High res. target image")
    plt.axis("off")

    # rescaled image
    plt.subplot(223)
    resized = resize(x_test[i,:,:,0], (96, 96), mode="constant")
    plt.imshow(resized, cmap="Greys_r")
    plt.title("Rescaled high res. image")
    plt.axis("off")

    # model2 predicted image
    plt.subplot(224)
    plt.imshow(y_pred[i,:,:,0], cmap="Greys_r")
    plt.title("Model high res. reconstruction")
    plt.axis("off")

    plt.show()

