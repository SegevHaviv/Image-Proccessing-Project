
# coding: utf-8

# In[1]:


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


# In[2]:


# Get images
X = []
for filename in os.listdir('C:/Users/carmel/Desktop/train/'):
    curr= (load_img('C:/Users/carmel/Desktop/train/'+filename))
    #curr=curr.resize((256,256))
    #imsave("C:/Users/carmel/Desktop/result/"+'1'+".jpg", curr)
    curr=img_to_array(curr)
    X.append(curr)
X = np.array(X, dtype=float)
Xtrain = 1.0/255*X



# In[3]:


#Load weights
inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('C:/Users/carmel/Desktop/weights/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
inception.graph = tf.get_default_graph()


# In[4]:


embed_input = Input(shape=(1000,))

#Encoder
encoder_input = Input(shape=(96, 96, 1,))
encoder_output = Conv2D(8, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(96, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(96, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(192, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(192, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(96, (3,3), activation='relu', padding='same')(encoder_output)

#Fusion
fusion_output = RepeatVector(12 * 12)(embed_input) 
fusion_output = Reshape(([12, 12, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
fusion_output = Conv2D(96, (1, 1), activation='relu', padding='same')(fusion_output) 

#Decoder
decoder_output = Conv2D(48, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(24, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(12, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(6, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
#plot_model(model, to_file='model.png')


# In[5]:


#Create embedding
def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

#Generate training data
batch_size = 20

def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)
        
#Train model      
#tensorboard = TensorBoard(log_dir="/output")
#Train model      
model.compile(optimizer='adam', loss='mse')
model.fit_generator(image_a_b_gen(batch_size), epochs=150, steps_per_epoch=21,shuffle=True)




# In[8]:


color_me = []
for filename in os.listdir('C:/Users/carmel/Desktop/test/'):
    curr= (load_img('C:/Users/carmel/Desktop/test/'+filename))
    #curr=curr.resize((256,256))
    #imsave("C:/Users/carmel/Desktop/result/"+'1'+".jpg", curr)
    curr=img_to_array(curr)
    color_me.append(curr)

    #color_me.append(img_to_array(load_img('C:/Users/carmel/Desktop/test/'+filename)))
color_me = np.array(color_me, dtype=float)
color_me = 1.0/255*color_me
color_me = gray2rgb(rgb2gray(color_me))
color_me_embed = create_inception_embedding(color_me)
color_me = rgb2lab(color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))


# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((96, 96,3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("C:/Users/carmel/Desktop/result/"+str(i)+".jpg", lab2rgb(cur))


# In[ ]:




