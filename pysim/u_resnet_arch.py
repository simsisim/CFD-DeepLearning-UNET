#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[5]:


def bn_act(x, act=True):
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tf.keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tf.keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    c = tf.keras.layers.Concatenate()([u, xskip])
    return c


f = [0, 8, 16, 32, 64, 128, 256, 512]
inputs = tf.keras.layers.Input((128, 256, 1))
    
    ## Encoder
e0 = inputs
inputs_reshape = tf.keras.layers.Lambda(lambda x : x[:, :, :, 0:1])(inputs)
e0 = tf.keras.layers.Lambda(lambda x: tf.where(x == 0, tf.ones_like(x),                                                tf.zeros_like(x)), name = "inputs_inverse")(inputs_reshape)   
    #e1 = stem(e0, f[0])
e1 = residual_block(e0, f[1], strides=2)
e2 = residual_block(e1, f[2], strides=2)
e3 = residual_block(e2, f[3], strides=2)
e4 = residual_block(e3, f[4], strides=2)
e5 = residual_block(e4, f[5], strides=2)  
e6 = residual_block(e5, f[6], strides=2) #2,4,256
e7 = residual_block(e6, f[7], strides=2) #1,2,512
    ## Bridge
b0 = conv_block(e7, f[7], strides=1)#1,2,512
    #b1 = conv_block(b0, f[7], strides=1)#1,1,512
    #b0 = residual_block(e7, f[7], strides=1)
    #b1 =  residual_block(b0, f[7], strides=1)
    #b1= keras.layers.UpSampling2D((2, 2))(b0)
    ## Decoder
u1 =  tf.keras.layers.UpSampling2D((1, 1))(b0)
u11 = tf.keras.layers.Concatenate()([u1, e7])
    #u1 = upsample_concat_block(b0, e7)
d1 = residual_block(u11, f[7])
    
u2 = upsample_concat_block(d1, e6)
d2 = residual_block(u2, f[6])
    
u3 = upsample_concat_block(d2, e5)
d3 = residual_block(u3, f[5])
    
u4 = upsample_concat_block(d3, e4)
d4 = residual_block(u4, f[4])

u5 = upsample_concat_block(d4, e3)
d5 = residual_block(u5, f[3])

u6 = upsample_concat_block(d5, e2)
d6 = residual_block(u6, f[2])

u7 = upsample_concat_block(d6, e1)
d7 = residual_block(u7, f[1], strides = 1)

u8 = upsample_concat_block(d7, e0)
d8 = residual_block(u8, 2, strides = 1)
x = tf.keras.layers.Conv2D(2, (1, 1), padding="same", activation="tanh")(d8)

    #inputs_reshape = keras.layers.Lambda(lambda x : x[:, :, :, 0:1])(inputs)
    #inputs_inverse = keras.layers.Lambda(lambda x: tf.where(x == 0, tf.ones_like(x),\
    #                                            tf.zeros_like(x)), name = "inputs_inverse")(inputs_reshape)                                                
outputs = tf.keras.layers.multiply([x, e0])
model = tf.keras.models.Model(inputs, outputs)


# In[6]:


#get_ipython().system('jupyter nbconvert --to script u_resnet_arch.ipynb')

