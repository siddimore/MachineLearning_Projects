from __future__ import print_function # for python 2.7 users

#%matplotlib inline
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D

from keras import backend as K
K.set_image_data_format( 'channels_last' ) # WARNING : important for images and tensors dimensions ordering

def convBlock(cdim, nb, bits=3):
    L = []

    for k in range(1, bits+1):
        convname = 'conv'+str(nb)+'-'+str(k)
        L.append(Convolution2D(cdim, kernel_size=(3,3),
        padding='same',activation='relu',name=convname))
        L.append(MaxPooling2D((2,2),strides=(2,2)))

    return L

def vgg_face_blank():

    withDo = True
    if True:

        model = Sequential()
        model.add( Permute((1,2,3), input_shape=(224,224,3)) )

        for l in convBlock(64, 1, bits=2):
            model.add(l)

        for l in convBlock(128, 2, bits=2):
            model.add(l)

        for l in convBlock(256, 3, bits=3):
            model.add(l)

        for l in convBlock(512, 4, bits=3):
            model.add(l)

        for l in convBlock(512, 5, bits=3):
            model.add(l)
        # Keras 1
        model.add(Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6'))

        if withDO:
            mdl.add( Dropout(0.5) )

        model.add( Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7'))

        if withDO:
            model.add( Dropout(0.5) )
        #mdl.add( Convolution2D(2622, 1, 1, name='fc8') ) # Keras 1
        model.add( Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8') ) # Keras 2
        model.add( Flatten() )
        model.add( Activation('softmax') )

        return model

    else:
        # See following link for a version based on Keras functional API :
        # gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9
        raise ValueError('not implemented')

faceModel = vgg_face_blank()

faceModel.summary()
