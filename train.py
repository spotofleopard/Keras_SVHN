from __future__ import print_function
import os,datetime,keras
import keras.backend as K
import tensorflow as tf,numpy as np
from keras.models import Sequential,Model
from keras.layers import Input,Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D,concatenate,LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.applications import imagenet_utils
from keras.utils import plot_model
from data_loader import *
from logger import TrainValTensorBoard
from evaluator import *

def build_model():
    feature_map_size=[48,64,128,160,192,192,192,192]
    data_in = Input(shape=input_shape)
    layer_input=data_in
    for i in range(len(feature_map_size)):
        conv_out=Conv2D(feature_map_size[i],5,activation='relu',padding='same')(layer_input)
        if(((i+1)%2)==0): 
            conv_out=MaxPooling2D()(conv_out)
        conv_out=Dropout(0.3)(conv_out)
        layer_input=conv_out
    #output size: (*,4,4,192)

    encoder_inputs=Reshape((16,192))(layer_input)
    encoder_outputs = LSTM(1024)(encoder_inputs)

    digits=[Dense(num_digit_classes,name='D{}'.format(i), activation='softmax')(encoder_outputs) for i in range(max_num_digits)]
    
    model = Model(inputs=data_in, outputs=[*digits])
    print(model.summary())
    return model

batch_size = 128
num_digit_classes = 11
max_num_digits=5
epochs = 50

# input image dimensions
input_shape = (64, 64, 3)

(x_train, y_len_train, y_digits_train)=load_svhn_tfrecords('c:/dataset/SVHN/train.tfrecords')
(x_test, y_len_test, y_digits_test) = load_svhn_tfrecords('c:/dataset/SVHN/test.tfrecords')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_digits_train=[keras.utils.to_categorical(y, num_digit_classes) for y in y_digits_train]
y_digits_test=[keras.utils.to_categorical(y, num_digit_classes) for y in y_digits_test]

model=build_model()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='AdaDelta',
              metrics=['accuracy'])

time_now=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
plot_model(model, to_file='c:/saved_models/SVHN/model_{}.png'.format(time_now))
filepath="c:/saved_models/SVHN/{}.hdf5".format(time_now)
checkpoint = ModelCheckpoint(filepath, monitor='val_digits_acc', verbose=1, save_best_only=True, mode='max')

model.fit(x_train, [*y_digits_train],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_data=(x_test, [*y_digits_test]),
          callbacks=[VectorLabelEvaluator(),TrainValTensorBoard(write_graph=False),checkpoint])
