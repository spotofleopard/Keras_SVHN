import numpy as np,tensorflow as tf
import os,sys
from PIL import Image

def load_svhn_tfrecords(file_name):
    record_iterator = tf.python_io.tf_record_iterator(path=file_name)
    images=[]
    lengths=[]
    digit_vectors=[]
    i=0
    print('loading {}'.format(file_name))
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        img_string = example.features.feature['image'].bytes_list.value[0]
        length = example.features.feature['length'].int64_list.value[0]
        digits = example.features.feature['digits'].int64_list.value
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        img = img_1d.reshape((64, 64, 3))
        images.append(img)   
        lengths.append(length)
        digit_vectors.append(digits)
        i+=1
        if((i%1000)==0):
            print('{: 4d}k'.format(i//1000), end='')
            print('\r', end='')
            
    images=np.array(images)
    lengths=np.array(lengths)
    digit_vectors=np.array(digit_vectors).swapaxes(0,1) 
    print('\nDone')
    return images,lengths,digit_vectors
