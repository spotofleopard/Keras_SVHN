from __future__ import print_function
import os,datetime,random,keras,PIL
import tensorflow as tf,numpy as np
from data_loader import *
from keras.models import load_model

(x_test, y_len_test, y_digits_test) = load_svhn_tfrecords('c:/dataset/SVHN/test.tfrecords')

model=load_model('c:/saved_models/SVHN/20180311121918.hdf5')
#randomly pick 100 samples
sample_indice=random.sample(range(x_test.shape[0]),100)
x_test=x_test[sample_indice]
y_digits_test=y_digits_test[:,sample_indice]
predictions=model.predict((x_test/128)-1)
output_dir='c:/tmp/svhn_test'
if not os.path.exists(output_dir): os.makedirs(output_dir)
html_str='<html><body><table border=\"1\"><tr><th>Image</th><th>Groud Truth</th><th>Prediction</th></tr>'
digits_labels=np.array(['0','1','2','3','4','5','6','7','8','9',''])
digits_predictions=np.array(predictions[1:])
digits_gt=np.swapaxes(y_digits_test,0,1)
max_idx=np.argmax(digits_predictions,axis=-1).swapaxes(0,1)
for i in range(max_idx.shape[0]):
    digits=''.join(digits_labels[max_idx[i]])
    gt=''.join(digits_labels[digits_gt[i]])
    img = x_test[i].reshape((64, 64, 3))
    picture = PIL.Image.fromarray(img)
    picture.save('{}/test{:04d}.png'.format(output_dir,i))
    html_str+='<tr bgcolor={}><td><img src=\"test{:04d}.png\"></td><td>{}</td><td>{}</td></tr>'.format('#e0ffe0' if(gt==digits) else '#ffe0e0',i,gt,digits)

html_str+='</table></body></html>'
with open('{}/test_result.html'.format(output_dir),'w') as f: f.write(html_str)