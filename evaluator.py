import os,datetime,keras
import tensorflow as tf,numpy as np,keras.backend as K

class VectorLabelEvaluator(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.array(self.model.predict(self.validation_data[0])[1:6])
        
        max_idx=np.argmax(y_pred,axis=-1)
        y_pred=keras.utils.to_categorical(max_idx, y_pred.shape[-1])
        y_true = np.array(self.validation_data[2:7])
        
        correct=K.all(K.equal(y_true,y_pred),axis=(0,-1))
        correct=tf.cast(correct, tf.float32)
        acc=K.mean(correct)

        logs['val_digits_acc']=K.eval(acc)





