import os,datetime,keras,tensorflow as tf
import keras.backend as K

#Modified from: https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure/48393723

class TrainValTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir=None, **kwargs):
        if(log_dir is None):
            time_now=datetime.datetime.now()
            log_dir='c:/keras_logs/'+time_now.strftime('%Y%m%d%H%M%S')
            if not os.path.exists(log_dir): os.makedirs(log_dir)
        self.training_log_dir = os.path.join(log_dir, 'train')
        self.val_log_dir = os.path.join(log_dir, 'val')
        super(TrainValTensorBoard, self).__init__( self.training_log_dir, **kwargs)  

    def set_model(self, model):
        self.model=model
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)

        self.val_writer.flush()

        #optimizer = self.model.optimizer
        #logs['learning_rate']=K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)


    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

