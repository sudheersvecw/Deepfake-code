import tensorflow as tf
from tensorflow import keras

class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1', threshold=0.5):
        super().__init__(name=name)
        self.threshold = threshold
        self.tp = self.add_weight('tp', initializer='zeros')
        self.fp = self.add_weight('fp', initializer='zeros')
        self.fn = self.add_weight('fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true*y_pred))
        self.fp.assign_add(tf.reduce_sum((1-y_true)*y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true*(1-y_pred)))

    def result(self):
        p = self.tp/(self.tp+self.fp+1e-7)
        r = self.tp/(self.tp+self.fn+1e-7)
        return 2*p*r/(p+r+1e-7)

    def reset_states(self):
        for v in self.variables:
            v.assign(0.0)
