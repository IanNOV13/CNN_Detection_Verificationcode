from keras.callbacks import Callback,LearningRateScheduler
import numpy as np
import tensorflow as tf
from keras import backend as K

class DelayedEarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=100, start_epoch=50):
        super(DelayedEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.start_epoch = start_epoch
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float('inf') if 'acc' not in self.monitor else -float('inf')
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if epoch > self.start_epoch:
            if (self.monitor == 'val_loss' and current < self.best) or (self.monitor != 'val_loss' and current > self.best):
                self.best = current
                self.wait = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f'\nEarly stopping at epoch {self.stopped_epoch + 1} restoring model weights from the end of the best epoch.')

def weighted_categorical_crossentropy(weights):
    """
    创建一个加权的交叉熵损失函数。
    ---
    Create a weighted categorical cross-entropy loss function.
    
    参数:
    weights：类别的权重。
    ---
    Parameters:
    weights: Weights for each class.
    """
    weights = K.variable(weights)
    
    def loss(y_true, y_pred):
        """
        计算加权的交叉熵损失。
        ---
        Compute the weighted cross-entropy loss.
        
        参数:
        y_true：真实标签。
        y_pred：预测值。
        ---
        Parameters:
        y_true: True labels.
        y_pred: Predicted values.
        """
        y_true = K.cast(y_true, y_pred.dtype)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_pred_logits = K.log(y_pred / (1 - y_pred))  # Convert probabilities to logits
        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred_logits, weights)
        return K.mean(loss)
    
    return loss

def cosine_decay_schedule(epoch, lr):
    initial_lr = 1e-3
    drop_rate = 0.25
    epochs_drop = 20.0
    alpha = 0.0
    cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch % epochs_drop) / epochs_drop))
    decayed = (1 - drop_rate) * cosine_decay + drop_rate
    new_lr = initial_lr * decayed
    return max(new_lr, alpha)
