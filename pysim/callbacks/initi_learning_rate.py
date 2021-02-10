import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


K = keras.backend
#from tensorflow.keras.callbacks import Callback

class showLR(tf.keras.callbacks.Callback) :
    def on_batch_begin(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        #print (" epoch={:02d}, lr={:.5f}".format( epoch, lr ))
        return lr

class ExponentialLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)
        #print("factor", self.factor)

def learning_rate_finder(model, train_dataset, epochs=1, batch_size=12, min_rate=10**-2, max_rate=10):
    init_weights = model.get_weights()
    stepsPerEpoch = 64
    #numBatchUpdates = epochs * stepsPerEpoch
    iterations =  3000 /(stepsPerEpoch)
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    print(factor)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(train_dataset, epochs=epochs, steps_per_epoch = 64, verbose = 0, 
                        callbacks=[exp_lr, showLR()])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), max(losses)])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    #plt.text(0.05, 700, "use turning_point/10 to select initial learning rate")
    plt.show()
