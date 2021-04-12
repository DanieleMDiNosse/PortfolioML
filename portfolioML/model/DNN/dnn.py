"""DNN model"""
import numpy as np
import pandas as pd
import logging
import argparse
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from model.split import split_Tperiod, get_train_set
from data.data_returns import read_filepath
from makedir import smart_makedir, go_up
from model.LSTM.lstm import all_data_LSTM

import matplotlib.pyplot as plt

def all_data_DNN(df_returns, df_binary, period, len_train=981, len_test=327):
    """
    Create a right input data for DNN starting from the right data of LSTM.
    Indeed the parameters are the same of the all_data_LSTM, these are changed select
    only m values (features) exctrated from the 240 values in the LSTM input data.
    """
    X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, period)

    m = list(range(0,240,20))+list(range(221,240))
    X_train = X_train[:,m,:]
    X_train = np.reshape(X_train, (X_train.shape[0], 31))

    X_test = X_test[:,m,:]
    X_test = np.reshape(X_test, (X_test.shape[0], 31))

    return X_train, y_train, X_test, y_test

def DNN_model(nodes_args, hidden=None , activation='tanh', loss='binary_crossentropy', optimizer='adam'):
    """
    DNN model with selected number of hidden layer for classification task.
    For more details about the model see the reference
    The model is maded by:

    - Input: shape = (feature), features are the numer of values taken from the past,
    follow the leterature the default is 31.

    - Hidden Layers: Dense(feature, activation=activation), sequential hidden layers full-connected 
    with different nodes. If hiddin is an integer the number of nodes for each layer follow a descrescent
    way from 31 to 5, note that the actual values of the nodes is determine by np.linspace(feature,5,hidden).
    
    - Output: Dense(1, activation='sigmoid'), the output is interpretated as the probability that 
    the input is grater than the cross-section median.

    Note that a suitable Dropout layers fill between the layers described above. Parameters of this 
    layers has been choosen following a "try and error" way to minimaze the shape of loss fuction
    (future version of this code will have the possibility to set this parameters).

    Reference: "doi:10.1016/j.ejor.2016.10.031"

    Parameters
    ----------

    nodes_args: list of integer
        Number of nodes for each layers.    

    hidden: integer(optional), default = None
        Number of hidden layers, the actual values of the nodes are fixed in descrescent way 
        from 3 to 5 through the np.linspace function (np.linspace(31,5,hidden)). 
        Follow some useful example:
        - 3: [31,18,5]
        - 4: [31,22,13,5]
        - 5: [31,24,18,11,5]
        - 6: [31,25,20,15,10,5]

    activation: string(optional)
        Activation function to use of hidden layers, default='tanh'
        Reference: https://keras.io/api/layers/core_layers/dense/

    loss: String (name of objective function), objective function or tf.keras.losses.Loss instance. See tf.keras.losses.
        Loss fuction, it must be a loss compatible with classification problem, defaul=binary_crossentropy'
        Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model

    optimater: string(optional)
        String (name of optimizer) or optimizer instance. See tf.keras.optimizers., default='adam'
        Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model


    Returns
    -------
    model: tensorflow.python.keras.engine.sequential.Sequential
        tensorflow model with selected hidden layers

    """
    model = Sequential()

    model.add(Input(shape=(31)))
    model.add(Dropout(0.1))

    if hidden is not None:
        logging.info("Nember of layers is determined by 'hidden',numebrs of neurons descrescent from 31 to 5")
        nodes = [int(i) for i in np.linspace(31,5,hidden)]
    else:
        nodes = nodes_args

    for nod in nodes:
        model.add(Dense(nod, activation=activation))
        model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    logging.info(model.summary()) 
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make DNN for classification task to predicti class label 0 or 1')
    parser.add_argument("nodes", type=int, nargs='+', help='Numerb of nodes in esch layers of DNN')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])

    #Read the data
    df_binary = go_up(2) + "/data/ReturnsBinary.csv"
    df_returns = go_up(2) + "/data/ReturnsData.csv"
    df_returns = read_filepath(df_returns)
    df_binary = read_filepath(df_binary)

    for per in range(5,10):
        model = DNN_model(args.nodes, optimizer='adam')
        #Splitting data for each period
        X_train, y_train, X_test, y_test = all_data_DNN(df_returns, df_binary, per)
        #Trainng
        es = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
        mc = ModelCheckpoint(f'DNN_test_period{per}.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train ,y_train, callbacks=[es,mc],validation_split=0.2, batch_size=256, epochs=400, verbose=1)

        #Elbow curve
        plt.figure(f'Loss and Accuracy period {per}')
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train_loss') 
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Loss vs Epochs')
        plt.grid()
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Accuracy vs Epochs')
        plt.grid()
        plt.legend()

    plt.show()

    
    plt.show()

    
