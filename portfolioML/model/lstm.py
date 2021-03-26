"""LSTM model"""
import numpy as np
import pandas as pd
import logging
import argparse
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model, Sequential
from keras.models import load_model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from split import split_sequences, split_Tperiod, get_train_set
from portfolioML.data.data_returns import read_filepath
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of input and output data for lstm classification problem')
    parser.add_argument('returns_file', type=str, help='Path to the returns input data')
    parser.add_argument('binary_file', type=str, help='Path to the binary target data')
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
    df_returns = read_filepath(args.returns_file)
    df_binary = read_filepath(args.binary_file)

    def all_data_LSTM(df_returns, df_binary, period, len_train=981, len_test=327):

        periods_returns, periods_binary = split_Tperiod(df_returns, df_binary)
        T1_input = periods_returns[period]
        T1_target = periods_binary[period]

        X_input_train, y_input_train = T1_input[:len_train], T1_target[:len_train]
        scaler = StandardScaler()
        X_input_train = scaler.fit_transform(X_input_train)

        X_test, y_test = T1_input[len_test:], T1_target[len_test:]

        X_train, y_train = get_train_set(X_input_train, y_input_train)
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_test, y_test = get_train_set(X_test, y_test)
        X_test = scaler.fit_transform(X_test)

        X_test, y_test = np.array(X_test), np.array(y_test)

        X_train = np.reshape(X_input_train, (X_input_train.shape[0], X_input_train.shape[1], 1))

        return X_train, y_train, X_test, y_test

    def LSTM_model():
        inputs = Input(shape= (240, 1))
        hidden = LSTM(25)(inputs)
        drop = Dropout(0.1)(hidden)
        outputs = Dense(2, activation='softmax')(drop)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    y_pred = []
    N = 2
    for i in range(N):
        model = LSTM_model()
        print(model.summary())

        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i)

        history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=1)

        model.save(f"LSTM_{i}_period")

        y_pred.append(model.predict(X_test))
        plt.figure()
        plt.plot(history.history['loss'], label='loss') #funzione a gomito dalla quale si deve scegliere il valore ottimale delle epoche
        plt.plot(history.history['val_loss'], label='val loss')
        plt.grid(True)
        plt.title(f'Period {i}')
        plt.show()

    # for i in range(0,3):
    #     model = load_model(f"LSTM_{i}_period")
    #     X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i+1)
    #     history = model.fit(X_train, y_train, epochs=1, batch_size=128, validation_split=0.2, verbose=1)
    #     plt.figure()
    #     plt.plot(history.history['loss'], label='loss')
    #     plt.plot(history.history['val_loss'], label='val_loss')
    #     plt.xlabel('Epochs')
    #     plt.legend()
    #     plt.title(f'Period {i}')
    #     plt.grid(True)
    #     # #Prediction
    #     # y_pred = model.predict(X_test)
    #     # for i,j in zip(y_test,y_pred):
    #     #     logging.info(i,j)
    #     model.save(f"LSTM_{i+1}_period.h5")


    # #Loss and Val-loss
    # print('loss sul traing in funzione delle epoche')
    # print(history.history['loss'])
    # print('\t')
    # print('loss sul validation in funzione delle epoche')
    # print(history.history['val_loss'])
    # print('\t')

    # plt.plot(history.history['loss']) #funzione a gomito dalla quale si deve scegliere il valore ottimale delle epoche
    # plt.plot(history.history['val_loss'])
    # plt.show()



