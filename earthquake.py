#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
import numpy as np
import json
import h5py
import csv
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Input, Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D



def load_data(path_to_file):
    V = []
    with open(path_to_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if '[' in row[0]:
                row[0] = row[0].replace('[', '')
            if ']|' in row[-1]:
                row[-1] = row[-1].split(']|')
            row = row[:-1] + row[-1]
            V.append(row)
    return np.array(V, dtype='float16')



def NET():
    weights = 'weights.hdf5'
    input = Input(shape=(512,))
    c = Reshape((512, 1))(input)
    c = Conv1D(16, 16, activation='relu', border_mode='same')(c)
    c = MaxPooling1D(pool_length=512/2)(c)
    fc = Flatten()(c)
    # fc = Dropout(0.5)(fc)
    output = Dense(1, activation='sigmoid')(fc)

    model = Model(inputs=input, output=output)
    model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
    ############################################################################################
    # # if previous file exist:
    # if os.path.isfile(weights):
    #     print 'loading weights file: ' + weights
    #     model.load_weights(weights)
    # else:
    #     raise RuntimeError(
    #         'VatBot Error: weights file (' + weights + ') not found...')
    ############################################################################################
    model.summary()
    return model


def binary_samples_weight(y):
    # sample weight, for uneven dataset:
    sw = 1. * np.count_nonzero(y) / len(y)
    SW = y.copy()
    SW[SW == 0] = sw
    SW[SW == 1] = 1 - sw
    return SW


def validation_split(x, y, validation=0.5):
    val_qtt = int(validation * len(y))
    train_qtt = len(y) - val_qtt
    sw = 1. * np.count_nonzero(y) / len(y)
    val_qtt0 = int(val_qtt * (1 - sw))
    # train_qtt0 = int(train_qtt * (1 - sw))
    val_qtt1 = val_qtt - val_qtt0
    # train_qtt1 = train_qtt - train_qtt0

    where_zeros = np.where(y == 0)[0]
    x0 = x[where_zeros]
    y0 = y[where_zeros]
    where_ones = np.where(y == 1)[0]
    x1 = x[where_ones]
    y1 = y[where_ones]

    val_x0 = x0[val_qtt0:]
    val_y0 = y0[val_qtt0:]
    train_x0 = x0[:val_qtt0]
    train_y0 = y0[:val_qtt0]

    val_x1 = x1[val_qtt1:]
    val_y1 = y1[val_qtt1:]
    train_x1 = x1[:val_qtt1]
    train_y1 = y1[:val_qtt1]

    train_x = np.vstack([train_x0, train_x1])
    train_y = np.hstack([train_y0, train_y1])

    val_x = np.vstack([val_x0, val_x1])
    val_y = np.hstack([val_y0, val_y1])

    print train_x.shape, train_y.shape, val_x.shape, val_y.shape
    return [train_x, train_y], [val_x, val_y]


def train():
    historyJSONname = 'history.json'
    weights = 'weights.hdf5'
    path_to_train_file = 'newtrain.csv'
    path_to_test_file = 'newtest.csv'
    size_batch = 8
    epochs = 1000
    train_data = load_data(path_to_train_file)
    x = (train_data[:, :-1] + 4) / 8
    y = train_data[:, -1]
    train_data = [x, y]
    val_data = load_data(path_to_test_file)
    x_ = (val_data[:, :-1] + 4) / 8
    y_ = val_data[:, -1]
    val_data = [x_, y_]
    # train_data, val_data = validation_split(x, y)
    print x.shape, y.shape, x_.shape, y_.shape
    SW = binary_samples_weight(train_data[1])

    model = NET()
    checkpointer = ModelCheckpoint(weights,
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='max')

    history = model.fit(train_data[0], train_data[1],
                        batch_size=size_batch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[checkpointer],
                        validation_data=val_data,
                        shuffle=True,
                        sample_weight=SW)

    # model.save_weights(weights, overwrite=True)
    h = history.__dict__
    H = {'params': h['params'], 'history': h['history'], 'epoch': h['epoch']}
    with open(historyJSONname, 'wb') as fp:
        json.dump(H, fp)

if __name__ == '__main__':
    train()

























































