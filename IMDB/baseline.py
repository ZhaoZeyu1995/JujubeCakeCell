import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Model

if __name__ == '__main__':
    max_len = 500
    num_words = 10000

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    print(x_train.shape)
    print(x_test.shape)

    inputs = Input((max_len,))
    x = Embedding(num_words, 32)(inputs)
    x = CuDNNLSTM(32)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, x)
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train,
              epochs=10,
              batch_size=128,
              validation_split=0.2)
