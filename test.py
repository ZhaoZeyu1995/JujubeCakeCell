from MyJujubeCakeCell import JujubeCakeCell, JujubeCake
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks

if __name__ == '__main__':
    max_len = 500
    num_words = 10000
    embedding_dim = 32
    sub_lstms = 5
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    inputs = Input((max_len,))
    x = Embedding(num_words, embedding_dim)(inputs)
    x = Reshape((int(max_len / sub_lstms), int(embedding_dim * sub_lstms)))(x)
    x = JujubeCake(16, sub_lstms)(x)
    x = Dense(46, activation='softmax')(x)
    model = Model(inputs, x)
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128,
              validation_split=0.2)
