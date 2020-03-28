import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from module import JujubeCakeCell, JujubeCake, MaskReshape


def test_on_reuters():
    max_len = 500
    embedding_dim = 128
    num_words = 10000
    sub_lstms = 5
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words)
    # word2idx = reuters.get_word_index()
    # idx2word = {idx:word for word,idx in word2idx.items()}

    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)
    # print(y_train.max())
    # print(y_train.min())

    inputs = keras.Input((max_len,))
    x = layers.Embedding(num_words+1,
                         embedding_dim,
                         mask_zero=True,
                         input_length=max_len)(inputs)
    x = MaskReshape(
        (int(max_len / sub_lstms), int(embedding_dim * sub_lstms)), factor=sub_lstms)(x)
    x = JujubeCake(16, sub_lstms)(x)
    x = layers.Dense(46, activation='softmax')(x)
    model = keras.Model(inputs, x)
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train,
              epochs=10,
              batch_size=128,
              validation_split=0.1)
    res = model.evaluate(x_test, y_test, batch_size=128)
    print('\n')
    print('#' + '#' * 70 + '#')
    print('#' + ('Evaluation result').center(70, ' ') + '#')
    print('#' + '#' * 70 + '#')
    print('\n')
    for metrics, value in zip(model.metrics_names, res):
        print('%s %.4f' % (metrics, value))

    inputs = keras.Input((max_len,))
    x = layers.Embedding(num_words+1,
                         embedding_dim,
                         mask_zero=True,
                         input_length=max_len)(inputs)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dense(46, activation='softmax')(x)
    model = keras.Model(inputs, x)
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train,
              epochs=10,
              batch_size=128,
              validation_split=0.1)
    res = model.evaluate(x_test, y_test, batch_size=128)
    print('\n')
    print('#' + '#' * 70 + '#')
    print('#' + ('Evaluation result').center(70, ' ') + '#')
    print('#' + '#' * 70 + '#')
    print('\n')
    for metrics, value in zip(model.metrics_names, res):
        print('%s %.4f' % (metrics, value))


def test():
    x = keras.Input((100, 128), name='inputs')
    y = JujubeCake(128, 2, name='jujube')(x)
    model = keras.Model(x, y)
    model.summary()


if __name__ == '__main__':
    # tf.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    test_on_reuters()
