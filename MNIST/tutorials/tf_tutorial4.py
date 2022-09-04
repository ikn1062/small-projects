import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # not meant for images
    model = keras.Sequential()
    model.add(keras.Input(shape=(None, 28)))
    model.add(layers.SimpleRNN(256, return_sequences=True, activation='tanh'))  # default activation is tanh
    model.add(layers.SimpleRNN(256, activation='tanh'))
    # to use GRU, just change SimpleRNN -> GRU
    # to use LSTM, just change SimpleRNN -> LSTM
    # to use bidirectional LSTM -> layers.Bidirectional(layers.LSTM(256, [return_sequences=True], activation='tanh'))
    model.add(layers.Dense(10))

    print(model.summary())

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
    model.evaluate(x_test, y_test, batch_size=64, verbose=2)


if __name__ == "__main__":
    main()
