import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train)
    print(y_train)

    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    # Sequential API (Convenient, not flexible) 1 input to 1 output
    model = func_api_model()
    print(model.summary())

    # If you remove sparse, 1 hot encoding needed
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)
    model.save_weights('saved-model/')  # can add: save_format='h5'


def keras_model():
    model = keras.Sequential()
    model.add(layers.Dense(512, input_shape=(28 * 28,), activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10))
    return model


def func_api_model():
    inputs = keras.Input(shape=(784,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    main()
