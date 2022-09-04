import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def my_pretrained():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    model = keras.models.load_model("complete_saved_model/")
    # model.trainable = False
    # for layer in model.layers[:-1]:
    #       assert layer.trainable == False
    #       layer.trainable = False

    base_inputs = model.layers[0].input
    base_outputs = model.layers[-1].input
    final_output = layers.Dense(10)(base_outputs)

    new_model = keras.Model(inputs=base_inputs, outputs=final_output)
    print(model.summary())

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)
    # base_outputs - model.layers[7]


def keras_pretrained():
    x = tf.random.normal(shape=(5, 299, 299, 3))
    y = tf.constant([0, 1, 2, 3, 4])

    model = keras.applications.InceptionV3(include_top=True)
    print(model.summary())

    base_input = model.layers[0].input
    base_outputs = model.layers[-2].output
    final_outputs = layers.Dense(5)(base_outputs)
    new_model = keras.Model(inputs=base_input, outputs=final_outputs)

    new_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    new_model.fit(x, y, epochs=15)


if __name__ == "__main__":
    # my_pretrained()
    keras_pretrained()
