import os
import matplotlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy


def mnist():
    (ds_train, ds_test), ds_info = tfds.load('mnist', split=["train", "test"], shuffle_files=True, as_supervised=True,
                                             with_info=True)
    print(ds_info)
    # fig = tfds.show_examples(ds_train, ds_info, rows=4, cols=4)  # needs to have as_supervised as false

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 64
    ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(1000)
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(AUTOTUNE)

    ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.prefetch(AUTOTUNE)

    model = keras_model()

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])

    model.fit(ds_train, epochs=5, verbose=2)
    model.evaluate(ds_test, batch_size=32, verbose=2)


def keras_model():
    model = keras.Sequential()
    model.add(layers.Dense(512, input_shape=(28 * 28,), activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10))
    return model


def txt_data():
    (ds_train, ds_test), ds_info = tfds.load("imdb_reviews", split=["train", "test"], shuffle_files=True, as_supervised=True,
                                             with_info=True)
    print(ds_info)

    # tokenise words
    tokenizer = tfds.features.text.Tokenizer()

    def build_vocab():
        vocab = set()
        for text, _ in ds_train:
            vocab.update(tokenizer.tokenize(text.numpy().lower()))

    vocab = build_vocab()
    encoder = tfds.features.text.TokenTextEncoder(
        vocab, oov_token="<UNK>", lowercase=True, tokenizer=tokenizer
    )

    def my_encoding(text_tensor, label):
        return encoder.encode(text_tensor.numpy()), label

    def encode_map(text, label):
        encoded_text, label = tf.py_function(my_encoding, inp=[text, label], Tout=(tf.int64, tf.int64))
        encoded_text.set.shape([None])
        label.set_shape([])
        return encoded_text, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_train = ds_train.map(encode_map, num_parallel_calls=AUTOTUNE.cache())
    ds_train = ds_train.shuffle()
    ds_train = ds_train.padded_batch(32, padded_shapes=([None], ()))
    ds_train = ds_train.prefetch(AUTOTUNE)

    ds_test = ds_test.map(encode_map)
    ds_test = ds_test.padded_batch(32, padded_shapes=([None], ()))

    model = keras.Sequential([
        layers.Masking(mask_value=0),
        layers.Embedding(input_dim=len(vocab)+2, output_dim=32),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1),
    ])

    model.compile(loss=BinaryCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1),
                  metrics=["accuracy"])


if __name__ == "__main__":
    # mnist()
    txt_data()
