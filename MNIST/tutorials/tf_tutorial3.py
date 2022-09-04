import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
    xtrain = xtrain.astype("float32") / 255.0
    xtest = xtest.astype("float32") / 255.0
    #####################
    ### DATA AUGMENTATION
    #####################

    def augment(image, label):
        new_height = new_width = 32
        image = tf.image.resize(image, (new_height, new_width))

        if tf.random.uniform((), minval=0, maxval=1) < 0.1:
            image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.1, upper=0.2)

        image = tf.image.random_flip_left_right(image) # 50% flip left and right
        # image = tf.image.random_flip_up_down(image)

        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE.cache())
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(AUTOTUNE)

    # NEW TF VERSION
    data_augmentation = keras.Sequential()(
        [
            layers.experimental.preprocessing.Resizing(height=32, width=32),
            layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
            layers.experimental.preprocessing.RandomContrast(factpr=0.1),
        ]
    )

    model = keras.Sequential()
    model.add(data_augmentation)
    model.add(layers.Input(shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, 3, padding='valid', activation='relu'))  # padding is a hyperparameter
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, 3, activation='relu'))  # padding is a hyperparameter
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, 3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model = my_model()
    print(model.summary())

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(learning_rate=3e-4), metrics=['accuracy'])

    model.fit(xtrain, ytrain, batch_size=64, epochs=150, verbose=2)
    model.evaluate(xtest, ytest, batch_size=64, verbose=2)


def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def my_model_better():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    main()


