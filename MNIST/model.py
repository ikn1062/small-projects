import tensorflow as tf
from keras import Sequential
from keras.utils import normalize
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


def make_model():
    # MNIST contains 28 x 28 images of hand-written digits from 0-9
    mnist = tf.keras.datasets.mnist

    # Load the data into training and testing data sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the training and testing set
    X_train = normalize(X_train, axis=1)
    X_test = normalize(X_test, axis=1)

    # Set up the classification model for MNSIT
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="tanh"))
    model.add(Dense(10, activation="softmax"))

    # Compiles the model with a loss function and an optimizer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Creates conditions for early stopping given epochs
    es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=2, verbose=0)
    mc = ModelCheckpoint("./mnist_model.model", monitor='val_accuracy', verbose=0, save_best_only=True)
    cb = [es, mc]

    # Creates conditions for early stopping given epochs
    model.fit(X_train, y_train, epochs=10, validation_split=0.15, callbacks=cb, verbose=1)

    # Outputs the loss and accuracy of the model
    val_loss, val_acc = model.evaluate(X_test, y_test)

