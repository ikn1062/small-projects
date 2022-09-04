from tensorflow import keras
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Print Data size
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # Visualize examples
    num_classes = 10
    f, ax = plt.subplots(1, num_classes, figsize=(40, 40))
    for i in range(0, num_classes):
        sample = x_train[y_train == i][0]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title(f"Label: {i}", fontsize=22)
    plt.show()

    # Categorical Labels (10x10 matrix)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Prepare Data -> everything from 0 to 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape Data -> all row vectors
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Fully Connected NN
    model = Sequential()
    model.add(Dense(units=128, input_shape=(784,), activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    batch_size = 512
    epochs = 10
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}, Test accuracy: {test_accuracy}")

    y_pred = model.predict(x_test)
    y_pred_clases = np.argmax(y_pred, axis=1)
    # print(y_pred)  # Probability
    # print(y_pred_clases)  # Classification
    y_true = np.argmax(y_test, axis=1)

    # Plotting confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred_clases)
    f, ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(confusion_mat, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.show()

    # Investigate errors
    errors = (y_pred_clases - y_true != 0)
    y_pred_classes_errors = y_pred_clases[errors]
    y_pred_errors = y_pred[errors]
    y_true_errors = y_true[errors]
    x_test_errors = x_test[errors]

    y_pred_errors_prob = np.max(y_pred_errors, axis=1)
    true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))
    diff_errors_pred_true = y_pred_errors_prob - true_prob_errors

    # Get list of sorted differenced
    sorted_idx_diff_errors = np.argsort(diff_errors_pred_true)
    top_idx_diff_errors = sorted_idx_diff_errors[-5:]

    # Show top errors
    num = len(top_idx_diff_errors)
    f, ax = plt.subplots(1, num, figsize=(30,30))
    for i in range(0, num):
        idx = top_idx_diff_errors[i]
        sample = x_test_errors[idx].reshape(28,28)
        y_t = y_true_errors[idx]
        y_p = y_pred_classes_errors[idx]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title(f"predicted label: {y_p}, true label: {y_t}", fontsize=22)
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    main()
