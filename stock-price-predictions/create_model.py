import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import datetime as dt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint


def create_model(data, prediction_days, scaler):
    # Prepare data
    scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

    X_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[x - prediction_days:x])
        y_train.append(scaled_data[x, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Creates conditions for early stopping given epochs
    es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=2, verbose=0)
    mc = ModelCheckpoint("./predict-stock.model", monitor='val_accuracy', verbose=0, save_best_only=True)
    cb = [es, mc]

    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.15, callbacks=cb, verbose=1)


def test_model(company, data, prediction_days, model, scaler, plots=True):
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()

    test_data = pdr.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Adj Close'].values

    total_dataset = pd.concat((data['Adj Close'], test_data['Adj Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    X_test = []

    for x in range(prediction_days, len(model_inputs)):
        X_test.append(model_inputs[x - prediction_days:x, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the test predictions
    if plots:
        plt.plot(actual_prices, color='black', label=f'Actual {company} Price')
        plt.plot(predicted_prices, color='green', label=f'Predicted {company} Price')
        plt.title(f'{company} Share Price')
        plt.xlabel("Time")
        plt.ylabel(f'{company} Share Price')
        plt.legend()
        plt.show()


def predict(company, data, prediction_days, model, scaler):
    test_start, test_end = dt.datetime(2020, 1, 1), dt.datetime.now()
    test_data = pdr.DataReader(company, 'yahoo', test_start, test_end)
    total_dataset = pd.concat((data['Adj Close'], test_data['Adj Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days: len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    np.reshape = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f'Prediction: {prediction}')






