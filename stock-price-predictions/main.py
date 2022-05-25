import pandas_datareader as pdr
import datetime as dt
import os.path
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

import create_model

# main:
if __name__ == "__main__":
    print("Enter stock Ticker: ")
    company = input()

    start, end = dt.datetime(2015, 1, 1), dt.datetime(2020, 1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = pdr.DataReader(company, 'yahoo', start, end)
    prediction_days = 60

    if not os.path.exists('predict-stock.model'):
        create_model.create_model(data, prediction_days, scaler)
    model = load_model("predict-stock.model")
    create_model.test_model(company, data, prediction_days, model, scaler)
    create_model.predict(company, data, prediction_days, model, scaler)




