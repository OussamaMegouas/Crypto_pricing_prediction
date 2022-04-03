'''
  This is a script to make prediction of cryptocurrencies
  using yahoo finance API, we will try to use machine learning to predict
  the prices of the major cryptocurrencies in the market
'''

import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1

d2 = date.today() - timedelta(days=730)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('BTC-USD',
                        start=start_date,
                        end=end_date,
                        progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.head())

## The code above is to collect the past period of prices

data.shape

import plotly.graph_objects as go
figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"],
                                        high=data["High"],
                                        low=data["Low"],
                                        close=data["Close"])])
figure.update_layout(title="Bitcoin Price Analysis",
                     xaxis_rangeslider_visible=False)
figure.show()

## the close column in the dataset contains the value we need to predict
## so we need to look at the correlation of all the columns in the data

correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))


## The price prediction model
## the problem of predicting the future prices of cryptocurrency is based
## on the problem of Time series analysis.

from autots import AutoTS
model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple')
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
prediction = model.predict()
forcast = prediction.forecast
print(forecast)