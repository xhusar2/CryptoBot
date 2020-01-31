from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pandas as pd
import backtrader as bt
import bitfinex
import datetime
import time
from PandasData import PandasData
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def fetch_data(start, stop, symbol, interval, tick_limit, step):
    # Create api instance
    api_v2 = bitfinex.bitfinex_v2.api_v2()
    data = []
    start = start - step
    while start < stop:
        start = start + step
        end = start + step
        res = api_v2.candles(symbol=symbol, interval=interval,
                             limit=tick_limit, start=start,
                             end=end)
        data.extend(res)
        time.sleep(1.2)
    return data


def get_data(pair, t_start, t_stop, bin_size):
    time_step = 60000000
    limit = 1000
    df = {}
    path = f'./data/{pair}_{t_start}-{t_stop}_{bin_size}.csv'
    if (os.path.exists(path)) and (os.path.isfile(path)):
        df = pd.read_csv(path, index_col=0)
    else:
        data = fetch_data(start=t_start, stop=t_stop, symbol=pair, interval=bin_size, tick_limit=limit, step=time_step)
        names = ['time', 'open', 'close', 'high', 'low', 'volume']
        df = pd.DataFrame(data, columns=names)
        df.drop_duplicates(inplace=True)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        print(df.head())
        df.to_csv(f'./data/{pair}_{t_start}-{t_stop}_{bin_size}.csv')
    return df


def main():
    t_start = datetime.datetime(2019, 11, 1, 0, 0)
    t_start = time.mktime(t_start.timetuple()) * 1000

    t_stop = datetime.datetime(2019, 12, 1, 0, 0)
    t_stop = time.mktime(t_stop.timetuple()) * 1000
    df = get_data(pair='btcusd', t_start=t_start, t_stop=t_stop, bin_size='1m')
    #print(df.head())

    # cerebro = bt.Cerebro()
    # data = PandasData(dataname=df, timeframe=1)
    # cerebro.adddata(data)
    # print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # cerebro.run()
    # print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    time_steps = 10
    tf.random.set_seed(13)

    #df.plot(subplots=True)
    #plt.show()

    df.astype('float').dtypes

    print(df.head())

    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    print(len(train), len(test))

    X_train, y_train = create_dataset(train, train.close, time_steps)
    X_test, y_test = create_dataset(test, test.close, time_steps)

    print(X_train.shape, y_train.shape)

    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=128,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(keras.layers.Dense(units=1))
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(0.001)
    )

    history = model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=16,
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )

    y_pred = model.predict(X_test)



def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

if __name__ == '__main__':
    main()
