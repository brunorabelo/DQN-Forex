import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

import os


def get_parent_path():
    os.path.abspath(os.path.join(os.curdir, os.pardir))


def get_absolute_path(path=""):
    return os.path.abspath(os.path.join(os.curdir, path))


def get_csv():
    yf.pdr_override()
    df_full = pdr.get_data_yahoo("EURBRL=X", start="2017-01-01").reset_index()
    # df_full = pdr.get_data_yahoo("INFY", start="2018-01-01").reset_index()
    df_full.to_csv('../data/EURBRL_2017.csv', index=False)


def plot_csv(path="../data/EURBRL_2017.csv"):
    data_df = pd.read_csv(path)
    values = data_df.Close.values
    values2 = values - min(values)
    plt.plot(values2, label="Data")
    plt.legend()
    plt.savefig(f"../plots/data.png")
    # normalize
    x = np.asarray(values)
    res = (x - x.mean()) / x.std()
    print(res.mean(), res.std())
    res -= min(res)
    # res /= res[0]
    plt.plot(res, label="norm variance 1")
    plt.legend()
    x = np.asarray(values)
    x /= x[0]
    plt.plot(x, label='divided by initial value')
    plt.legend()
    # plt.savefig(f"../plots/data_norm")

    x = np.asarray(values)
    res = (x - x.mean()) / x.std()
    print(res.mean(), res.std())
    res -= min(res)
    res /= res[0]
    plt.plot(res, label="norm variance 1 divided by inital value")
    plt.legend()
    plt.savefig(f"../plots/data_norm")


print(get_absolute_path("results/"))
