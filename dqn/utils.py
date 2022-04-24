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
    df_full = pdr.get_data_yahoo("EURUSD=X", start="2015-01-01", end="2020-01-31").reset_index()
    # df_full = pdr.get_data_yahoo("INFY", start="2018-01-01").reset_index()
    df_full.to_csv('../data/EURUSD_2015-2019.csv', index=False)


def plot_csv(path="../data/EURBRL_2019.csv"):
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

def plot_results(df):
    #
    print(df)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)

    # Plot

    ax1.plot(df["balance"], marker='o')

    ax2.plot(df["inventory"])

    ax3.plot(df["cash"])

    ax4.plot(df["sharpe"])

    ax5.plot(df['sortino'])

    #     ax1.xlabel("Date")
    #     ax1.ylabel("Total balance")
    #     ax1.title("Balance")
    # ax2.xlabel("Date")
    # ax2.ylabel("Inventory")
    # ax2.title("Inventory")
    # labeling
    # ax3.xlabel("Date")
    # ax3.ylabel("Cash")
    # ax3.title("Cash")
    # # labeling
    # ax4.xlabel("Date")
    # ax4.ylabel("Sharpe")
    # ax4.title("Sharpe")
    # # labeling
    # ax5.xlabel("Date")
    # ax5.ylabel("Sortino")
    # ax5.title("Sortino")


    fig.savefig(
        f'{get_absolute_path("testing_results")}/'
        f'ddqn_{model_name}'
        f'.png')

    print(get_absolute_path("results/"))
# get_csv()
# plot_csv()
