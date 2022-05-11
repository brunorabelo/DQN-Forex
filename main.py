import datetime

import numpy as np
from matplotlib import pyplot as plt

from dqn.env import Env
from dqn import train, run_model
from dqn.rewards import DailyReturnReward, SortinoRatioReward, SharpeRatioReward
from dqn.utils import get_absolute_path, paired_t_test
from tabulate import tabulate


def prepare_models():
    # Select the number of episodes Sugestion: 5000
    episodes = 2000

    # Select the reward type. Uncomment the one you want and comment the others
    # reward_type = DailyReturnReward
    # reward_type = SortinoRatioReward
    # reward_type = SharpeRatioReward

    # env = Env(window_size=30, reward_type=reward_type)
    # train.train(episodes=episodes, env=env)

    reward_type = SortinoRatioReward
    env = Env(window_size=30, reward_type=reward_type)
    train.train(episodes=episodes, env=env)

    # reward_type = SharpeRatioReward
    #
    # env = Env(window_size=30, reward_type=reward_type)
    # train.train(episodes=episodes, env=env)


def execute_models():
    reward_type = DailyReturnReward
    # reward_type = SortinoRatioReward
    # reward_type = SharpeRatioReward
    env = Env(window_size=30, reward_type=reward_type)
    weight_path = 'checkpoint_ddqn_2022_04_21-10:59:02_PM__REWARD_DailyReturn__EPISODES_2000__FINAL_DATE_2016-07-01.pth'
    pnl = run_model.run(env, weight_path)

    # reward_type = SortinoRatioReward
    # # reward_type = SharpeRatioReward
    # env = Env(window_size=30, reward_type=reward_type)
    # weight_path = 'checkpoint_ddqn_2022_04_25-02:13:23_PM__REWARD_SortinoRatio__EPISODES_600__FINAL_DATE_2016-07-01.pth'
    # sortino = run_model.run(env, weight_path)


    reward_type = DailyReturnReward
    # reward_type = SortinoRatioReward
    # reward_type = SharpeRatioReward
    env = Env(window_size=30, reward_type=reward_type)
    weight_path =  'checkpoint_ddqn_2022_04_25-02:13:23_PM__REWARD_SortinoRatio__EPISODES_400__FINAL_DATE_2016-07-01.pth'
    sortino = run_model.run(env, weight_path)


    reward_type = SharpeRatioReward
    # reward_type = SortinoRatioReward
    env = Env(window_size=30, reward_type=reward_type)
    weight_path = 'checkpoint_ddqn_2022_04_25-07:20:52_AM__REWARD_SharpeRatio__EPISODES_1000__FINAL_DATE_2016-07-01.pth'
    sharpe = run_model.run(env, weight_path)

    env = Env(window_size=30, reward_type=reward_type)
    random = run_model.random_model(env)

    column = 'balance'

    plot_fig(column, pnl, random, sharpe, sortino)

    column = 'inventory'
    plot_fig(column, pnl, random, sharpe, sortino)

    column = 'sharpe'
    plot_fig(column, pnl, random, sharpe, sortino)

    column = 'sortino'
    plot_fig(column, pnl, random, sharpe, sortino)

    print("PnL")
    extract_info_table(pnl)
    print("-----------------------------------------------------")

    print("Sortino")
    extract_info_table(sortino)
    print("-----------------------------------------------------")

    print("Sharpe")
    extract_info_table(sharpe)
    print("-----------------------------------------------------")

    print("Random")
    extract_info_table(random)
    print("-----------------------------------------------------")

    statistics(pnl, random)
    pnl.to_csv(f'{get_absolute_path("testing_results")}/'
               f'pnl.csv')
    random.to_csv(f'{get_absolute_path("testing_results")}/'
                  f'random.csv')
    sharpe.to_csv(f'{get_absolute_path("testing_results")}/'
                  f'sharpe.csv')


def statistics(df1, df2):
    df1 = df1.loc['2015-01-01':'2019-12-31']
    df2 = df2.loc['2015-01-01':'2019-12-31']
    t, pvalue = paired_t_test(df1['sharpe'], df2['sharpe'])
    print(f"P-value (sharpe): {pvalue}")
    arr = np.array(df1['balance'])
    daily_returns1 = arr[1:] / arr[:-1] - 1
    arr = np.array(df2['balance'])
    daily_returns2 = arr[1:] / arr[:-1] - 1
    t, pvalue = paired_t_test(daily_returns1, daily_returns2)
    print(f"P-value ( daily returns): {pvalue}")


def extract_info_table(df):
    backtest_row = df.loc[["20160630"]]
    balance = backtest_row['balance'].values[0]
    backtest_cumulative_return = balance / 10_000.0 - 1
    backtest_annual = (1 + backtest_cumulative_return) ** (1 / 1.5) - 1
    backtest_sharp = backtest_row['sharpe'].values[0]
    backtest_sortino = backtest_row['sortino'].values[0]
    arr = np.array(df.loc[df.index <= '20160630']['balance'])
    daily_returns = arr[1:] / arr[:-1] - 1
    backtest_volatility = np.std(daily_returns) * np.sqrt(252)

    final_test_row = df.loc[["20191231"]]
    balance = final_test_row['balance'].values[0]
    final_cumulative_return = balance / 10_000.0 - 1
    final_annual = (1 + final_cumulative_return) ** (1 / 5.0) - 1
    final_sharp = final_test_row['sharpe'].values[0]
    final_sortino = final_test_row['sortino'].values[0]
    arr = np.array(df.loc[df.index <= '20191231']['balance'])
    daily_returns = arr[1:] / arr[:-1] - 1
    final_volatility = np.std(daily_returns) * np.sqrt(252)

    outofsample_row = df.loc['2016-07-01':'2019-12-31']
    balance = outofsample_row['balance'].values[-1]
    outofsample_cumulative_return = balance / outofsample_row['balance'].values[0] - 1
    outofsample_annual = (1 + outofsample_cumulative_return) ** (1 / 1.5) - 1
    arr = np.array(outofsample_row['balance'])
    daily_returns = arr[1:] / arr[:-1] - 1
    outofsample_volatility = np.std(daily_returns) * np.sqrt(252)
    rp = np.mean(daily_returns)
    std = np.std(daily_returns)
    if std == 0:
        std = 1
    outofsample_sharp = rp / std
    rp = np.mean(daily_returns)
    std = np.std(daily_returns[daily_returns > 0])
    if std == 0:
        std = 1
    outofsample_sortino = rp / std

    res = [
        ["cumulative_return", backtest_cumulative_return, final_cumulative_return, outofsample_cumulative_return],
        ["annual_return", backtest_annual, final_annual, outofsample_annual],
        ["annual_volatility", backtest_volatility, final_volatility, outofsample_volatility],
        ["sharp", backtest_sharp, final_sharp, outofsample_sharp],
        ["sortino", backtest_sortino, final_sortino, outofsample_sortino],
    ]
    print(tabulate(res, headers=["statistics", "backtest", "total", "outofsample"], tablefmt="latex"))


def plot_fig(column, pnl, random, sharpe, sortino):
    fig, (ax1) = plt.subplots(1, 1)
    # Plot
    ax1.plot(pnl[column])
    ax1.plot(sortino[column])
    ax1.plot(sharpe[column])
    ax1.plot(random[column])
    ax1.legend(["Daily Return", "Sortino Ratio", "Sharpe Ratio", "Random"])
    ax1.axvline(x=datetime.datetime(2016, 7, 1), color='r', linestyle='--', label="End of training")
    fig.savefig(
        f'{get_absolute_path("testing_results")}/'
        f'{column}'
        f'.png')


# execute_models()
prepare_models()
