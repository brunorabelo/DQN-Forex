import numpy as np


def daily_return(daily_balance, risk_free=0):
    arr = np.array(daily_balance)
    daily_returns = arr[1:] / arr[:-1] - 1
    return daily_returns[-1] if len(daily_returns) > 0 else 0


def sharpe_ratio(daily_balance, risk_free=0):
    arr = np.array(daily_balance)
    if len(arr) > 21:
        arr = arr[-21:]
    daily_returns = arr[1:] / arr[:-1] - 1
    rp = np.mean(daily_returns)
    std = np.std(daily_returns)
    if std == 0:
        std = 1
    ratio = (rp - risk_free) / std
    ratio = ratio if not np.isnan(ratio) and not np.isinf(ratio) else 0
    return ratio


def sortino_ratio(daily_balance, risk_free=0):
    arr = np.array(daily_balance)
    if len(arr) > 21:
        arr = arr[-21:]
    daily_returns = arr[1:] / arr[:-1] - 1
    rp = np.mean(daily_returns)
    std = np.std(daily_returns[daily_returns > 0])
    if std == 0:
        std = 1
    ratio = (rp - risk_free) / std
    ratio = ratio if not np.isnan(ratio) and not np.isinf(ratio) else 0
    return ratio


class SortinoRatioReward:
    name = "SortinoRatio"
    reward_function = sortino_ratio


class SharpeRatioReward:
    name = "SharpeRatio"
    reward_function = sharpe_ratio


class DailyReturnReward:
    name = "DailyReturn"
    reward_function = daily_return
