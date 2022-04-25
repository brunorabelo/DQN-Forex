from datetime import datetime
import random
import numpy as np
import pandas as pd
from .agent import Agent as DDQNAgent
import matplotlib.pyplot as plt
import dqn.rewards as rewards
# episodes = 5_000  # Number of episodes used for training
from .utils import get_absolute_path

INITIAL_MONEY = 10_000
BATCH_SIZE = 64
# EPSILON_DECAY = 0.9995
START_EPSILON = 1.0
FINAL_EPSILON = 0.01
TESTING_END_DATE = '2019-12-31'
# TESTING_END_DATE = '2022-02-01'
RENDER = False


def run(env, model_name):
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    epsilon_delta = 0
    agent = DDQNAgent(state_size=env.window_size, action_size=len(env.actions), epsilon=0, epsilon_delta=epsilon_delta)
    agent.load(f'{get_absolute_path("weights")}/{model_name}')
    state = env.reset()
    cumulative_reward = 0.0
    current_balance = env.balance()

    current_date = env.get_current_date()

    data = {
        "date": [current_date],
        "balance": [current_balance],
        "inventory": [0],
        "cash": [INITIAL_MONEY],
        'sharpe': [0],
        "sortino": [0]
    }

    while current_date <= TESTING_END_DATE:
        if RENDER:
            env.status()

        action = agent.act(state)

        next_state, reward, done, info = env.step(action, render=RENDER)

        state = next_state

        current_balance = info.get("balance")
        current_date = info.get("date")
        cumulative_reward = agent.gamma * cumulative_reward + reward

        data['date'].append(current_date)
        data['balance'].append(current_balance)
        data['inventory'].append(info.get("inventory"))
        data['cash'].append(info.get("cash"))

        sharpe = rewards.sharpe_ratio(data['balance'], last_21=False)
        sortino = rewards.sortino_ratio(data['balance'], last_21=False)

        data['sharpe'].append(sharpe)
        data['sortino'].append(sortino)

        if done:
            break

    df = pd.DataFrame(data, columns=['date', 'balance', 'inventory', 'cash', 'sharpe', 'sortino'])
    df["date"] = df["date"].astype("datetime64")
    df = df.set_index("date")

    return df
    # print(f'{model_name} - {current_balance}')


def random_model(env):
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    state = env.reset()
    action_size = len(env.actions)
    cumulative_reward = 0.0
    current_balance = env.balance()

    current_date = env.get_current_date()

    data = {
        "date": [current_date],
        "balance": [current_balance],
        "inventory": [0],
        "cash": [INITIAL_MONEY],
        'sharpe': [0],
        "sortino": [0]
    }

    while current_date <= TESTING_END_DATE:
        if RENDER:
            env.status()

        action = random.choice(np.arange(action_size))

        next_state, reward, done, info = env.step(action, render=RENDER)

        state = next_state

        current_balance = info.get("balance")
        current_date = info.get("date")

        data['date'].append(current_date)
        data['balance'].append(current_balance)
        data['inventory'].append(info.get("inventory"))
        data['cash'].append(info.get("cash"))

        sharpe = rewards.sharpe_ratio(data['balance'])
        sortino = rewards.sortino_ratio(data['balance'])

        data['sharpe'].append(sharpe)
        data['sortino'].append(sortino)

        if done:
            break

    df = pd.DataFrame(data, columns=['date', 'balance', 'inventory', 'cash', 'sharpe', 'sortino'])
    df["date"] = df["date"].astype("datetime64")
    df = df.set_index("date")

    return df
    # print(f'{model_name} - {current_balance}')
