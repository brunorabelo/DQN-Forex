from copy import copy
import sqlite3

import pandas as pd
import numpy as np

# data_path = 'data/EURUSD_2010.csv'

# df = pd.read_csv(data_path)


class Env:

    def __init__(self, window_size, initial_money=10_000, reward_type=None, path="data/EURBRL_2017.csv"):
        self.actions = [0, 1, 2]
        self.window_size = window_size
        self.inventory = 0
        self.initial_money = initial_money
        self.money = initial_money
        self.time = 1  # count the amount of days
        self.data = self.get_data(path=path)
        self.reward_type = reward_type
        self.daily_balance = []

    def get_data(self, path):
        # cnx = sqlite3.connect(path)
        data_df = pd.read_csv(path)
        values = data_df.Close.values
        # normalize
        # x = np.asarray(values)
        # res = (x - x.mean()) / x.std()
        # res -= min(res)
        # print(res.mean(), res.std())
        res = values / values[0]
        return res

    def reset(self):
        self.money = self.initial_money
        self.inventory = 0
        self.time = 1
        return self.get_state()

    def index(self):
        return self.time + self.window_size

    def get_current_price(self):
        return self.data[self.time + self.window_size]

    def get_state(self):
        # starts out actually in the day 30° to have the past data available
        res = self.data[self.time - 1: self.time + self.window_size]
        res = np.array(res)

        res = np.ediff1d(res)

        return res

    def status(self):
        print(f"money: {self.money}")

    def step(self, action, render=True):
        return self.step_1_unit(action, render=render)

    def step_1_unit(self, action, render=True):
        """
        :param action:
        0 1 2 => {hold, buy, sell}
        :param render:
        0 1 2 => {hold, buy, sell}
        :return:
        """
        msg = ""
        price = self.get_current_price()
        if action == 1 and self.money > price:
            # buy
            self.money -= price
            self.inventory += 1
            msg = f'day {self.time}: buy 1 unit at price {price} | money_left: {self.money} | total balance {self.balance()}'
        elif action == 2 and self.inventory > 0:
            # sell
            self.money += price
            self.inventory -= 1
            msg = f'day {self.time}: sold 1 unit at price {price} | money_left: {self.money} | total balance {self.balance()}'
        else:
            msg = f"day {self.time}: hold | money_left: {self.money} | total balance: {self.balance()}"
        if render:
            print(msg)

        self.time += 1
        next_state = self.get_state()
        self.daily_balance.append(self.balance())
        reward = self.reward_type.reward_function(self.daily_balance, 0)
        done = self.money < 0
        return next_state, reward, done, {"balance": self.balance()}

    def step_all_units(self, action):
        pass

    def balance(self):
        return self.money + self.inventory * self.get_current_price()

    def get_state_size(self):
        return self.window_size

    def __str__(self):
        return F"ENV_{self.reward_type.name}"
