import pandas as pd
import numpy as np


# data_path = 'data/EURUSD_2010.csv'

# df = pd.read_csv(data_path)


class Env:

    def __init__(self, window_size, initial_money=10_000, reward_type=None, path="data/EURUSD_2015-2022.csv"):
        self.actions = [0, 1, 2, 3, 4, 5]
        self.window_size = window_size
        self.inventory = 0
        self.initial_money = initial_money
        self.money = initial_money
        self.time = 1  # count the amount of days
        self.data = []
        self.dates = []
        self.reward_type = reward_type
        self.daily_balance = []

        self.get_data_and_dates(path=path)

    def get_data_and_dates(self, path):
        data_df = pd.read_csv(path)
        values = data_df.Close.values
        res = values / values[0]
        self.data = res
        self.dates = data_df.Date.values

    def reset(self):
        self.money = self.initial_money
        self.inventory = 0
        self.time = 1
        return self.get_state()

    def index(self):
        return self.time + self.window_size

    def get_current_price(self):
        return self.data[self.time + self.window_size]

    def get_current_date(self):
        return self.dates[self.time + self.window_size]

    def get_state(self):
        # starts out actually in the day 30° to have the past data available
        res = self.data[self.time - 1: self.time + self.window_size]
        res = np.array(res)

        res = np.ediff1d(res)

        return res

    def status(self):
        print(f"money: {self.money}")

    def step(self, action, render=True):
        """
        :param action:
        0 1 2 3 => {hold, buy, sell, close}
        :return:
        """
        msg = ""
        price = self.get_current_price()
        if action == 1 and self.money > price:
            # buy 1 unit
            units = 1
            self.money -= price * units
            self.inventory += units
            msg = f'day {self.time}: buy {units} unit at price {price} | money_left: {self.money} | total balance {self.balance()}'
        elif action == 2 and self.inventory > 0:
            # sell 1 unit
            units = 1
            self.money += price * units
            self.inventory -= units
            msg = f'day {self.time}: sold {units} unit at price {price} | money_left: {self.money} | total balance {self.balance()}'
        elif action == 3 and self.inventory > 0:
            # close
            units = self.inventory
            self.money += price * units
            self.inventory = 0
            msg = f'day {self.time}: sold {units} unit at price {price} | money_left: {self.money} | total balance {self.balance()}'
        elif action == 4 and self.inventory > 0:
            # sell 10% inventory
            units = self.inventory // 10
            units = min(self.inventory, units)
            self.money += price * units
            self.inventory -= units
            msg = f'day {self.time}: sold {units} unit at price {price} | money_left: {self.money} | total balance {self.balance()}'
        elif action == 5 and self.money > 0:
            # buy 10% money
            investing_value = self.money * 10.0 / 100.0
            investing_value = min(self.money, investing_value)
            units = investing_value // price
            self.money -= price * units
            self.inventory += units
            msg = f'day {self.time}: buy {units} unit at price {price} | money_left: {self.money} | total balance {self.balance()}'
        else:
            msg = f"day {self.time}: hold | money_left: {self.money} | total balance: {self.balance()}"
        if render:
            print(msg)

        self.time += 1
        next_state = self.get_state()
        self.daily_balance.append(self.balance())
        reward = self.reward_type.reward_function(self.daily_balance, 0)
        done = self.money < 0 or self.time > len(self.data) - self.window_size
        return next_state, reward, done, {"balance": self.balance(), 'date': self.get_current_date(),
                                          "inventory": self.inventory,
                                          'cash': self.money}

    def balance(self):
        return self.money + self.inventory * self.get_current_price()

    def get_state_size(self):
        return self.window_size

    def __str__(self):
        return F"ENV_{self.reward_type.name}"
