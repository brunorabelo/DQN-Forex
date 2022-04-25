import time
from datetime import datetime, timedelta

import numpy as np

from .agent import Agent as DDQNAgent
import matplotlib.pyplot as plt
# episodes = 5_000  # Number of episodes used for training
from .utils import get_absolute_path

INITIAL_MONEY = 10_000
BATCH_SIZE = 64
# EPSILON_DECAY = 0.9995
START_EPSILON = 1.0
FINAL_EPSILON = 0.01
TRAINING_END_DATE = '2016-07-01'
RENDER = False
elapsed_times = []


def train(episodes, env):
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    action_size = len(env.actions)

    epsilon_delta = (START_EPSILON - FINAL_EPSILON) / episodes
    agent = DDQNAgent(state_size=env.window_size, action_size=action_size, seed=0, batch_size=BATCH_SIZE,
                      epsilon=START_EPSILON, epsilon_min=FINAL_EPSILON, epsilon_delta=epsilon_delta)

    done = False
    reward_history = []
    final_balance_history = []
    data_size = len(env.data)
    for episode in range(1, episodes + 1):
        st = time.perf_counter()
        state = env.reset()
        cumulative_reward = 0.0
        final_balance = 0.0
        current_training_date = env.get_current_date()
        while current_training_date <= TRAINING_END_DATE:
            if RENDER:
                env.status()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action, render=RENDER)

            agent.step(state, action, reward, next_state, done)

            state = next_state

            final_balance = info.get("balance")
            current_training_date = info.get("date")

            cumulative_reward = agent.gamma * cumulative_reward + reward

            if done:
                print("done: episode: {}/{}, end_date: {}, score: {:.6}, epsilon: {:.3}"
                      .format(episode, episodes, current_training_date, cumulative_reward, agent.epsilon))
                break

        elapsed_time = time.perf_counter() - st
        elapsed_times.append(elapsed_time)
        left_episodes = episodes - episode
        left_time = elapsed_time * left_episodes
        print(
            f"episode: {episode}/{episodes}, "
            f"end_date: {current_training_date}, "
            f"score: {cumulative_reward}, "
            f"balance: {final_balance}, "
            f"epsilon: {agent.epsilon}, "
            f"left_time: {str(timedelta(seconds=left_time))}"
        )
        final_balance_history.append(final_balance)
        reward_history.append(cumulative_reward)
        agent.update_epsilon()
        # Every 10 episodes, update the plot for training monitoring
        if episode % 20 == 0:
            mean_time = np.mean(elapsed_times)
            elapsed_times.clear()
            left_episodes = episodes - episode
            left_time = mean_time * left_episodes
            print(f"left_episodes: {left_episodes}, left_time: {str(timedelta(seconds=left_time))}")

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(final_balance_history, 'r')
            ax2.plot(reward_history, 'b')
            ax1.set(xlabel='Episode')
            ax1.set(ylabel='balance')
            ax2.set(xlabel='Episode')
            ax2.set(ylabel='score')
            fig.savefig(
                f'{get_absolute_path("results")}/'
                f'ddqn_{date}'
                f'__REWARD_{env.reward_type.name}'
                f'__EPISODES_{episodes}'
                f'__FINAL_DATE_{TRAINING_END_DATE}'
                f'__BATCH_SIZE_{BATCH_SIZE}'
                f'__EPSILON_{START_EPSILON}'
                f'__DECAY_{epsilon_delta}'
                f'.png')
            # Saving the model to disk
            # agent.save("trained_model_reward.h5")

        if episode % 200 == 0:
            agent.save(f'{get_absolute_path("weights")}/'
                       f'checkpoint_ddqn_{date}'
                       f'__REWARD_{env.reward_type.name}'
                       f'__EPISODES_{episode}'
                       f'__FINAL_DATE_{TRAINING_END_DATE}'
                       f'.pth')
