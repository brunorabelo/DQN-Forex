from datetime import datetime
from .agent import Agent as DDQNAgent
import matplotlib.pyplot as plt
# episodes = 5_000  # Number of episodes used for training
from .utils import get_absolute_path

INITIAL_MONEY = 10_000
BATCH_SIZE = 64
# EPSILON_DECAY = 0.9995
START_EPSILON = 1.0
FINAL_EPSILON = 0.01
TESTING_END_DATE = '2019-12-31'
RENDER = False


def run(env, model_name):
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    epsilon_delta = (START_EPSILON - FINAL_EPSILON) / 100
    agent = DDQNAgent(state_size=env.window_size, action_size=len(env.actions), epsilon=0, epsilon_delta=epsilon_delta)
    agent.load(f'{get_absolute_path("weights")}/{model_name}')
    state = env.reset()
    cumulative_reward = 0.0
    current_balance = env.balance()
    balance_history = []
    date_history = []
    reward_history = []

    current_date = env.get_current_date()
    balance_history.append(current_balance)
    date_history.append(current_date)
    while current_date <= TESTING_END_DATE:
        if RENDER:
            env.status()

        action = agent.act(state)

        next_state, reward, done, info = env.step(action, render=RENDER)

        state = next_state

        current_balance = info.get("balance")
        current_date = info.get("date")
        cumulative_reward = agent.gamma * cumulative_reward + reward

        balance_history.append(current_balance)
        reward_history.append(cumulative_reward)
        date_history.append(current_date)

        if done:
            break

    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(date_history, balance_history)
    fig.savefig(
        f'{get_absolute_path("testing_results")}/'
        f'ddqn_{model_name}'
        f'.png')
    print(f'{model_name} - {current_balance}')


def train(episodes, env):
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    action_size = len(env.actions)

    epsilon_delta = (START_EPSILON - FINAL_EPSILON) / episodes
    agent = DDQNAgent(state_size=env.window_size, action_size=action_size, seed=0, batch_size=BATCH_SIZE,
                      epsilon=START_EPSILON, epsilon_min=FINAL_EPSILON, epsilon_delta=epsilon_delta)

    done = False
    return_history = []
    cumulative_returns = []
    data_size = len(env.data)

    for episode in range(1, episodes + 1):
        state = env.reset()
        cumulative_reward = 0.0
        final_balance = 0.0
        current_training_date = env.get_current_date()
        while TESTING_END_DATE < current_training_date:
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

        print(
            f"episode: {episode}/{episodes}, end_date: {current_training_date}, score: {cumulative_reward}, balance: {final_balance} epsilon: {agent.epsilon}")

        cumulative_returns.append(final_balance)
        return_history.append(cumulative_reward)
        agent.update_epsilon()
        # Every 10 episodes, update the plot for training monitoring
        if episode % 20 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(cumulative_returns, 'r')
            ax2.plot(return_history, 'b')
            ax1.set(xlabel='Episode')
            ax1.set(ylabel='balance')
            ax2.set(xlabel='Episode')
            ax2.set(ylabel='score')
            fig.savefig(
                f'{get_absolute_path("results")}/'
                f'ddqn_{date}'
                f'__REWARD_{env.reward_type.name}'
                f'__EPISODES_{episodes}'
                f'__FINAL_DATE_{TESTING_END_DATE}'
                f'__BATCH_SIZE_{BATCH_SIZE}'
                f'__EPSILON_{START_EPSILON}'
                f'__DECAY_{epsilon_delta}'
                f'.png')
            # Saving the model to disk
            # agent.save("trained_model_reward.h5")
            agent.save(f'{get_absolute_path("weights")}/checkpoint_ddqn_{date}.pth')
