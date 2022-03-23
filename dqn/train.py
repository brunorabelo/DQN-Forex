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
TOTAL_DAYS = 91
RENDER = False


def train(episodes, env):
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    action_size = len(env.actions)

    epsilon_delta = (START_EPSILON - FINAL_EPSILON) / episodes
    agent = DDQNAgent(state_size=env.window_size, action_size=action_size, seed=0, batch_size=BATCH_SIZE,
                      epsilon=START_EPSILON, epsilon_min=FINAL_EPSILON, epsilon_delta=epsilon_delta)

    done = False
    return_history = []
    balances = []
    data_size = len(env.data)
    for episode in range(1, episodes + 1):
        state = env.reset()
        cumulative_reward = 0.0

        final_balance = 0.0

        for time in range(1, min(TOTAL_DAYS + 1, data_size - 1)):
            if RENDER:
                env.status()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action, render=RENDER)

            agent.step(state, action, reward, next_state, done)

            state = next_state

            final_balance = info.get("balance")

            cumulative_reward = agent.gamma * cumulative_reward + reward

            if done:
                print("done: episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                      .format(episode, episodes, time, cumulative_reward, agent.epsilon))
                break

        print(
            f"episode: {episode}/{episodes}, time: {time}, score: {cumulative_reward}, balance: {final_balance} epsilon: {agent.epsilon}")

        balances.append(final_balance)
        return_history.append(cumulative_reward)
        agent.update_epsilon()
        # Every 10 episodes, update the plot for training monitoring
        if episode % 20 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(balances, 'r')
            ax2.plot(return_history, 'b')
            ax1.set(xlabel='Episode')
            ax1.set(ylabel='balance')
            ax2.set(xlabel='Episode')
            ax2.set(ylabel='score')
            fig.savefig(
                f'{get_absolute_path("results")}/ddqn_{date}__REWARD_{env.reward_type.name}__EPISODES_{episodes}__TOTAL_DAYS_{TOTAL_DAYS}__BATCH_SIZE_{BATCH_SIZE}__EPSILON_{START_EPSILON}__DECAY_{epsilon_delta}.png')
            # Saving the model to disk
            # agent.save("trained_model_reward.h5")
            agent.save(f'{get_absolute_path("weights")}/checkpoint_ddqn_{date}.pth')
