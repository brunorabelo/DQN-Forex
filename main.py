from dqn.env import Env
from dqn import train
from dqn.rewards import DailyReturnReward, SortinoRatioReward, SharpeRatioReward

# Select the number of episodes Sugestion: 5000
episodes = 5000

# Select the reward type. Uncomment the one you want and comment the others
reward_type = DailyReturnReward
# reward_type = SortinoRatioReward
# reward_type = SharpeRatioReward

env = Env(window_size=30, reward_type=reward_type)
train.train(episodes=episodes, env=env)