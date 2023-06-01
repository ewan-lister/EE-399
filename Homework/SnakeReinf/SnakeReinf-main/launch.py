from game.Snake import Snake
from reinf.SnakeEnv import SnakeEnv
from reinf.utils import perform_mc, show_games

# Winning everytime hyperparameters
grid_length = 4
n_episodes = 1000000
epsilon = 0.04
gamma = 0.55
rewards = [-100000000, -30000000, 40, 100]
# 1st -100000, -40000, 10, 10
# 2nd -100, -1, 1, 10
# [Losing move, inefficient move, efficient move, winning move]

# Playing part
#game = Snake((800, 800), grid_length)
#game.start_interactive_game()

print("Epsilon: " + str(epsilon) + " Gamma: " + str(gamma))
print("Rewards: " + str(rewards))
print(f"Number of episodes: {n_episodes}")

# Training part
# use pickle library to save q_table
env = SnakeEnv(grid_length=grid_length, with_rendering=False)
q_table = perform_mc(env, n_episodes, epsilon, gamma, rewards)


# Viz part
env = SnakeEnv(grid_length=grid_length, with_rendering=True)
show_games(env, 100, q_table)
