from input import *
import core
import results
import csv
from learning import *
from ui import *
import matplotlib.pyplot as plt
from scipy.stats import norm

# LIGHT SCENARIO
ALPHA = 0.2
GAMMA = 0.4
EPSILON = 1.00
EPSILON_RATE = 0.5
train_episodes = 10
test_episodes = 1
SG_REQ_THRESHOLD = REQUESTS_MAX - 1
SG_SLACK_THRESHOLD = 1.5

# TRAIN

env = core.SimulationEnv()


initial_Q_table = np.zeros([DIM[0], DIM[1], DIM[2], 2])

Q_table, ep_rewards, percent_reject_curve, explore, \
    episode_count, ep_rew_comps, ep_rew_comps_std = q_learn(env, initial_Q_table, EPSILON, EPSILON_RATE, ALPHA, GAMMA,
                                                            train_episodes)

# TEST

# hw = [[]] * 3
# delays = [[]] * 3
# req_rej = [[]] * 2
# denied = [[]] * 3
# terminal_load = [[]] * 3
#
# hw[0], delays[0], denied[0], terminal_load[0] = test_do_nothing(env, test_episodes)
# hw[1], delays[1], req_rej[0], sg_trajectories, denied[1], terminal_load[1] = test_smart_greedy(env, test_episodes, SG_REQ_THRESHOLD, SG_SLACK_THRESHOLD)
# hw[2], delays[2], req_rej[1], sars, q_trajectories, denied[2], terminal_load[2] = q_test(env, Q_table, test_episodes)
#
# # PLOT
# results.plot_reward_components(ep_rew_comps, ep_rew_comps_std, episode_count)
# results.plot_learning_curve(episode_count, ep_rewards, percent_reject_curve, explore)
# results.plot_whiskers([hw, delays, req_rej])
# results.plot_whiskers2([denied, terminal_load])
# results.plot_multi_trajectories([sg_trajectories, q_trajectories])
