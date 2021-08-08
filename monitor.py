from input import *
import core
import results
import csv
from learning import *
from ui import *
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib.ticker import PercentFormatter
import timeit

start = timeit.default_timer()

# TRAIN

env = core.SimulationEnv()

initial_Q_table = np.zeros([DIM[0], DIM[1], DIM[2], DIM[3], NR_ACTIONS])
dummy_table = np.zeros([DIM[0], DIM[1], DIM[2], DIM[3], NR_ACTIONS])
dummy_table_OP = np.zeros([DIM[0], DIM[1], DIM[2], DIM[3], NR_ACTIONS])

Q_table, mean_ep_rew, reject_ratio, epsilons = q_learn_(env, initial_Q_table, dummy_table, REWARD_WEIGHTS)
headways_rl, delays_rl, tr_rl, rr_rl, tt_rl, otp_rl = q_test(env, Q_table)

eps = [i for i in range(0, TRAIN_EPISODES, PULL_TRAIN_DATA_EVERY)]

headways_sg, delays_sg, tr_sg, rr_sg, tt_sg, otp_sg = test_smart_greedy(env)

# delays are PER ROUTE!!! LIST OF LENGTH 2

stop = timeit.default_timer()

np.save('out/params/headways SG', np.array(headways_sg))
np.save('out/params/headways PQL peak', np.array(headways_rl))
np.save('out/params/delays SG', np.array(delays_sg))
np.save('out/params/delays PQL peak', np.array(delays_rl))
np.save('out/params/Q table peak', Q_table)
np.save('out/params/epsilons', np.array(eps))
np.save('out/params/mean ep reward peak', np.array(mean_ep_rew))


lbls = ['SG', 'PQL']
clrs = ['lightcoral', 'deepskyblue']

lbls_OP = ['peak', 'off-peak']
clrs_OP = ['deepskyblue', 'limegreen']

results.plot_2hw_cv(headways_sg, headways_rl, 'out/figs/headway comparison.png', lbls, clrs)
results.plot_2whiskers(delays_sg, delays_rl,  'out/figs/delay comparison.png', 'delay (s)', lbls, clrs)
results.plot_training_info(eps, mean_ep_rew, epsilons, 'out/figs/training info.png')
results.plot_policy_extract(Q_table, 'out/figs/policy_sample.png')
results.plot_requests(tr_sg, rr_sg, tr_rl, rr_rl, 'out/figs/rejections.png')

Q_table_OP, mean_ep_rew_OP, reject_ratio_OP, epsilons_OP = q_learn_(env, initial_Q_table, dummy_table_OP, REWARD_WEIGHTS_OFF_PEAK)

headways_rl_OP, delays_rl_OP, tr_rl_OP, rr_rl_OP, tt_rl_OP, otp_rl_OP = q_test(env, Q_table_OP)
results.plot_2hw_cv(headways_rl, headways_rl_OP,'out/figs/OP headway comparison.png', lbls_OP, clrs_OP)
results.plot_2whiskers(delays_rl,  delays_rl_OP,'out/figs/OP delay comparison.png', 'delay (s)', lbls_OP, clrs_OP)
results.plot_policy_extract(Q_table_OP, 'out/figs/policy_sample_OP.png')
results.write_results(tr_sg, rr_sg, tr_rl, rr_rl, tr_rl_OP, rr_rl_OP,stop - start, 'out/txt/test results.dat')

np.save('out/params/headways PQL off peak', np.array(headways_rl_OP))
np.save('out/params/delays PQL off peak', np.array(delays_rl_OP))
np.save('out/params/Q table off peak', Q_table_OP)
np.save('out/params/mean ep reward off peak', np.array(mean_ep_rew_OP))

