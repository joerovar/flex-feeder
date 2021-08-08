from scipy.stats import lognorm
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
import results

lbls_OP = ['peak', 'off-peak']
clrs_OP = ['lightcoral', 'deepskyblue']

headways_rl = np.load('out/saved results/params/headways PQL peak.npy', allow_pickle=True).item()
headways_rl_OP = np.load('out/saved results/params/headways PQL off peak.npy', allow_pickle=True).item()
delays_rl = np.load('out/saved results/params/delays PQL peak.npy', allow_pickle=True).tolist()
delays_rl_OP = np.load('out/saved results/params/delays PQL off peak.npy', allow_pickle=True).tolist()

results.plot_2hw_cv(headways_rl, headways_rl_OP,'out/figs/OP headway comparison.png', lbls_OP, clrs_OP)
results.plot_2whiskers(delays_rl,  delays_rl_OP,'out/figs/OP delay comparison.png', 'delay (s)', lbls_OP, clrs_OP)