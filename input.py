import numpy as np
import csv
from scipy.stats import lognorm
from numpy import random
import matplotlib.pyplot as plt
from input_fns import *

# SIMULATION
# time units in seconds
FIXED_ROUTES_BASE = ['A', 'B']
NR_DEVIATED_STOPS = 2
FACTOR_FIXED_OD = [0.8*2.3, 2.3]
FACTOR_REQ_OD = [0.8*0.36, 0.36]
ROUTE_STOPS, LINK_TIMES = extract_network('in/routes.dat')
OD_REQ, O_REQ, D_REQ, ARR_RATES, PROB_DROP, ONS_FIXED, OFFS_FIXED, PEAK_VOL = extract_demand('in/od.dat', FACTOR_FIXED_OD, FACTOR_REQ_OD)
DEV_STOPS = {'A': O_REQ[0][-NR_DEVIATED_STOPS:],
             'B': D_REQ[1][-NR_DEVIATED_STOPS:]}

tot_dev = sum(list(OD_REQ[1].flat))
tot_fixed = sum(list(ONS_FIXED[1].flat))
demand_ratio = round(tot_dev / tot_fixed * 100, 1)
STOPPING_DELAY = 3
BOARDING_DELAY = 2
ALIGHTING_DELAY = 1
HEADWAY = 15 * 60
MIN_LAYOVER = 60
CAPACITY = 25
HEADWAY_MAX = round(CAPACITY/PEAK_VOL*60*60)
assert HEADWAY <= HEADWAY_MAX
ROUTE_TIMES = compute_route_times(ROUTE_STOPS, LINK_TIMES, STOPPING_DELAY, BOARDING_DELAY, ALIGHTING_DELAY, HEADWAY, ONS_FIXED, OFFS_FIXED)
FIXED_CYCLE_TIME = ROUTE_TIMES['A0']+ROUTE_TIMES['B0']
N_BUSES = round(FIXED_CYCLE_TIME/HEADWAY)
DELAY_TOLERANCE = 2.0 * 60
SLACK_MIN = 1.2 * 60
ROUTE_TIMES_MINS = {}
for k in ROUTE_TIMES:
    ROUTE_TIMES_MINS[k] = round(ROUTE_TIMES[k] / 60, 1)
# print(f'peak volume {PEAK_VOL} pax/hr')
# print(f'total request demand {round(tot_dev)} pax/hr')
# print(f'total fixed demand {round(tot_fixed)} pax/hr')
# print(f'demand ratio {demand_ratio} %')
# print(f'hmax: {round(HEADWAY_MAX/60)} mins')
# print(f'route times (mins): {ROUTE_TIMES_MINS}')
# print(f'n buses: {N_BUSES}')
MAX_SIMUL_TIME = 180.0 * 60
FINAL_TRIP_DEPARTURE = MAX_SIMUL_TIME-MAX_SIMUL_TIME % HEADWAY+HEADWAY

TTD = 'LOGNORMAL'

CV = 0.33
CV_REQ = 0.75 * CV
LOGN_S = np.sqrt(np.log(np.power(CV, 2)+1))
LOGN_S_REQ = np.sqrt(np.log(np.power(CV_REQ, 2)+1))


# AS EXAMPLE
NMEAN = 2.5
NMEAN_REQ = 1.9

# LEARNING

ALPHA = 0.1
GAMMA = 0.05
STARTING_EPSILON = 1.00
ENDING_EPSILON = 0.02
PERCENT_EPS_EXPLORE = 0.7
TRAIN_EPISODES = 900
TEST_EPISODES = 25
PULL_TRAIN_DATA_EVERY = 15

REWARD_WEIGHTS = [15, 5, 25]
REWARD_WEIGHTS_OFF_PEAK = [40, 5, 25]

NR_ROUTES = 2
REQUESTS_MAX = 2
DELAY_HIGH = 5
DELAY_LOW = 0
OBS_HIGH = np.array([NR_ROUTES-1, REQUESTS_MAX, REQUESTS_MAX, DELAY_HIGH])
OBS_LOW = np.array([0, 0, 0, -DELAY_LOW])

STATE_STEP_SIZE = np.array([1, 1, 1, 1])

DIM = np.divide(OBS_HIGH - OBS_LOW, STATE_STEP_SIZE)
DIM = np.add(DIM, 1)
DIM = DIM.astype(int)
NR_ACTIONS = 4

