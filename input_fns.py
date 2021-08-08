import csv
import numpy as np
from numpy import inf
from itertools import combinations_with_replacement
from scipy.stats import lognorm
import matplotlib.pyplot as plt


def extract_demand(path, factor_fixed_od, factor_req_od):
    with open(path, 'r') as f:
        rf = csv.reader(f, delimiter=' ')
        o_rows = []
        d_cols = []
        fixed_od = []
        requests_od = []
        i = 0
        for row in rf:
            if row[0] == 'fixed_od':
                i = 1
                fixed_od.append([])
                o_rows.append([])
                continue
            if row[0] == 'requests_od':
                i = 2
                requests_od.append([])
                o_rows.append([])
                continue
            if row[0] == 'OD':
                d_cols.append(row[1:])
                continue
            if i == 1:
                o_rows[-1].append(row[0])
                fixed_od[-1].append(row[1:])
            if i == 2:
                o_rows[-1].append(row[0])
                requests_od[-1].append(row[1:])
    fixed_od = np.multiply(np.array(fixed_od).astype(float), np.array([[[factor_fixed_od[0]]],[[factor_fixed_od[1]]]]))
    requests_od = np.multiply(np.array(requests_od).astype(float), np.array([[[factor_req_od[0]]],[[factor_req_od[1]]]]))
    # arrival rates

    arr_rates = {}
    ons = fixed_od.sum(axis=2)
    origins = list(np.array(o_rows[:2]).flat)
    arr = list(ons.flat)

    for i, j in zip(origins, arr):
        arr_rates[i] = j

    prob_drop = {}
    offs = fixed_od.sum(axis=1)
    ons = np.concatenate((ons,[[0],[0]]), axis=1)
    offs = np.concatenate(([[0], [0]], offs), axis=1)
    delta = np.subtract(ons, offs)
    dep_vol = [[delta[0][0]], [delta[1][0]]]
    for i in range(len(delta)):
        for j in range(1, len(delta[0])-1):
            dep_vol[i].append(dep_vol[i][-1]+delta[i][j])
    peak_vol = np.max(dep_vol)
    offs = np.delete(offs, 0, 1)
    p_drop = list(np.around(np.divide(offs, dep_vol),3).flat)
    destinations = list(np.array(d_cols[:2]).flat)
    for i, j in zip(destinations, p_drop):
        prob_drop[i] = j
    return requests_od, o_rows[2:], d_cols[2:], arr_rates, prob_drop, ons, offs, peak_vol


def extract_network(path):
    with open(path, 'r') as f:
        rf = csv.reader(f, delimiter=' ')
        route_stops = {}
        link_times = {}
        i = 0
        for row in rf:
            if row[0] == 'route_name':
                i = 1
                continue
            if row[0] == 'o-d':
                i = 2
                continue
            if i == 1:
                route_stops[row[0]] = row[1:]
            if i == 2:
                link_times[row[0]] = float(row[1])

    return route_stops, link_times


def compute_route_times(route_stops, link_times, stopping_delay, boarding_delay, alighting_delay, headway, ons,offs):
    route_times = {}

    for route in route_stops:
        run_times = []
        for i in range(1, len(route_stops[route])):
            next_stop = route_stops[route][i]
            prev_stop = route_stops[route][i-1]
            run_time = float(link_times[prev_stop+'-'+next_stop])
            run_times.append(run_time)
        route_times[route] = round(sum(run_times), 1)
    freq = 60*60 / headway

    # DWELL TIMES
    tot_ons_offs = ons.sum(axis=1)
    pax_delay = tot_ons_offs * (boarding_delay + alighting_delay) / freq
    expected_ons = np.divide(ons, freq)
    expected_offs = np.concatenate(([[0],[0]],np.divide(offs, freq)),axis=1)
    expected_ons_offs = np.add(expected_ons, expected_offs)
    prob_stopping = np.add(1, -np.exp(-expected_ons_offs))
    stop_delay = np.multiply(prob_stopping, stopping_delay).sum(axis=1)
    tot_dwell_delay = pax_delay + stop_delay # only for fixed stops
    # RUN TIMES
    idx = {'A': 0, 'B': 1}

    for route in route_stops:
        # here we distinguish deviated routes
        if int(route[1]):
            idx_dev = {'1': 1, '2': 1, '3': 2}
            added_delay = (idx_dev[route[1]])*(stopping_delay + boarding_delay)
            placeholder = route_times[route]
            route_times[route] = round(placeholder + tot_dwell_delay[idx[route[0]]] + added_delay,1)
            # probability of 1 to stop twice and we assume in average one boarding per stop for requests
            continue
        placeholder = route_times[route]
        route_times[route] = round(placeholder + tot_dwell_delay[idx[route[0]]], 1)
    return route_times


def compare_distributions(cv1, nmean, logn_s):
    x = cv1 * nmean * np.random.randn(10000) + nmean  # normally  distributed values
    y = lognorm.rvs(logn_s, scale=nmean, size=10000)
    plt.hist(x, density=True, alpha=0.2, label='normal')
    plt.hist(y, density=True, alpha=0.3, label='lognormal')
    plt.title('main route')
    plt.legend()
    plt.show()
    return


def compare_distributions2(cv2, nmean2, logn_s2):
    x = cv2 * nmean2 * np.random.randn(10000) + nmean2  # normally  distributed values
    y = lognorm.rvs(logn_s2, scale=nmean2, size=10000)
    plt.hist(x, density=True, alpha=0.2, label='normal')
    plt.hist(y, density=True, alpha=0.3, label='lognormal')
    plt.title('deviations')
    plt.legend()
    plt.show()
    return
