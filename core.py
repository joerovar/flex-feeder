import numpy as np
from input import *
from numpy import random
from scipy.stats import lognorm
from random import sample
import csv
from scipy.stats import norm
import matplotlib.pyplot as plt
import results


class SimulationEnv:
    def __init__(self):
        # THE ONLY NECESSARY TRIP INFORMATION TO CARRY THROUGHOUT SIMULATION
        # SIMUL PARAMS
        # NEW
        self.routes = []
        self.trip_id = []
        self.last_stop = []
        self.next_stop = []
        self.dep_t = []
        self.terminal_dep_t = []
        self.arr_t = []
        self.load = []
        self.bus_idx = 0
        self.event_type = 0
        self.next_instance_time = []
        # (0=END 1=STOP -1=DISPATCH)
        self.time = 0.0  # keep track of time
        self.next_trip_ids = {}
        self.last_bus_time = {}
        self.next_departures = {}
        self.o_reqs = []
        self.d_reqs = []
        # RL PARAMS
        self.s0 = []
        self.act = []
        self.sars = []
        # only updated when a sars is completed

        # RECORDINGS
        self.trip_trajectories = []
        self.recorded_headway = {}
        self.recorded_delays = []
        self.recorded_trip_times = []

    def get_pax(self, headway):
        i = self.bus_idx
        stop = self.last_stop[i]
        arrival_rate = ARR_RATES[stop]
        pax_at_stop = np.random.poisson(lam=arrival_rate * headway / 3600)
        return pax_at_stop

    def get_travel_time(self):
        i = self.bus_idx
        prev_stop = self.last_stop[i]
        next_stop = self.next_stop[i]
        route = self.routes[i]
        mean_runtime = LINK_TIMES[str(prev_stop)+'-'+str(next_stop)]
        if next_stop in DEV_STOPS[route[0]] or prev_stop in DEV_STOPS[route[0]]:
            s = LOGN_S_REQ
        else:
            s = LOGN_S
        runtime = lognorm.rvs(s, scale=mean_runtime)

        assert TTD == 'LOGNORMAL'

        return runtime

    def record_trajectories(self):
        i = self.bus_idx
        # IN THE END I WANT IT LIKE THIS
        if self.event_type == 2:
            trajectory = [self.next_stop[i], round(self.arr_t[i],2), 0]
            self.trip_trajectories[i][-1].append(trajectory)
        if self.event_type == 1:
            trajectory = [self.last_stop[i], round(self.dep_t[i],2), self.load[i]]
            self.trip_trajectories[i][-1].append(trajectory)
        if self.event_type == 0:
            trajectory = [[self.last_stop[i], round(self.dep_t[i],2), self.load[i]]]
            self.trip_trajectories[i].append(trajectory)
        return

    def dev_stop_arrival(self):
        i = self.bus_idx
        curr_route = self.routes[i]
        curr_stop_idx = ROUTE_STOPS[curr_route].index(self.next_stop[i])
        self.last_stop[i] = ROUTE_STOPS[curr_route][curr_stop_idx]
        self.next_stop[i] = ROUTE_STOPS[curr_route][curr_stop_idx + 1]
        assert self.last_stop[i] in self.o_reqs[i] or self.last_stop[i] in self.d_reqs[i]
        pickups = self.o_reqs[i].count(self.last_stop[i])
        dropoffs = self.d_reqs[i].count(self.last_stop[i])
        dwell_time = round(STOPPING_DELAY + pickups * BOARDING_DELAY + dropoffs * ALIGHTING_DELAY, 1)
        self.load[i] += pickups
        self.load[i] -= dropoffs
        self.dep_t[i] = self.time + dwell_time
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime

        self.record_trajectories()
        return

    def fixed_stop_arrival(self):
        i = self.bus_idx
        curr_route = self.routes[i]
        curr_stop_idx = ROUTE_STOPS[curr_route].index(self.next_stop[i])
        self.last_stop[i] = ROUTE_STOPS[curr_route][curr_stop_idx]
        self.next_stop[i] = ROUTE_STOPS[curr_route][curr_stop_idx + 1]
        req_dropoffs = self.d_reqs.count(self.last_stop[i])
        self.load[i] -= req_dropoffs
        dropoffs = int(round(PROB_DROP[self.last_stop[i]] * self.load[i]))
        self.load[i] -= dropoffs
        assert self.load[i] >= 0
        if int(self.last_bus_time[self.last_stop[i]]):
            headway = self.time - self.last_bus_time[self.last_stop[i]]
            if headway < 0:
                headway = 0
        else:
            headway = HEADWAY
        self.recorded_headway[self.last_stop[i]].append(headway)
        pickups = self.get_pax(headway)
        dwell_time = round(STOPPING_DELAY + pickups * BOARDING_DELAY + dropoffs * ALIGHTING_DELAY, 1)
        dwell_time = (pickups + dropoffs > 0) * dwell_time
        self.load[i] += pickups
        self.dep_t[i] = self.time + dwell_time
        self.last_bus_time[self.last_stop[i]] = self.dep_t[i]

        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime

        self.record_trajectories()
        return

    def get_requests(self, headway):
        i = self.bus_idx
        route = self.routes[i]
        dev_stops = DEV_STOPS[route[0]]
        route_idx = FIXED_ROUTES_BASE.index(self.routes[i][0])
        od = OD_REQ[route_idx]
        expected_arr = np.multiply(headway / 3600, od)
        req = random.poisson(lam=expected_arr)
        idxs = np.array([[],
                         []])
        while not np.all((req == 0)):
            idxs = np.append(idxs, np.array(req.nonzero()), axis=1)
            np.subtract(req, 1, where=req > 0, out=req)
        o_req_idx, d_req_idx = idxs.astype(int)
        o_req_idx.tolist()
        d_req_idx.tolist()
        o_req = []
        d_req = []
        for i in o_req_idx:
            o_req.append(O_REQ[route_idx][i])
        for j in d_req_idx:
            d_req.append(D_REQ[route_idx][j])

        reqs = [0, 0]
        for o in o_req:
            if o in dev_stops:
                idx = dev_stops.index(o)
                reqs[idx] += 1
        for d in d_req:
            if d in dev_stops:
                idx = dev_stops.index(d)
                reqs[idx] += 1
        return reqs, o_req, d_req

    def _get_reward(self, dep_delay, excess_tt, reward_weights):
        [w1, w2, w3] = reward_weights
        i = self.bus_idx
        action = self.act[i]
        s0 = self.s0[i]
        factors = {0: [1, 1], 1: [0, 1], 2: [1, 0], 3: [0, 0]}
        fs = factors[action]
        # in this notation reqs is 1 if single and 2 if multiple
        reqs = sum([s0[1], s0[2]])
        reqs_rejected = sum([f * r for (f, r) in zip(fs, [s0[1], s0[2]])])
        prev_dep_delay = s0[3]
        if reqs:
            rew_reqs = - reqs_rejected
            if round(prev_dep_delay):
                rew_reqs = rew_reqs / prev_dep_delay

            deviation_time = [1.4, 1.4]
            rew_excess_tt = - sum([(1-f) * dt for (f, dt) in zip(fs, deviation_time)])

            if (dep_delay/60) - (DELAY_TOLERANCE/60) > 0 and action:
                rew_delay = - ((dep_delay/60) - (DELAY_TOLERANCE/60))
            else:
                rew_delay = 0

            rew_components = [w1 * rew_reqs, w2 * rew_excess_tt, w3 * rew_delay]
            reward = sum(rew_components)
        else:
            reward = 0
        # print(s0)
        # print(action)
        # print(dep_delay/60)
        # print(reward)
        return reward

    def terminal_arrival(self, reward_weights):
        # change route to A if B and B if A (and deal with terminal trips)
        i = self.bus_idx
        self.arr_t[i] = self.time
        self.record_trajectories()
        prev_route_idx = FIXED_ROUTES_BASE.index(self.routes[i][0])
        next_route_idx = 1 - prev_route_idx
        prev_route = FIXED_ROUTES_BASE[prev_route_idx]
        next_route = FIXED_ROUTES_BASE[next_route_idx]
        self.last_stop[i] = ROUTE_STOPS[prev_route + '0'][-1]
        self.next_stop[i] = ROUTE_STOPS[next_route + '0'][0]

        req_headway = self.time - self.last_bus_time[self.last_stop[i]]
        self.recorded_headway[self.last_stop[i]].append(req_headway)

        if not self.next_departures[next_route]:
            assert not self.next_trip_ids[next_route]
            actual_route_time = self.time - self.terminal_dep_t[i]
            fixed_route_time = ROUTE_TIMES[self.routes[i][0] + '0']
            excess_travel_time = max(0, actual_route_time - fixed_route_time)

            dep_time = max(self.time + MIN_LAYOVER, FINAL_TRIP_DEPARTURE)
            dep_delay = dep_time - FINAL_TRIP_DEPARTURE

            rew = self._get_reward(dep_delay, excess_travel_time, reward_weights)
            s1 = [next_route_idx, 0, 0, round(dep_delay/60, 2)]

            self.sars[i].append([self.s0[i], self.act[i], rew, s1])
            self.next_instance_time[i] = MAX_SIMUL_TIME * 10
            self.last_bus_time[self.last_stop[i]] = self.time
            self.load[i] = 0

        else:
            actual_route_time = self.time - self.terminal_dep_t[i]
            fixed_route_time = ROUTE_TIMES[self.routes[i][0] + '0']
            excess_travel_time = max(0, actual_route_time - fixed_route_time)
            self.routes[i] = next_route + '0'
            self.trip_id[i] = self.next_trip_ids[next_route][0]
            self.next_trip_ids[next_route].pop(0)
            scheduled_dep_time = self.next_departures[next_route][0]
            self.next_departures[next_route].pop(0)

            dep_time = max(self.time + MIN_LAYOVER, scheduled_dep_time)

            dep_delay = dep_time - scheduled_dep_time
            rew = self._get_reward(dep_delay, excess_travel_time, reward_weights)

            self.last_bus_time[self.last_stop[i]] = self.time
            assert req_headway >= 0
            if req_headway < 0:
                req_headway = 0

            [req1, req2], o_req, d_req = self.get_requests(req_headway)
            self.o_reqs[i] = o_req
            self.d_reqs[i] = d_req
            [s_req1, s_req2] = [min(req1, 2), min(req2, 2)]
            s1 = [FIXED_ROUTES_BASE.index(next_route[0]), s_req1, s_req2, round(dep_delay/60, 2)]
            self.sars[i].append([self.s0[i], self.act[i], rew, s1])
            self.s0[i] = s1
            self.load[i] = 0
            self.next_instance_time[i] = dep_time
        self.recorded_trip_times[prev_route_idx].append(actual_route_time)
        self.recorded_delays[next_route_idx].append(dep_delay)
        return

    def update_served_requests(self):
        i = self.bus_idx
        route = self.routes[i]
        dev_stops = DEV_STOPS[route[0]]

        stops_served_per_action = {0: [], 1: [dev_stops[0]], 2: [dev_stops[1]], 3: dev_stops}
        stops_served = stops_served_per_action[self.act[i]]

        for stop in self.o_reqs[i]:
            if stop in dev_stops and stop not in stops_served:
                idx = self.o_reqs[i].index(stop)
                self.o_reqs[i].pop(idx)
                self.d_reqs[i].pop(idx)
        for stop in self.d_reqs[i]:
            if stop in dev_stops and stop not in stops_served:
                idx = self.d_reqs[i].index(stop)
                self.o_reqs[i].pop(idx)
                self.d_reqs[i].pop(idx)
        return

    def terminal_departure(self):
        # route is set to action so now just set the right stop and compute travel time
        i = self.bus_idx
        self.last_stop[i] = ROUTE_STOPS[self.routes[i]][0]
        self.next_stop[i] = ROUTE_STOPS[self.routes[i]][1]
        self.dep_t[i] = self.time
        self.terminal_dep_t[i] = self.time
        if int(self.time):
            headway = self.dep_t[i] - self.last_bus_time[self.last_stop[i]]
        else:
            headway = HEADWAY
        self.recorded_headway[self.last_stop[i]].append(headway)
        pickups = self.get_pax(headway)
        self.load[i] += pickups
        req_pickups = len(self.o_reqs[i])
        self.load[i] += req_pickups
        self.record_trajectories()
        self.last_bus_time[self.last_stop[i]] = self.dep_t[i]
        runtime = self.get_travel_time()
        self.next_instance_time[i] = self.dep_t[i] + runtime
        return

    def next_event(self):
        self.time = min(self.next_instance_time)
        self.bus_idx = self.next_instance_time.index(self.time)
        curr_route = self.routes[self.bus_idx]
        next_stop = self.next_stop[self.bus_idx]
        event_types = {ROUTE_STOPS[curr_route][0]: 0,
                       ROUTE_STOPS[curr_route][-1]: 2}
        if next_stop in event_types.keys():
            self.event_type = event_types[next_stop]
        else:
            self.event_type = 1
        return

    def reset_simulation(self):
        # SIMUL PARAMS
        self.routes = [FIXED_ROUTES_BASE[0] + '0', FIXED_ROUTES_BASE[1] + '0']
        scheduled_departures1 = [i for i in range(0, int(MAX_SIMUL_TIME), int(HEADWAY))]
        scheduled_departures2 = [i for i in range(0, int(MAX_SIMUL_TIME), int(HEADWAY))]
        self.next_departures = {FIXED_ROUTES_BASE[0]: scheduled_departures1,
                                FIXED_ROUTES_BASE[1]: scheduled_departures2}
        self.next_trip_ids = {FIXED_ROUTES_BASE[0]: [i for i in range(1, 1 + len(scheduled_departures1))],
                              FIXED_ROUTES_BASE[1]: [i for i in range(101, 101 + len(scheduled_departures2))]}
        route_bus1 = self.routes[0]
        route_bus2 = self.routes[1]
        self.trip_id = [self.next_trip_ids[route_bus1[0]][0], self.next_trip_ids[route_bus2[0]][0]]
        self.next_trip_ids[route_bus1[0]].pop(0)
        self.next_trip_ids[route_bus2[0]].pop(0)
        self.next_departures[route_bus1[0]].pop(0)
        self.next_departures[route_bus2[0]].pop(0)
        self.last_stop = [0, 0]
        self.next_stop = [ROUTE_STOPS[route_bus1][0], ROUTE_STOPS[route_bus2][0]]
        self.load = [0, 0]
        self.dep_t = [0, 0]
        self.arr_t = [0, 0]
        self.next_instance_time = [0, 0]
        self.terminal_dep_t = [0, 0]
        self.o_reqs = [[], []]
        self.d_reqs = [[], []]

        # bus_specific = [self.last_stop, self.next_stop,self.load,self.last_dep_t,self.next_arr_t,self.bus_at_terminal]
        # for i in bus_specific:
        #     i.extend([0]*N_BUSES)
        self.event_type = 0
        self.bus_idx = 0
        self.time = 0.0
        for r in [ROUTE_STOPS[FIXED_ROUTES_BASE[0] + '0'], ROUTE_STOPS[FIXED_ROUTES_BASE[1] + '0']]:
            for s in r:
                self.recorded_headway[s] = []
                self.last_bus_time[s] = 0.0
        # RL PARAMS
        self.s0 = [[], []]
        self.act = [[], []]
        self.sars = [[], []]
        # only updated when a sars is completed

        # RECORDINGS
        self.trip_trajectories = [[], []]
        self.recorded_delays = [[], []]
        self.recorded_trip_times = [[], []]

        # set first departure delay
        req_headway = HEADWAY
        for i in range(2):
            self.bus_idx = i
            [req1, req2], o_req, d_req = self.get_requests(req_headway)
            self.o_reqs[i] = o_req
            self.d_reqs[i] = d_req
            [s_req1, s_req2] = [min(req1, 2), min(req2, 2)]
            s0 = [FIXED_ROUTES_BASE.index(self.routes[i][0]), s_req1, s_req2, 0]
            self.s0[i] = s0
        self.bus_idx = 0
        return 0

    def prep(self, rewards=REWARD_WEIGHTS):
        self.next_event()
        i = self.bus_idx

        if self.event_type == 2:
            self.terminal_arrival(rewards)
            final_trips = [i > (2 * MAX_SIMUL_TIME) for i in self.next_instance_time]
            return False not in final_trips

        if self.event_type == 1:
            if self.next_stop[i] in DEV_STOPS[self.routes[i][0]]:
                self.dev_stop_arrival()
            else:
                self.fixed_stop_arrival()
            return self.prep(rewards)

        if self.event_type == 0:
            self.terminal_departure()
            return self.prep(rewards)

    def set_route(self, action, action2=0, double=0):
        i = self.bus_idx
        route_base = self.routes[i][0]
        self.act[i] = action
        self.routes[i] = route_base + str(action)
        self.update_served_requests()
        if double:
            self.bus_idx = 1 - i
            j = self.bus_idx
            route_base = self.routes[j][0]
            self.act[j] = action2
            self.routes[j] = route_base + str(action2)
            self.update_served_requests()
        return


# allowed_actions = {(0, 0): [0],
#                        (1, 0): [0, 1],
#                        (0, 1): [0, 2],
#                        (1, 1): [0, 1, 2, 3],
#                        (1, 2): [0, 1, 2, 3],
#                        (2, 1): [0, 1, 2, 3],
#                        (2, 0): [0, 1],
#                        (0, 2): [0, 2],
#                        (2, 2): [0, 1, 2, 3]}
#
# env = SimulationEnv()
# done = env.reset_simulation()
# reqs1 = [env.s0[0][1], env.s0[0][2]]
# reqs2 = [env.s0[1][1], env.s0[1][2]]
# next_legal_actions1 = allowed_actions[(reqs1[0], reqs1[1])]
# next_legal_actions2 = allowed_actions[(reqs2[0], reqs2[1])]
# act1 = sample(next_legal_actions1, 1)[0]
# act2 = sample(next_legal_actions2, 1)[0]
# env.set_route(act1, act2, double=1)
# count = 0
# while not done:
#     done = env.prep()
#     i = env.bus_idx
#     sars = env.sars[i][-1]
#     s1 = sars[3]
#     reqs1 = s1[1:3]
#     next_legal_actions = allowed_actions[(reqs1[0], reqs1[1])]
#     act = sample(next_legal_actions, 1)[0]
#     count += len(env.o_reqs[i])
#     env.set_route(act)
#
# results.write_csv_sars(env.sars)
# results.write_csv_trajectories(env.trip_trajectories)

