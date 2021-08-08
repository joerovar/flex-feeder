from input import *
from random import sample


def trajectories_if_error(results):
    for element in results:
        stops = []
        trip_times = []
        i = 0  # count within single trip data
        for item in element:
            if item[0] == '00':
                stops.append(i)
                trip_times.append(item[2])
            if item[0] == '01':
                stops.append(i)
                trip_times.append(item[1])
            if (item[0] != '00') & (item[0] != '01'):
                stops.append(i)
                stops.append(i)
                trip_times.append(item[1])
                trip_times.append(item[2])
            i += 1
        plt.plot(trip_times, stops)
    plt.tight_layout()
    plt.savefig('out/trajectories_error.png')
    return


def get_idx_q_table(obs):
    obs = np.array(obs)
    ceil = np.divide(OBS_HIGH - OBS_LOW, STATE_STEP_SIZE).astype(int)
    floor = 0
    idx = np.divide(obs - OBS_LOW, STATE_STEP_SIZE)
    idx = np.around(idx, 0)
    idx = idx.astype(int)
    try:
        idx[3]
    except IndexError:
        print(idx)
        raise
    if idx[3] > ceil[3]:
        idx[3] = ceil[3]
    if idx[3] < 0:
        idx[3] = 0
    assert idx.all() >= floor
    assert (idx <= ceil).all()
    return idx


def consult_q_table(q_table, obs, allowed_actions):
    i = get_idx_q_table(obs)
    q_values = q_table[i[0], i[1], i[2], i[3]]
    q_allowed = q_values[allowed_actions]
    max_sub_idx = np.argmax(q_allowed)
    best_action = np.arange(len(q_values))[allowed_actions][max_sub_idx]

    return best_action


def update_q_table_(q_table, sars, alpha, gamma, dummy_table, next_allowed_actions):
    s0, a, r, s1 = sars
    assert r <= 0
    i = get_idx_q_table(s0)
    j = get_idx_q_table(s1)
    old_q = q_table[i[0], i[1], i[2], i[3], a]
    next_q_allowed = q_table[j[0], j[1], j[2], j[3]][next_allowed_actions]
    next_max = np.max(next_q_allowed)
    new_q = (1 - alpha) * old_q + alpha * (r + gamma * next_max)
    assert (r + gamma*next_max) <= 0, q_table[j[0], j[1], j[2], j[3]]
    q_table[i[0], i[1], i[2], i[3], a] = new_q
    if not dummy_table[i[0], i[1], i[2], i[3], a]:
        dummy_table[i[0], i[1], i[2], i[3], a] = 1

    return q_table, dummy_table


def q_learn_(env, initial_q_table, table_checker, reward_weights):
    q_table = np.copy(initial_q_table)
    dim = table_checker.size
    allowed_actions = {(0, 0): [0],
                       (1, 0): [0, 1],
                       (0, 1): [0, 2],
                       (1, 1): [0, 1, 2, 3],
                       (1, 2): [0, 1, 2, 3],
                       (2, 1): [0, 1, 2, 3],
                       (2, 0): [0, 1],
                       (0, 2): [0, 2],
                       (2, 2): [0, 1, 2, 3]}
    exploration_episodes = PERCENT_EPS_EXPLORE * TRAIN_EPISODES
    epsilon_slope = (ENDING_EPSILON - STARTING_EPSILON) / exploration_episodes
    epsilon = STARTING_EPSILON

    mean_ep_reward = []
    epsilons = []
    reject_ratio = []

    for i in range(TRAIN_EPISODES):
        tot_reqs = 0
        rejected_reqs = 0

        done = env.reset_simulation()
        act1 = sample(allowed_actions[(env.s0[0][1], env.s0[0][2])], 1)[0]
        act2 = sample(allowed_actions[(env.s0[1][1], env.s0[1][2])], 1)[0]

        reqs_received = len(env.o_reqs[0]) + len(env.o_reqs[1])
        env.set_route(act1, action2=act2, double=1) # here we discard the rejected reqs
        reqs_served = len(env.o_reqs[0]) + len(env.o_reqs[1])
        rejected_reqs += (reqs_received - reqs_served)
        tot_reqs += reqs_received

        ep_reward = []

        while not done:
            done = env.prep(rewards=reward_weights)
            idx = env.bus_idx
            sars = env.sars[idx][-1]

            s0 = sars[0]
            reqs0 = s0[1:3]
            s1 = sars[3]
            reqs1 = s1[1:3]
            next_legal_actions = allowed_actions[(reqs1[0], reqs1[1])]
            if reqs0[0] or reqs0[1]:
                q_table, table_checker = update_q_table_(q_table, sars, ALPHA, GAMMA, table_checker, next_legal_actions)
                ep_reward.append(sars[2])
            if not done:
                # S1 or env.s0[i] is the state that requires the next action
                if np.random.uniform(0, 1) > epsilon:
                    if reqs1[0] or reqs1[1]:
                        act = consult_q_table(q_table, s1, next_legal_actions)
                    else:
                        act = 0
                else:
                    act = sample(next_legal_actions, 1)[0]

                reqs_received = len(env.o_reqs[idx])
                env.set_route(act)
                reqs_served = len(env.o_reqs[idx])
                rejected_reqs += (reqs_received - reqs_served)
                tot_reqs += reqs_received

        if not (i % PULL_TRAIN_DATA_EVERY):
            ep_reward = np.array(ep_reward).mean()
            mean_ep_reward.append(ep_reward)

            reject_ratio.append(rejected_reqs/tot_reqs * 100)

            progress = np.count_nonzero(table_checker)/dim * 100 / float(2/3)
            print(f'{round(progress,1)} %')
            epsilons.append(100*epsilon)

        if epsilon >= ENDING_EPSILON:
            epsilon = STARTING_EPSILON + epsilon_slope * (i + 1)

    return q_table, mean_ep_reward, reject_ratio, epsilons


def q_test(env, q_table):
    allowed_actions = {(0, 0): [0],
                       (1, 0): [0, 1],
                       (0, 1): [0, 2],
                       (1, 1): [0, 1, 2, 3],
                       (1, 2): [0, 1, 2, 3],
                       (2, 1): [0, 1, 2, 3],
                       (2, 0): [0, 1],
                       (0, 2): [0, 2],
                       (2, 2): [0, 1, 2, 3]}

    recorded_headway = {}
    reject_ratio = []
    recorded_reqs = []
    recorded_rejected_reqs = []
    on_time_performance = [[], []]

    for r in [ROUTE_STOPS[FIXED_ROUTES_BASE[0] + '0'], ROUTE_STOPS[FIXED_ROUTES_BASE[1] + '0']]:
        for s in r:
            recorded_headway[s] = []
    recorded_delays = [[], []]
    recorded_trip_times = [[], []]
    for i in range(TEST_EPISODES):
        tot_reqs = 0
        rejected_reqs = 0
        done = env.reset_simulation()
        s0_1 = env.s0[0]
        s0_2 = env.s0[1]
        reqs1 = tuple(s0_1[1:3])
        reqs2 = tuple(s0_2[1:3])
        next_legal_actions1 = allowed_actions[reqs1]
        next_legal_actions2 = allowed_actions[reqs2]
        act1 = consult_q_table(q_table, s0_1, next_legal_actions1)
        act2 = consult_q_table(q_table, s0_2, next_legal_actions2)
        reqs_received = len(env.o_reqs[0]) + len(env.o_reqs[1])
        env.set_route(act1, action2=act2, double=1)  # here we discard the rejected reqs
        reqs_served = len(env.o_reqs[0]) + len(env.o_reqs[1])
        rejected_reqs += (reqs_received - reqs_served)
        tot_reqs += reqs_received
        while not done:
            done = env.prep()
            idx = env.bus_idx
            sars = env.sars[idx][-1]
            s1 = sars[3]
            reqs1 = tuple(s1[1:3])
            next_legal_actions = allowed_actions[reqs1]
            if not done:
                if reqs1[0] or reqs1[1]:
                    act = consult_q_table(q_table, s1, next_legal_actions)
                else:
                    act = 0
                reqs_received = len(env.o_reqs[idx])
                env.set_route(act)
                reqs_served = len(env.o_reqs[idx])
                rejected_reqs += (reqs_received - reqs_served)
                tot_reqs += reqs_received

        recorded_reqs.append(tot_reqs)
        recorded_rejected_reqs.append(rejected_reqs)
        recorded_delays[0].extend(env.recorded_delays[0])
        recorded_delays[1].extend(env.recorded_delays[1])
        recorded_trip_times[0].extend(env.recorded_trip_times[0])
        recorded_trip_times[1].extend(env.recorded_trip_times[1])
        j = 0
        for d in env.recorded_delays:
            delays = np.array(d)
            otp = (delays <= DELAY_TOLERANCE).sum() / delays.size * 100
            on_time_performance[j].append(otp)
            j += 1
        for h in env.recorded_headway:
            recorded_headway[h].extend(env.recorded_headway[h])
    return recorded_headway, recorded_delays, recorded_reqs, recorded_rejected_reqs, recorded_trip_times, on_time_performance


def test_do_nothing(env):
    recorded_headway = {}
    for r in [ROUTE_STOPS[FIXED_ROUTES_BASE[0] + '0'], ROUTE_STOPS[FIXED_ROUTES_BASE[1] + '0']]:
        for s in r:
            recorded_headway[s] = []
    recorded_delays = [[], []]
    recorded_trip_times = [[], []]
    on_time_performance = [[], []]
    for i in range(TEST_EPISODES):
        done = env.reset_simulation()
        env.set_route(0, action2=0, double=1)
        while not done:
            done = env.prep()
            env.set_route(0)
        recorded_delays[0].extend(env.recorded_delays[0])
        recorded_delays[1].extend(env.recorded_delays[1])
        recorded_trip_times[0].extend(env.recorded_trip_times[0])
        recorded_trip_times[1].extend(env.recorded_trip_times[1])
        for h in env.recorded_headway:
            recorded_headway[h].extend(env.recorded_headway[h])
        j = 0
        for d in env.recorded_delays:
            delays = np.array(d)
            otp = (delays <= DELAY_TOLERANCE).sum() / delays.size * 100
            on_time_performance[j].append(otp)
            j += 1
    return recorded_headway, recorded_delays, recorded_trip_times, on_time_performance


def smart_greedy_policy(s, policy_on_time, policy_min_requests):
    reqs = tuple(s[1:3])
    delay = s[3]

    on_time_condition = not round(delay)
    min_request_condition = reqs[0] == 2 or reqs[1] == 2
    acceptable_delay_condition = round(delay) < DELAY_TOLERANCE/60
    if on_time_condition:
        act = policy_on_time[reqs]
    elif min_request_condition and acceptable_delay_condition:
        act = policy_min_requests[reqs]
    else:
        act = 0

    return act


def test_smart_greedy(env):
    # this will be the allowed actions if you're on time (rounded delay is 0)
    actions_on_time = {(0, 0): 0,
                       (1, 0): 1,
                       (0, 1): 2,
                       (1, 1): 3,
                       (1, 2): 3,
                       (2, 1): 3,
                       (2, 0): 1,
                       (0, 2): 2,
                       (2, 2): 3}

    # this is the policy if you're below the delay tolerance
    actions_min_reqs = {(0, 0): 0,
                        (1, 0): 0,
                        (0, 1): 0,
                        (1, 1): 0,
                        (1, 2): 2,
                        (2, 1): 1,
                        (2, 0): 1,
                        (0, 2): 2,
                        (2, 2): 3}
    recorded_headway = {}
    on_time_performance = [[], []]
    for r in [ROUTE_STOPS[FIXED_ROUTES_BASE[0] + '0'], ROUTE_STOPS[FIXED_ROUTES_BASE[1] + '0']]:
        for s in r:
            recorded_headway[s] = []
    recorded_delays = [[], []]
    recorded_trip_times = [[], []]
    recorded_reqs = []
    recorded_rej_reqs = []
    for i in range(TEST_EPISODES):
        tot_reqs = 0
        rejected_reqs = 0
        done = env.reset_simulation()
        s0_1 = env.s0[0]
        act1 = smart_greedy_policy(s0_1, actions_on_time, actions_min_reqs)
        s0_2 = env.s0[1]
        act2 = smart_greedy_policy(s0_2, actions_on_time, actions_min_reqs)
        reqs_received = len(env.o_reqs[0]) + len(env.o_reqs[1])
        env.set_route(act1, action2=act2, double=1)  # here we discard the rejected reqs
        reqs_served = len(env.o_reqs[0]) + len(env.o_reqs[1])
        rejected_reqs += (reqs_received - reqs_served)
        tot_reqs += reqs_received
        while not done:
            done = env.prep()
            if not done:
                idx = env.bus_idx
                sars = env.sars[idx][-1]
                s1 = sars[3]
                act = smart_greedy_policy(s1, actions_on_time, actions_min_reqs)
                reqs_received = len(env.o_reqs[idx])
                env.set_route(act)
                reqs_served = len(env.o_reqs[idx])
                rejected_reqs += (reqs_received - reqs_served)
                tot_reqs += reqs_received
        recorded_reqs.append(tot_reqs)
        recorded_rej_reqs.append(rejected_reqs)
        recorded_delays[0].extend(env.recorded_delays[0])
        recorded_delays[1].extend(env.recorded_delays[1])
        recorded_trip_times[0].extend(env.recorded_trip_times[0])
        recorded_trip_times[1].extend(env.recorded_trip_times[1])
        j = 0
        for d in env.recorded_delays:
            delays = np.array(d)
            otp = (delays <= DELAY_TOLERANCE).sum() / delays.size * 100
            on_time_performance[j].append(otp)
            j += 1
        for h in env.recorded_headway:
            recorded_headway[h].extend(env.recorded_headway[h])
    return recorded_headway, recorded_delays, recorded_reqs, recorded_rej_reqs, recorded_trip_times, on_time_performance


