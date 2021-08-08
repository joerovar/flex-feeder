import results


def begin_message(env,obs):
    print('BEGIN SIMULATION------')
    print(f'requesting action for trip {env.active_trips[env.trip_idx]} departing terminal')
    print(f'{obs[0]} request(s)\n{obs[1]} pax capacity\n{obs[2]} mins behind schedule')
    print(f'Time {env.time} mins')
    print('---------')
    return


def show_runtime(env,i,nr_events,action,obs):
    if i>=0.999*nr_events:
        if action:
            print(obs)
            print(f'action taken: deviate')
            run = env.active_trips_next_arr_t[env.trip_idx]-env.active_trips_last_dep_t[env.trip_idx]
            print(f'run time {round(run,1)}')
        else:
            print(obs)
            print(f'action taken: continue fixed')
            run = env.active_trips_next_arr_t[env.trip_idx] - env.active_trips_last_dep_t[env.trip_idx]
            print(f'run time {round(run, 1)}')
    print(f'next arrival time {env.active_trips_next_arr_t[env.trip_idx]}')
    print('----------')
    return


def message_action(env,next_obs):
    if env.trip_type == 1:
        print(f'reward received for trip {env.active_trips[env.trip_idx]} was {env.active_trips_reward[env.trip_idx]}')
        print(f'----------------------')
    print(f'active trips {env.active_trips}')
    if env.trip_type == -1:
        print(f'requesting action for trip {env.active_trips[env.trip_idx]} departing terminal')
    if env.trip_type == 1:
        print(f'requesting action for trip {env.active_trips[env.trip_idx]} departing'
              f' stop {env.active_trips_last_stop[env.trip_idx]}')
    print(f'{next_obs[0]} request(s)\n{next_obs[1]} pax capacity\n{next_obs[2]} mins behind schedule')
    print(f'Time {env.time} mins')
    print(f'Departing at {env.active_trips_last_dep_t[env.trip_idx]} mins')
    print('-----------------')
    return


def get_plots(env):
    first_relevant_trip=10
    last_relevant_trip=25
    trip_info = env.recorded_trip_info
    trip_info = results.get_only_trips(trip_info, first_relevant_trip, last_relevant_trip)
    results.write_simulation_trip_info(trip_info)
    results.plot_time_trajectories(trip_info)
    results.plot_load_over_time(trip_info)

    return

