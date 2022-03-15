# Authors: Joshua Seaton, Philip Brown
# March 14, 2022
#
# The following source code was used to generate Figures 4 & 5 in the manuscript
# All Stable Equilibria Have Improved Performance Guarantees in Submodular Maximization
#     with Communication-Denied Agents

from SetCoverEnv import SetCoverGame, LogLinear, MarginalValue
from matplotlib import axes, scale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import exp


def plot_temp_data(x_data, y_data, k):
    plt.figure(1)
    plt.plot(x_data, y_data)
    plt.xscale("log")
    plt.xlabel("Temperature")
    plt.ylabel(r"Average Empirical Value of $W(a)\ /\ W\left(a^{opt}\right)$")
    plt.title(r'$\left|K\right|={}$'.format(k))
    plt.show()


def plot_poa_data(x_data, poa_data, labels, temp_data=None, temp_label=''):
    ax1: axes.Axes
    ax2: axes.Axes
    fig, ax1 = plt.subplots()
    for i in range(len(poa_data)):
        style = '-'
        if i == 0:
            style = '--'
        elif i == 1:
            style = 'k-'
        elif i == 2:
            style = 'r+'
        elif i == 3:
            style = '-.'
        ax1.plot(x_data, poa_data[i], style, label=labels[i])
    ax1.set_ylim(ymin=0, ymax=1)
    ax1.set_xlabel("Stability")
    ax1.set_ylabel(r"System Objective as a Proportion of Optimal")
    if temp_data is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel(temp_label)
        ax2.plot(x_data, temp_data)
    plt.legend(loc='upper left')
    plt.minorticks_on()
    plt.margins(x=0.0, y=0.0)
    plt.show()


def create_threshold_game(values: list, zero_nodes: int) -> SetCoverGame:
    gb = SetCoverGame.build(MarginalValue.Factory)
    i = 0
    for value in values:
        res = gb.create_resource(f'R{i}', value)
        agb = gb.create_agent(f'P{i}')
        ab = agb.create_action()
        ab.add_resource(res)
        ab.build()
        for j in range(zero_nodes):
            zero = gb.create_resource(f'R{i}_{j}', 0)
            ab = agb.create_action()
            ab.add_resource(zero)
            ab.build()
        if i > 0:
            ab = agb.create_action()
            ab.add_resource(gb.resource('R0'))
            ab.build()
            agent = agb.build()
            gb.add_information_edge(agent, gb.agent('P0'))
        else:
            agb.build()
        i += 1
    return gb.build()


def create_hi_lo_values(k: int, distance: float, max_value: float):
    hi_lo = [1.0]
    total_allowance = distance * k
    for i in range(k):
        next_value = min(max_value, total_allowance)
        hi_lo.append(next_value)
        total_allowance -= next_value
    return hi_lo


def create_linear_values(k: int, distance: float):
    center = 1.0 - distance / k
    error = 1.0 - center
    if center < .5:
        error = center
    start = center - error
    stop = center + error
    return [1.0] + np.linspace(start, stop, k).tolist()


def create_homogenous_values(k: int, distance: float) -> list:
    value = 1 - distance / k
    return [1.0] + [value for x in range(k)]


def simulate_temp_range(sim: SetCoverGame.Simulator, temperatures: np.ndarray, num_actions, seed=0) -> np.ndarray:
    obj_values = np.empty(num_actions)
    avg_obj_values = np.empty(len(temperatures))

    for i in range(len(temperatures)):
        sim.initialize(LogLinear(temperatures[i]), seed)
        while sim.t < num_actions:
            obj_values[sim.t] = sim.objective_function_value()
            sim.next()
        avg_obj_values[i] = np.average(obj_values[(num_actions // 4):])

        print('.', end='')
    return avg_obj_values


def do_full_simulation():
    # The following code segment produces Figure 5 in the manuscript.
    # To produce the figure, the following parameters were used in the simulation:
    # random_seed = 0, k = 10, num_zero_nodes = 3, low_temp = -2.3, high_temp = 2.3,
    # temp_steps = 60, distance_steps = 50, num_actions = 200000
    random_seed = 0
    k = 10
    num_zero_nodes = 3
    distance_steps = 50
    num_actions = 200000

    low_temp = -2.3
    high_temp = 2.3
    temp_steps = 60
    temperatures = np.logspace(low_temp, high_temp, temp_steps)

    num_plots = 4
    distances = np.linspace(0, k - 1, distance_steps)
    poa_data = np.empty((num_plots, len(distances)))
    sims_completed = 0

    print(f"\n   ***   {(num_plots - 3) * temp_steps * distance_steps} simulations will be executed.   ***\n")

    for d in range(len(distances)):
        distance = distances[d]

        resources_values = list()
        resources_values.append(create_homogenous_values(k, distance))
        # resources_values.append(create_linear_values(k, distance))
        # resources_values.append(create_hi_lo_values(k, distance, 1.0))

        poa_data[0, d] = 1 / (k + 1)
        poa_data[1, d] = 1 / (k + 1 - distance)

        for m in range(len(resources_values)):
            game = create_threshold_game(resources_values[m], num_zero_nodes)
            opt_obj_value = sum(resources_values[m])
            sim = game.simulator
            avg_obj_values = simulate_temp_range(sim, temperatures, num_actions, random_seed)
            sims_completed += temp_steps
            if sims_completed % (temp_steps * len(resources_values)) == 0:
                print(f'  ***  {sims_completed} simulations completed.  ***')

            # poa_data[2 + m, d] = np.max(avg_obj_values) / opt_obj_value
            poa_data[2, d] = np.min(avg_obj_values) / opt_obj_value
            poa_data[3, d] = np.max(avg_obj_values) / opt_obj_value
            # print(f'np.argmax: {np.argmax(avg_obj_values)}')
            # print(f'temperatures[np.argmax(avg_obj_values){temperatures[np.argmax(avg_obj_values)]}')
            # poa_data[3, d] = temperatures[np.argmax(avg_obj_values)]

    plot_poa_data(distances, poa_data, [r'$1\ /\ (1 + K)$',
                                        r'$1\ /\ \left(1 + K -D(G)\right)$',
                                        'Empirical Minimum Avg Value w/Noise',
                                        'Empirical Maximum Avg Value w/Noise'])

    return distances, poa_data


def do_fixed_temp_sim():
    k = 10
    num_zero_nodes = 3
    temp_steps = 50
    temperatures = np.logspace(-2.3, 2.3, temp_steps)
    temperatures = temperatures.reshape((temp_steps, 1))

    distance_steps = 50
    num_actions = 200

    num_plots = 3
    distances = np.linspace(0, k - 1, distance_steps)
    poa_data = np.empty((num_plots, len(distances)))
    to_analyze = np.empty((3, temp_steps), dtype=float)
    sims_completed = 0

    print(f"\n   ***   {(num_plots - 2) * distance_steps} simulations will be executed.   ***\n")
    for temperature in temperatures:
        i = 0
        for d in range(len(distances)):
            distance = distances[d]

            resources_values = list()
            resources_values.append(create_homogenous_values(k, distance))
            # resources_values.append(create_linear_values(k, distance))
            # resources_values.append(create_hi_lo_values(k, value, 1.0))

            poa_data[0, d] = 1 / (k + 1)

            for m in range(len(resources_values)):
                game = create_threshold_game(resources_values[m], num_zero_nodes)
                opt_obj_value = sum(resources_values[m])
                sim = game.simulator
                avg_obj_values = simulate_temp_range(sim, temperature, num_actions)
                sims_completed += 1
                if sims_completed % 50 == 0:
                    print(f'  ***  {sims_completed} simulations completed.  ***')

                # poa_data[2 + m, d] = np.max(avg_obj_values[1]) / opt_obj_value
                poa_data[2, d] = np.max(avg_obj_values) / opt_obj_value
                poa_data[1, d] = 1 / opt_obj_value
        to_analyze[0, i] = temperature[0]
        to_analyze[1, i] = np.average(poa_data[2])
        to_analyze[2, i] = np.average(poa_data[2] - poa_data[1])
        i += 1
    print(f'temperature at np.argmax of dimension 1: {to_analyze[0, np.argmax(to_analyze[1])]}')
    print(f'temperature at np.argmax of dimension 2: {to_analyze[0, np.argmax(to_analyze[2])]}')


def do_multi_temp_simulation():
    # To reproduce Figure 4, use the following parameters:
    # random_seed = 0, k = 10, distance = 3, num_zero_nodes = 3, num_actions = 200000,
    #    low_temp = -2.3, high_temp = 2.3, temp_steps = 60

    random_seed = 0
    k = 10
    distance = 3
    num_zero_nodes = 3
    num_actions = 200000

    low_temp = -2.3
    high_temp = 2.3
    temp_steps = 60
    temperatures = np.logspace(low_temp, high_temp, temp_steps)

    print(f"\n   ***   {temp_steps} simulations will be executed.   ***\n")
    resources_values = create_homogenous_values(k, distance)
    game = create_threshold_game(resources_values, num_zero_nodes)
    opt_obj_value = sum(resources_values)
    sim = game.simulator
    avg_obj_values = simulate_temp_range(sim, temperatures, num_actions, random_seed)
    print(f'  ***  {temp_steps} simulations completed.  ***')
    avg_obj_values = avg_obj_values / opt_obj_value

    print(f'Max avg objective function value as a ratio of optimal: {np.max(avg_obj_values)}')
    print(f'Min avg objective function value as a ratio of optimal: {np.min(avg_obj_values)}')
    # poa_data[2, d] = np.max(avg_obj_values) / opt_obj_value
    print(f'np.argmax: {np.argmax(avg_obj_values)}')
    print(f'temperatures[np.argmax(avg_obj_values)]: {temperatures[np.argmax(avg_obj_values)]}')
    # poa_data[3, d] = temperatures[np.argmax(avg_obj_values)]

    plot_temp_data(temperatures, avg_obj_values, k)


if __name__ == "__main__":
    # To produce Figure 4, call do_full_simulation()
    do_full_simulation()
    # do_fixed_temp_sim()
    # To produce Figure 3, call do_multi_temp_simulation()
    # do_multi_temp_simulation()
