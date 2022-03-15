# lcss-2022
## Producing figures for *All Stable Equilibria Have Improved Performance Guarantees in Submodular Maximization with Communication-Denied Agents*

EnvBase.py, SetCover.py, and SetCoverEnv.py contain base classes and other classes for the simulations in SetCoverSims.py

To produce figure 3 from the manuscript, call the function `do_multi_temp_simulation` from the file SetCoverSims.py.
The parameters that were used to produce the figure are as follows: random_seed = 0, k = 10, distance = 3,
num_zero_nodes = 3, num_actions = 200000, low_temp = -2.3, high_temp = 2.3, temp_steps = 60

To produce figure 4 from the manuscript, call the function `do_full_simulation` that is contained in SetCoverSims.py.
The parameters that were used to produce the figure are as follows: random_seed = 0, k = 10, num_zero_nodes = 3,
low_temp = -2.3, high_temp = 2.3, temp_steps = 60, distance_steps = 50, num_actions = 200000