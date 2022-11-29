# lcss-2022
## Producing figures for *All Stable Equilibria Have Improved Performance Guarantees in Submodular Maximization with Communication-Denied Agents*

EnvBase.py, SetCover.py, and SetCoverEnv.py contain base classes and other classes for the simulations in SetCoverSims.py

To produce figure 3 from the manuscript, call the function `do_multi_temp_simulation` from the file SetCoverSims.py.
The parameters that were used to produce the figure are as follows: random_seed = 0, k = 10, distance = 3,
num_zero_nodes = 3, num_actions = 200000, low_temp = -2.3, high_temp = 2.3, temp_steps = 60

To produce figure 4 from the manuscript, call the function `do_full_simulation` that is contained in SetCoverSims.py.
The parameters that were used to produce the figure are as follows: random_seed = 0, k = 10, num_zero_nodes = 3,
low_temp = -2.3, high_temp = 2.3, temp_steps = 60, distance_steps = 50, num_actions = 200000

If you use this code, please cite<br>
J. H. Seaton and P. N. Brown, "All Stable Equilibria Have Improved Performance Guarantees in Submodular Maximization With Communication-Denied Agents," in IEEE Control Systems Letters, vol. 6, pp. 2491-2496, 2022, doi: 10.1109/LCSYS.2022.3166748.

This material is based upon work supported by the National Science Foundation under Grant Number ECCS-2013779. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
