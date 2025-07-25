# FILE_NAME: src/sapiens_sim/core/simulation.py
# CODE_BLOCK_ID: SapiensSim-v0.3-simulation.py

import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def simulation_tick(agents: np.ndarray, hunger_rate: float, starvation_rate: float):
    """
    Executes one tick of the simulation.

    This function is JIT-compiled by Numba for high performance.
    The `nopython=True` argument means Numba will convert this entire function
    to highly optimized machine code, with no Python interpreter overhead.
    `cache=True` will save the compiled function to disk to speed up subsequent runs.

    Args:
        agents (np.ndarray): The NumPy structured array of agents.
        hunger_rate (float): The amount to increase hunger by.
        starvation_rate (float): The amount to decrease health by if starving.
    """
    # Loop through every agent. Numba will optimize this loop heavily.
    for i in range(len(agents)):
        # We can't access agents by field name (e.g., agents[i]['health'])
        # inside a Numba-jitted function in the same way.
        # It's better to work with the tuple representation.
        # This will be refactored later for clarity, but shows the core concept.
        
        # Increase hunger for all agents
        agents[i].hunger += hunger_rate
        
        # Clamp hunger at 100
        if agents[i].hunger > 100.0:
            agents[i].hunger = 100.0
            
        # Apply starvation penalty if hunger is high
        if agents[i].hunger > 90.0:
            agents[i].health -= starvation_rate
            
        # Clamp health at 0 - agent is dead but remains in the array for now
        if agents[i].health < 0:
            agents[i].health = 0
            
    return agents