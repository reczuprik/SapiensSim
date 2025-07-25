# FILE_NAME: src/sapiens_sim/core/simulation.py
# CODE_BLOCK_ID: SapiensSim-v0.4-simulation.py

import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def simulation_tick(
    agents: np.ndarray,
    world_height: int,
    world_width: int,
    move_speed: float,
    hunger_rate: float,
    starvation_rate: float
):
    """
    Executes one tick of the simulation, including agent movement and biology.
    """
    # Loop through every agent. Numba will optimize this loop heavily.
    for i in range(len(agents)):
        agent = agents[i]

        # --- MOVEMENT ---
        # Generate a random direction vector
        # np.random.randn() gives a random number from a standard normal distribution
        direction_y = np.random.randn()
        direction_x = np.random.randn()
        
        # Normalize the vector to have a length of 1
        norm = np.sqrt(direction_y**2 + direction_x**2)
        if norm > 0:
            direction_y /= norm
            direction_x /= norm
            
        # Update position based on random direction and speed
        agent.pos[0] += direction_y * move_speed
        agent.pos[1] += direction_x * move_speed

        # Boundary check to keep agents within the world
        # Clamp Y position
        if agent.pos[0] < 0: agent.pos[0] = 0
        if agent.pos[0] > world_height - 1: agent.pos[0] = world_height - 1
        # Clamp X position
        if agent.pos[1] < 0: agent.pos[1] = 0
        if agent.pos[1] > world_width - 1: agent.pos[1] = world_width - 1

        # --- BIOLOGY ---
        # Increase hunger for all agents
        agent.hunger += hunger_rate
        if agent.hunger > 100.0:
            agent.hunger = 100.0
            
        # Apply starvation penalty if hunger is high
        if agent.hunger > 90.0:
            agent.health -= starvation_rate
        if agent.health < 0:
            agent.health = 0
            
    return agents