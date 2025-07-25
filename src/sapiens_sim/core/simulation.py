# FILE_NAME: src/sapiens_sim/core/simulation.py
# CODE_BLOCK_ID: SapiensSim-v0.5-simulation-FIXED-2.py

import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def simulation_tick(
    agents: np.ndarray,
    world: np.ndarray,
    move_speed: float,
    hunger_rate: float,
    starvation_rate: float,
    foraging_threshold: float,
    eat_rate: float,
    resource_regrowth_rate: float
):
    """
    Executes one tick of the simulation, including foraging, movement, and biology.
    """
    world_height, world_width = world.shape

    # --- WORLD UPDATE ---
    # Explicit loop for Numba compatibility
    for y in range(world_height):
        for x in range(world_width):
            world[y, x].resources += resource_regrowth_rate
            if world[y, x].resources > 100.0:
                world[y, x].resources = 100.0

    # --- AGENT UPDATE LOOP ---
    for i in range(len(agents)):
        agent = agents[i]

        if agent.health <= 0:
            continue

        direction_y, direction_x = 0.0, 0.0

        # --- BEHAVIOR ---
        if agent.hunger > foraging_threshold:
            # FORAGE: Find nearest food
            best_food_y, best_food_x = -1, -1
            min_dist_sq = -1
            for y in range(world_height):
                for x in range(world_width):
                    if world[y, x].resources > 10:
                        dist_sq = (agent.pos[0] - y)**2 + (agent.pos[1] - x)**2
                        if min_dist_sq == -1 or dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            best_food_y, best_food_x = y, x
            
            if best_food_y != -1:
                direction_y = best_food_y - agent.pos[0]
                direction_x = best_food_x - agent.pos[1]
        else:
            # WANDER: Move randomly
            direction_y = np.random.randn()
            direction_x = np.random.randn()

        # --- MOVEMENT ---
        norm = np.sqrt(direction_y**2 + direction_x**2)
        if norm > 0:
            direction_y /= norm
            direction_x /= norm
            
        agent.pos[0] += direction_y * move_speed
        agent.pos[1] += direction_x * move_speed

        # Boundary check using Numba-friendly explicit if-statements
        # THIS IS THE FIX
        if agent.pos[0] < 0:
            agent.pos[0] = 0
        elif agent.pos[0] > world_height - 1:
            agent.pos[0] = world_height - 1

        if agent.pos[1] < 0:
            agent.pos[1] = 0
        elif agent.pos[1] > world_width - 1:
            agent.pos[1] = world_width - 1

        # --- EATING ---
        tile_y, tile_x = int(agent.pos[0]), int(agent.pos[1])
        if world[tile_y, tile_x].resources > 0:
            eaten_amount = min(world[tile_y, tile_x].resources, eat_rate)
            world[tile_y, tile_x].resources -= eaten_amount
            agent.hunger -= eaten_amount
            if agent.hunger < 0: agent.hunger = 0

        # --- BIOLOGY ---
        agent.hunger += hunger_rate
        if agent.hunger > 100.0:
            agent.hunger = 100.0
            
        if agent.hunger > 90.0:
            agent.health -= starvation_rate
        if agent.health < 0:
            agent.health = 0
            
    return agents, world