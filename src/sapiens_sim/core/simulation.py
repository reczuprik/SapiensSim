# FILE_NAME: src/sapiens_sim/core/simulation.py
# CODE_BLOCK_ID: SapiensSim-v0.8-simulation.py

import numpy as np
from numba import jit
@jit(nopython=True)
def _handle_birth(
    agents: np.ndarray,
    mother_index: int,
    next_agent_id: int,
    newborn_health: float,
    newborn_hunger: float,
    mother_health_penalty: float
) -> int:
    """
    Handles the logic of a birth, activating an inactive agent slot.
    Returns the next available agent ID.
    """
    # Find the first inactive agent slot
    for i in range(len(agents)):
        if agents[i].health <= 0:
            # --- ACTIVATE NEWBORN ---
            newborn = agents[i]
            newborn.id = next_agent_id
            newborn.health = newborn_health
            newborn.hunger = newborn_hunger  # Set base hunger value
            newborn.age = 0
            newborn.sex = np.random.choice(np.array([0, 1], dtype=np.int8)) # Must pass array to choice
            newborn.is_fertile = False
            newborn.is_pregnant = 0
            newborn.mating_desire = 0
            
            # Place newborn next to the mother with a slight offset
            mother_pos = agents[mother_index].pos
            newborn.pos[0] = mother_pos[0] + np.random.randn()
            newborn.pos[1] = mother_pos[1] + np.random.randn()

            # --- PENALIZE MOTHER ---
            agents[mother_index].health -= mother_health_penalty

            # Return the next ID to be used
            return next_agent_id + 1
            
    # If no inactive slots are found, return the current ID (population is full)
    return next_agent_id

@jit(nopython=True, cache=True)
def simulation_tick(
    agents: np.ndarray,
    world: np.ndarray,
    next_agent_id: int,
    move_speed: float,
    hunger_rate: float,
    starvation_rate: float,
    foraging_threshold: float,
    eat_rate: float,
    resource_regrowth_rate: float,
    min_reproduction_age: int,
    reproduction_rate: float,
    gestation_period: int,
    reproduction_threshold: float,
    mating_desire_rate: float,
    newborn_health: float,
    newborn_hunger: float,
    mother_health_penalty: float
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
    # Track which agents are born this tick
    born_this_tick = np.zeros(len(agents), dtype=np.bool_)

    for i in range(len(agents)):
        agent = agents[i]

        if agent.health <= 0:
            continue

        # Skip aging and other updates for newborns in their birth tick
        if not born_this_tick[i]:
            # --- AGING and FERTILITY ---
            agent.age += 1 # Agents age each tick
            if not agent.is_fertile and agent.age >= min_reproduction_age:
                agent.is_fertile = True
            
        # --- GESTATION ---
        if agent.is_pregnant > 0:
            agent.is_pregnant -= 1
            if agent.is_pregnant == 0:
                # Mark the newborn as born this tick
                prev_next_agent_id = next_agent_id
                next_agent_id = _handle_birth(
                    agents, i, next_agent_id,
                    newborn_health, newborn_hunger, mother_health_penalty
                )
                # Find and mark the newborn
                for j in range(len(agents)):
                    if agents[j].id == prev_next_agent_id:
                        born_this_tick[j] = True
                        break

        # Skip biology updates for newborns in their birth tick
        if born_this_tick[i]:
            continue

        # --- MATING DESIRE ---
        # Desire increases for fertile, non-pregnant, healthy agents
        if agent.is_fertile and not agent.is_pregnant and agent.hunger < reproduction_threshold:
            agent.mating_desire += mating_desire_rate
            if agent.mating_desire > 100.0:
                agent.mating_desire = 100.0

        # --- REPRODUCTION (Symmetrical Search) ---
        # An agent with high desire will actively seek a mate
        if agent.is_fertile and not agent.is_pregnant and agent.mating_desire > 80.0:
            for j in range(len(agents)):
                if i == j: continue
                
                partner = agents[j]
                # Check if partner is a suitable mate (opposite sex, also fertile and ready)
                if (
                    partner.health > 0 and
                    agent.sex != partner.sex and
                    partner.is_fertile and
                    not partner.is_pregnant and
                    partner.mating_desire > 80.0
                ):
                    dist_sq = (agent.pos[0] - partner.pos[0])**2 + (agent.pos[1] - partner.pos[1])**2
                    if dist_sq < 25: # If within range
                        if np.random.rand() < reproduction_rate:
                            # --- SUCCESSFUL MATING ---
                            # Identify the female
                            if agent.sex == 1: # agent 'i' is female
                                agent.is_pregnant = gestation_period
                            else: # partner 'j' is female
                                partner.is_pregnant = gestation_period
                            
                            # Reset desire for BOTH partners
                            agent.mating_desire = 0
                            partner.mating_desire = 0
                            
                            break # Agent 'i' has mated, stop searching
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
            
    return agents, world, next_agent_id
    return agents, world, next_agent_id
