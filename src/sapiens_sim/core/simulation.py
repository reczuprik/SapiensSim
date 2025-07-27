# FILE_NAME: src/sapiens_sim/core/simulation.py
# CODE_BLOCK_ID: SapiensSim-NEAT-v1.0-simulation.py

import numpy as np
from numba import jit
from .agent_manager import AgentManager, SEX_FEMALE

# Define terrain constants here if they aren't already, so Numba can see them
TERRAIN_PLAINS = 0
TERRAIN_FOREST = 1
TERRAIN_WATER = 2
TERRAIN_MOUNTAIN = 3

def simulation_tick(
    agent_population: AgentManager,
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
    mother_health_penalty: float,
    max_agent_age: int,
    terrain_cost_plains: float,
    terrain_cost_forest: float,
    terrain_cost_mountain: float
) -> tuple:
    """
    Enhanced simulation tick that uses NEAT brains for agent decision making.
    This version follows a Think -> Act -> Update State logical flow.
    """
    agents = agent_population.agents
    world_height, world_width = world.shape

    _update_world_resources(world, resource_regrowth_rate)
    
    born_this_tick = np.zeros(len(agents), dtype=np.bool_)
 # --- PHASE 1: BIRTHS ---
    for i in range(len(agents)):
        agent = agents[i]
        if agent['health'] > 0 and agent['is_pregnant'] > 0:
            agent['is_pregnant'] -= 1
            if agent['is_pregnant'] == 0 and agent['sex'] == SEX_FEMALE:
                father_idx = -1
                for j in range(len(agents)):
                    if agents[j]['health'] > 0 and agents[j]['id'] == agent['partner_id']:
                        father_idx = j
                        break
                
                father_idx = i if father_idx == -1 else father_idx

                offspring_pos = agent['pos'] + np.random.randn(2)
                offspring_idx = agent_population.create_offspring(
                    i, father_idx, next_agent_id, offspring_pos
                )
                
                # Simplified logic: create_offspring now handles the health penalty.
                if offspring_idx != -1:
                    next_agent_id += 1
                    agent['partner_id'] = -1

    # --- PHASE 2: AGENT THINK/ACT/UPDATE ---
    for i in range(len(agents)):
        agent = agents[i]
        
        # An agent with age=0 is a newborn and does nothing for one tick.
        if agent['health'] <= 0 or agent['age'] == 0:
            continue

        # 1. THINK
        decision = agent_population.make_decision(i, world, agents)

        # 2. ACT (Movement, Eating, Conception)
        # ... (This logic is correct and remains unchanged) ...
        direction_y, direction_x = 0.0, 0.0
        # 1. First, check for the overriding "Rest" action.
        if decision['rest'] > 0.7 and decision['rest'] > decision['seek_food'] and decision['rest'] > decision['seek_mate']:
            # Winner-Take-All for Resting:
            direction_y = 0.0
            direction_x = 0.0
            if agent['health'] < 90.0:
                agent['health'] += 0.5
                agent_population.update_fitness(i, 0.02)
        # 2. If not resting, use your blended logic for all other actions.
        elif decision['seek_food'] > 0.5 and agent['hunger'] > foraging_threshold:
            food_direction = _get_food_direction(agent, world)
            direction_y = 0.7 * food_direction[0] + 0.3 * decision['move_y']
            direction_x = 0.7 * food_direction[1] + 0.3 * decision['move_x']
            if np.dot([direction_y, direction_x], food_direction) > 0:
                agent_population.update_fitness(i, 0.1)
        
        elif decision['seek_mate'] > 0.5 and agent['mating_desire'] > 50.0:
            mate_direction = _get_mate_direction(i, agents)
            direction_y = 0.6 * mate_direction[0] + 0.4 * decision['move_y']
            direction_x = 0.6 * mate_direction[1] + 0.4 * decision['move_x']
            if agent['mating_desire'] > 80.0:
                agent_population.update_fitness(i, 0.05)
        
        else:
            direction_y = decision['move_y']
            direction_x = decision['move_x']

        _move_agent(agent, direction_y, direction_x, move_speed, world_height, world_width)
        eaten_amount = _handle_eating(agent, world, eat_rate)
        if eaten_amount > 0:
            reward = eaten_amount / eat_rate * (1.0 + agent['hunger'] / 100.0)
            agent_population.update_fitness(i, reward * 0.1)

        if (agent['is_fertile'] and not agent['is_pregnant'] and 
            agent['mating_desire'] > 80.0 and decision['seek_mate'] > 0.6):
            
            mate_idx = _find_suitable_mate(i, agents)
            if mate_idx != -1 and np.random.rand() < reproduction_rate:
                if agent['sex'] == SEX_FEMALE:
                    female_agent = agent
                    father_id = agents[mate_idx]['id']
                else:
                    female_agent = agents[mate_idx]
                    father_id = agent['id']
                female_agent['is_pregnant'] = gestation_period
                female_agent['partner_id'] = father_id
                agent['mating_desire'] = 0
                agents[mate_idx]['mating_desire'] = 0
                agent_population.update_fitness(i, 2.0)
                agent_population.update_fitness(mate_idx, 2.0)

        # 3. UPDATE STATE (Biology, Age, Fitness)
        _update_agent_biology(agent,
                              world, 
                              hunger_rate,
                              starvation_rate,
                              terrain_cost_plains,
                              terrain_cost_forest,
                              terrain_cost_mountain)

        if agent['is_fertile'] and not agent['is_pregnant'] and agent['hunger'] < reproduction_threshold:
            agent['mating_desire'] += mating_desire_rate
            if agent['mating_desire'] > 100.0:
                agent['mating_desire'] = 100.0

        agent['age'] += 1
        if not agent['is_fertile'] and agent['age'] >= min_reproduction_age:
            agent['is_fertile'] = True

        agent_population.update_fitness(i, 0.01)
        if agent['health'] > 75.0: agent_population.update_fitness(i, 0.05)
        if agent['age'] > 50: agent_population.update_fitness(i, 0.02)
        if agent['hunger'] > 90.0: agent_population.update_fitness(i, -0.1)
        if agent['health'] <= 0: agent_population.update_fitness(i, -5.0)
    
    # We must age all living agents *after* the main loop, so newborns
    # from this tick become age 1 for the next tick.
    for i in range(len(agents)):
        if agents[i]['health'] > 0:
            agents[i]['age'] += 1

            
    return agents, world, next_agent_id
@jit(nopython=True)
def _update_world_resources(world: np.ndarray, regrowth_rate: float):
    """Update world resources (Numba optimized)"""
    world_height, world_width = world.shape
    for y in range(world_height):
        for x in range(world_width):
            world[y, x]['resources'] += regrowth_rate
            if world[y, x]['resources'] > 100.0:
                world[y, x]['resources'] = 100.0

def _get_food_direction(agent: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Get direction to nearest food source"""
    agent_pos = agent['pos']
    world_height, world_width = world.shape
    
    best_direction = np.array([0.0, 0.0])
    min_dist_sq = float('inf')
    
    # Sample subset of world for performance
    for _ in range(50):  # Check 50 random tiles
        y = np.random.randint(0, world_height)
        x = np.random.randint(0, world_width)
        
        if world[y, x]['resources'] > 10:
            dist_sq = (agent_pos[0] - y)**2 + (agent_pos[1] - x)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                direction = np.array([y - agent_pos[0], x - agent_pos[1]])
                norm = np.sqrt(direction[0]**2 + direction[1]**2)
                if norm > 0:
                    best_direction = direction / norm
    
    return best_direction

def _get_mate_direction(agent_idx: int, agents: np.ndarray) -> np.ndarray:
    """Get direction to nearest suitable mate"""
    agent = agents[agent_idx]
    best_direction = np.array([0.0, 0.0])
    min_dist_sq = float('inf')
    
    for i, other_agent in enumerate(agents):
        if (i != agent_idx and other_agent['health'] > 0 and 
            other_agent['sex'] != agent['sex'] and other_agent['is_fertile'] and
            not other_agent['is_pregnant'] and other_agent['mating_desire'] > 50.0):
            
            dist_sq = ((agent['pos'][0] - other_agent['pos'][0])**2 + 
                      (agent['pos'][1] - other_agent['pos'][1])**2)
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                direction = other_agent['pos'] - agent['pos']
                norm = np.sqrt(direction[0]**2 + direction[1]**2)
                if norm > 0:
                    best_direction = direction / norm
    
    return best_direction

def _find_suitable_mate(agent_idx: int, agents: np.ndarray) -> int:
    """Find a suitable mate within range"""
    agent = agents[agent_idx]
    
    for i, other_agent in enumerate(agents):
        if (i != agent_idx and other_agent['health'] > 0 and 
            other_agent['sex'] != agent['sex'] and other_agent['is_fertile'] and
            not other_agent['is_pregnant'] and other_agent['mating_desire'] > 80.0):
            
            dist_sq = ((agent['pos'][0] - other_agent['pos'][0])**2 + 
                      (agent['pos'][1] - other_agent['pos'][1])**2)
            
            if dist_sq < 25:  # Within mating range
                return i
    
    return -1

@jit(nopython=True)
def _move_agent(agent: np.ndarray, direction_y: float, direction_x: float, 
                move_speed: float, world_height: int, world_width: int):
    """Move an agent (Numba optimized)"""
    # Normalize direction
    norm = np.sqrt(direction_y**2 + direction_x**2)
    if norm > 0:
        direction_y /= norm
        direction_x /= norm
    
    # Apply movement
    agent['pos'][0] += direction_y * move_speed
    agent['pos'][1] += direction_x * move_speed
    
    # Boundary checking
    if agent['pos'][0] < 0:
        agent['pos'][0] = 0
    elif agent['pos'][0] > world_height - 1:
        agent['pos'][0] = world_height - 1
    
    if agent['pos'][1] < 0:
        agent['pos'][1] = 0
    elif agent['pos'][1] > world_width - 1:
        agent['pos'][1] = world_width - 1

@jit(nopython=True)
def _handle_eating(agent: np.ndarray, world: np.ndarray, eat_rate: float) -> float:
    """Handle agent eating (Numba optimized)"""
    tile_y, tile_x = int(agent['pos'][0]), int(agent['pos'][1])
    
    if world[tile_y, tile_x]['resources'] > 0:
        eaten_amount = min(world[tile_y, tile_x]['resources'], eat_rate)
        world[tile_y, tile_x]['resources'] -= eaten_amount
        agent['hunger'] -= eaten_amount
        if agent['hunger'] < 0:
            agent['hunger'] = 0
        return eaten_amount
    
    return 0.0

@jit(nopython=True)
def _update_agent_biology(
    agent: np.ndarray,
    world: np.ndarray, # <-- NEW
    hunger_rate: float,
    starvation_rate: float,
    # --- NEW TERRAIN COST PARAMS ---
    cost_plains: float,
    cost_forest: float,
    cost_mountain: float
):    
    """Update agent's biological state, with hunger cost modified by terrain."""
    
    # Determine the hunger cost for this tick
    current_tile_y = int(agent['pos'][0])
    current_tile_x = int(agent['pos'][1])
    terrain_type = world[current_tile_y, current_tile_x]['terrain']
    
    terrain_multiplier = 1.0
    if terrain_type == TERRAIN_PLAINS:
        terrain_multiplier = cost_plains
    elif terrain_type == TERRAIN_FOREST:
        terrain_multiplier = cost_forest
    elif terrain_type == TERRAIN_MOUNTAIN:
        terrain_multiplier = cost_mountain
    
    # Apply hunger cost
    agent['hunger'] += hunger_rate * terrain_multiplier

    """Update agent's biological state (Numba optimized)"""
    agent['hunger'] += hunger_rate
    if agent['hunger'] > 100.0:
        agent['hunger'] = 100.0
    
    if agent['hunger'] > 90.0:
        agent['health'] -= starvation_rate
    
    if agent['health'] < 0:
        agent['health'] = 0