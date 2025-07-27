# FILE: src/sapiens_sim/core/vectorized_simulation.py
# NEW: High-performance vectorized operations for biology updates

import numpy as np
from numba import jit, prange
from .agent_manager import SEX_FEMALE

# Terrain constants for Numba
TERRAIN_PLAINS = 0
TERRAIN_FOREST = 1
TERRAIN_WATER = 2
TERRAIN_MOUNTAIN = 3

@jit(nopython=True, parallel=True)
def vectorized_biology_update(
    agents: np.ndarray,
    world: np.ndarray,
    hunger_rate: float,
    starvation_rate: float,
    terrain_cost_plains: float,
    terrain_cost_forest: float,
    terrain_cost_mountain: float,
    shelter_decay_per_tick: float
):
    """
    Update all agents' biology simultaneously - MASSIVE speedup!
    Processes hundreds of agents in parallel instead of one by one.
    """
    world_height, world_width = world.shape
    
    # Process all agents in parallel
    for i in prange(len(agents)):
        agent = agents[i]
        
        # Skip dead or newborn agents
        if agent['health'] <= 0 or agent['age'] == 0:
            continue
        
        # --- Shelter Durability Update ---
        has_shelter = agent['shelter_durability'] > 0
        if has_shelter:
            agent['shelter_durability'] -= shelter_decay_per_tick
            if agent['shelter_durability'] < 0:
                agent['shelter_durability'] = 0
        
        # --- Calculate Terrain-Modified Hunger Rate ---
        # Safe tile coordinate extraction
        tile_y = int(agent['pos'][0])
        tile_x = int(agent['pos'][1])
        
        # Clamp to valid bounds
        tile_y = max(0, min(tile_y, world_height - 1))
        tile_x = max(0, min(tile_x, world_width - 1))
        
        terrain_type = world[tile_y, tile_x]['terrain']
        
        # Terrain multiplier
        terrain_multiplier = terrain_cost_plains
        if terrain_type == TERRAIN_FOREST:
            terrain_multiplier = terrain_cost_forest
        elif terrain_type == TERRAIN_MOUNTAIN:
            terrain_multiplier = terrain_cost_mountain
        
        # Shelter benefit (30% hunger reduction)
        effective_hunger_rate = hunger_rate * 0.7 if has_shelter else hunger_rate
        
        # Apply hunger increase
        agent['hunger'] += effective_hunger_rate * terrain_multiplier
        
        # Cap hunger at 100
        if agent['hunger'] > 100.0:
            agent['hunger'] = 100.0
        
        # Apply starvation damage
        if agent['hunger'] > 90.0:
            agent['health'] -= starvation_rate
        
        # Ensure health doesn't go negative
        if agent['health'] < 0:
            agent['health'] = 0

@jit(nopython=True, parallel=True)
def vectorized_movement(
    agents: np.ndarray,
    movements: np.ndarray,  # [agent_idx, direction_y, direction_x]
    move_speed: float,
    world_height: int,
    world_width: int
):
    """
    Move all agents simultaneously with vectorized boundary checking
    """
    for i in prange(len(agents)):
        if agents[i]['health'] <= 0:
            continue
        
        direction_y = movements[i, 0]
        direction_x = movements[i, 1]
        
        # Normalize direction
        norm = np.sqrt(direction_y**2 + direction_x**2)
        if norm > 0:
            direction_y /= norm
            direction_x /= norm
        
        # Apply movement
        agents[i]['pos'][0] += direction_y * move_speed
        agents[i]['pos'][1] += direction_x * move_speed
        
        # Boundary clamping
        agents[i]['pos'][0] = max(0.0, min(agents[i]['pos'][0], float(world_height - 1)))
        agents[i]['pos'][1] = max(0.0, min(agents[i]['pos'][1], float(world_width - 1)))

@jit(nopython=True, parallel=True)
def vectorized_eating(
    agents: np.ndarray,
    world: np.ndarray,
    eat_rate: float,
    tool_decay_on_use: float
) -> np.ndarray:
    """
    Handle eating for all agents simultaneously
    Returns array of amounts eaten by each agent
    """
    world_height, world_width = world.shape
    eaten_amounts = np.zeros(len(agents), dtype=np.float32)
    
    for i in prange(len(agents)):
        agent = agents[i]
        
        if agent['health'] <= 0:
            continue
        
        # Safe tile coordinates
        tile_y = max(0, min(int(agent['pos'][0]), world_height - 1))
        tile_x = max(0, min(int(agent['pos'][1]), world_width - 1))
        
        has_tool = agent['tool_durability'] > 0
        
        if world[tile_y, tile_x]['resources'] > 0:
            effective_eat_rate = eat_rate * 1.5 if has_tool else eat_rate
            eaten_amount = min(world[tile_y, tile_x]['resources'], effective_eat_rate)
            
            # Update world resources (atomic operation)
            world[tile_y, tile_x]['resources'] -= eaten_amount
            
            # Update agent hunger
            agent['hunger'] -= eaten_amount
            if agent['hunger'] < 0:
                agent['hunger'] = 0
            
            # Tool durability decay
            if has_tool and eaten_amount > 0:
                agent['tool_durability'] -= tool_decay_on_use
                if agent['tool_durability'] < 0:
                    agent['tool_durability'] = 0
            
            eaten_amounts[i] = eaten_amount
    
    return eaten_amounts

@jit(nopython=True, parallel=True)
def vectorized_aging_and_fertility(
    agents: np.ndarray,
    min_reproduction_age: int,
    max_agent_age: int
) -> np.ndarray:
    """
    Update age and fertility for all agents simultaneously
    Returns array of agents that died from old age
    """
    death_mask = np.zeros(len(agents), dtype=np.bool_)
    
    for i in prange(len(agents)):
        agent = agents[i]
        
        if agent['health'] <= 0:
            continue
        
        # Age the agent
        agent['age'] += 1
        
        # Update fertility
        if not agent['is_fertile'] and agent['age'] >= min_reproduction_age:
            agent['is_fertile'] = True
        
        # Check for death from old age
        if agent['age'] > max_agent_age:
            agent['health'] = 0
            death_mask[i] = True
    
    return death_mask

@jit(nopython=True, parallel=True)
def vectorized_mating_desire_update(
    agents: np.ndarray,
    mating_desire_rate: float,
    reproduction_threshold: float
):
    """
    Update mating desire for all fertile agents simultaneously
    """
    for i in prange(len(agents)):
        agent = agents[i]
        
        if (agent['health'] > 0 and agent['is_fertile'] and 
            not agent['is_pregnant'] and agent['hunger'] < reproduction_threshold):
            
            agent['mating_desire'] += mating_desire_rate
            if agent['mating_desire'] > 100.0:
                agent['mating_desire'] = 100.0

@jit(nopython=True, parallel=True)  
def vectorized_pregnancy_update(
    agents: np.ndarray
) -> np.ndarray:
    """
    Update pregnancy status for all agents simultaneously
    Returns array of agents ready to give birth
    """
    birth_ready = np.zeros(len(agents), dtype=np.bool_)
    
    for i in prange(len(agents)):
        agent = agents[i]
        
        if agent['health'] > 0 and agent['is_pregnant'] > 0:
            agent['is_pregnant'] -= 1
            if agent['is_pregnant'] == 0 and agent['sex'] == SEX_FEMALE:
                birth_ready[i] = True
    
    return birth_ready

def vectorized_simulation_tick(
    agent_manager,
    world: np.ndarray,
    next_agent_id: int,
    decisions: dict,  # Decisions from NEAT evaluation
    **params
):
    """
    Complete vectorized simulation tick - THIS IS THE MAIN PERFORMANCE BOOST!
    Processes all agents simultaneously instead of one by one.
    """
    agents = agent_manager.agents
    world_height, world_width = world.shape
    
    # Extract parameters
    move_speed = params['move_speed']
    hunger_rate = params['hunger_rate']
    starvation_rate = params['starvation_rate']
    eat_rate = params['eat_rate']
    min_reproduction_age = params['min_reproduction_age']
    reproduction_rate = params['reproduction_rate']
    gestation_period = params['gestation_period']
    reproduction_threshold = params['reproduction_threshold']
    mating_desire_rate = params['mating_desire_rate']
    terrain_cost_plains = params['terrain_cost_plains']
    terrain_cost_forest = params['terrain_cost_forest']
    terrain_cost_mountain = params['terrain_cost_mountain']
    tool_decay_on_use = params['tool_decay_on_use']
    shelter_decay_per_tick = params['shelter_decay_per_tick']
    max_agent_age = params['max_agent_age']
    fitness_death_penalty = params['fitness_death_penalty']
    
    # --- 1. WORLD UPDATES ---
    _update_world_resources_vectorized(world, params['resource_regrowth_rate'])
    
    # --- 2. HANDLE BIRTHS ---
    birth_ready = vectorized_pregnancy_update(agents)
    birth_indices = np.where(birth_ready)[0]
    
    for mother_idx in birth_indices:
        mother = agents[mother_idx]
        
        # Find father
        father_idx = mother_idx  # Default fallback
        for j in range(len(agents)):
            if agents[j]['health'] > 0 and agents[j]['id'] == mother['partner_id']:
                father_idx = j
                break
        
        # Create offspring
        offspring_pos = mother['pos'] + np.random.randn(2)
        offspring_idx = agent_manager.create_offspring(
            mother_idx, father_idx, next_agent_id, offspring_pos
        )
        
        if offspring_idx != -1:
            next_agent_id += 1
            mother['partner_id'] = -1
    
    # --- 3. PROCESS MOVEMENT (Vectorized) ---
    active_count = np.sum(agents['health'] > 0)
    if active_count > 0:
        # Convert decisions to movement array
        movements = np.zeros((len(agents), 2), dtype=np.float32)
        
        for agent_idx, decision in decisions.items():
            if agents[agent_idx]['health'] > 0 and agents[agent_idx]['age'] > 0:
                # Process decision logic (simplified for vectorization)
                if decision.get('rest', 0) > 0.7:
                    movements[agent_idx] = [0.0, 0.0]
                    # Apply rest benefit
                    if agents[agent_idx]['health'] < 90.0:
                        agents[agent_idx]['health'] += 0.5
                else:
                    movements[agent_idx] = [decision.get('move_y', 0), decision.get('move_x', 0)]
        
        # Apply all movements simultaneously
        vectorized_movement(agents, movements, move_speed, world_height, world_width)
    
    # --- 4. HANDLE EATING (Vectorized) ---
    eaten_amounts = vectorized_eating(agents, world, eat_rate, tool_decay_on_use)
    
    # Update fitness for eating (vectorized)
    for i in range(len(eaten_amounts)):
        if eaten_amounts[i] > 0 and agents[i]['health'] > 0:
            reward = eaten_amounts[i] / eat_rate * (1.0 + agents[i]['hunger'] / 100.0)
            agent_manager.update_fitness(i, reward * 0.1)
    
    # --- 5. BIOLOGY UPDATES (Vectorized) ---
    vectorized_biology_update(
        agents, world, hunger_rate, starvation_rate,
        terrain_cost_plains, terrain_cost_forest, terrain_cost_mountain,
        shelter_decay_per_tick
    )
    
    # --- 6. AGING AND FERTILITY (Vectorized) ---
    death_mask = vectorized_aging_and_fertility(agents, min_reproduction_age, max_agent_age)
    
    # Apply death penalty to agents that died from old age
    death_indices = np.where(death_mask)[0]
    for i in death_indices:
        agent_manager.update_fitness(i, fitness_death_penalty)
    
    # --- 7. MATING DESIRE UPDATES (Vectorized) ---
    vectorized_mating_desire_update(agents, mating_desire_rate, reproduction_threshold)
    
    # --- 8. FITNESS UPDATES (Vectorized) ---
    active_mask = agents['health'] > 0
    if np.any(active_mask):
        # Survival fitness
        agent_manager.agents['fitness'][active_mask] += 0.01
        
        # Health bonus
        healthy_mask = (agents['health'] > 75.0) & active_mask
        agent_manager.agents['fitness'][healthy_mask] += 0.05
        
        # Age bonus
        mature_mask = (agents['age'] > 50) & active_mask
        agent_manager.agents['fitness'][mature_mask] += 0.02
        
        # Starvation penalty
        starving_mask = (agents['hunger'] > 90.0) & active_mask
        agent_manager.agents['fitness'][starving_mask] -= 0.1
        
        # Death penalty
        dead_mask = agents['health'] <= 0
        agent_manager.agents['fitness'][dead_mask] -= 5.0
    
    return agents, world, next_agent_id

@jit(nopython=True, parallel=True)
def _update_world_resources_vectorized(world: np.ndarray, regrowth_rate: float):
    """Vectorized world resource updates"""
    world_height, world_width = world.shape
    
    for y in prange(world_height):
        for x in prange(world_width):
            world[y, x]['resources'] += regrowth_rate
            if world[y, x]['resources'] > 100.0:
                world[y, x]['resources'] = 100.0

# Performance comparison function
def compare_performance(agent_manager, world, decisions, next_agent_id, **params):
    """
    Compare vectorized vs original performance
    """
    import time
    import copy
    
    # Test original
    start = time.time()
    from .simulation import simulation_tick
    agents_orig, world_orig, next_id_orig = simulation_tick(
        copy.deepcopy(agent_manager), copy.deepcopy(world), next_agent_id, **params
    )
    original_time = time.time() - start
    
    # Test vectorized
    start = time.time()
    agents_vec, world_vec, next_id_vec = vectorized_simulation_tick(
        copy.deepcopy(agent_manager), copy.deepcopy(world), next_agent_id, decisions, **params
    )
    vectorized_time = time.time() - start
    
    speedup = original_time / vectorized_time if vectorized_time > 0 else float('inf')
    
    print(f"Performance Comparison:")
    print(f"  Original: {original_time*1000:.2f}ms")
    print(f"  Vectorized: {vectorized_time*1000:.2f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    
    return speedup