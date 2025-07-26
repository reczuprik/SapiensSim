# FILE_NAME: src/sapiens_sim/core/agents.py
# CODE_BLOCK_ID: SapiensSim-v0.8-agents.py


import numpy as np
from .. import config

# Define sex types as integer constants for performance
SEX_MALE = 0
SEX_FEMALE = 1
def create_agents(
    count: int,
    max_population: int,
    world_width: int,
    world_height: int,
    min_reproduction_age: int, # <-- THE FIX IS HERE
    initial_health: float = 100.0,
    initial_hunger: float = 0.0
) -> np.ndarray:
    """
    Creates the population of agents as a 1D NumPy structured array.

    Args:
        count (int): The number of agents to create.
        world_width (int): The width of the world, for placing agents.
        world_height (int): The height of the world, for placing agents.
        initial_health (float): The starting health for all agents.
        initial_hunger (float): The starting hunger for all agents.

    Returns:
        np.ndarray: A 1D array where each element is an agent.
    """
    # Define the data type (dtype) for a single agent.
    # This structure holds all the state for one agent.
    # We use precise types (like float32, int32) for memory efficiency.
    agent_dtype = np.dtype([
        ('id', np.int32),
        ('pos', np.float32, (2,)),
        ('health', np.float32),
        ('hunger', np.float32),
        ('age', np.int32),
        ('sex', np.int8),
        ('is_fertile', bool),
        ('is_pregnant', np.int32),
        ('mating_desire', np.float32),
    ])

    # Create the 1D array, initialized with zeros for all fields.
    agents = np.zeros(max_population, dtype=agent_dtype)

    # --- Initialize Agent Properties ---
    active_agents = agents[:count]

    # Assign unique IDs from 0 to count-1
    active_agents['id'] = np.arange(count, dtype=np.int32)
    
    # Assign random starting positions within the world boundaries
    active_agents['pos'][:, 0] = np.random.uniform(0, world_height, size=count) # Y-coordinates
    active_agents['pos'][:, 1] = np.random.uniform(0, world_width, size=count)  # X-coordinates

    # Set initial biological states
    active_agents['health'] = initial_health
    active_agents['hunger'] = initial_hunger
    active_agents['age'] = np.random.randint(15, 50, size=count) # Start as adults
    
    # Assign sex randomly (approx. 50/50 split)
    active_agents['sex'] = np.random.choice([SEX_MALE, SEX_FEMALE], size=count)
    
    # Set fertility based on age (for now, all adults are fertile)
    active_agents['is_fertile'] = active_agents['age'] >= min_reproduction_age

    print(f"{count} agents created.")
    print(f"Male count: {np.sum(active_agents['sex'] == SEX_MALE)}")
    print(f"Female count: {np.sum(active_agents['sex'] == SEX_FEMALE)}")
    print(f"Fertile agents: {np.sum(active_agents['is_fertile'])}")
    print(f"{max_population} agent slots created. {count} are active.")

    return agents