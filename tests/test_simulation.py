# FILE_NAME: tests/test_simulation.py

import pytest
import numpy as np

from sapiens_sim.core.agent_manager import AgentManager
from sapiens_sim.core.world import create_world
from sapiens_sim.core.simulation import simulation_tick # The newly renamed function

# --- Test Constants ---
TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT = 100, 100
TEST_MOVE_SPEED, TEST_HUNGER_RATE, TEST_STARVATION_RATE = 1.0, 0.5, 1.0
TEST_FORAGING_THRESHOLD, TEST_EAT_RATE, TEST_REGROWTH_RATE = 30.0, 20.0, 0.1
TEST_MIN_REPRODUCTION_AGE, TEST_GESTATION_PERIOD = 18, 10
TEST_REPRODUCTION_RATE, TEST_REPRODUCTION_THRESHOLD = 1.0, 40.0
TEST_MATING_DESIRE_RATE = 25.0
TEST_NEWBORN_HEALTH, TEST_NEWBORN_HUNGER, TEST_MOTHER_HEALTH_PENALTY = 50.0, 30.0, 20.0
TEST_MAX_POPULATION = 20

# Helper function to run the simulation tick with all default test params
def run_test_tick(manager, world, next_id):
    return simulation_tick(
        agent_population=manager, world=world, next_agent_id=next_id,
        move_speed=TEST_MOVE_SPEED, hunger_rate=TEST_HUNGER_RATE,
        starvation_rate=TEST_STARVATION_RATE, foraging_threshold=TEST_FORAGING_THRESHOLD,
        eat_rate=TEST_EAT_RATE, resource_regrowth_rate=TEST_REGROWTH_RATE,
        min_reproduction_age=TEST_MIN_REPRODUCTION_AGE, reproduction_rate=TEST_REPRODUCTION_RATE,
        gestation_period=TEST_GESTATION_PERIOD, reproduction_threshold=TEST_REPRODUCTION_THRESHOLD,
        mating_desire_rate=TEST_MATING_DESIRE_RATE, newborn_health=TEST_NEWBORN_HEALTH,
        newborn_hunger=TEST_NEWBORN_HUNGER, mother_health_penalty=TEST_MOTHER_HEALTH_PENALTY
    )

@pytest.fixture
def manager_and_world():
    """Provides a manager and world for testing."""
    manager = AgentManager(max_population=TEST_MAX_POPULATION)
    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    return manager, world


def test_population_stops_at_max_capacity(manager_and_world):
    """
    Tests that no new agents are born when the agent array is full.
    """
    # --- ARRANGE ---
    manager, world = manager_and_world
    # Manually fill the entire population with healthy agents
    manager.agents['health'] = 100
    manager.agents['id'] = np.arange(TEST_MAX_POPULATION)
    # Make one agent pregnant and due to give birth
    manager.agents[0]['is_pregnant'] = 1
    next_agent_id = TEST_MAX_POPULATION + 1

    # --- ACT ---
    agents, world, next_agent_id = run_test_tick(manager, world, next_agent_id)

    # --- ASSERT ---
    final_population = np.sum(agents['health'] > 0)
    assert final_population == TEST_MAX_POPULATION, "Population exceeded max capacity."
def test_agent_gives_birth(manager_and_world):
    manager, world = manager_and_world
    
    # --- ARRANGE ---
    mother_idx, father_idx = 0, 1
    mother_id, father_id = 1, 2
    
    # 1. Manually create the parent agents' data in the numpy array
    manager.agents[mother_idx] = (mother_id, (50, 50), 100.0, 20.0, 25, 1, True, 1, father_id, 0, 10.0, 1)
    manager.agents[father_idx] = (father_id, (52, 52), 100.0, 20.0, 26, 0, True, 0, -1, 0, 10.0, 1)

    # 2. THE CRUCIAL FIX: Create and assign genomes for the parents
    from sapiens_sim.core.neat_brain import create_random_genome, NEATBrain
    mother_genome = create_random_genome()
    father_genome = create_random_genome()
    manager.genomes[mother_idx] = mother_genome
    manager.brains[mother_idx] = NEATBrain(mother_genome)
    manager.genomes[father_idx] = father_genome
    manager.brains[father_idx] = NEATBrain(father_genome)
    
    initial_mother_health = manager.agents[mother_idx]['health']
    next_agent_id = 3 # Newborn ID

    # --- ACT ---
    agents, world, next_agent_id = run_test_tick(manager, world, next_agent_id)

    # --- ASSERT ---
    active_agents = agents[agents['health'] > 0]
    
    # Population should now be 3 (mother, father, child)
    assert len(active_agents) == 3, "Population did not increase to 3."
    
    # Find the newborn in the array
    newborn_agent_data = active_agents[active_agents['id'] == 3]
    assert len(newborn_agent_data) == 1, "Newborn was not created correctly."

    # Find the index of the newborn to check its genome
    newborn_idx = np.where(agents['id'] == 3)[0][0]
    assert manager.genomes[newborn_idx] is not None, "Newborn genome was not created."
    assert manager.brains[newborn_idx] is not None, "Newborn brain was not created."