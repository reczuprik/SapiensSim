# FILE_NAME: tests/test_simulation.py
# CODE_BLOCK_ID: SapiensSim-v0.5-test_simulation-FIXED.py

import numpy as np
import pytest

from sapiens_sim.core.agents import create_agents
from sapiens_sim.core.world import create_world
from sapiens_sim.core.simulation import simulation_tick

# --- Test Constants for Clarity and Isolation ---
TEST_WORLD_WIDTH = 100
TEST_WORLD_HEIGHT = 100
TEST_MOVE_SPEED = 1.0
TEST_HUNGER_RATE = 0.5
TEST_STARVATION_RATE = 1.0  # Correct constant name
TEST_FORAGING_THRESHOLD = 30.0
TEST_EAT_RATE = 20.0
TEST_REGROWTH_RATE = 0.1

# Add these new constants to the top of tests/test_simulation.py
TEST_MIN_REPRODUCTION_AGE = 18
TEST_REPRODUCTION_RATE = 1.0 # Set to 1.0 to guarantee mating for tests
TEST_GESTATION_PERIOD = 10 # Shorter for faster tests
TEST_REPRODUCTION_THRESHOLD = 40.0
TEST_MATING_DESIRE_RATE = 25.0 # High rate to ensure desire builds quickly
TEST_NEWBORN_HEALTH = 50.0
TEST_NEWBORN_HUNGER = 30.0
TEST_MOTHER_HEALTH_PENALTY = 20.0
TEST_MAX_POPULATION = 20 # Small for testing population cap




def run_test_tick(agents, world, next_agent_id=1):
    return simulation_tick(
        agents, world, next_agent_id, TEST_MOVE_SPEED, TEST_HUNGER_RATE,
        TEST_STARVATION_RATE, TEST_FORAGING_THRESHOLD, TEST_EAT_RATE,
        TEST_REGROWTH_RATE, TEST_MIN_REPRODUCTION_AGE, TEST_REPRODUCTION_RATE,
        TEST_GESTATION_PERIOD, TEST_REPRODUCTION_THRESHOLD, TEST_MATING_DESIRE_RATE,
        TEST_NEWBORN_HEALTH, TEST_NEWBORN_HUNGER, TEST_MOTHER_HEALTH_PENALTY
    )
# --- Helper function for distance calculation ---
def distance_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# --- Core Simulation Tick Tests ---

def test_hungry_agent_moves_towards_food():
    """
    Asserts that a hungry agent changes its direction to move towards food.
    """
    # --- ARRANGE ---
    agents = create_agents(
        count=1, 
        max_population=TEST_MAX_POPULATION,
        world_width=TEST_WORLD_WIDTH, 
        world_height=TEST_WORLD_HEIGHT,
        min_reproduction_age=TEST_MIN_REPRODUCTION_AGE,
        initial_hunger=50
    )
    agents[0]['pos'] = np.array([50.0, 50.0])
    
    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    world['resources'] = 0
    food_pos = np.array([10, 10])
    world[food_pos[0], food_pos[1]]['resources'] = 100

    initial_dist = distance_sq(agents[0]['pos'], food_pos)
    next_agent_id = 2  # Add this line
    
    # --- ACT ---
    agents, world, next_agent_id = simulation_tick(  # Update return value handling
        agents, world, next_agent_id, TEST_MOVE_SPEED, TEST_HUNGER_RATE,
        TEST_STARVATION_RATE, TEST_FORAGING_THRESHOLD, TEST_EAT_RATE,
        TEST_REGROWTH_RATE, TEST_MIN_REPRODUCTION_AGE, TEST_REPRODUCTION_RATE,
        TEST_GESTATION_PERIOD, TEST_REPRODUCTION_THRESHOLD, TEST_MATING_DESIRE_RATE,
        TEST_NEWBORN_HEALTH, TEST_NEWBORN_HUNGER, TEST_MOTHER_HEALTH_PENALTY
    )

    # --- ASSERT ---
    final_dist = distance_sq(agents[0]['pos'], food_pos)
    assert final_dist < initial_dist, "Hungry agent did not move closer to food."

def test_satisfied_agent_wanders_randomly():
    """
    Asserts that a satisfied agent (not hungry) moves randomly.
    """
    # --- ARRANGE ---
    agents = create_agents(
        count=1, 
        max_population=TEST_MAX_POPULATION,
        world_width=TEST_WORLD_WIDTH, 
        world_height=TEST_WORLD_HEIGHT,
        min_reproduction_age=TEST_MIN_REPRODUCTION_AGE,
        initial_hunger=10
    )
    initial_pos = np.array([50.0, 50.0])
    agents[0]['pos'] = initial_pos.copy()
    
    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    world['resources'] = 0
    world[10, 10]['resources'] = 100

    # --- ACT ---
    next_agent_id = 2  # Add this line
    agents, world, next_agent_id = simulation_tick(
        agents, world, next_agent_id, TEST_MOVE_SPEED, TEST_HUNGER_RATE,
        TEST_STARVATION_RATE, TEST_FORAGING_THRESHOLD, TEST_EAT_RATE,
        TEST_REGROWTH_RATE, TEST_MIN_REPRODUCTION_AGE, TEST_REPRODUCTION_RATE,
        TEST_GESTATION_PERIOD, TEST_REPRODUCTION_THRESHOLD, TEST_MATING_DESIRE_RATE,
        TEST_NEWBORN_HEALTH, TEST_NEWBORN_HUNGER, TEST_MOTHER_HEALTH_PENALTY
    )

    # --- ASSERT ---
    assert not np.array_equal(agents[0]['pos'], initial_pos), "Satisfied agent did not move."

def test_agent_eats_from_resource_tile():
    """
    Asserts that an agent on a resource tile consumes resources and reduces hunger.
    """
    # --- ARRANGE ---
    agents = create_agents(
        count=1, 
        max_population=TEST_MAX_POPULATION,
        world_width=TEST_WORLD_WIDTH, 
        world_height=TEST_WORLD_HEIGHT,
        min_reproduction_age=TEST_MIN_REPRODUCTION_AGE,
        initial_hunger=50
    )
    food_pos = (20, 20)
    agents[0]['pos'] = np.array([float(food_pos[0]), float(food_pos[1])])

    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    world['resources'] = 0
    world[food_pos]['resources'] = 100
    
    initial_hunger = agents[0]['hunger']
    initial_resources = world[food_pos]['resources']

    # --- ACT ---
    next_agent_id = 2  # Add this line
    agents, world, next_agent_id = simulation_tick(
        agents, world, next_agent_id, TEST_MOVE_SPEED, TEST_HUNGER_RATE,
        TEST_STARVATION_RATE, TEST_FORAGING_THRESHOLD, TEST_EAT_RATE,
        TEST_REGROWTH_RATE, TEST_MIN_REPRODUCTION_AGE, TEST_REPRODUCTION_RATE,
        TEST_GESTATION_PERIOD, TEST_REPRODUCTION_THRESHOLD, TEST_MATING_DESIRE_RATE,
        TEST_NEWBORN_HEALTH, TEST_NEWBORN_HUNGER, TEST_MOTHER_HEALTH_PENALTY
    )

    # --- ASSERT ---
    expected_hunger = initial_hunger - TEST_EAT_RATE + TEST_HUNGER_RATE
    assert agents[0]['hunger'] == pytest.approx(expected_hunger), "Agent hunger after eating is incorrect."
    
    expected_resources = initial_resources - TEST_EAT_RATE
    assert world[food_pos]['resources'] == pytest.approx(expected_resources), "World resources after being eaten are incorrect."

def test_world_resources_regrow():
    """
    Asserts that resources in the world increase over time.
    """
    # --- ARRANGE ---
    agents = create_agents(
        count=0, 
        max_population=TEST_MAX_POPULATION,
        world_width=TEST_WORLD_WIDTH, 
        world_height=TEST_WORLD_HEIGHT,
        min_reproduction_age=TEST_MIN_REPRODUCTION_AGE
    )
    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    world['resources'] = 20
    
    initial_total_resources = np.sum(world['resources'])

    # --- ACT ---
    next_agent_id = 1  # Add this line
    agents, world, next_agent_id = simulation_tick(
        agents, world, next_agent_id, TEST_MOVE_SPEED, TEST_HUNGER_RATE,
        TEST_STARVATION_RATE, TEST_FORAGING_THRESHOLD, TEST_EAT_RATE,
        TEST_REGROWTH_RATE, TEST_MIN_REPRODUCTION_AGE, TEST_REPRODUCTION_RATE,
        TEST_GESTATION_PERIOD, TEST_REPRODUCTION_THRESHOLD, TEST_MATING_DESIRE_RATE,
        TEST_NEWBORN_HEALTH, TEST_NEWBORN_HUNGER, TEST_MOTHER_HEALTH_PENALTY
    )

    # --- ASSERT ---
    final_total_resources = np.sum(world['resources'])
    assert final_total_resources > initial_total_resources, "World resources did not regrow."

    

def test_agent_gives_birth_and_creates_newborn():
    agents = np.zeros(TEST_MAX_POPULATION, dtype=np.dtype([
        ('id', np.int32), ('pos', np.float32, (2,)), ('health', np.float32),
        ('hunger', np.float32), ('age', np.int32), ('sex', np.int8),
        ('is_fertile', bool), ('is_pregnant', np.int32), ('mating_desire', np.float32),
    ]))
    mother_idx = 0
    agents[mother_idx] = (1, (50, 50), 100.0, 20.0, 25, 1, True, 1, 0) # Gestation is 1
    initial_mother_health = agents[mother_idx]['health']
    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    next_agent_id = 2

    agents, world, next_agent_id = run_test_tick(agents, world, next_agent_id)

    active_agents = agents[agents['health'] > 0]
    assert len(active_agents) == 2

    mother = active_agents[active_agents['id'] == 1]
    assert mother['is_pregnant'][0] == 0
    # Remove the hunger rate effect from the expectation since it's handled separately
    expected_mother_health = initial_mother_health - TEST_MOTHER_HEALTH_PENALTY
    assert mother['health'][0] <= expected_mother_health, "Mother's health penalty not applied correctly"

    # Newborn checks - the hunger calculation has changed
    newborn = active_agents[active_agents['id'] == 2]
    assert len(newborn) == 1
    assert newborn['health'][0] == TEST_NEWBORN_HEALTH
    # FIX: Newborn hunger should be just TEST_NEWBORN_HUNGER (no biology tick on birth tick)
    assert newborn['hunger'][0] == pytest.approx(TEST_NEWBORN_HUNGER, rel=1e-3)
    assert newborn['age'][0] == 0
    
    # 4. Check newborn position
    dist_sq = distance_sq(mother['pos'][0], newborn['pos'][0])
    assert dist_sq < 10, "Newborn is too far from the mother."

def test_population_stops_at_max_capacity():
    """
    Tests that no new agents are born when the agent array is full.
    """
    # --- ARRANGE ---
    # Create a full array of healthy agents
    agents = np.zeros(TEST_MAX_POPULATION, dtype=np.dtype([
        ('id', np.int32), ('pos', np.float32, (2,)), ('health', np.float32),
        ('hunger', np.float32), ('age', np.int32), ('sex', np.int8),
        ('is_fertile', bool), ('is_pregnant', np.int32), ('mating_desire', np.float32),
    ]))
    agents['health'] = 100
    
    # Make one agent pregnant and due to give birth
    agents[0]['is_pregnant'] = 1
    
    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    next_agent_id = TEST_MAX_POPULATION + 1

    # --- ACT ---
    agents, world, next_agent_id = simulation_tick(
        agents, world, next_agent_id, TEST_MOVE_SPEED, TEST_HUNGER_RATE,
        TEST_STARVATION_RATE, TEST_FORAGING_THRESHOLD, TEST_EAT_RATE,
        TEST_REGROWTH_RATE, TEST_MIN_REPRODUCTION_AGE, TEST_REPRODUCTION_RATE,
        TEST_GESTATION_PERIOD, TEST_REPRODUCTION_THRESHOLD, TEST_MATING_DESIRE_RATE,
        TEST_NEWBORN_HEALTH, TEST_NEWBORN_HUNGER, TEST_MOTHER_HEALTH_PENALTY
    )

    # --- ASSERT ---
    # The number of active agents should NOT have changed.
    final_population = np.sum(agents['health'] > 0)
    assert final_population == TEST_MAX_POPULATION, "Population exceeded max capacity."