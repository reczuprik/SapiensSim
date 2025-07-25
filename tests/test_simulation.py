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

# --- Helper function for distance calculation ---
def distance_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# --- Core Simulation Tick Tests ---

def test_hungry_agent_moves_towards_food():
    """
    Asserts that a hungry agent changes its direction to move towards food.
    """
    # --- ARRANGE ---
    agents = create_agents(1, TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT, initial_hunger=50)
    agents[0]['pos'] = np.array([50.0, 50.0])
    
    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    world['resources'] = 0
    food_pos = np.array([10, 10])
    world[food_pos[0], food_pos[1]]['resources'] = 100

    initial_dist = distance_sq(agents[0]['pos'], food_pos)
    
    # --- ACT ---
    agents, world = simulation_tick(
        agents, world, TEST_MOVE_SPEED, TEST_HUNGER_RATE, TEST_STARVATION_RATE, # <-- FIX
        TEST_FORAGING_THRESHOLD, TEST_EAT_RATE, TEST_REGROWTH_RATE
    )

    # --- ASSERT ---
    final_dist = distance_sq(agents[0]['pos'], food_pos)
    assert final_dist < initial_dist, "Hungry agent did not move closer to food."

def test_satisfied_agent_wanders_randomly():
    """
    Asserts that a satisfied agent (not hungry) moves randomly.
    """
    # --- ARRANGE ---
    agents = create_agents(1, TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT, initial_hunger=10)
    initial_pos = np.array([50.0, 50.0])
    agents[0]['pos'] = initial_pos.copy()
    
    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    world['resources'] = 0
    world[10, 10]['resources'] = 100

    # --- ACT ---
    agents, world = simulation_tick(
        agents, world, TEST_MOVE_SPEED, TEST_HUNGER_RATE, TEST_STARVATION_RATE, # <-- FIX
        TEST_FORAGING_THRESHOLD, TEST_EAT_RATE, TEST_REGROWTH_RATE
    )

    # --- ASSERT ---
    assert not np.array_equal(agents[0]['pos'], initial_pos), "Satisfied agent did not move."

def test_agent_eats_from_resource_tile():
    """
    Asserts that an agent on a resource tile consumes resources and reduces hunger.
    """
    # --- ARRANGE ---
    agents = create_agents(1, TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT, initial_hunger=50)
    food_pos = (20, 20)
    agents[0]['pos'] = np.array([float(food_pos[0]), float(food_pos[1])])

    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    world['resources'] = 0
    world[food_pos]['resources'] = 100
    
    initial_hunger = agents[0]['hunger']
    initial_resources = world[food_pos]['resources']

    # --- ACT ---
    agents, world = simulation_tick(
        agents, world, TEST_MOVE_SPEED, TEST_HUNGER_RATE, TEST_STARVATION_RATE, # <-- FIX
        TEST_FORAGING_THRESHOLD, TEST_EAT_RATE, TEST_REGROWTH_RATE
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
    agents = create_agents(0, TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    world = create_world(TEST_WORLD_WIDTH, TEST_WORLD_HEIGHT)
    world['resources'] = 20
    
    initial_total_resources = np.sum(world['resources'])

    # --- ACT ---
    agents, world = simulation_tick(
        agents, world, TEST_MOVE_SPEED, TEST_HUNGER_RATE, TEST_STARVATION_RATE, # <-- FIX
        TEST_FORAGING_THRESHOLD, TEST_EAT_RATE, TEST_REGROWTH_RATE
    )

    # --- ASSERT ---
    final_total_resources = np.sum(world['resources'])
    assert final_total_resources > initial_total_resources, "World resources did not regrow."