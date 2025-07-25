# FILE_NAME: tests/test_simulation.py
# CODE_BLOCK_ID: SapiensSim-v0.4-test_simulation.py

import numpy as np
import pytest

from sapiens_sim.core.agents import create_agents
from sapiens_sim.core.simulation import simulation_tick

# --- Constants for test clarity ---
TEST_HUNGER_RATE = 0.5
TEST_STARVATION_RATE = 1.0
TEST_MOVE_SPEED = 1.0
TEST_WORLD_WIDTH = 100
TEST_WORLD_HEIGHT = 100


# --- Previous tests for biology (keep them) ---
def test_simulation_tick_increases_hunger():
    agents = create_agents(count=10, world_width=50, world_height=50, initial_hunger=20.0)
    initial_hunger = agents['hunger'].copy()
    agents = simulation_tick(agents, TEST_WORLD_HEIGHT, TEST_WORLD_WIDTH, TEST_MOVE_SPEED, TEST_HUNGER_RATE, TEST_STARVATION_RATE)
    expected_hunger = initial_hunger + TEST_HUNGER_RATE
    assert np.all(agents['hunger'] == expected_hunger)

def test_simulation_tick_applies_starvation():
    agents = create_agents(count=10, world_width=50, world_height=50, initial_health=100.0, initial_hunger=95.0)
    initial_health = agents['health'].copy()
    agents = simulation_tick(agents, TEST_WORLD_HEIGHT, TEST_WORLD_WIDTH, TEST_MOVE_SPEED, TEST_HUNGER_RATE, TEST_STARVATION_RATE)
    expected_health = initial_health - TEST_STARVATION_RATE
    assert np.all(agents['health'] == expected_health)

def test_simulation_tick_does_not_apply_starvation_when_not_hungry():
    agents = create_agents(count=10, world_width=50, world_height=50, initial_health=100.0, initial_hunger=50.0)
    initial_health = agents['health'].copy()
    agents = simulation_tick(agents, TEST_WORLD_HEIGHT, TEST_WORLD_WIDTH, TEST_MOVE_SPEED, TEST_HUNGER_RATE, TEST_STARVATION_RATE)
    assert np.all(agents['health'] == initial_health)


# --- NEW TESTS FOR MOVEMENT ---

def test_simulation_tick_moves_agents():
    """
    Tests that agents' positions change after a simulation tick.
    """
    # --- ARRANGE ---
    agents = create_agents(count=10, world_width=TEST_WORLD_WIDTH, world_height=TEST_WORLD_HEIGHT)
    # Important: make a deep copy of the original positions
    initial_positions = agents['pos'].copy()

    # --- ACT ---
    agents = simulation_tick(agents, TEST_WORLD_HEIGHT, TEST_WORLD_WIDTH, TEST_MOVE_SPEED, TEST_HUNGER_RATE, TEST_STARVATION_RATE)

    # --- ASSERT ---
    # Check that the new positions are NOT the same as the old ones.
    # np.array_equal is the correct way to compare two numpy arrays for equality.
    assert not np.array_equal(agents['pos'], initial_positions), "Agents did not move."

def test_movement_respects_world_boundaries():
    """
    Tests that agent movement is clamped to the world boundaries.
    """
    # --- ARRANGE ---
    # Create a few agents and run the simulation for many steps to ensure
    # they have plenty of opportunities to hit the walls.
    agents = create_agents(count=5, world_width=TEST_WORLD_WIDTH, world_height=TEST_WORLD_HEIGHT)
    
    # --- ACT ---
    # Run the simulation for 200 ticks
    for _ in range(200):
        agents = simulation_tick(agents, TEST_WORLD_HEIGHT, TEST_WORLD_WIDTH, TEST_MOVE_SPEED, TEST_HUNGER_RATE, TEST_STARVATION_RATE)

    # --- ASSERT ---
    # Check that all agent Y and X positions are within the valid range.
    # The valid range is from 0 up to (but not including) the dimension size.
    assert np.all(agents['pos'][:, 0] >= 0), "Agent Y position is below 0."
    assert np.all(agents['pos'][:, 0] < TEST_WORLD_HEIGHT), "Agent Y position is out of bounds."
    assert np.all(agents['pos'][:, 1] >= 0), "Agent X position is below 0."
    assert np.all(agents['pos'][:, 1] < TEST_WORLD_WIDTH), "Agent X position is out of bounds."