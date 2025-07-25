# FILE_NAME: tests/test_simulation.py
# CODE_BLOCK_ID: SapiensSim-v0.3-test_simulation.py

import numpy as np
import pytest

from sapiens_sim.core.agents import create_agents
from sapiens_sim.core.simulation import simulation_tick

# Define test constants to make tests clear and independent
TEST_HUNGER_RATE = 0.5
TEST_STARVATION_RATE = 1.0

def test_simulation_tick_increases_hunger():
    """
    Tests that a single simulation tick correctly increases agent hunger.
    """
    # --- ARRANGE ---
    # Create a small population of 10 agents for this test
    agents = create_agents(count=10, world_width=50, world_height=50, initial_hunger=20.0)
    initial_hunger = agents['hunger'].copy() # Make a copy, not a reference

    # --- ACT ---
    # Run the function we are testing
    agents = simulation_tick(agents, TEST_HUNGER_RATE, TEST_STARVATION_RATE)

    # --- ASSERT ---
    # Check that hunger increased by the correct amount for all agents
    expected_hunger = initial_hunger + TEST_HUNGER_RATE
    assert np.all(agents['hunger'] == expected_hunger), "Hunger did not increase correctly."

def test_simulation_tick_applies_starvation():
    """
    Tests that starvation is correctly applied when hunger is high.
    """
    # --- ARRANGE ---
    # Create agents that are already very hungry
    agents = create_agents(count=10, world_width=50, world_height=50, initial_health=100.0, initial_hunger=95.0)
    initial_health = agents['health'].copy()

    # --- ACT ---
    # Run the function we are testing
    agents = simulation_tick(agents, TEST_HUNGER_RATE, TEST_STARVATION_RATE)

    # --- ASSERT ---
    # Check that health decreased due to starvation
    expected_health = initial_health - TEST_STARVATION_RATE
    assert np.all(agents['health'] == expected_health), "Starvation did not decrease health correctly."

def test_simulation_tick_does_not_apply_starvation_when_not_hungry():
    """
    Tests that health does NOT decrease when hunger is not high.
    """
    # --- ARRANGE ---
    # Create agents with low hunger
    agents = create_agents(count=10, world_width=50, world_height=50, initial_health=100.0, initial_hunger=50.0)
    initial_health = agents['health'].copy()

    # --- ACT ---
    # Run the function
    agents = simulation_tick(agents, TEST_HUNGER_RATE, TEST_STARVATION_RATE)

    # --- ASSERT ---
    # Check that health has NOT changed
    assert np.all(agents['health'] == initial_health), "Health should not decrease when not starving."