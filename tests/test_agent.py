# FILE_NAME: tests/test_agents.py
# CODE_BLOCK_ID: SapiensSim-v0.8-test_agents-FIXED.py

import numpy as np
import pytest

from sapiens_sim.core.agents import create_agents, SEX_MALE, SEX_FEMALE

@pytest.fixture
def agent_creation_params():
    """A pytest fixture to provide standard parameters for agent creation tests."""
    return {
        "count": 150,
        "max_population": 200,
        "world_width": 200,
        "world_height": 200,
        "min_reproduction_age": 18,
        "initial_health": 95.0,
        "initial_hunger": 10.0
    }

def test_create_agents_properties(agent_creation_params):
    """
    Tests the properties of a newly created agent population.
    """
    params = agent_creation_params
    agents = create_agents(**params)

    # --- ASSERT ---
    # 1. Check the total size of the pre-allocated array
    assert len(agents) == params["max_population"], "Total array size should equal max_population."

    # 2. Check the number of ACTIVE agents
    active_agents = agents[agents['health'] > 0]
    assert len(active_agents) == params["count"], "Incorrect number of active agents created."

    # 3. Check properties of the ACTIVE agents
    assert np.all(active_agents['health'] == params["initial_health"])
    assert np.all(active_agents['id'] < params["count"]) # IDs should be in the initial range  
def test_create_agents_data_integrity(agent_creation_params):
    """
    Tests the integrity and validity of the data in a new agent population.
    """
    # 1. Call the function we are testing
    agents = create_agents(**agent_creation_params)
    params = agent_creation_params

    # 2. Assert data integrity
    
    # Check that all IDs are unique
    assert len(np.unique(agents['id'])) == params["count"], "Agent IDs are not unique."

    # Check that positions are within world bounds
    assert np.all(agents['pos'][:, 0] >= 0) and np.all(agents['pos'][:, 0] <= params["world_height"]), "Agent Y-positions are out of bounds."
    assert np.all(agents['pos'][:, 1] >= 0) and np.all(agents['pos'][:, 1] <= params["world_width"]), "Agent X-positions are out of bounds."

    # Check that sex is assigned to a valid value
    allowed_sexes = [SEX_MALE, SEX_FEMALE]
    assert np.all(np.isin(agents['sex'], allowed_sexes)), "Agents have invalid sex values."
