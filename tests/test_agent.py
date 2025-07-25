# FILE_NAME: tests/test_agents.py
# CODE_BLOCK_ID: SapiensSim-v0.2-test_agents.py

import numpy as np
import pytest

from sapiens_sim.core.agents import create_agents, SEX_MALE, SEX_FEMALE

@pytest.fixture
def agent_creation_params():
    """A pytest fixture to provide standard parameters for agent creation tests."""
    return {
        "count": 150,
        "world_width": 200,
        "world_height": 200,
        "initial_health": 95.0,
        "initial_hunger": 10.0
    }

def test_create_agents_basic_properties(agent_creation_params):
    """
    Tests that agents are created with the correct basic properties.
    """
    # 1. Call the function we are testing using the fixture
    agents = create_agents(**agent_creation_params)
    params = agent_creation_params

    # 2. Assert the outcomes are what we expect
    
    # Check that the number of agents is correct
    assert len(agents) == params["count"], "Incorrect number of agents created."

    # Check that the dtype is what we expect
    expected_fields = ['id', 'pos', 'health', 'hunger', 'age', 'sex', 'is_fertile', 'is_pregnant']
    for field in expected_fields:
        assert field in agents.dtype.names, f"Agent dtype is missing '{field}' field."

    # Check initial values
    assert np.all(agents['health'] == params["initial_health"]), "Initial health not set correctly."
    assert np.all(agents['hunger'] == params["initial_hunger"]), "Initial hunger not set correctly."
    assert np.all(agents['is_fertile'] == True), "Agents should be initialized as fertile."
    
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
