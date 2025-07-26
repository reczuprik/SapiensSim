# FILE_NAME: tests/test_neuroevolution.py

import pytest
import numpy as np

# We need to import the new classes and functions we are testing
from sapiens_sim.core.agent_manager import AgentManager
from sapiens_sim.core.neat_brain import NEATGenome, NEATBrain, NUM_INPUTS

@pytest.fixture
def agent_manager_fixture():
    """
    Provides a freshly initialized AgentManager for tests.
    """
    # Use smaller numbers for faster testing
    manager = AgentManager(max_population=50)
    manager.create_initial_population(
        count=10,
        world_width=100,
        world_height=100,
        min_reproduction_age=18
    )
    return manager

def test_agent_manager_initialization(agent_manager_fixture):
    """
    Tests that the AgentManager initializes the correct number of agents,
    genomes, and brains.
    """
    manager = agent_manager_fixture
    initial_count = 10
    max_pop = 50
    
    # 1. Check array sizes
    assert len(manager.agents) == max_pop
    assert len(manager.genomes) == max_pop
    assert len(manager.brains) == max_pop
    
    # 2. Check that the initial agents have brains and genomes
    for i in range(initial_count):
        assert manager.agents[i]['health'] > 0, f"Agent {i} should be active."
        assert manager.genomes[i] is not None, f"Genome for agent {i} should not be None."
        assert manager.brains[i] is not None, f"Brain for agent {i} should not be None."
        
    # 3. Check that the remaining slots are empty
    for i in range(initial_count, max_pop):
        assert manager.agents[i]['health'] == 0, f"Agent {i} should be inactive."
        assert manager.genomes[i] is None, f"Genome for agent {i} should be None."
        assert manager.brains[i] is None, f"Brain for agent {i} should be None."

def test_neat_brain_evaluation(agent_manager_fixture):
    """
    Tests that a brain can be created and evaluated, returning an output
    of the correct shape.
    """
    manager = agent_manager_fixture
    
    # Get the brain of the first agent
    brain = manager.brains[0]
    assert brain is not None, "Test requires a valid brain."

    # Create a random input vector of the correct size
    # In the real simulation, these inputs would come from the agent's senses
    random_inputs = np.random.rand(NUM_INPUTS).astype(np.float32)
    
    # Activate the brain
    try:
        outputs = brain.evaluate(random_inputs)
    except Exception as e:
        pytest.fail(f"Brain evaluation failed with an exception: {e}")
        
    # Check the output
    assert outputs is not None, "Brain output should not be None."
    assert isinstance(outputs, np.ndarray), "Brain output should be a NumPy array."
    # The number of outputs is defined in your neat_brain.py file
    # We can import it or hardcode it here for the test
    from sapiens_sim.core.neat_brain import NUM_OUTPUTS
    assert len(outputs) == NUM_OUTPUTS, "Brain produced an incorrect number of outputs."