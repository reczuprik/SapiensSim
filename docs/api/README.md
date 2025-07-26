# API Reference

## Core Modules

### AgentManager
```python
class AgentManager:
    """Manages a population of agents with NEAT brains"""
    def __init__(self, max_population: int)
    def create_initial_population(self, count: int, ...)
    def create_offspring(self, parent1_idx: int, parent2_idx: int, ...)
```

### NEATBrain
```python
class NEATBrain:
    """Neural network brain that can be evaluated efficiently"""
    def __init__(self, genome: NEATGenome)
    def evaluate(self, inputs: np.ndarray) -> np.ndarray
```

### Simulation
```python
def simulation_tick(agent_population: AgentManager, world: np.ndarray, ...) -> tuple:
    """Executes one tick of the simulation"""
```

See individual module documentation for complete API details.
