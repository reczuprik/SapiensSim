# SapiensSim Documentation

## Overview
SapiensSim is a sophisticated agent-based simulation platform that models early human societies using NEAT (NeuroEvolution of Augmenting Topologies) for agent decision making. The simulation focuses on survival, resource management, tool usage, and social interactions between agents in a dynamic environment.

## Key Features
- Neural network-based agent decision making
- Dynamic resource management and regeneration
- Complex agent interactions including mating and reproduction
- Tool crafting and shelter building mechanics
- Terrain-based movement costs
- Optimized performance using NumPy and Numba

## Table of Contents

1. [Installation](installation.md)
2. [Architecture](architecture.md)
3. [Core Components](components.md)
4. [NEAT Implementation](neat.md)
5. [Configuration Guide](configuration.md)
6. [Testing Guide](testing.md)
7. [API Reference](api/README.md)
8. [Tutorial](tutorial.md)

## Quick Start
```python
from sapiens_sim.main import run_simulation

# Basic simulation with default parameters
run_simulation()

# Custom simulation with specific parameters
run_simulation(
    world_size=(200, 200),
    initial_population=100,
    simulation_length=5000
)
```

## Requirements
- Python 3.8+
- NumPy
- Numba
- neat-python

## License
MIT License - See LICENSE file for details
