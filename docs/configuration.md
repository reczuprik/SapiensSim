# Configuration Guide

## Overview
SapiensSim uses a centralized configuration system in `config.py`. All simulation parameters are grouped into logical sections and can be validated for consistency.

## Configuration Sections

### World Configuration
```python
WORLD_WIDTH = 200           # World grid width
WORLD_HEIGHT = 200         # World grid height
RESOURCE_REGROWTH_RATE = 0.05  # Resource regeneration rate
```

### Population & Simulation
```python
MAX_POPULATION_SIZE = 300   # Maximum agents allowed
AGENT_INITIAL_COUNT = 100   # Starting agents
SIMULATION_TICKS = 5000     # Simulation duration
```

### Agent Biology
```python
MOVE_SPEED = 1.0           # Base movement speed
HUNGER_RATE = 0.5         # Hunger increase rate
STARVATION_RATE = 1.5     # Health loss when starving
```

### Technology
```python
TOOL_DECAY_ON_USE = 2.5    # Tool durability loss
SHELTER_DECAY_PER_TICK = 0.1  # Shelter durability loss
```

## Validation
Configuration can be validated using:
```python
from sapiens_sim.config import validate_config
errors, warnings = validate_config()
```

## Performance Tuning
Adjust optimization thresholds based on your hardware:
```python
OPTIMIZATION_THRESHOLDS = {
    'spatial_grid': 50,
    'batch_neat': 75,
    'vectorized_ops': 50,
    'lazy_world': 150,
}
```
