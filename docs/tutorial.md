# SapiensSim Tutorial

## Basic Simulation
```python
from sapiens_sim.main import run_simulation
from sapiens_sim.config import print_config_summary

# Print current configuration
print_config_summary()

# Run basic simulation
result = run_simulation()
```

## Custom World Generation
```python
from sapiens_sim.world import generate_world

world = generate_world(
    width=200,
    height=200,
    resource_density=0.7,
    terrain_complexity=0.5
)
```

## Agent Customization
```python
from sapiens_sim.agent_manager import AgentManager

# Create custom agent population
manager = AgentManager(max_population=300)
manager.create_initial_population(
    count=100,
    mean_health=80.0,
    mean_hunger=20.0
)
```

## Data Analysis
```python
from sapiens_sim.analysis import analyze_simulation

# Analyze simulation results
stats = analyze_simulation(result)
stats.plot_population_over_time()
stats.plot_resource_distribution()
```

## Advanced Features

### Custom Neural Networks
```python
from sapiens_sim.neat import CustomNEATConfig

neat_config = CustomNEATConfig(
    num_inputs=10,
    num_outputs=4,
    hidden_layers=[5, 3]
)
```

### Performance Optimization
```python
from sapiens_sim.config import get_optimization_config

# Enable CUDA acceleration
config = get_optimization_config()
config['enable_cuda'] = True
```
