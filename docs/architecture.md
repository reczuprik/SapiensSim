# SapiensSim Architecture

## Core Components

### Agent Manager
- Manages population of agents
- Handles agent creation, death, and reproduction
- Maintains NEAT genomes and brains

### Simulation Core
- Manages world state
- Handles agent interactions
- Updates resources and environment

### NEAT Implementation
- Evolves neural networks for agent decision making
- Handles crossover and mutation
- Manages speciation and fitness

## Data Flow
```
World Creation → Agent Initialization → Simulation Loop
  ↳ Agent Decisions → World Updates → State Changes
```

## Performance Optimizations
- NumPy for vectorized operations
- Numba for JIT compilation
- Optimized data structures
