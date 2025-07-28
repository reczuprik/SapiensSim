"""
Configuration module for SapiensSim

This module contains all configuration parameters and validation logic for the simulation.
Parameters are grouped into logical sections and can be validated for consistency.

Sections:
- World Configuration: Basic world parameters
- Population & Simulation: Core simulation settings
- Agent Biology & Reproduction: Agent lifecycle parameters
- Fitness Function Weights: Evolution scoring parameters
- Terrain & Technology: Environmental factors
- Optimization Settings: Performance tuning parameters
"""

# --- World Configuration ---
WORLD_WIDTH = 200           # Width of the simulation world grid
WORLD_HEIGHT = 200          # Height of the simulation world grid
RESOURCE_REGROWTH_RATE = 0.05  # Rate at which food resources regenerate per tick (0-1)

# --- Population & Simulation ---
MAX_POPULATION_SIZE = 300   # Maximum number of agents allowed in simulation
AGENT_INITIAL_COUNT = 100   # Starting number of agents
SIMULATION_TICKS = 5000     # Total simulation duration in ticks
CULLING_INTERVAL = 100      # Frequency of dead agent cleanup
MAX_AGENT_AGE = 4000        # Maximum agent lifespan in ticks (40 "years" at 100 ticks/year)

# --- Agent Biology & Reproduction (Tuned for more challenge) ---
MOVE_SPEED = 1.0            # Base movement speed of agents
HUNGER_RATE = 0.5           # Rate at which agents become hungry per tick
STARVATION_RATE = 1.5       # Health loss rate when starving
EAT_RATE = 20.0            # Amount of hunger satisfied by one meal
FORAGING_THRESHOLD = 30.0   # Hunger level that triggers food-seeking behavior
MIN_REPRODUCTION_AGE = 18   # Minimum age for reproduction
REPRODUCTION_RATE = 0.02    # Base chance of successful reproduction
GESTATION_PERIOD = 20       # Time required for pregnancy
REPRODUCTION_THRESHOLD = 40.0  # Maximum hunger level for reproduction
MATING_DESIRE_RATE = 0.2    # Rate at which mating desire increases

# --- Newborn Configuration ---
NEWBORN_HEALTH = 50.0
NEWBORN_HUNGER = 30.0
MOTHER_HEALTH_PENALTY = 20.0

# --- Fitness Function Weights ---
FITNESS_SURVIVAL = 0.01
FITNESS_EATING = 0.1
FITNESS_REPRODUCTION = 2.0
FITNESS_HEALTH = 0.05
FITNESS_AGE = 0.02
FITNESS_STARVATION_PENALTY = -0.1
FITNESS_DEATH_PENALTY = -15.0

# --- Terrain Movement Costs ---
# Multiplier for hunger_rate based on terrain. 1.0 is normal cost.
TERRAIN_COST_PLAINS = 1.0
TERRAIN_COST_FOREST = 1.5   # Moving through dense forest is harder
TERRAIN_COST_MOUNTAIN = 5.0  # Moving through mountains is extremely costly

# --- Technology Durability ---
TOOL_DECAY_ON_USE = 2.5     # Tool loses 2.5% of its durability with each use
SHELTER_DECAY_PER_TICK = 0.1  # Shelter loses 0.1% of its durability each tick

# --- Optimization Settings ---
OPTIMIZATION_THRESHOLDS = {
    'spatial_grid': 50,       # Use spatial grid with 50+ agents
    'batch_neat': 75,         # Batch evaluation with 75+ agents  
    'vectorized_ops': 50,     # Vectorized operations with 50+ agents
    'lazy_world': 150,        # Lazy world updates with 150+ agents
}

# --- Performance Monitoring ---
ENABLE_PERFORMANCE_LOGGING = True
PERFORMANCE_LOG_INTERVAL = 500  # Log detailed performance every N ticks
ENABLE_OPTIMIZATION_SWITCHING = True  # Allow dynamic optimization switching

# FIXED: Configuration validation
def validate_config() -> tuple[list[str], list[str]]:
    """
    Validate configuration parameters for consistency and potential issues.
    
    Performs checks for:
    - Critical errors (invalid parameter values)
    - Performance warnings (suboptimal configurations)
    - Logical conflicts between parameters
    
    Returns:
        tuple[list[str], list[str]]: Lists of error and warning messages
    """
    errors = []
    warnings = []
    
    # Critical validations
    if MAX_POPULATION_SIZE < AGENT_INITIAL_COUNT:
        errors.append(f"MAX_POPULATION_SIZE ({MAX_POPULATION_SIZE}) must be >= AGENT_INITIAL_COUNT ({AGENT_INITIAL_COUNT})")
    
    if HUNGER_RATE <= 0:
        errors.append(f"HUNGER_RATE ({HUNGER_RATE}) must be positive")
        
    if EAT_RATE <= 0:
        errors.append(f"EAT_RATE ({EAT_RATE}) must be positive")
        
    if RESOURCE_REGROWTH_RATE < 0:
        errors.append(f"RESOURCE_REGROWTH_RATE ({RESOURCE_REGROWTH_RATE}) must be non-negative")
    
    # Performance warnings
    if HUNGER_RATE >= EAT_RATE:
        warnings.append(f"HUNGER_RATE ({HUNGER_RATE}) >= EAT_RATE ({EAT_RATE}) - agents may starve quickly")
        
    if RESOURCE_REGROWTH_RATE > 1.0:
        warnings.append(f"RESOURCE_REGROWTH_RATE ({RESOURCE_REGROWTH_RATE}) > 1.0 - resources may grow very fast")
        
    if SIMULATION_TICKS > 10000 and AGENT_INITIAL_COUNT > 200:
        warnings.append(f"Large simulation ({SIMULATION_TICKS} ticks, {AGENT_INITIAL_COUNT} agents) may be slow without CUDA")
    
    return errors, warnings

def get_optimization_config() -> dict:
    """
    Get the current optimization configuration.
    
    Returns:
        dict: Dictionary containing optimization thresholds and flags
    """
    return {
        'thresholds': OPTIMIZATION_THRESHOLDS,
        'enable_logging': ENABLE_PERFORMANCE_LOGGING,
        'log_interval': PERFORMANCE_LOG_INTERVAL,
        'enable_switching': ENABLE_OPTIMIZATION_SWITCHING
    }

def print_config_summary() -> None:
    """
    Print a formatted summary of current configuration values.
    Useful for debugging and verification.
    """
    print("\n=== SapiensSim Configuration ===")
    print(f"World Size: {WORLD_WIDTH}x{WORLD_HEIGHT}")
    print(f"Initial Population: {AGENT_INITIAL_COUNT}/{MAX_POPULATION_SIZE}")
    print(f"Simulation Length: {SIMULATION_TICKS} ticks")
    print("\nOptimization Settings:")
    for key, value in OPTIMIZATION_THRESHOLDS.items():
        print(f"  {key}: {value}+ agents")

# Auto-validate on import
if __name__ != "__main__":
    errors, warnings = validate_config()
    if errors:
        raise ValueError("Configuration errors found:\n" + "\n".join(f"  - {e}" for e in errors))
    if warnings and ENABLE_PERFORMANCE_LOGGING:
        print("Configuration warnings:")
        for w in warnings:
            print(f"  ⚠ {w}")

# FIXED: Export all configuration for easy importing
__all__ = [
    'WORLD_WIDTH', 'WORLD_HEIGHT', 'RESOURCE_REGROWTH_RATE',
    'MAX_POPULATION_SIZE', 'AGENT_INITIAL_COUNT', 'SIMULATION_TICKS', 'CULLING_INTERVAL', 'MAX_AGENT_AGE',
    'MOVE_SPEED', 'HUNGER_RATE', 'STARVATION_RATE', 'EAT_RATE', 'FORAGING_THRESHOLD',
    'MIN_REPRODUCTION_AGE', 'REPRODUCTION_RATE', 'GESTATION_PERIOD', 'REPRODUCTION_THRESHOLD', 'MATING_DESIRE_RATE',
    'NEWBORN_HEALTH', 'NEWBORN_HUNGER', 'MOTHER_HEALTH_PENALTY',
    'FITNESS_SURVIVAL', 'FITNESS_EATING', 'FITNESS_REPRODUCTION', 'FITNESS_HEALTH', 'FITNESS_AGE', 
    'FITNESS_STARVATION_PENALTY', 'FITNESS_DEATH_PENALTY',
    'TERRAIN_COST_PLAINS', 'TERRAIN_COST_FOREST', 'TERRAIN_COST_MOUNTAIN',
    'TOOL_DECAY_ON_USE', 'SHELTER_DECAY_PER_TICK',
    'OPTIMIZATION_THRESHOLDS', 'ENABLE_PERFORMANCE_LOGGING', 'PERFORMANCE_LOG_INTERVAL', 
    'ENABLE_OPTIMIZATION_SWITCHING', 'validate_config', 'get_optimization_config', 'print_config_summary'
]

