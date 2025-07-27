# FILE_NAME: src/sapiens_sim/config.py
# FIXED: Removed duplicates and inconsistencies

# --- World Configuration ---
WORLD_WIDTH = 200
WORLD_HEIGHT = 200
# FIXED: Removed duplicate - using the more challenging value
RESOURCE_REGROWTH_RATE = 0.05  # Food is scarce and grows back slowly

# --- Population & Simulation ---
MAX_POPULATION_SIZE = 300
AGENT_INITIAL_COUNT = 100
SIMULATION_TICKS = 5000
CULLING_INTERVAL = 100  # How often to clean up dead agents
# --- Lifespan ---
MAX_AGENT_AGE = 4000  # Approx 40 "years" if 1 year = 100 ticks

# --- Agent Biology & Reproduction (Tuned for more challenge) ---
MOVE_SPEED = 1.0
HUNGER_RATE = 0.5           # Daily energy cost
STARVATION_RATE = 1.5       # Starving is dangerous
EAT_RATE = 20.0             # Energy from a single meal
FORAGING_THRESHOLD = 30.0   # FIXED: Added comment - hunger level to start seeking food
MIN_REPRODUCTION_AGE = 18
REPRODUCTION_RATE = 0.02
GESTATION_PERIOD = 20
REPRODUCTION_THRESHOLD = 40.0  # Max hunger level to be able to reproduce
MATING_DESIRE_RATE = 0.2

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

# FIXED: Added optimization configuration section
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
def validate_config():
    """Validate configuration values for common issues"""
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
    'ENABLE_OPTIMIZATION_SWITCHING', 'validate_config'
]