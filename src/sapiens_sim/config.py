# FILE_NAME: src/sapiens_sim/config.py

# --- World Configuration ---
WORLD_WIDTH = 200
WORLD_HEIGHT = 200
RESOURCE_REGROWTH_RATE = 0.15

# --- Population & Simulation ---
MAX_POPULATION_SIZE = 300
AGENT_INITIAL_COUNT = 100
SIMULATION_TICKS = 5000
CULLING_INTERVAL = 100 # How often to clean up dead agents
# --- NEW: LIFESPAN ---
MAX_AGENT_AGE = 4000 # Approx 40 "years" if 1 year = 100 ticks

# --- Agent Biology & Reproduction (Tuned for more challenge) ---


RESOURCE_REGROWTH_RATE = 0.05 # Food is much scarcer and grows back slower
# --- Agent Biology & Reproduction ---
MOVE_SPEED = 1.0
HUNGER_RATE = 0.5           # Increased daily energy cost

STARVATION_RATE = 1.5       # Starving is more dangerous
EAT_RATE = 20.0             # Reduced energy from a single meal
FORAGING_THRESHOLD = 30.0  # <-- THE MISSING PARAMETER
MIN_REPRODUCTION_AGE = 18
REPRODUCTION_RATE = 0.02
GESTATION_PERIOD = 20
REPRODUCTION_THRESHOLD = 40.0 # Max hunger level to be able to reproduce
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


# --- NEW: TERRAIN MOVEMENT COSTS ---
# Multiplier for hunger_rate based on terrain. 1.0 is normal cost.
TERRAIN_COST_PLAINS = 1.0
TERRAIN_COST_FOREST = 1.5   # Moving through dense forest is harder
TERRAIN_COST_MOUNTAIN = 5.0 # Moving through mountains is extremely costly