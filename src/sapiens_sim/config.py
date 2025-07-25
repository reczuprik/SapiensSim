# FILE_NAME: src/sapiens_sim/config.py
# CODE_BLOCK_ID: SapiensSim-v0.5-config.py

# --- World Configuration ---
WORLD_WIDTH = 200
WORLD_HEIGHT = 200

# --- Agent Configuration ---
AGENT_INITIAL_COUNT = 150
MOVE_SPEED = 1.0 # Max distance an agent can move in one tick.

# --- Simulation Configuration ---
SIMULATION_TICKS = 1000 # The total number of steps the simulation will run for.

# --- Biological Rates ---
HUNGER_RATE = 0.5
STARVATION_RATE = 1.0
# NEW PARAMETERS FOR FORAGING
FORAGING_THRESHOLD = 30.0 # Hunger level at which agents start seeking food.
EAT_RATE = 20.0 # Amount of hunger restored when eating.
RESOURCE_REGROWTH_RATE = 0.1 # Amount of resources that regrow on a tile each tick.