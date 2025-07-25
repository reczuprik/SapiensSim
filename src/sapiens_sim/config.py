# FILE_NAME: src/sapiens_sim/config.py
# CODE_BLOCK_ID: SapiensSim-v0.3-config.py

# --- World Configuration ---
WORLD_WIDTH = 200
WORLD_HEIGHT = 200

# --- Agent Configuration ---
AGENT_INITIAL_COUNT = 150

# --- Simulation Configuration ---
SIMULATION_TICKS = 1000 # The total number of steps the simulation will run for.

# --- Biological Rates ---
# How much hunger increases per tick.
HUNGER_RATE = 0.5
# If hunger is > 90, health decreases by this amount per tick.
STARVATION_RATE = 1.0