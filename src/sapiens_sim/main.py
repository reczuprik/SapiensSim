# FILE_NAME: src/sapiens_sim/main.py
# CODE_BLOCK_ID: SapiensSim-v0.3-main.py

import time
import numpy as np

# Import our configuration variables and core functions
from . import config
from .core.world import create_world
from .core.agents import create_agents
from .core.simulation import simulation_tick

def run_simulation():
    """
    Sets up and runs the main simulation loop.
    """
    print("--- SapiensSim Initialization ---")
    
    # --- Setup ---
    # Create the world and agent populations using our functions
    world = create_world(config.WORLD_WIDTH, config.WORLD_HEIGHT)
    agents = create_agents(
        count=config.AGENT_INITIAL_COUNT,
        world_width=config.WORLD_WIDTH,
        world_height=config.WORLD_HEIGHT
    )
    
    # Store starting health for comparison at the end
    initial_total_health = np.sum(agents['health'])
    
    print("\n--- Simulation Starting ---")
    print(f"Running for {config.SIMULATION_TICKS} ticks.")
    
    start_time = time.time()
    
    # --- Main Loop ---
    for tick in range(config.SIMULATION_TICKS):
        # This is the core of our simulation.
        # We pass the agents array and config values to our fast, Numba-jitted function.
        agents = simulation_tick(
            agents=agents,
            hunger_rate=config.HUNGER_RATE,
            starvation_rate=config.STARVATION_RATE
        )

        # We can add logging here to see progress, but for performance,
        # it's best to do it infrequently.
        if (tick + 1) % 100 == 0:
            print(f"Tick {tick+1}/{config.SIMULATION_TICKS} complete.")

    end_time = time.time()
    
    # --- Simulation Results ---
    print("\n--- Simulation Finished ---")
    print(f"Total runtime: {end_time - start_time:.4f} seconds.")
    
    final_total_health = np.sum(agents['health'])
    
    print(f"\nInitial total health: {initial_total_health:.2f}")
    print(f"Final total health:   {final_total_health:.2f}")
    
    # Count how many agents are still "alive" (health > 0)
    survivors = np.sum(agents['health'] > 0)
    print(f"Survivors: {survivors}/{config.AGENT_INITIAL_COUNT}")

# This is a standard Python construct. If you execute this file directly
# (e.g., python src/sapiens_sim/main.py), it will call run_simulation().
# If you import it into another file, it won't run automatically.
if __name__ == '__main__':
    run_simulation()