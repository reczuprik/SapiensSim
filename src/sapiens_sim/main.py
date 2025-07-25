#FILE_NAME: src/sapiens_sim/main.py
#CODE_BLOCK_ID: SapiensSim-v0.5-main.py
import time
import numpy as np
from . import config
from .core.world import create_world
from .core.agents import create_agents
from .core.simulation import simulation_tick
def run_simulation():
    print("--- SapiensSim Initialization ---")
    world = create_world(config.WORLD_WIDTH, config.WORLD_HEIGHT)
    agents = create_agents(
        count=config.AGENT_INITIAL_COUNT,
        world_width=config.WORLD_WIDTH,
        world_height=config.WORLD_HEIGHT
    )

    initial_total_health = np.sum(agents['health'])

    print("\n--- Simulation Starting ---")
    print(f"Running for {config.SIMULATION_TICKS} ticks.")

    start_time = time.time()

    # --- Main Loop ---
    for tick in range(config.SIMULATION_TICKS):
        # The world state is now passed into and returned from the function
        agents, world = simulation_tick(
            agents=agents,
            world=world,
            move_speed=config.MOVE_SPEED,
            hunger_rate=config.HUNGER_RATE,
            starvation_rate=config.STARVATION_RATE,
            foraging_threshold=config.FORAGING_THRESHOLD,
            eat_rate=config.EAT_RATE,
            resource_regrowth_rate=config.RESOURCE_REGROWTH_RATE
        )

        if (tick + 1) % 100 == 0:
            print(f"Tick {tick+1}/{config.SIMULATION_TICKS} complete.")
            # Let's check the status of Agent 0
            agent_0_health = agents[0]['health']
            agent_0_hunger = agents[0]['hunger']
            print(f"  Agent 0: Health={agent_0_health:.1f}, Hunger={agent_0_hunger:.1f}")

    end_time = time.time()

    print("\n--- Simulation Finished ---")
    print(f"Total runtime: {end_time - start_time:.4f} seconds.")

    final_total_health = np.sum(agents['health'])

    print(f"\nInitial total health: {initial_total_health:.2f}")
    print(f"Final total health:   {final_total_health:.2f}")

    survivors = np.sum(agents['health'] > 0)
    print(f"Survivors: {survivors}/{config.AGENT_INITIAL_COUNT}")

if __name__ == "__main__":
    run_simulation()