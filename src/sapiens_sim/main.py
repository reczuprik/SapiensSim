#FILE_NAME: src/sapiens_sim/main.py
#CODE_BLOCK_ID: SapiensSim-v0.7-main.py
import time
import numpy as np
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
    world = create_world(config.WORLD_WIDTH, config.WORLD_HEIGHT)
    # Update the create_agents call to include age configuration
    agents = create_agents(
        count=config.AGENT_INITIAL_COUNT,
        max_population=config.MAX_POPULATION_SIZE,
        world_width=config.WORLD_WIDTH,
        world_height=config.WORLD_HEIGHT,
        min_reproduction_age=config.MIN_REPRODUCTION_AGE
    )
    
    # Initialize the ID for the next newborn
    next_agent_id = config.AGENT_INITIAL_COUNT

    initial_total_health = np.sum(agents['health'])
    
    print("\n--- Simulation Starting ---")
    print(f"Running for {config.SIMULATION_TICKS} ticks.")
    
    start_time = time.time()
    
    # --- Main Loop ---
    for tick in range(config.SIMULATION_TICKS):
        # Update the simulation_tick call with all new parameters
        agents, world, next_agent_id  = simulation_tick(
            agents=agents,
            world=world,
            next_agent_id=next_agent_id,
            move_speed=config.MOVE_SPEED,
            hunger_rate=config.HUNGER_RATE,
            starvation_rate=config.STARVATION_RATE,
            foraging_threshold=config.FORAGING_THRESHOLD,
            eat_rate=config.EAT_RATE,
            resource_regrowth_rate=config.RESOURCE_REGROWTH_RATE,
            min_reproduction_age=config.MIN_REPRODUCTION_AGE,
            reproduction_rate=config.REPRODUCTION_RATE,
            gestation_period=config.GESTATION_PERIOD,
            reproduction_threshold=config.REPRODUCTION_THRESHOLD,
            mating_desire_rate=config.MATING_DESIRE_RATE,
            newborn_health=config.NEWBORN_HEALTH,
            newborn_hunger=config.NEWBORN_HUNGER,
            mother_health_penalty=config.MOTHER_HEALTH_PENALTY
        )

        if (tick + 1) % 100 == 0:
            active_agents = np.sum(agents['health'] > 0)
            print(f"Tick {tick+1}/{config.SIMULATION_TICKS} complete. Population: {active_agents}")
        
        
            agent_0_health = agents[0]['health']
            agent_0_hunger = agents[0]['hunger']
            print(f"  Agent 0: Health={agent_0_health:.1f}, Hunger={agent_0_hunger:.1f}")

    
    end_time = time.time()

    print("\n--- Simulation Finished ---")
    print(f"Total runtime: {end_time - start_time:.4f} seconds.")

    final_total_health = np.sum(agents['health'])

    print(f"\nInitial total health: {initial_total_health:.2f}")
    print(f"Final total health:   {final_total_health:.2f}")

    

    final_population = np.sum(agents['health'] > 0)
    print(f"Initial Population: {config.AGENT_INITIAL_COUNT}")
    print(f"Final Population:   {final_population}")

if __name__ == "__main__":
    run_simulation()