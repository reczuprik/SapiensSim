#CODE_BLOCK_ID: SapiensSim-v1.14-main-FINAL.py
#FILE_NAME: src/sapiens_sim/main.py
import time
import numpy as np
from . import config
from .core.world import create_world
from .core.agent_manager import AgentManager # <-- IMPORT AgentManager
from .core.simulation import simulation_tick # <-- IMPORT the correct tick function
def run_simulation():
    """
    Sets up and runs the main simulation loop using the AgentManager.
    """
    print("--- SapiensSim with Custom Neuroevolution ---")
    # --- Setup ---
    world = create_world(config.WORLD_WIDTH, config.WORLD_HEIGHT)

    # 1. Create the AgentManager, which will hold everything
    agent_manager = AgentManager(max_population=config.MAX_POPULATION_SIZE)

    # 2. Use the manager to create the initial population
    agents = agent_manager.create_initial_population(
        count=config.AGENT_INITIAL_COUNT,
        world_width=config.WORLD_WIDTH,
        world_height=config.WORLD_HEIGHT,
        min_reproduction_age=config.MIN_REPRODUCTION_AGE
    )

    next_agent_id = config.AGENT_INITIAL_COUNT

    print(f"\n--- Simulation Starting ---")
    print(f"Running for {config.SIMULATION_TICKS} ticks.")

    start_time = time.time()
    # --- Stats Header ---
    print("\n" + "="*80)
    print(" TICK | POP | AVG FIT | MAX FIT | AVG AGE | MAX AGE | AVG GEN | BRAIN (N/C)")
    print("="*80)

    # --- Main Loop ---
    for tick in range(config.SIMULATION_TICKS):
        # 3. The call signature is now correct. We pass the manager itself.
        agents, world, next_agent_id = simulation_tick(
            agent_population=agent_manager, # <-- PASS THE MANAGER OBJECT
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

        # 4. Periodically cull the dead to free up slots
        if (tick + 1) % config.CULLING_INTERVAL == 0:
            agent_manager.cull_the_dead()

        # --- Logging ---
        if (tick + 1) % 100 == 0:
            stats = agent_manager.get_population_stats()
            if stats['population'] > 0:
                print(
                    f" {tick+1:<4} |"
                    f" {stats['population']:<3} |"
                    f" {stats['avg_fitness']:<7.2f} |"
                    f" {stats['max_fitness']:<7.2f} |"
                    f" {stats['avg_age']:<7.1f} |"
                    f" {stats['max_age']:<7} |"
                    f" {stats['avg_generation']:<7.1f} |"
                    f" {stats['avg_nodes']:.1f}/{stats['avg_connections']:.1f}"
                )
            else:
                print(f" {tick+1:<4} | EXTINCTION")
                break # Stop the simulation if everyone is dead

    end_time = time.time()

    # --- Final Report ---
    print("\n--- Simulation Finished ---")
    print(f"Total runtime: {end_time - start_time:.2f} seconds.")

    final_population = np.sum(agents['health'] > 0)
    print(f"Initial Population: {config.AGENT_INITIAL_COUNT}")
    print(f"Final Population:   {final_population}")

if __name__ == "__main__":
    run_simulation()