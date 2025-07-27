# FILE: src/sapiens_sim/smart_main.py  
# Smart main that uses adaptive optimization

import time
import numpy as np
from . import config
from .core.world import create_world
from .core.agent_manager import AgentManager
from .core.adaptive_optimization import HybridSimulation

def run_smart_simulation():
    """
    Run simulation with intelligent optimization selection
    """
    print("=== Smart Adaptive SapiensSim ===")
    
    # Create components
    world = create_world(config.WORLD_WIDTH, config.WORLD_HEIGHT)
    agent_manager = AgentManager(max_population=config.MAX_POPULATION_SIZE)
    
    # Initialize population
    agents = agent_manager.create_initial_population(
        count=config.AGENT_INITIAL_COUNT,
        world_width=config.WORLD_WIDTH,
        world_height=config.WORLD_HEIGHT,
        min_reproduction_age=config.MIN_REPRODUCTION_AGE
    )
    
    # Create hybrid simulation that adapts to scale
    hybrid_sim = HybridSimulation(
        config.WORLD_WIDTH,
        config.WORLD_HEIGHT, 
        config.MAX_POPULATION_SIZE
    )
    
    # Initialize with adaptive strategy
    hybrid_sim.initialize_simulation(
        agent_manager, 
        config.AGENT_INITIAL_COUNT,
        config.SIMULATION_TICKS
    )
    
    next_agent_id = config.AGENT_INITIAL_COUNT
    start_time = time.time()
    
    print(f"Running adaptive simulation for {config.SIMULATION_TICKS} ticks...")
    
    # Main simulation loop
    for tick in range(config.SIMULATION_TICKS):
        
        agents, world, next_agent_id = hybrid_sim.adaptive_simulation_tick(
            agent_manager=agent_manager,
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
        
        # Periodic cleanup and reporting
        if (tick + 1) % config.CULLING_INTERVAL == 0:
            agent_manager.cull_the_dead()
        
        if (tick + 1) % 100 == 0:
            active_population = np.sum(agents['health'] > 0)
            print(f"Tick {tick+1}/{config.SIMULATION_TICKS} | Population: {active_population}")
    
    total_time = time.time() - start_time
    final_population = np.sum(agents['health'] > 0)
    
    print(f"\n=== Smart Simulation Completed ===")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Initial Population: {config.AGENT_INITIAL_COUNT}")
    print(f"Final Population: {final_population}")
    print(f"Strategy Used: {hybrid_sim.current_strategy}")
    
    return agents, world, hybrid_sim

if __name__ == "__main__":
    run_smart_simulation()