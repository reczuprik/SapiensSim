# FILE: src/sapiens_sim/main.py  
# FIXED: Enhanced main runner with proper optimization tracking

import time
import numpy as np
from . import config
from .core.world import create_world
from .core.agent_manager import AgentManager
from .core.adaptive_optimization import HybridSimulation

def run_smart_simulation():
    """
    Run simulation with intelligent optimization selection
    FIXED: Better optimization tracking and error handling
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
    
    # FIXED: Enhanced stats header with optimization tracking
    print("\n" + "="*95)
    print(" TICK | POP | AVG FIT | MAX FIT | AVG AGE | MAX AGE | AVG GEN | BRAIN (N/C) | OPT")
    print("="*95)

    # FIXED: Track optimization usage
    optimization_switches = 0
    last_strategy = None
    performance_log = []
    tick_times = []

    # Main simulation loop
    for tick in range(config.SIMULATION_TICKS):
        tick_start = time.time()
        
        # FIXED: Check for critical config issues before running
        if tick == 0:
            _validate_config()
        
        # Separate core simulation parameters from additional parameters
        all_params = {
            'move_speed': config.MOVE_SPEED,
            'hunger_rate': config.HUNGER_RATE,
            'starvation_rate': config.STARVATION_RATE,
            'foraging_threshold': config.FORAGING_THRESHOLD,
            'eat_rate': config.EAT_RATE,
            'resource_regrowth_rate': config.RESOURCE_REGROWTH_RATE,
            'min_reproduction_age': config.MIN_REPRODUCTION_AGE,
            'reproduction_rate': config.REPRODUCTION_RATE,
            'gestation_period': config.GESTATION_PERIOD,
            'reproduction_threshold': config.REPRODUCTION_THRESHOLD,
            'mating_desire_rate': config.MATING_DESIRE_RATE,
            'newborn_health': config.NEWBORN_HEALTH,
            'newborn_hunger': config.NEWBORN_HUNGER,
            'mother_health_penalty': config.MOTHER_HEALTH_PENALTY,
            'terrain_cost_plains': config.TERRAIN_COST_PLAINS,
            'terrain_cost_forest': config.TERRAIN_COST_FOREST,
            'terrain_cost_mountain': config.TERRAIN_COST_MOUNTAIN,
            'tool_decay_on_use': config.TOOL_DECAY_ON_USE,
            'shelter_decay_per_tick': config.SHELTER_DECAY_PER_TICK,
            'max_agent_age': config.MAX_AGENT_AGE,
            'fitness_death_penalty': config.FITNESS_DEATH_PENALTY
        }

        try:
            agents, world, next_agent_id = hybrid_sim.adaptive_simulation_tick(
                agent_manager=agent_manager,
                world=world,
                next_agent_id=next_agent_id,
                **all_params
            )
        except Exception as e:
            print(f"ERROR in simulation tick {tick}: {e}")
            print("Attempting to continue with fallback...")
            # FIXED: Try to continue with simple simulation
            try:
                from .core.simulation import simulation_tick
                agents, world, next_agent_id = simulation_tick(
                    agent_manager, world, next_agent_id, **all_params
                )
            except Exception as e2:
                print(f"FATAL: Fallback also failed: {e2}")
                break

        # FIXED: Track optimization strategy changes
        current_strategy = hybrid_sim.current_strategy
        if last_strategy != current_strategy:
            optimization_switches += 1
            last_strategy = current_strategy.copy()

        tick_time = time.time() - tick_start
        tick_times.append(tick_time)

        # Periodic cleanup and reporting
        if (tick + 1) % config.CULLING_INTERVAL == 0:
            agent_manager.cull_the_dead()
        
        # FIXED: Enhanced reporting with optimization info
        if (tick + 1) % 100 == 0:
            stats = agent_manager.get_population_stats()
            perf_stats = hybrid_sim.get_performance_stats()
            
            # Create optimization status string
            opt_status = ""
            if current_strategy.get('use_batch_neat', False):
                opt_status += "B"
            if current_strategy.get('use_spatial_grid', False):
                opt_status += "S"  
            if current_strategy.get('use_vectorized_ops', False):
                opt_status += "V"
            if current_strategy.get('use_lazy_world', False):
                opt_status += "L"
            if not opt_status:
                opt_status = "None"
            
            # Calculate average tick time for recent period
            recent_tick_time = np.mean(tick_times[-100:]) if tick_times else 0
            
            if stats['population'] > 0:
                print(
                    f" {tick+1:<4} |"
                    f" {stats['population']:<3} |"
                    f" {stats['avg_fitness']:<7.2f} |"
                    f" {stats['max_fitness']:<7.2f} |"
                    f" {stats['avg_age']:<7.1f} |"
                    f" {stats['max_age']:<7} |"
                    f" {stats['avg_generation']:<7.1f} |"
                    f" {stats['avg_nodes']:.1f}/{stats['avg_connections']:.1f} |"
                    f" {opt_status:<4}"
                )
                
                # FIXED: Log detailed performance every 500 ticks
                if (tick + 1) % 500 == 0:
                    print(f"    └─ Perf: {recent_tick_time*1000:.1f}ms/tick, "
                          f"Batch: {perf_stats['batch_evaluations']}, "
                          f"Spatial: {perf_stats['spatial_queries']}, "
                          f"Fallbacks: {perf_stats['fallback_count']}")
                    
                    # Reset performance stats
                    hybrid_sim.reset_performance_stats()
                    
            else:
                print(f" {tick+1:<4} | EXTINCTION - Final tick time: {recent_tick_time*1000:.1f}ms")
                break # Stop simulation if everyone is dead
    
    total_time = time.time() - start_time
    final_population = np.sum(agents['health'] > 0)
    avg_tick_time = np.mean(tick_times) if tick_times else 0
    
    print(f"\n=== Smart Simulation Completed ===")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Tick Time: {avg_tick_time*1000:.2f} ms")
    print(f"Initial Population: {config.AGENT_INITIAL_COUNT}")
    print(f"Final Population: {final_population}")
    print(f"Optimization Switches: {optimization_switches}")
    print(f"Final Strategy: {hybrid_sim.current_strategy}")
    
    # FIXED: Final performance summary
    final_perf = hybrid_sim.get_performance_stats()
    print(f"Performance Summary:")
    print(f"  - Batch Evaluations: {final_perf['batch_evaluations']}")
    print(f"  - Spatial Queries: {final_perf['spatial_queries']}")  
    print(f"  - Fallbacks: {final_perf['fallback_count']}")
    
    if final_perf['fallback_count'] > 0:
        print(f"  ⚠ Warning: {final_perf['fallback_count']} optimization failures occurred")
    
    return agents, world, hybrid_sim

def _validate_config():
    """FIXED: Validate configuration for common issues"""
    issues = []
    
    # Check for the duplicate RESOURCE_REGROWTH_RATE issue
    if hasattr(config, 'RESOURCE_REGROWTH_RATE'):
        print(f"✓ RESOURCE_REGROWTH_RATE = {config.RESOURCE_REGROWTH_RATE}")
    else:
        issues.append("RESOURCE_REGROWTH_RATE not found in config")
    
    # Check for reasonable values
    if config.MAX_POPULATION_SIZE < config.AGENT_INITIAL_COUNT:
        issues.append(f"MAX_POPULATION_SIZE ({config.MAX_POPULATION_SIZE}) < AGENT_INITIAL_COUNT ({config.AGENT_INITIAL_COUNT})")
    
    if config.HUNGER_RATE <= 0:
        issues.append(f"HUNGER_RATE ({config.HUNGER_RATE}) should be positive")
        
    if config.EAT_RATE <= 0:
        issues.append(f"EAT_RATE ({config.EAT_RATE}) should be positive")
    
    if issues:
        print("⚠ Configuration Issues Detected:")
        for issue in issues:
            print(f"    - {issue}")
        print("Continuing anyway, but results may be unexpected...\n")
    else:
        print("✓ Configuration validation passed\n")

def run_performance_test():
    """FIXED: Performance test to validate optimizations work"""
    print("=== Performance Test Mode ===")
    
    # Test with different population sizes to verify optimizations kick in
    test_populations = [25, 75, 150, 300]
    
    for pop_size in test_populations:
        print(f"\nTesting with {pop_size} agents...")
        
        # Override config for this test
        original_initial = config.AGENT_INITIAL_COUNT
        original_ticks = config.SIMULATION_TICKS
        
        config.AGENT_INITIAL_COUNT = pop_size
        config.SIMULATION_TICKS = 100  # Short test
        
        try:
            start_time = time.time()
            agents, world, hybrid_sim = run_smart_simulation()
            test_time = time.time() - start_time
            
            strategy = hybrid_sim.current_strategy
            perf_stats = hybrid_sim.get_performance_stats()
            
            print(f"  Time: {test_time:.2f}s")
            print(f"  Strategy: {strategy}")
            print(f"  Batch Evals: {perf_stats['batch_evaluations']}")
            print(f"  Spatial Queries: {perf_stats['spatial_queries']}")
            print(f"  Fallbacks: {perf_stats['fallback_count']}")
            
        finally:
            # Restore original config
            config.AGENT_INITIAL_COUNT = original_initial
            config.SIMULATION_TICKS = original_ticks

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_performance_test()
    else:
        run_smart_simulation()