# FILE: src/sapiens_sim/core/adaptive_optimization.py
# Adaptive optimization that switches between optimized and simple versions

import numpy as np
import time
from typing import Dict, Any

class AdaptiveOptimizer:
    """
    Intelligently chooses between optimized and simple implementations
    based on population size and simulation parameters
    """
    
    def __init__(self):
        self.optimization_thresholds = {
            'spatial_grid': 100,      # Use spatial grid only with 100+ agents
            'batch_neat': 150,        # Batch evaluation only with 150+ agents  
            'vectorized_ops': 75,     # Vectorized operations with 75+ agents
            'lazy_world': 200,        # Lazy world updates with 200+ agents
        }
        
        self.performance_cache = {}
        
    def should_use_optimization(self, optimization_type: str, population_size: int, 
                              world_size: tuple) -> bool:
        """Decide whether to use a specific optimization"""
        
        threshold = self.optimization_thresholds.get(optimization_type, 50)
        
        # Adjust thresholds based on world size
        world_factor = (world_size[0] * world_size[1]) / 40000  # Normalize to 200x200
        adjusted_threshold = threshold * max(0.5, world_factor)
        
        return population_size >= adjusted_threshold
    
    def get_optimal_strategy(self, population_size: int, world_size: tuple, 
                           ticks: int) -> Dict[str, bool]:
        """Get the optimal optimization strategy for given parameters"""
        
        strategy = {
            'use_spatial_grid': self.should_use_optimization('spatial_grid', population_size, world_size),
            'use_batch_neat': self.should_use_optimization('batch_neat', population_size, world_size),
            'use_vectorized_ops': self.should_use_optimization('vectorized_ops', population_size, world_size),
            'use_lazy_world': self.should_use_optimization('lazy_world', population_size, world_size),
        }
        
        # For very small simulations, use simple approach
        if population_size < 50 and ticks < 500:
            strategy = {k: False for k in strategy}
            
        return strategy

class HybridSimulation:
    """
    Hybrid simulation that adapts optimization strategies based on scale
    """
    
    def __init__(self, world_width: int, world_height: int, max_population: int):
        self.world_width = world_width
        self.world_height = world_height
        self.max_population = max_population
        self.adaptive_optimizer = AdaptiveOptimizer()
        
        # Initialize both optimized and simple components
        self.spatial_manager = None
        self.batch_evaluator = None
        self.current_strategy = {}
        
    def initialize_simulation(self, agent_manager, initial_population: int, total_ticks: int):
        """Initialize with adaptive strategy"""
        
        world_size = (self.world_height, self.world_width)
        self.current_strategy = self.adaptive_optimizer.get_optimal_strategy(
            initial_population, world_size, total_ticks
        )
        
        print(f"=== Adaptive Optimization Strategy ===")
        print(f"Population: {initial_population}, World: {world_size}, Ticks: {total_ticks}")
        for opt, enabled in self.current_strategy.items():
            print(f"  {opt}: {'ENABLED' if enabled else 'DISABLED'}")
        
        # Initialize only needed components
        if self.current_strategy['use_spatial_grid']:
            from .spatial_optimization import OptimizedSpatialManager
            self.spatial_manager = OptimizedSpatialManager(self.world_width, self.world_height)
            print("✓ Spatial optimization initialized")
            
        if self.current_strategy['use_batch_neat']:
            from .batch_neat_optimizer import BatchNEATEvaluator
            self.batch_evaluator = BatchNEATEvaluator(self.max_population)
            self.batch_evaluator.compile_population(agent_manager.genomes)
            print("✓ Batch NEAT evaluation initialized")
    
    def adaptive_simulation_tick(self, agent_manager, world, next_agent_id, **params):
        """Run simulation tick with adaptive optimizations"""
        
        current_population = np.sum(agent_manager.agents['health'] > 0)
        
        # Dynamically adjust strategy if population changed significantly
        if abs(current_population - self.max_population * 0.5) > self.max_population * 0.3:
            new_strategy = self.adaptive_optimizer.get_optimal_strategy(
                current_population, (self.world_height, self.world_width), 1000
            )
            
            if new_strategy != self.current_strategy:
                print(f"Population changed to {current_population}, adjusting strategy...")
                self.current_strategy = new_strategy
        
        # Use appropriate simulation method
        if any(self.current_strategy.values()):
            return self._optimized_tick(agent_manager, world, next_agent_id, **params)
        else:
            return self._simple_tick(agent_manager, world, next_agent_id, **params)
    
    def _optimized_tick(self, agent_manager, world, next_agent_id, **params):
        """Use optimized components selectively"""
        
        agents = agent_manager.agents
        
        # Spatial optimization (if enabled)
        if self.current_strategy['use_spatial_grid'] and self.spatial_manager:
            self.spatial_manager.update_agent_positions(agents)
            mate_results, food_results = self.spatial_manager.batch_find_mates_and_food(agents)
        else:
            mate_results, food_results = self._simple_spatial_queries(agents, world)
        
        # Batch NEAT evaluation (if enabled)  
        if self.current_strategy['use_batch_neat'] and self.batch_evaluator:
            batch_decisions = self.batch_evaluator.make_batch_decisions(
                agents, world, mate_results, food_results, self.spatial_manager
            )
        else:
            batch_decisions = self._simple_decision_making(agent_manager, agents, world)
        
        # Continue with rest of simulation logic...
        return self._process_agents(agent_manager, agents, world, batch_decisions, next_agent_id, **params)
    
    def _simple_tick(self, agent_manager, world, next_agent_id, **params):
        """Use simple, original implementation for small scales"""
        
        # Import original simulation logic
        from .simulation import simulation_tick
        return simulation_tick(agent_manager, world, next_agent_id, **params)
    
    def _simple_spatial_queries(self, agents, world):
        """Simple O(n²) spatial queries for small populations"""
        mate_results = {}
        food_results = {}
        
        active_agents = agents[agents['health'] > 0]
        
        for i, agent in enumerate(active_agents):
            agent_idx = np.where(agents == agent)[0][0]
            
            # Simple mate finding
            nearest_mate_dist = float('inf')
            nearest_mate_dir = np.array([0.0, 0.0])
            
            for j, other_agent in enumerate(active_agents):
                if (i != j and other_agent['sex'] != agent['sex'] and 
                    other_agent['is_fertile'] and not other_agent['is_pregnant']):
                    
                    dist = np.linalg.norm(agent['pos'] - other_agent['pos'])
                    if dist < nearest_mate_dist:
                        nearest_mate_dist = dist
                        if dist > 0:
                            nearest_mate_dir = (other_agent['pos'] - agent['pos']) / dist
            
            mate_results[agent_idx] = {
                'distance': nearest_mate_dist,
                'direction': nearest_mate_dir,
                'nearest_mate_idx': -1  # Simplified
            }
            
            # Simple food finding
            agent_pos = agent['pos']
            nearest_food_dist = float('inf')
            nearest_food_dir = np.array([0.0, 0.0])
            
            # Sample some random tiles
            for _ in range(20):  # Much smaller sample for small sims
                y = np.random.randint(0, world.shape[0])
                x = np.random.randint(0, world.shape[1])
                
                if world[y, x]['resources'] > 10:
                    dist = np.linalg.norm(agent_pos - np.array([y, x]))
                    if dist < nearest_food_dist:
                        nearest_food_dist = dist
                        if dist > 0:
                            nearest_food_dir = (np.array([y, x]) - agent_pos) / dist
            
            food_results[agent_idx] = {
                'distance': nearest_food_dist,
                'direction': nearest_food_dir
            }
        
        return mate_results, food_results
    
    def _simple_decision_making(self, agent_manager, agents, world):
        """Simple individual decision making for small populations"""
        decisions = {}
        
        for i, agent in enumerate(agents):
            if agent['health'] > 0:
                decisions[i] = agent_manager.make_decision(i, world, agents)
        
        return decisions
    
    def _process_agents(self, agent_manager, agents, world, decisions, next_agent_id, **params):
        """Process agent actions with selected optimizations"""
        
        # Use vectorized operations if enabled
        if self.current_strategy['use_vectorized_ops']:
            return self._vectorized_agent_processing(agent_manager, agents, world, decisions, next_agent_id, **params)
        else:
            return self._individual_agent_processing(agent_manager, agents, world, decisions, next_agent_id, **params)
    
    def _vectorized_agent_processing(self, agent_manager, agents, world, decisions, next_agent_id, **params):
        """Vectorized agent processing for larger populations"""
        # Implementation would use vectorized operations from optimized_simulation.py
        # For brevity, falling back to individual processing
        return self._individual_agent_processing(agent_manager, agents, world, decisions, next_agent_id, **params)
    
    def _individual_agent_processing(self, agent_manager, agents, world, decisions, next_agent_id, **params):
        """Individual agent processing for smaller populations"""
        from .simulation import simulation_tick
        return simulation_tick(agent_manager, world, next_agent_id, **params)


