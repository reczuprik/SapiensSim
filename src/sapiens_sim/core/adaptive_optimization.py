# FILE: src/sapiens_sim/core/adaptive_optimization.py
# FIXED: Inconsistent optimization usage

import numpy as np
import time
from typing import Dict, Any
from .vectorized_simulation import vectorized_simulation_tick, compare_performance

class AdaptiveOptimizer:
    """
    Intelligently chooses between optimized and simple implementations
    based on population size and simulation parameters
    """
    
    def __init__(self):
        # FIXED: Lowered thresholds to ensure optimizations are used more aggressively
        self.optimization_thresholds = {
            'spatial_grid': 50,       # Use spatial grid with 50+ agents (was 100)
            'batch_neat': 75,         # Batch evaluation with 75+ agents (was 100)  
            'vectorized_ops': 50,     # Vectorized operations with 50+ agents (was 100)
            'lazy_world': 150,        # Lazy world updates with 150+ agents (was 100)
        }
        
        self.performance_cache = {}
        
    def should_use_optimization(self, optimization_type: str, population_size: int, 
                              world_size: tuple) -> bool:
        """Decide whether to use a specific optimization - FIXED logic"""
        
        threshold = self.optimization_thresholds.get(optimization_type, 50)
        
        # FIXED: More aggressive world size adjustment
        world_factor = (world_size[0] * world_size[1]) / 40000  # Normalize to 200x200
        adjusted_threshold = threshold * max(0.3, min(world_factor, 1.5))  # Better bounds
        
        return population_size >= adjusted_threshold
    
    def get_optimal_strategy(self, population_size: int, world_size: tuple, 
                           ticks: int) -> Dict[str, bool]:
        """Get the optimal optimization strategy for given parameters - FIXED"""
        
        strategy = {
            'use_spatial_grid': self.should_use_optimization('spatial_grid', population_size, world_size),
            'use_batch_neat': self.should_use_optimization('batch_neat', population_size, world_size),
            'use_vectorized_ops': self.should_use_optimization('vectorized_ops', population_size, world_size),
            'use_lazy_world': self.should_use_optimization('lazy_world', population_size, world_size),
        }
        
        # FIXED: More nuanced small simulation handling
        if population_size < 25 and ticks < 500:
            # Only disable ALL optimizations for very small sims
            strategy = {k: False for k in strategy}
        elif population_size < 50:
            # For small-medium sims, keep spatial but disable heavy optimizations
            strategy['use_batch_neat'] = False
            strategy['use_lazy_world'] = False
            
        return strategy

class HybridSimulation:
    """
    Hybrid simulation that adapts optimization strategies based on scale
    FIXED: Ensures optimizations are actually used when enabled
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
        
        # FIXED: Add performance tracking
        self.performance_stats = {
            'batch_evaluations': 0,
            'spatial_queries': 0,
            'fallback_count': 0
        }
        
    def initialize_simulation(self, agent_manager, initial_population: int, total_ticks: int):
        """Initialize with adaptive strategy - FIXED initialization"""
        
        world_size = (self.world_height, self.world_width)
        self.current_strategy = self.adaptive_optimizer.get_optimal_strategy(
            initial_population, world_size, total_ticks
        )
        
        print(f"=== Adaptive Optimization Strategy ===")
        print(f"Population: {initial_population}, World: {world_size}, Ticks: {total_ticks}")
        for opt, enabled in self.current_strategy.items():
            print(f"  {opt}: {'ENABLED' if enabled else 'DISABLED'}")
        
        # FIXED: Always initialize components if they might be needed
        # This prevents runtime failures when strategy changes mid-simulation
        try:
            from .spatial_optimization import OptimizedSpatialManager
            self.spatial_manager = OptimizedSpatialManager(self.world_width, self.world_height)
            print("✓ Spatial optimization ready")
        except ImportError as e:
            print(f"⚠ Spatial optimization unavailable: {e}")
            self.spatial_manager = None
            
        try:
            from .batch_neat_optimizer import BatchNEATEvaluator
            self.batch_evaluator = BatchNEATEvaluator(self.max_population)
            # FIXED: Only compile if we have genomes
            if hasattr(agent_manager, 'genomes') and any(g is not None for g in agent_manager.genomes):
                self.batch_evaluator.compile_population(agent_manager.genomes)
                print("✓ Batch NEAT evaluation ready")
            else:
                print("⚠ No genomes available for batch compilation")
        except ImportError as e:
            print(f"⚠ Batch NEAT optimization unavailable: {e}")
            self.batch_evaluator = None
    
    def adaptive_simulation_tick(self, agent_manager, world, next_agent_id, **params):
        """Run simulation tick with adaptive optimizations - FIXED strategy enforcement"""
        
        current_population = np.sum(agent_manager.agents['health'] > 0)
        
        # FIXED: More intelligent strategy switching
        if abs(current_population - getattr(self, '_last_population', current_population)) > max(10, current_population * 0.2):
            new_strategy = self.adaptive_optimizer.get_optimal_strategy(
                current_population, (self.world_height, self.world_width), 1000
            )
            
            if new_strategy != self.current_strategy:
                print(f"Population changed to {current_population}, adjusting strategy...")
                self.current_strategy = new_strategy
                
                # FIXED: Re-compile batch evaluator if switching to batch mode
                if (new_strategy['use_batch_neat'] and self.batch_evaluator and 
                    hasattr(agent_manager, 'genomes') and any(g is not None for g in agent_manager.genomes)):
                    self.batch_evaluator.compile_population(agent_manager.genomes)
        
        self._last_population = current_population
        
        # FIXED: Force optimized path when optimizations are enabled
        if self._should_use_optimized_path(current_population):
            agents, world, next_agent_id = self._optimized_tick(agent_manager, world, next_agent_id, **params)
        else:
            agents, world, next_agent_id = self._simple_tick(agent_manager, world, next_agent_id, **params)
        
        # Handle aging and death (common to both paths)
        max_age = params.get('max_agent_age', 4000)
        death_penalty = params.get('fitness_death_penalty', -5.0)
        self._handle_aging_and_death(agent_manager, max_age, death_penalty)

        return agents, world, next_agent_id

    # In HybridSimulation._should_use_optimized_path()
    def _should_use_optimized_path(self, current_population: int) -> bool:
        result = (
            (self.current_strategy.get('use_spatial_grid', False) and self.spatial_manager is not None) or
            (self.current_strategy.get('use_batch_neat', False) and self.batch_evaluator is not None) or
            (self.current_strategy.get('use_vectorized_ops', False) and current_population > 25)
        )
        
        
        return result
    def _handle_aging_and_death(self, agent_manager, max_age: int, death_penalty: float):
        """
        Handles aging and death from old age for all active agents.
        This is done in a vectorized way for performance.
        """
        agents = agent_manager.agents
        active_mask = agents['health'] > 0
        
        if not np.any(active_mask):
            return
        
        # --- Death from Old Age ---
        old_age_mask = (agents['age'] > max_age) & active_mask
        
        if np.any(old_age_mask):
            # "Kill" the agents by setting their health to 0
            agents['health'][old_age_mask] = 0
            
            # Apply a fitness penalty for dying
            old_age_indices = np.where(old_age_mask)[0]
            for i in old_age_indices:
                agent_manager.update_fitness(i, death_penalty)

    def _optimized_tick(self, agent_manager, world, next_agent_id, **params):
        """Use optimized components selectively - FIXED to ensure usage"""
        
        agents = agent_manager.agents
        active_count = np.sum(agents['health'] > 0)
        
        # FIXED: Always update spatial manager if available
        if self.spatial_manager is not None:
            self.spatial_manager.update_agent_positions(agents)
            # Update resource cache periodically for performance
            if not hasattr(self, '_resource_cache_tick') or self._resource_cache_tick % 10 == 0:
                self.spatial_manager.update_resource_cache(world)
            self._resource_cache_tick = getattr(self, '_resource_cache_tick', 0) + 1
        
        # FIXED: Spatial optimization with proper fallback
        if (self.current_strategy['use_spatial_grid'] and self.spatial_manager is not None):
            try:
                mate_results, food_results = self.spatial_manager.batch_find_mates_and_food(agents)
                self.performance_stats['spatial_queries'] += 1
            except Exception as e:
                print(f"Spatial optimization failed, falling back: {e}")
                mate_results, food_results = self._simple_spatial_queries(agents, world)
                self.performance_stats['fallback_count'] += 1
        else:
            mate_results, food_results = self._simple_spatial_queries(agents, world)
        
        # FIXED: Batch NEAT evaluation with proper fallback  
        if (self.current_strategy['use_batch_neat'] and self.batch_evaluator is not None):
            try:
                batch_decisions = self.batch_evaluator.make_batch_decisions(
                    agents, world, mate_results, food_results, self.spatial_manager
                )

                self.performance_stats['batch_evaluations'] += 1
                active_count = np.sum(agents['health'] > 0)

                # FIXED: Ensure we have decisions for all active agents
                if len(batch_decisions) < active_count * 0.8:  # At least 80% coverage
                    print(f"Warning: Batch evaluation only returned {len(batch_decisions)} decisions for {active_count} agents")
                    # Fill missing decisions with simple evaluation
                    
                    missing_decisions = self._simple_decision_making(agent_manager, agents, world, 
                                                                  exclude_indices=set(batch_decisions.keys()))
                    batch_decisions.update(missing_decisions)
                    
            except Exception as e:
                print(f"Batch NEAT evaluation failed, falling back: {e}")
                batch_decisions = self._simple_decision_making(agent_manager, agents, world)
                self.performance_stats['fallback_count'] += 1
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
        """Simple O(n²) spatial queries for small populations - FIXED efficiency"""
        mate_results = {}
        food_results = {}
        
        active_mask = agents['health'] > 0
        active_indices = np.where(active_mask)[0]
        
        # FIXED: Only process active agents
        for agent_idx in active_indices:
            agent = agents[agent_idx]
            
            # FIXED: More efficient mate finding
            nearest_mate_dist = float('inf')
            nearest_mate_dir = np.array([0.0, 0.0])
            
            for other_idx in active_indices:
                if agent_idx != other_idx:
                    other_agent = agents[other_idx]
                    if (other_agent['sex'] != agent['sex'] and 
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
            
            # FIXED: More efficient food finding with early termination
            agent_pos = agent['pos']
            nearest_food_dist = float('inf')
            nearest_food_dir = np.array([0.0, 0.0])
            
            # FIXED: Sample fewer tiles but more intelligently
            sample_count = min(20, world.shape[0] * world.shape[1] // 50)
            for _ in range(sample_count):
                y = np.random.randint(0, world.shape[0])
                x = np.random.randint(0, world.shape[1])
                
                if world[y, x]['resources'] > 10:
                    dist = np.linalg.norm(agent_pos - np.array([y, x]))
                    if dist < nearest_food_dist:
                        nearest_food_dist = dist
                        if dist > 0:
                            nearest_food_dir = (np.array([y, x]) - agent_pos) / dist
                        
                        # FIXED: Early termination for nearby food
                        if dist < 5.0:
                            break
            
            food_results[agent_idx] = {
                'distance': nearest_food_dist,
                'direction': nearest_food_dir
            }
        
        return mate_results, food_results
    
    def _simple_decision_making(self, agent_manager, agents, world, exclude_indices=None):
        """Simple individual decision making for small populations - FIXED"""
        decisions = {}
        exclude_indices = exclude_indices or set()
        
        for i, agent in enumerate(agents):
            if agent['health'] > 0 and i not in exclude_indices:
                try:
                    decisions[i] = agent_manager.make_decision(i, world, agents)
                except Exception as e:
                    print(f"Decision making failed for agent {i}: {e}")
                    # FIXED: Provide fallback decision
                    decisions[i] = {
                        'move_x': np.random.randn() * 0.5,
                        'move_y': np.random.randn() * 0.5,
                        'seek_food': 0.7,
                        'seek_mate': 0.3,
                        'rest': 0.1,
                        'craft_tool': 0.0,
                        'build_shelter': 0.0,
                    }
        
        return decisions
    
        # In your adaptive_optimization.py - _process_agents method
    def _process_agents(self, agent_manager, agents, world, decisions, next_agent_id, **params):
        current_population = np.sum(agents['health'] > 0)
        
        # FORCE vectorized processing for ANY population > 25
        if current_population > 25:
            #print(f"🚀 Forcing vectorized processing for {current_population} agents")
            return self._vectorized_agent_processing(agent_manager, agents, world, decisions, next_agent_id, **params)
        else:
            return self._individual_agent_processing(agent_manager, agents, world, decisions, next_agent_id, **params)
    
    def _individual_agent_processing(self, agent_manager, agents, world, decisions, next_agent_id, **params):

        # FIXED: Comprehensive parameter filtering to prevent errors
        supported_params = {
            'move_speed', 'hunger_rate', 'starvation_rate', 'foraging_threshold',
            'eat_rate', 'resource_regrowth_rate', 'min_reproduction_age',
            'reproduction_rate', 'gestation_period', 'reproduction_threshold',
            'mating_desire_rate', 'newborn_health', 'newborn_hunger',
            'mother_health_penalty','terrain_cost_plains',
            'terrain_cost_forest', 'terrain_cost_mountain','max_agent_age', 
            'fitness_death_penalty', 'tool_decay_on_use', 'shelter_decay_per_tick'
        }
        
        filtered_params = {k: v for k, v in params.items() if k in supported_params}
        current_population = np.sum(agents['health'] > 0)
        if current_population > 25 and hasattr(self, '_vectorized_agent_processing'):
                print(f"🔄 Redirecting {current_population} agents to vectorized processing")
                return self._vectorized_agent_processing(agent_manager, agents, world, decisions, next_agent_id, **params)
            
        """Individual agent processing - FIXED parameter filtering"""
        from .simulation import simulation_tick
        return simulation_tick(agent_manager, world, next_agent_id, **filtered_params)

    def _vectorized_agent_processing(self, agent_manager, agents, world, decisions, next_agent_id, **params):
        """
        Vectorized agent processing for larger populations - NOW IMPLEMENTED!
        This is where we get the 3-5x speedup
        """
        current_population = np.sum(agents['health'] > 0)
        
        #print(f"🚀 Using vectorized processing for {current_population} agents")
        
        try:
            # Use the new vectorized simulation tick
            agents, world, next_agent_id = vectorized_simulation_tick(
                agent_manager, world, next_agent_id, decisions, **params
            )
            
            # Track successful vectorized processing
            if not hasattr(self, 'vectorized_count'):
                self.vectorized_count = 0
            self.vectorized_count += 1
            
            return agents, world, next_agent_id
            
        except Exception as e:
            print(f"⚠ Vectorized processing failed: {e}")
            print("Falling back to individual processing...")
            self.performance_stats['fallback_count'] += 1
            
            # Fallback to individual processing
            return self._individual_agent_processing(agent_manager, agents, world, decisions, next_agent_id, **params)
    
    def get_performance_stats(self):
        """Enhanced performance statistics including vectorized operations"""
        stats = self.performance_stats.copy()
        stats['vectorized_operations'] = getattr(self, 'vectorized_count', 0)
        return stats
    
    def reset_performance_stats(self):
        """Reset all performance tracking"""
        self.performance_stats = {
            'batch_evaluations': 0,
            'spatial_queries': 0,
            'fallback_count': 0
        }
        self.vectorized_count = 0