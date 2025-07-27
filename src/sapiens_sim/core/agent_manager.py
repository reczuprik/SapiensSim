# File: src/sapiens_sim/core/agent_manager.py
# Reference ID: SapiensSim-v1.2-agent_manager-FIXED-offspring.py


import numpy as np
from typing import List, Optional
from .. import config
from .neat_brain import (
    NEATGenome, NEATBrain, create_random_genome, mutate_genome, crossover,
    NUM_INPUTS, NUM_OUTPUTS,
    INPUT_HUNGER, INPUT_HEALTH, INPUT_AGE, INPUT_MATING_DESIRE,
    INPUT_NEAREST_FOOD_DISTANCE, INPUT_NEAREST_FOOD_DIRECTION_X, INPUT_NEAREST_FOOD_DIRECTION_Y,
    INPUT_NEAREST_MATE_DISTANCE, INPUT_NEAREST_MATE_DIRECTION_X, INPUT_NEAREST_MATE_DIRECTION_Y,
    INPUT_POPULATION_DENSITY, INPUT_CURRENT_RESOURCES,INPUT_TERRAIN_TYPE,
    OUTPUT_MOVE_X, OUTPUT_MOVE_Y, OUTPUT_SEEK_FOOD, OUTPUT_SEEK_MATE, OUTPUT_REST
)
from .neat_brain import NEATGenome, NEATBrain, create_random_genome, mutate_genome, crossover

# Define sex types as integer constants for performance
SEX_MALE = 0
SEX_FEMALE = 1

class AgentManager:
    """Manages a population of agents with NEAT brains"""
    
    def __init__(self, max_population: int):
        self.max_population = max_population
        self.genomes: List[Optional[NEATGenome]] = [None] * max_population
        self.brains: List[Optional[NEATBrain]] = [None] * max_population
        self.generation = 0
        
        # Create the numpy array for agent data (same as before but without hardcoded behavior)
        self.agent_dtype = np.dtype([
            ('id', np.int32),
            ('pos', np.float32, (2,)),
            ('health', np.float32),
            ('hunger', np.float32),
            ('age', np.int32),
            ('sex', np.int8),
            ('is_fertile', bool),
            ('is_pregnant', np.int32),
            ('partner_id', np.int32), # <-- NEW FIELD to store father's ID
            ('mating_desire', np.float32),
            ('fitness', np.float32),
            ('generation', np.int32),
        ])
        
        self.agents = np.zeros(max_population, dtype=self.agent_dtype)
    
    def create_initial_population(self, count: int, world_width: int, world_height: int, 
                            min_reproduction_age: int, initial_health: float = 100.0, 
                            initial_hunger: float = 0.0):
        """Create initial population with random NEAT brains"""
        
        # Create the basic agent data
        active_agents = self.agents[:count]
        
        # Assign unique IDs
        active_agents['id'] = np.arange(count, dtype=np.int32)
        
        # Random positions
        active_agents['pos'][:, 0] = np.random.uniform(0, world_height, size=count)
        active_agents['pos'][:, 1] = np.random.uniform(0, world_width, size=count)
        
        # Initial biological states
        active_agents['health'] = initial_health
        active_agents['hunger'] = initial_hunger
        active_agents['age'] = np.random.randint(15, 50, size=count)
        
        # Random sex
        active_agents['sex'] = np.random.choice([SEX_MALE, SEX_FEMALE], size=count)
        
        # Fertility based on age
        active_agents['is_fertile'] = active_agents['age'] >= min_reproduction_age
        
        # Initialize fitness and generation
        active_agents['fitness'] = 0.0
        active_agents['generation'] = 0
        
        # Create random NEAT brains for each agent
        for i in range(count):
            # --- THE CORRECT FIX ---
            # 1. Create the genome first.
            genome = create_random_genome()
            
            # 2. Assign it to the list. We know it's not None here.
            self.genomes[i] = genome
            
            # 3. Now that self.genomes[i] is guaranteed to be a NEATGenome,
            #    we can safely create the brain without a type error.
            self.brains[i] = NEATBrain(genome)
        
        print(f"{count} agents with NEAT brains created.")
        print(f"Male count: {np.sum(active_agents['sex'] == SEX_MALE)}")
        print(f"Female count: {np.sum(active_agents['sex'] == SEX_FEMALE)}")
        print(f"Fertile agents: {np.sum(active_agents['is_fertile'])}")
        return self.agents
    
    def get_brain_inputs(self, agent_idx: int, world: np.ndarray, agents: np.ndarray) -> np.ndarray:
        """Generate inputs for an agent's brain based on current state"""
        agent = agents[agent_idx]
        inputs = np.zeros(NUM_INPUTS, dtype=np.float32)
        
        # Basic agent state (normalized to 0-1)
        inputs[INPUT_HUNGER] = min(agent['hunger'] / 100.0, 1.0)
        inputs[INPUT_HEALTH] = min(agent['health'] / 100.0, 1.0)
        inputs[INPUT_AGE] = min(agent['age'] / 100.0, 1.0)  # Assuming max age ~100
        inputs[INPUT_MATING_DESIRE] = min(agent['mating_desire'] / 100.0, 1.0)
        
        # Find nearest food
        nearest_food_dist, nearest_food_dir = self._find_nearest_food(agent, world)
        inputs[INPUT_NEAREST_FOOD_DISTANCE] = min(nearest_food_dist / 100.0, 1.0)  # Normalize
        inputs[INPUT_NEAREST_FOOD_DIRECTION_X] = nearest_food_dir[0]
        inputs[INPUT_NEAREST_FOOD_DIRECTION_Y] = nearest_food_dir[1]
        
        # Find nearest potential mate
        nearest_mate_dist, nearest_mate_dir = self._find_nearest_mate(agent_idx, agents)
        inputs[INPUT_NEAREST_MATE_DISTANCE] = min(nearest_mate_dist / 100.0, 1.0)  # Normalize
        inputs[INPUT_NEAREST_MATE_DIRECTION_X] = nearest_mate_dir[0]
        inputs[INPUT_NEAREST_MATE_DIRECTION_Y] = nearest_mate_dir[1]
        
        # Population density around agent
        inputs[INPUT_POPULATION_DENSITY] = self._get_local_population_density(agent_idx, agents)
        
        # Current tile resources
        tile_y, tile_x = int(agent['pos'][0]), int(agent['pos'][1])
        tile_y = max(0, min(world.shape[0] - 1, tile_y))
        tile_x = max(0, min(world.shape[1] - 1, tile_x))
        inputs[INPUT_CURRENT_RESOURCES] = min(world[tile_y, tile_x]['resources'] / 100.0, 1.0)
        
        terrain_type = world[tile_y, tile_x]['terrain']
        # Normalize the terrain type (0-3) to a value between 0 and 1
        inputs[INPUT_TERRAIN_TYPE] = terrain_type / 3.0
        
        return inputs
    
    def _find_nearest_food(self, agent: np.ndarray, world: np.ndarray) -> tuple:
        """Find nearest food source and return distance and direction"""
        agent_pos = agent['pos']
        min_dist = float('inf')
        best_direction = np.array([0.0, 0.0])
        
        world_height, world_width = world.shape
        
        # Sample a subset of world tiles for performance
        sample_size = min(100, world_height * world_width // 10)
        sampled_positions = []
        
        for _ in range(sample_size):
            y = np.random.randint(0, world_height)
            x = np.random.randint(0, world_width)
            if world[y, x]['resources'] > 10:
                sampled_positions.append((y, x))
        
        for y, x in sampled_positions:
            dist_sq = (agent_pos[0] - y)**2 + (agent_pos[1] - x)**2
            if dist_sq < min_dist:
                min_dist = dist_sq
                direction = np.array([y - agent_pos[0], x - agent_pos[1]])
                norm = np.sqrt(direction[0]**2 + direction[1]**2)
                if norm > 0:
                    best_direction = direction / norm
        
        return np.sqrt(min_dist), best_direction
    
    def _find_nearest_mate(self, agent_idx: int, agents: np.ndarray) -> tuple:
        """Find nearest potential mate and return distance and direction"""
        agent = agents[agent_idx]
        min_dist = float('inf')
        best_direction = np.array([0.0, 0.0])
        
        for i, other_agent in enumerate(agents):
            if (i != agent_idx and other_agent['health'] > 0 and 
                other_agent['sex'] != agent['sex'] and other_agent['is_fertile'] and
                not other_agent['is_pregnant']):
                
                dist_sq = ((agent['pos'][0] - other_agent['pos'][0])**2 + 
                          (agent['pos'][1] - other_agent['pos'][1])**2)
                
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    direction = other_agent['pos'] - agent['pos']
                    norm = np.sqrt(direction[0]**2 + direction[1]**2)
                    if norm > 0:
                        best_direction = direction / norm
        
        return np.sqrt(min_dist), best_direction
    
    def _get_local_population_density(self, agent_idx: int, agents: np.ndarray, radius: float = 20.0) -> float:
        """Get population density around an agent"""
        agent = agents[agent_idx]
        count = 0
        
        for i, other_agent in enumerate(agents):
            if i != agent_idx and other_agent['health'] > 0:
                dist_sq = ((agent['pos'][0] - other_agent['pos'][0])**2 + 
                          (agent['pos'][1] - other_agent['pos'][1])**2)
                if dist_sq <= radius**2:
                    count += 1
        
        # Normalize by area
        return min(count / (np.pi * radius**2) * 1000, 1.0)  # Scale and cap at 1.0
    
    def make_decision(self, agent_idx: int, world: np.ndarray, agents: np.ndarray) -> dict:
        """Use NEAT brain to make decisions for an agent"""
        # --- FIX 2: Check for None *before* accessing the attribute ---
        brain = self.brains[agent_idx]
        if brain is None:
            # Fallback to random behavior if no brain
            return {
                'move_x': np.random.randn(),
                'move_y': np.random.randn(), 
                'seek_food': 0.5,
                'seek_mate': 0.5,
                'rest': 0.5
            }
        
        # Get inputs for the brain
        inputs = self.get_brain_inputs(agent_idx, world, agents)
        
        # Evaluate the neural network
        outputs = brain.evaluate(inputs)
        
        # Convert outputs to decisions
        return {
            'move_x': np.tanh(outputs[OUTPUT_MOVE_X]),
            'move_y': np.tanh(outputs[OUTPUT_MOVE_Y]),
            'seek_food': outputs[OUTPUT_SEEK_FOOD],
            'seek_mate': outputs[OUTPUT_SEEK_MATE],
            'rest': outputs[OUTPUT_REST]
        }

    # In class AgentManager:
    def create_offspring(self, mother_idx: int, father_idx: int, 
                        offspring_id: int, offspring_pos: np.ndarray) -> int:
        """
        Creates an offspring, handling genetics and initialization.
        This function is now guaranteed to succeed if a slot is available.
        """
        # 1. Find an empty slot for the newborn.
        offspring_idx = -1
        for i in range(self.max_population):
            if self.agents[i]['health'] <= 0:
                offspring_idx = i
                break
        
        if offspring_idx == -1:
            return -1  # Population is full, birth fails.

        # 2. Handle Genetics
        parent1_genome = self.genomes[mother_idx]
        parent2_genome = self.genomes[father_idx]
        
        offspring_genome = None
        if parent1_genome is not None and parent2_genome is not None:
            offspring_genome = crossover(parent1_genome, parent2_genome)
            mutate_genome(offspring_genome)
        
        # Fallback: If crossover fails or parents have no genome, create a random one.
        if offspring_genome is None:
            offspring_genome = create_random_genome()

        self.genomes[offspring_idx] = offspring_genome
        self.brains[offspring_idx] = NEATBrain(offspring_genome)

        # 3. Initialize the newborn's state in the NumPy array.
        offspring = self.agents[offspring_idx]
        offspring['id'] = offspring_id
        offspring['pos'] = offspring_pos
        offspring['health'] = config.NEWBORN_HEALTH
        offspring['hunger'] = config.NEWBORN_HUNGER
        offspring['age'] = 0
        offspring['sex'] = np.random.choice([SEX_MALE, SEX_FEMALE])
        offspring['is_fertile'] = False
        offspring['is_pregnant'] = 0
        offspring['partner_id'] = -1
        offspring['mating_desire'] = 0.0
        offspring['fitness'] = 0.0
        offspring['generation'] = self.generation + 1 # Child is of the next generation

        # 4. Apply penalty to the mother.
        self.agents[mother_idx]['health'] -= config.MOTHER_HEALTH_PENALTY

        return offspring_idx
    def update_fitness(self, agent_idx: int, fitness_delta: float):
        """
        Updates the fitness score for a specific agent and its corresponding genome.

        Args:
            agent_idx (int): The index of the agent in the numpy array.
            fitness_delta (float): The amount to add (or subtract) from the fitness.
        """
        # Ensure the agent index is valid
        if 0 <= agent_idx < self.max_population:
            # Update the fitness score in the NumPy array
            self.agents[agent_idx]['fitness'] += fitness_delta
            
            # Also update the fitness score on the genome object itself,
            # so it can be used for crossover and evolution.
            genome = self.genomes[agent_idx]
            if genome is not None:
                genome.fitness += fitness_delta
    # In class AgentManager:

    def cull_the_dead(self):
        """
        Finds all agents with health <= 0, resets their genome and brain,
        making their slots available for new births.
        """
        dead_indices = np.where(self.agents['health'] <= 0)[0]
        
        # We only need to clear the slots if there are any dead agents
        if len(dead_indices) == 0:
            return

        print(f"Culling {len(dead_indices)} dead agents.")
        
        for i in dead_indices:
            # An agent that is already dead might not have a genome
            if self.genomes[i] is not None:
                # Optional: You could log the fitness of the dying agent here
                # print(f"Agent {self.agents[i]['id']} died with fitness {self.genomes[i].fitness}")
                self.genomes[i] = None
                self.brains[i] = None
        
        # Note: We don't need to touch the numpy array data (health, pos, etc.)
        # The create_offspring function will overwrite it completely.
    # In class AgentManager:

    def get_population_stats(self) -> dict:
        """
        Analyzes the current population and returns a dictionary of key statistics.
        """
        active_indices = np.where(self.agents['health'] > 0)[0]
        
        if len(active_indices) == 0:
            return {'population': 0}
        
        active_agents = self.agents[active_indices]
        
        # --- Genetic Stats ---
        generations = active_agents['generation']
        
        # --- Brain Complexity Stats ---
        num_nodes = []
        num_connections = []
        for i in active_indices:
            genome = self.genomes[i]
            if genome:
                num_nodes.append(len(genome.nodes))
                num_connections.append(len([c for c in genome.connections.values() if c.enabled]))

        return {
            'population': len(active_agents),
            'avg_fitness': np.mean(active_agents['fitness']),
            'max_fitness': np.max(active_agents['fitness']),
            'avg_age': np.mean(active_agents['age']),
            'max_age': np.max(active_agents['age']),
            'avg_generation': np.mean(generations),
            'max_generation': np.max(generations),
            'avg_nodes': np.mean(num_nodes) if num_nodes else 0,
            'avg_connections': np.mean(num_connections) if num_connections else 0,
        }