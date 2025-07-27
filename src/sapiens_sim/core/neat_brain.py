# FILE_NAME: src/sapiens_sim/core/neat_brain.py
# CODE_BLOCK_ID: SapiensSim-NEAT-v1.0-neat_brain.py

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from numba import jit, types
from numba.typed import Dict as NumbaDict
import math

# Activation functions
@jit(nopython=True)
def sigmoid(x: float) -> float:
    """Sigmoid activation function"""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

@jit(nopython=True)
def tanh_activation(x: float) -> float:
    """Tanh activation function"""
    return math.tanh(max(-500, min(500, x)))

@jit(nopython=True)
def relu(x: float) -> float:
    """ReLU activation function"""
    return max(0.0, x)

@jit(nopython=True)
def linear(x: float) -> float:
    """Linear activation function"""
    return x

# Innovation number tracking for NEAT
class InnovationTracker:
    def __init__(self):
        self.connection_innovations: Dict[Tuple[int, int], int] = {}
        self.node_innovations: Dict[Tuple[int, int], int] = {}
        self.current_innovation = 0
    
    def get_connection_innovation(self, from_node: int, to_node: int) -> int:
        """Get innovation number for a connection"""
        key = (from_node, to_node)
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.current_innovation
            self.current_innovation += 1
        return self.connection_innovations[key]
    
    def get_node_innovation(self, from_node: int, to_node: int) -> int:
        """Get innovation number for a new node splitting a connection"""
        key = (from_node, to_node)
        if key not in self.node_innovations:
            self.node_innovations[key] = self.current_innovation
            self.current_innovation += 1
        return self.node_innovations[key]

# Global innovation tracker
GLOBAL_INNOVATION = InnovationTracker()

@dataclass
class ConnectionGene:
    """Represents a connection between two nodes"""
    from_node: int
    to_node: int
    weight: float
    enabled: bool
    innovation: int
    
    def copy(self) -> 'ConnectionGene':
        return ConnectionGene(
            self.from_node, 
            self.to_node, 
            self.weight, 
            self.enabled, 
            self.innovation
        )

@dataclass
class NodeGene:
    """Represents a node in the network"""
    node_id: int
    node_type: str  # 'input', 'hidden', 'output'
    activation: str = 'sigmoid'  # activation function type
    
    def copy(self) -> 'NodeGene':
        return NodeGene(self.node_id, self.node_type, self.activation)

class NEATGenome:
    """NEAT genome representing a neural network"""
    
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.species_id = 0
        
        # Create input and output nodes
        self._initialize_nodes()
        
        # Create initial random connections
        self._initialize_connections()
    
    def _initialize_nodes(self):
        """Initialize input and output nodes"""
        # Input nodes (0 to input_size-1)
        for i in range(self.input_size):
            self.nodes[i] = NodeGene(i, 'input', 'linear')
        
        # Output nodes (input_size to input_size+output_size-1)
        for i in range(self.output_size):
            node_id = self.input_size + i
            self.nodes[node_id] = NodeGene(node_id, 'output', 'sigmoid')
    
    def _initialize_connections(self):
        """Create initial random connections from inputs to outputs"""
        for input_id in range(self.input_size):
            for output_id in range(self.input_size, self.input_size + self.output_size):
                if random.random() < 0.5:  # 50% chance of initial connection
                    innovation = GLOBAL_INNOVATION.get_connection_innovation(input_id, output_id)
                    weight = random.uniform(-2.0, 2.0)
                    self.connections[innovation] = ConnectionGene(
                        input_id, output_id, weight, True, innovation
                    )
    
    def add_connection_mutation(self):
        """Add a new connection between two nodes"""
        possible_connections = []
        
        # Find all possible connections that don't exist
        for from_id in self.nodes:
            for to_id in self.nodes:
                if from_id != to_id and not self._connection_exists(from_id, to_id):
                    # Prevent connections from output to input/output
                    if (self.nodes[from_id].node_type != 'output' and 
                        self.nodes[to_id].node_type != 'input'):
                        possible_connections.append((from_id, to_id))
        
        if possible_connections:
            from_id, to_id = random.choice(possible_connections)
            innovation = GLOBAL_INNOVATION.get_connection_innovation(from_id, to_id)
            weight = random.uniform(-2.0, 2.0)
            self.connections[innovation] = ConnectionGene(
                from_id, to_id, weight, True, innovation
            )
    
    def add_node_mutation(self):
        """Add a new node by splitting an existing connection"""
        enabled_connections = [conn for conn in self.connections.values() if conn.enabled]
        
        if enabled_connections:
            # Choose random connection to split
            conn = random.choice(enabled_connections)
            
            # Disable the original connection
            conn.enabled = False
            
            # Create new node
            new_node_id = max(self.nodes.keys()) + 1
            self.nodes[new_node_id] = NodeGene(new_node_id, 'hidden', 'sigmoid')
            
            # Create two new connections
            # Connection from original source to new node (weight = 1.0)
            innovation1 = GLOBAL_INNOVATION.get_connection_innovation(conn.from_node, new_node_id)
            self.connections[innovation1] = ConnectionGene(
                conn.from_node, new_node_id, 1.0, True, innovation1
            )
            
            # Connection from new node to original target (original weight)
            innovation2 = GLOBAL_INNOVATION.get_connection_innovation(new_node_id, conn.to_node)
            self.connections[innovation2] = ConnectionGene(
                new_node_id, conn.to_node, conn.weight, True, innovation2
            )
    
    def weight_mutation(self, mutation_rate: float = 0.8, perturbation_rate: float = 0.9):
        """Mutate connection weights"""
        for conn in self.connections.values():
            if random.random() < mutation_rate:
                if random.random() < perturbation_rate:
                    # Small perturbation
                    conn.weight += random.uniform(-0.5, 0.5)
                else:
                    # Complete replacement
                    conn.weight = random.uniform(-2.0, 2.0)
                
                # Clamp weights
                conn.weight = max(-5.0, min(5.0, conn.weight))
    
    def _connection_exists(self, from_id: int, to_id: int) -> bool:
        """Check if a connection already exists"""
        return any(conn.from_node == from_id and conn.to_node == to_id 
                  for conn in self.connections.values())
    
    def copy(self) -> 'NEATGenome':
        """Create a copy of this genome"""
        new_genome = NEATGenome(self.input_size, self.output_size)
        new_genome.nodes = {k: v.copy() for k, v in self.nodes.items()}
        new_genome.connections = {k: v.copy() for k, v in self.connections.items()}
        new_genome.fitness = self.fitness
        new_genome.adjusted_fitness = self.adjusted_fitness
        new_genome.species_id = self.species_id
        return new_genome

class NEATBrain:
    """Neural network brain that can be evaluated efficiently"""
    
    def __init__(self, genome: NEATGenome):
        self.genome = genome
        self.input_size = genome.input_size
        self.output_size = genome.output_size
        
        # Pre-compute network structure for fast evaluation
        self._compile_network()
    
    def _compile_network(self):
        """Compile the network into arrays for fast numba evaluation"""
        # Get all nodes in topological order
        self.node_order = self._topological_sort()
        
        # Create mapping from node_id to index in arrays
        self.node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_order)}
        
        # Create connection arrays
        connections = [conn for conn in self.genome.connections.values() if conn.enabled]
        
        self.connection_from = np.array([self.node_to_idx[conn.from_node] for conn in connections], dtype=np.int32)
        self.connection_to = np.array([self.node_to_idx[conn.to_node] for conn in connections], dtype=np.int32)
        self.connection_weights = np.array([conn.weight for conn in connections], dtype=np.float32)
        
        # Node activation types
        self.node_activations = np.array([
            0 if self.genome.nodes[node_id].activation == 'sigmoid' else
            1 if self.genome.nodes[node_id].activation == 'tanh' else
            2 if self.genome.nodes[node_id].activation == 'relu' else
            3  # linear
            for node_id in self.node_order
        ], dtype=np.int32)
        
        self.num_nodes = len(self.node_order)
    
    def _topological_sort(self) -> List[int]:
        """Sort nodes in topological order for evaluation"""
        # Simple topological sort
        in_degree = {node_id: 0 for node_id in self.genome.nodes}
        
        # Calculate in-degrees
        for conn in self.genome.connections.values():
            if conn.enabled:
                in_degree[conn.to_node] += 1
        
        # Start with nodes that have no incoming connections
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            # Reduce in-degree of connected nodes
            for conn in self.genome.connections.values():
                if conn.enabled and conn.from_node == node_id:
                    in_degree[conn.to_node] -= 1
                    if in_degree[conn.to_node] == 0:
                        queue.append(conn.to_node)
        
        return result
    
    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the network with given inputs"""
        return self._evaluate_numba(
            inputs, 
            self.connection_from,
            self.connection_to, 
            self.connection_weights,
            self.node_activations,
            self.num_nodes,
            self.input_size,
            self.output_size
        )
    
    @staticmethod
    @jit(nopython=True)
    def _evaluate_numba(inputs, connection_from, connection_to, connection_weights, 
                       node_activations, num_nodes, input_size, output_size):
        """Numba-compiled network evaluation"""
        # Initialize node values
        node_values = np.zeros(num_nodes, dtype=np.float32)
        
        # Set input values
        for i in range(input_size):
            node_values[i] = inputs[i]
        
        # Forward propagation
        for conn_idx in range(len(connection_from)):
            from_idx = connection_from[conn_idx]
            to_idx = connection_to[conn_idx]
            weight = connection_weights[conn_idx]
            
            node_values[to_idx] += node_values[from_idx] * weight
        
        # Apply activation functions
        for i in range(input_size, num_nodes):  # Skip input nodes
            activation_type = node_activations[i]
            if activation_type == 0:  # sigmoid
                node_values[i] = sigmoid(node_values[i])
            elif activation_type == 1:  # tanh
                node_values[i] = tanh_activation(node_values[i])
            elif activation_type == 2:  # relu
                node_values[i] = relu(node_values[i])
            # else: linear (no change)
        
        # Extract outputs
        outputs = np.zeros(output_size, dtype=np.float32)
        for i in range(output_size):
            outputs[i] = node_values[input_size + i]
        
        return outputs

def crossover(parent1: NEATGenome, parent2: NEATGenome) -> NEATGenome:
    """Create offspring through crossover of two genomes"""
    # Ensure parent1 has higher fitness
    if parent2.fitness > parent1.fitness:
        parent1, parent2 = parent2, parent1
    
    child = NEATGenome(parent1.input_size, parent1.output_size)
    child.nodes = {}
    child.connections = {}
    
    # Copy all nodes from both parents
    all_nodes = {}
    all_nodes.update(parent1.nodes)
    all_nodes.update(parent2.nodes)
    child.nodes = {k: v.copy() for k, v in all_nodes.items()}
    
    # Handle connections
    all_innovations = set(parent1.connections.keys()) | set(parent2.connections.keys())
    
    for innovation in all_innovations:
        conn1 = parent1.connections.get(innovation)
        conn2 = parent2.connections.get(innovation)
        
        if conn1 and conn2:
            # Matching gene - randomly choose from either parent
            chosen_conn = random.choice([conn1, conn2])
            child.connections[innovation] = chosen_conn.copy()
            
            # If one parent has disabled gene, 75% chance child is disabled
            if not conn1.enabled or not conn2.enabled:
                if random.random() < 0.75:
                    child.connections[innovation].enabled = False
        elif conn1:
            # Disjoint/Excess gene from fitter parent
            child.connections[innovation] = conn1.copy()
        # Ignore excess/disjoint genes from less fit parent
    
    return child

def compatibility_distance(genome1: NEATGenome, genome2: NEATGenome, 
                         c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
    """Calculate compatibility distance between two genomes"""
    innovations1 = set(genome1.connections.keys())
    innovations2 = set(genome2.connections.keys())
    
    # Count disjoint and excess genes
    all_innovations = innovations1 | innovations2
    matching = innovations1 & innovations2
    disjoint_excess = len(all_innovations - matching)
    
    # Average weight difference of matching connections
    weight_diff = 0.0
    if matching:
        for innovation in matching:
            weight_diff += abs(genome1.connections[innovation].weight - 
                             genome2.connections[innovation].weight)
        weight_diff /= len(matching)
    
    # Normalize by genome size
    N = max(len(genome1.connections), len(genome2.connections), 1)
    
    return (c1 * disjoint_excess / N) + (c3 * weight_diff)

# Input/Output definitions for agent brains
INPUT_HUNGER = 0
INPUT_HEALTH = 1
INPUT_AGE = 2
INPUT_MATING_DESIRE = 3
INPUT_NEAREST_FOOD_DISTANCE = 4
INPUT_NEAREST_FOOD_DIRECTION_X = 5
INPUT_NEAREST_FOOD_DIRECTION_Y = 6
INPUT_NEAREST_MATE_DISTANCE = 7
INPUT_NEAREST_MATE_DIRECTION_X = 8
INPUT_NEAREST_MATE_DIRECTION_Y = 9
INPUT_POPULATION_DENSITY = 10
INPUT_CURRENT_RESOURCES = 11
INPUT_TERRAIN_TYPE = 12 
INPUT_CURRENT_STONE = 13   # Senses stone on the current tile
INPUT_TOOL_DURABILITY = 14   # RENAMED from INPUT_HAS_TOOL
INPUT_SHELTER_DURABILITY = 15  # RENAMED from INPUT_HAS_SHELTER

NUM_INPUTS = 16 


OUTPUT_MOVE_X = 0
OUTPUT_MOVE_Y = 1
OUTPUT_SEEK_FOOD = 2
OUTPUT_SEEK_MATE = 3
OUTPUT_REST = 4
OUTPUT_CRAFT_TOOL = 5      # The decision to make a tool
OUTPUT_BUILD_SHELTER = 6   # The decision to build a shelter
NUM_OUTPUTS = 7

def create_random_genome() -> NEATGenome:
    """Create a random NEAT genome for an agent"""
    return NEATGenome(NUM_INPUTS, NUM_OUTPUTS)

def mutate_genome(genome: NEATGenome, 
                 add_connection_rate: float = 0.05,
                 add_node_rate: float = 0.03,
                 weight_mutation_rate: float = 0.8):
    """Apply mutations to a genome"""
    # Weight mutations
    genome.weight_mutation(weight_mutation_rate)
    
    # Structural mutations
    if random.random() < add_connection_rate:
        genome.add_connection_mutation()
    
    if random.random() < add_node_rate:
        genome.add_node_mutation()