# FILE: src/sapiens_sim/core/batch_neat_optimizer.py
# High-performance batch NEAT brain evaluation system

import numpy as np
from numba import jit, prange, types
from numba.typed import Dict as NumbaDict
import math
from typing import List, Dict, Tuple, Optional

from .neat_brain import (
    NUM_INPUTS, NUM_OUTPUTS,
    INPUT_HUNGER, INPUT_HEALTH, INPUT_AGE, INPUT_MATING_DESIRE,
    INPUT_NEAREST_FOOD_DISTANCE, INPUT_NEAREST_FOOD_DIRECTION_X, INPUT_NEAREST_FOOD_DIRECTION_Y,
    INPUT_NEAREST_MATE_DISTANCE, INPUT_NEAREST_MATE_DIRECTION_X, INPUT_NEAREST_MATE_DIRECTION_Y,
    INPUT_POPULATION_DENSITY, INPUT_CURRENT_RESOURCES,
    OUTPUT_MOVE_X, OUTPUT_MOVE_Y, OUTPUT_SEEK_FOOD, OUTPUT_SEEK_MATE, OUTPUT_REST,
    sigmoid, tanh_activation, relu, linear
)

@jit(nopython=True)
def sigmoid_batch(x):
    """Vectorized sigmoid activation"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

@jit(nopython=True)  
def tanh_batch(x):
    """Vectorized tanh activation"""
    return np.tanh(np.clip(x, -500, 500))

@jit(nopython=True)
def relu_batch(x):
    """Vectorized ReLU activation"""
    return np.maximum(0.0, x)

@jit(nopython=True)
def apply_activation_batch(values, activation_types):
    """Apply activation functions to batch of values"""
    result = np.zeros_like(values)
    
    for i in prange(len(values)):
        if activation_types[i] == 0:  # sigmoid
            result[i] = sigmoid(values[i])
        elif activation_types[i] == 1:  # tanh
            result[i] = tanh_activation(values[i])
        elif activation_types[i] == 2:  # relu
            result[i] = relu(values[i])
        else:  # linear
            result[i] = values[i]
    
    return result

class BatchNetworkCompiler:
    """Compiles NEAT networks into batch-processable format"""
    
    def __init__(self, max_population: int):
        self.max_population = max_population
        self.max_nodes_per_network = 50  # Reasonable upper bound
        self.max_connections_per_network = 200
        
        # Pre-allocated arrays for maximum performance
        self.reset_batch_arrays()
        
    def reset_batch_arrays(self):
        """Reset all batch processing arrays"""
        # Network structure arrays
        self.network_node_counts = np.zeros(self.max_population, dtype=np.int32)
        self.network_connection_counts = np.zeros(self.max_population, dtype=np.int32)
        
        # Node data: [network_idx, local_node_idx] -> global properties
        self.node_activations = np.zeros((self.max_population, self.max_nodes_per_network), dtype=np.int32)
        self.node_values = np.zeros((self.max_population, self.max_nodes_per_network), dtype=np.float32)
        
        # Connection data: [network_idx, connection_idx] -> (from_node, to_node, weight)
        self.connection_from = np.full((self.max_population, self.max_connections_per_network), -1, dtype=np.int32)
        self.connection_to = np.full((self.max_population, self.max_connections_per_network), -1, dtype=np.int32)
        self.connection_weights = np.zeros((self.max_population, self.max_connections_per_network), dtype=np.float32)
        
        # Topological ordering for each network
        self.evaluation_order = np.full((self.max_population, self.max_nodes_per_network), -1, dtype=np.int32)
        
        # Input/output mappings
        self.input_node_indices = np.zeros((self.max_population, NUM_INPUTS), dtype=np.int32)
        self.output_node_indices = np.zeros((self.max_population, NUM_OUTPUTS), dtype=np.int32)
        
        # Network validity mask
        self.valid_networks = np.zeros(self.max_population, dtype=np.bool_)
    
    def compile_network(self, network_idx: int, genome):
        """Compile a single NEAT genome into batch format"""
        if genome is None:
            self.valid_networks[network_idx] = False
            return
        
        try:
            # Get sorted nodes for topological evaluation
            node_order = self._topological_sort(genome)
            
            if len(node_order) > self.max_nodes_per_network:
                print(f"Warning: Network {network_idx} has too many nodes ({len(node_order)}), truncating")
                node_order = node_order[:self.max_nodes_per_network]
            
            # Map genome node IDs to local indices
            node_id_to_local = {node_id: i for i, node_id in enumerate(node_order)}
            
            # Store network metadata
            self.network_node_counts[network_idx] = len(node_order)
            self.valid_networks[network_idx] = True
            
            # Store node data
            for local_idx, node_id in enumerate(node_order):
                node = genome.nodes[node_id]
                if node.activation == 'sigmoid':
                    self.node_activations[network_idx, local_idx] = 0
                elif node.activation == 'tanh':
                    self.node_activations[network_idx, local_idx] = 1
                elif node.activation == 'relu':
                    self.node_activations[network_idx, local_idx] = 2
                else:  # linear
                    self.node_activations[network_idx, local_idx] = 3
                
                self.evaluation_order[network_idx, local_idx] = local_idx
            
            # Store input/output mappings
            for i in range(NUM_INPUTS):
                if i in node_id_to_local:
                    self.input_node_indices[network_idx, i] = node_id_to_local[i]
            
            for i in range(NUM_OUTPUTS):
                output_node_id = NUM_INPUTS + i
                if output_node_id in node_id_to_local:
                    self.output_node_indices[network_idx, i] = node_id_to_local[output_node_id]
            
            # Store connection data
            conn_idx = 0
            for connection in genome.connections.values():
                if connection.enabled and conn_idx < self.max_connections_per_network:
                    from_local = node_id_to_local.get(connection.from_node, -1)
                    to_local = node_id_to_local.get(connection.to_node, -1)
                    
                    if from_local >= 0 and to_local >= 0:
                        self.connection_from[network_idx, conn_idx] = from_local
                        self.connection_to[network_idx, conn_idx] = to_local
                        self.connection_weights[network_idx, conn_idx] = connection.weight
                        conn_idx += 1
            
            self.network_connection_counts[network_idx] = conn_idx
            
        except Exception as e:
            print(f"Failed to compile network {network_idx}: {e}")
            self.valid_networks[network_idx] = False
    
    def _topological_sort(self, genome) -> List[int]:
        """Topological sort of genome nodes"""
        in_degree = {node_id: 0 for node_id in genome.nodes}
        
        # Calculate in-degrees
        for conn in genome.connections.values():
            if conn.enabled:
                in_degree[conn.to_node] += 1
        
        # Start with nodes that have no incoming connections
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            # Reduce in-degree of connected nodes
            for conn in genome.connections.values():
                if conn.enabled and conn.from_node == node_id:
                    in_degree[conn.to_node] -= 1
                    if in_degree[conn.to_node] == 0:
                        queue.append(conn.to_node)
        
        return result

    def compile_all_networks(self, genomes):
        """Compile all networks in the population"""
        for i, genome in enumerate(genomes):
            if i < self.max_population:
                self.compile_network(i, genome)

@jit(nopython=True)
def batch_evaluate_networks(
    batch_inputs,
    active_mask,
    network_node_counts,
    network_connection_counts,
    node_activations,
    connection_from,
    connection_to,
    connection_weights,
    input_node_indices,
    output_node_indices,
    max_depth=10
):
    """
    Evaluate all networks in batch - MASSIVE performance improvement
    """
    batch_size = len(batch_inputs)
    max_nodes = node_activations.shape[1]
    
    # Pre-allocate node values array
    node_values = np.zeros((batch_size, max_nodes), dtype=np.float32)
    outputs = np.zeros((batch_size, NUM_OUTPUTS), dtype=np.float32)
    
    # Set input values for all networks
    for net_idx in prange(batch_size):
        if not active_mask[net_idx]:
            continue
            
        # Set input node values
        for input_idx in range(NUM_INPUTS):
            local_node_idx = input_node_indices[net_idx, input_idx]
            if local_node_idx >= 0:
                node_values[net_idx, local_node_idx] = batch_inputs[net_idx, input_idx]
    
    # Forward propagation for all networks simultaneously
    for depth in range(max_depth):
        for net_idx in prange(batch_size):
            if not active_mask[net_idx]:
                continue
                
            num_connections = network_connection_counts[net_idx]
            
            # Process all connections for this network
            for conn_idx in range(num_connections):
                from_node = connection_from[net_idx, conn_idx]
                to_node = connection_to[net_idx, conn_idx]
                weight = connection_weights[net_idx, conn_idx]
                
                if from_node >= 0 and to_node >= 0:
                    node_values[net_idx, to_node] += node_values[net_idx, from_node] * weight
        
        # Apply activation functions
        for net_idx in prange(batch_size):
            if not active_mask[net_idx]:
                continue
                
            num_nodes = network_node_counts[net_idx]
            
            # Skip input nodes (first NUM_INPUTS nodes)
            for node_idx in range(NUM_INPUTS, num_nodes):
                activation_type = node_activations[net_idx, node_idx]
                value = node_values[net_idx, node_idx]
                
                if activation_type == 0:  # sigmoid
                    node_values[net_idx, node_idx] = sigmoid(value)
                elif activation_type == 1:  # tanh
                    node_values[net_idx, node_idx] = tanh_activation(value)
                elif activation_type == 2:  # relu
                    node_values[net_idx, node_idx] = relu(value)
                # Linear activation: no change needed
    
    # Extract outputs
    for net_idx in prange(batch_size):
        if not active_mask[net_idx]:
            continue
            
        for output_idx in range(NUM_OUTPUTS):
            local_node_idx = output_node_indices[net_idx, output_idx]
            if local_node_idx >= 0:
                outputs[net_idx, output_idx] = node_values[net_idx, local_node_idx]
    
    return outputs

class BatchNEATEvaluator:
    """High-performance batch NEAT brain evaluator"""
    
    def __init__(self, max_population: int):
        self.max_population = max_population
        self.compiler = BatchNetworkCompiler(max_population)
        
        # Pre-allocated arrays
        self.batch_inputs = np.zeros((max_population, NUM_INPUTS), dtype=np.float32)
        self.batch_outputs = np.zeros((max_population, NUM_OUTPUTS), dtype=np.float32)
        self.active_mask = np.zeros(max_population, dtype=np.bool_)
        
        self.networks_compiled = False
    
    def compile_population(self, genomes):
        """Compile all genomes for batch processing"""
        print("Compiling population for batch evaluation...")
        self.compiler.compile_all_networks(genomes)
        self.networks_compiled = True
        
        valid_count = np.sum(self.compiler.valid_networks)
        print(f"Successfully compiled {valid_count}/{len(genomes)} networks")
    
    def generate_batch_inputs(self, agents, world, mate_results, food_results, spatial_manager):
        """Generate input vectors for all active agents simultaneously"""
        active_indices = []
        batch_idx = 0
        
        for i, agent in enumerate(agents):
            if agent['health'] > 0 and batch_idx < self.max_population:
                active_indices.append(i)
                
                # Basic agent state (normalized to 0-1)
                self.batch_inputs[batch_idx, INPUT_HUNGER] = min(agent['hunger'] / 100.0, 1.0)
                self.batch_inputs[batch_idx, INPUT_HEALTH] = min(agent['health'] / 100.0, 1.0)
                self.batch_inputs[batch_idx, INPUT_AGE] = min(agent['age'] / 100.0, 1.0)
                self.batch_inputs[batch_idx, INPUT_MATING_DESIRE] = min(agent['mating_desire'] / 100.0, 1.0)
                
                # Food information
                if i in food_results:
                    food_data = food_results[i]
                    self.batch_inputs[batch_idx, INPUT_NEAREST_FOOD_DISTANCE] = min(food_data['distance'] / 100.0, 1.0)
                    self.batch_inputs[batch_idx, INPUT_NEAREST_FOOD_DIRECTION_X] = food_data['direction'][0]
                    self.batch_inputs[batch_idx, INPUT_NEAREST_FOOD_DIRECTION_Y] = food_data['direction'][1]
                
                # Mate information
                if i in mate_results:
                    mate_data = mate_results[i]
                    self.batch_inputs[batch_idx, INPUT_NEAREST_MATE_DISTANCE] = min(mate_data['distance'] / 100.0, 1.0)
                    self.batch_inputs[batch_idx, INPUT_NEAREST_MATE_DIRECTION_X] = mate_data['direction'][0]
                    self.batch_inputs[batch_idx, INPUT_NEAREST_MATE_DIRECTION_Y] = mate_data['direction'][1]
                
                # Population density
                nearby_agents = spatial_manager.get_nearby_agents(agent['pos'], radius=20.0)
                density = min(len(nearby_agents) / 50.0, 1.0)  # Normalize by expected max
                self.batch_inputs[batch_idx, INPUT_POPULATION_DENSITY] = density
                
                # Current tile resources
                tile_y, tile_x = int(agent['pos'][0]), int(agent['pos'][1])
                tile_y = max(0, min(world.shape[0] - 1, tile_y))
                tile_x = max(0, min(world.shape[1] - 1, tile_x))
                self.batch_inputs[batch_idx, INPUT_CURRENT_RESOURCES] = min(world[tile_y, tile_x]['resources'] / 100.0, 1.0)
                
                self.active_mask[batch_idx] = True
                batch_idx += 1
        
        # Clear remaining slots
        for i in range(batch_idx, self.max_population):
            self.active_mask[i] = False
        
        return active_indices, batch_idx
    
    def evaluate_batch(self, active_indices, batch_size):
        """Evaluate all active networks in batch"""
        if not self.networks_compiled:
            raise RuntimeError("Networks not compiled. Call compile_population() first.")
        
        # Batch evaluation using Numba-compiled function
        self.batch_outputs = batch_evaluate_networks(
            self.batch_inputs[:batch_size],
            self.active_mask[:batch_size],
            self.compiler.network_node_counts,
            self.compiler.network_connection_counts,
            self.compiler.node_activations,
            self.compiler.connection_from,
            self.compiler.connection_to,
            self.compiler.connection_weights,
            self.compiler.input_node_indices,
            self.compiler.output_node_indices
        )
        
        # Convert outputs to decision dictionaries
        decisions = {}
        for batch_idx in range(batch_size):
            if self.active_mask[batch_idx]:
                agent_idx = active_indices[batch_idx]
                outputs = self.batch_outputs[batch_idx]
                
                decisions[agent_idx] = {
                    'move_x': np.tanh(outputs[OUTPUT_MOVE_X]),
                    'move_y': np.tanh(outputs[OUTPUT_MOVE_Y]),
                    'seek_food': outputs[OUTPUT_SEEK_FOOD],
                    'seek_mate': outputs[OUTPUT_SEEK_MATE],
                    'rest': outputs[OUTPUT_REST]
                }
        
        return decisions
    
    def make_batch_decisions(self, agents, world, mate_results, food_results, spatial_manager):
        """Complete batch decision making pipeline"""
        # Generate inputs for all agents
        active_indices, batch_size = self.generate_batch_inputs(
            agents, world, mate_results, food_results, spatial_manager
        )
        
        if batch_size == 0:
            return {}
        
        # Evaluate all networks
        decisions = self.evaluate_batch(active_indices, batch_size)
        
        return decisions