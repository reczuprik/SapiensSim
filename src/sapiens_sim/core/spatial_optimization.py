# FILE: src/sapiens_sim/core/spatial_optimization.py
# High-performance spatial indexing system for massive speedup

import numpy as np
from numba import jit, types, prange
from numba.typed import Dict as NumbaDict, List as NumbaList
import math

@jit(nopython=True)
def euclidean_distance_squared(pos1, pos2):
    """Fast squared distance calculation"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return dx * dx + dy * dy

@jit(nopython=True)
def normalize_direction(direction):
    """Fast direction normalization"""
    norm = math.sqrt(direction[0]**2 + direction[1]**2)
    if norm > 0:
        return np.array([direction[0] / norm, direction[1] / norm])
    return np.array([0.0, 0.0])

class SpatialGrid:
    """High-performance spatial grid for O(1) neighbor queries"""
    
    def __init__(self, world_width, world_height, cell_size=15):
        self.cell_size = cell_size
        self.grid_width = world_width // cell_size + 1
        self.grid_height = world_height // cell_size + 1
        self.world_width = world_width
        self.world_height = world_height
        
        # Pre-allocate grid cells - much faster than dynamic dict
        self.grid = {}
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                self.grid[(i, j)] = []
        
        # Cache for agent positions to avoid recalculation
        self.agent_positions = {}
        
    def _get_cell_coords(self, pos):
        """Convert world position to grid coordinates"""
        grid_x = min(max(int(pos[1] // self.cell_size), 0), self.grid_width - 1)
        grid_y = min(max(int(pos[0] // self.cell_size), 0), self.grid_height - 1)
        return grid_y, grid_x
    
    def update_agent(self, agent_idx, new_pos):
        """Update agent position in grid"""
        # Remove from old cell if exists
        if agent_idx in self.agent_positions:
            old_cell = self.agent_positions[agent_idx]
            if agent_idx in self.grid[old_cell]:
                self.grid[old_cell].remove(agent_idx)
        
        # Add to new cell
        new_cell = self._get_cell_coords(new_pos)
        self.grid[new_cell].append(agent_idx)
        self.agent_positions[agent_idx] = new_cell
    
    def remove_agent(self, agent_idx):
        """Remove agent from grid"""
        if agent_idx in self.agent_positions:
            cell = self.agent_positions[agent_idx]
            if agent_idx in self.grid[cell]:
                self.grid[cell].remove(agent_idx)
            del self.agent_positions[agent_idx]
    
    def get_nearby_agents(self, pos, radius=20):
        """Get agents within radius - MASSIVE speedup vs O(n) search"""
        cells_to_check = self._get_cells_in_radius(pos, radius)
        nearby_agents = []
        
        for cell in cells_to_check:
            nearby_agents.extend(self.grid.get(cell, []))
        
        return nearby_agents
    
    def _get_cells_in_radius(self, pos, radius):
        """Get grid cells that intersect with radius"""
        center_y, center_x = self._get_cell_coords(pos)
        cell_radius = int(math.ceil(radius / self.cell_size))
        
        cells = []
        for dy in range(-cell_radius, cell_radius + 1):
            for dx in range(-cell_radius, cell_radius + 1):
                grid_y = center_y + dy
                grid_x = center_x + dx
                
                if (0 <= grid_y < self.grid_height and 
                    0 <= grid_x < self.grid_width):
                    cells.append((grid_y, grid_x))
        
        return cells

class ResourceQuadTree:
    """Spatial indexing for food resources"""
    
    def __init__(self, world_width, world_height, max_depth=6):
        self.world_width = world_width
        self.world_height = world_height
        self.max_depth = max_depth
        self.root = self._build_quadtree(0, 0, world_width, world_height, 0)
        
    def _build_quadtree(self, x, y, width, height, depth):
        """Build quadtree recursively"""
        node = {
            'x': x, 'y': y, 'width': width, 'height': height,
            'depth': depth, 'resources': [], 'children': None
        }
        
        if depth < self.max_depth and width > 4 and height > 4:
            # Create children
            half_w, half_h = width // 2, height // 2
            node['children'] = [
                self._build_quadtree(x, y, half_w, half_h, depth + 1),
                self._build_quadtree(x + half_w, y, width - half_w, half_h, depth + 1),
                self._build_quadtree(x, y + half_h, half_w, height - half_h, depth + 1),
                self._build_quadtree(x + half_w, y + half_h, width - half_w, height - half_h, depth + 1)
            ]
        
        return node
    
    def update_resources(self, world):
        """Rebuild resource index from world state"""
        self._clear_resources(self.root)
        world_height, world_width = world.shape
        
        for y in range(world_height):
            for x in range(world_width):
                if world[y, x]['resources'] > 0:
                    self._insert_resource(self.root, x, y, world[y, x]['resources'])
    
    def _clear_resources(self, node):
        """Clear all resources from quadtree"""
        node['resources'].clear()
        if node['children']:
            for child in node['children']:
                self._clear_resources(child)
    
    def _insert_resource(self, node, x, y, amount):
        """Insert resource into quadtree"""
        if not self._point_in_bounds(node, x, y):
            return
        
        node['resources'].append((x, y, amount))
        
        if node['children']:
            for child in node['children']:
                self._insert_resource(child, x, y, amount)
    
    def _point_in_bounds(self, node, x, y):
        """Check if point is within node bounds"""
        return (node['x'] <= x < node['x'] + node['width'] and
                node['y'] <= y < node['y'] + node['height'])
    
    def query_radius(self, pos, radius, min_resources=10):
        """Find resources within radius"""
        results = []
        self._query_radius_recursive(self.root, pos, radius, min_resources, results)
        return results
    
    def _query_radius_recursive(self, node, pos, radius, min_resources, results):
        """Recursive radius query"""
        if not self._circle_intersects_rect(pos, radius, node):
            return
        
        # Check resources in this node
        for x, y, amount in node['resources']:
            if amount >= min_resources:
                dist_sq = (pos[1] - x)**2 + (pos[0] - y)**2
                if dist_sq <= radius**2:
                    results.append((x, y, amount, math.sqrt(dist_sq)))
        
        # Recurse to children
        if node['children']:
            for child in node['children']:
                self._query_radius_recursive(child, pos, radius, min_resources, results)
    
    def _circle_intersects_rect(self, pos, radius, node):
        """Check if circle intersects rectangle"""
        cx, cy = pos[1], pos[0]  # Convert to world coordinates
        
        # Find closest point on rectangle to circle center
        closest_x = max(node['x'], min(cx, node['x'] + node['width']))
        closest_y = max(node['y'], min(cy, node['y'] + node['height']))
        
        # Check if distance is within radius
        dist_sq = (cx - closest_x)**2 + (cy - closest_y)**2
        return dist_sq <= radius**2

@jit(nopython=True)
def batch_find_nearest_mates(positions, sexes, fertile, health, ages, mating_desires, 
                           min_age=18, max_distance=50.0):
    """Vectorized mate finding - MASSIVE speedup"""
    n_agents = len(positions)
    nearest_mates = np.full(n_agents, -1, dtype=np.int32)
    nearest_distances = np.full(n_agents, np.inf, dtype=np.float32)
    mate_directions = np.zeros((n_agents, 2), dtype=np.float32)
    
    for i in prange(n_agents):
        if health[i] <= 0 or not fertile[i] or mating_desires[i] < 50.0:
            continue
            
        agent_pos = positions[i]
        agent_sex = sexes[i]
        min_dist_sq = max_distance * max_distance
        best_mate_idx = -1
        best_direction = np.zeros(2, dtype=np.float32)
        
        for j in range(n_agents):
            if (i != j and health[j] > 0 and fertile[j] and 
                sexes[j] != agent_sex and ages[j] >= min_age and
                mating_desires[j] > 50.0):
                
                # Fast squared distance
                dx = positions[j][0] - agent_pos[0]
                dy = positions[j][1] - agent_pos[1]
                dist_sq = dx * dx + dy * dy
                
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_mate_idx = j
                    
                    # Normalized direction
                    dist = math.sqrt(dist_sq)
                    if dist > 0:
                        best_direction[0] = dx / dist
                        best_direction[1] = dy / dist
        
        if best_mate_idx >= 0:
            nearest_mates[i] = best_mate_idx
            nearest_distances[i] = math.sqrt(min_dist_sq)
            mate_directions[i] = best_direction
    
    return nearest_mates, nearest_distances, mate_directions

@jit(nopython=True)
def batch_find_nearest_food(agent_positions, resource_positions, resource_amounts, 
                          max_distance=100.0, min_resources=10.0):
    """Vectorized food finding using pre-built resource arrays"""
    n_agents = len(agent_positions)
    n_resources = len(resource_positions)
    
    nearest_food_distances = np.full(n_agents, np.inf, dtype=np.float32)
    food_directions = np.zeros((n_agents, 2), dtype=np.float32)
    
    for i in prange(n_agents):
        agent_pos = agent_positions[i]
        min_dist_sq = max_distance * max_distance
        best_direction = np.zeros(2, dtype=np.float32)
        
        for j in range(n_resources):
            if resource_amounts[j] >= min_resources:
                dx = resource_positions[j][0] - agent_pos[0]
                dy = resource_positions[j][1] - agent_pos[1]
                dist_sq = dx * dx + dy * dy
                
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    dist = math.sqrt(dist_sq)
                    if dist > 0:
                        best_direction[0] = dx / dist
                        best_direction[1] = dy / dist
        
        if min_dist_sq < max_distance * max_distance:
            nearest_food_distances[i] = math.sqrt(min_dist_sq)
            food_directions[i] = best_direction
    
    return nearest_food_distances, food_directions

class OptimizedSpatialManager:
    """Combines all spatial optimizations"""
    
    def __init__(self, world_width, world_height):
        self.spatial_grid = SpatialGrid(world_width, world_height)
        self.resource_quadtree = ResourceQuadTree(world_width, world_height)
        self.world_width = world_width
        self.world_height = world_height
        
        # Cache for resource positions
        self.resource_positions = np.zeros((0, 2), dtype=np.float32)
        self.resource_amounts = np.zeros(0, dtype=np.float32)
        self.resource_cache_valid = False
        
    def update_agent_positions(self, agents):
        """Update all agent positions in spatial grid"""
        for i, agent in enumerate(agents):
            if agent['health'] > 0:
                self.spatial_grid.update_agent(i, agent['pos'])
            else:
                self.spatial_grid.remove_agent(i)
    
    def update_resource_cache(self, world):
        """Update cached resource positions for fast access"""
        positions = []
        amounts = []
        
        world_height, world_width = world.shape
        for y in range(world_height):
            for x in range(world_width):
                if world[y, x]['resources'] > 0:
                    positions.append([y, x])
                    amounts.append(world[y, x]['resources'])
        
        self.resource_positions = np.array(positions, dtype=np.float32)
        self.resource_amounts = np.array(amounts, dtype=np.float32)
        self.resource_cache_valid = True
        
        # Also update quadtree
        self.resource_quadtree.update_resources(world)
    
    def batch_find_mates_and_food(self, agents):
        """Find nearest mates and food for all agents simultaneously"""
        active_mask = agents['health'] > 0
        active_agents = agents[active_mask]
        
        if len(active_agents) == 0:
            return {}, {}
        
        # Extract positions and properties
        positions = active_agents['pos']
        sexes = active_agents['sex']
        fertile = active_agents['is_fertile']
        health = active_agents['health']
        ages = active_agents['age']
        mating_desires = active_agents['mating_desire']
        
        # Batch mate finding
        nearest_mates, mate_distances, mate_directions = batch_find_nearest_mates(
            positions, sexes, fertile, health, ages, mating_desires
        )
        
        # Batch food finding
        if not self.resource_cache_valid:
            food_distances = np.full(len(active_agents), np.inf, dtype=np.float32)
            food_directions = np.zeros((len(active_agents), 2), dtype=np.float32)
        else:
            food_distances, food_directions = batch_find_nearest_food(
                positions, self.resource_positions, self.resource_amounts
            )
        
        # Create result dictionaries
        active_indices = np.where(active_mask)[0]
        
        mate_results = {}
        food_results = {}
        
        for i, agent_idx in enumerate(active_indices):
            mate_results[agent_idx] = {
                'nearest_mate_idx': nearest_mates[i] if nearest_mates[i] >= 0 else -1,
                'distance': mate_distances[i],
                'direction': mate_directions[i]
            }
            
            food_results[agent_idx] = {
                'distance': food_distances[i],
                'direction': food_directions[i]
            }
        
        return mate_results, food_results
    
    def get_nearby_agents(self, pos, radius=20):
        """Get agents near position using spatial grid"""
        return self.spatial_grid.get_nearby_agents(pos, radius)
    
    def get_nearby_resources(self, pos, radius=50, min_resources=10):
        """Get resources near position using quadtree"""
        return self.resource_quadtree.query_radius(pos, radius, min_resources)