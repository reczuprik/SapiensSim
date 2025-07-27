import numpy as np
from .neat_brain import linear # Assuming this is where terrain constants are/will be

# Define terrain types as integer constants for performance.
TERRAIN_PLAINS = 0
TERRAIN_FOREST = 1
TERRAIN_WATER = 2
TERRAIN_MOUNTAIN = 3

def create_world(width: int, height: int) -> np.ndarray:
    """
    Creates a world with more realistic geography, including a mountain
    range and a fertile river valley.
    """
    world_dtype = np.dtype([
        ('resources', np.float32),
        ('terrain', np.int8)
    ])
    world = np.zeros((height, width), dtype=world_dtype)

    # --- Generate a Mountain Range ---
    # Create a wavy line for the mountain range spine
    mountain_x = np.arange(width)
    mountain_y = (height / 2) + np.sin(mountain_x / 20) * 15 + np.random.randn(width) * 5
    
    # Create a grid of coordinates
    y_coords, x_coords = np.indices((height, width))
    
    # Calculate distance from each point to the mountain spine
    dist_to_mountain = np.abs(y_coords - mountain_y[:, np.newaxis].T)
    
    # Create mountain and high-altitude plains zones
    mountain_zone = dist_to_mountain < 10
    high_plains_zone = (dist_to_mountain >= 10) & (dist_to_mountain < 25)

    world['terrain'][mountain_zone] = TERRAIN_MOUNTAIN
    # Mountains are barren
    world['resources'][mountain_zone] = 0

    # --- Generate a Fertile River Valley ---
    # The valley runs parallel to the mountains
    valley_zone = (dist_to_mountain >= 25) & (dist_to_mountain < 50)
    world['terrain'][valley_zone] = TERRAIN_PLAINS
    
    # Forests grow near the river (center of the valley)
    forest_zone = (dist_to_mountain >= 30) & (dist_to_mountain < 45)
    world['terrain'][forest_zone] = TERRAIN_FOREST
    
    # Populate resources based on biome
    # Forests are very rich
    world['resources'][forest_zone] = np.random.uniform(60, 100, np.sum(forest_zone))
    # Plains in the valley are moderately rich
    world['resources'][valley_zone & (world['terrain'] == TERRAIN_PLAINS)] = np.random.uniform(20, 40, np.sum(valley_zone & (world['terrain'] == TERRAIN_PLAINS)))
    # High plains are sparse
    world['resources'][high_plains_zone] = np.random.uniform(0, 10, np.sum(high_plains_zone))

    print(f"Realistic world created with size {width}x{height}.")
    print(f"Forest cells: {np.sum(forest_zone)}, Mountain cells: {np.sum(mountain_zone)}")
    
    return world