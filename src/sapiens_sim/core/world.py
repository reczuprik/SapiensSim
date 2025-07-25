# FILE_NAME: src/sapiens_sim/core/world.py
# CODE_BLOCK_ID: SapiensSim-v0.2-world.py

import numpy as np

# Define terrain types as integer constants for performance.
# Using integers is much faster and more memory-efficient than strings.
TERRAIN_PLAINS = 0
TERRAIN_FOREST = 1
TERRAIN_WATER = 2
TERRAIN_MOUNTAIN = 3

def create_world(width: int, height: int) -> np.ndarray:
    """
    Creates the simulation world as a 2D NumPy structured array.

    Each cell in the world grid contains information about its terrain
    and available resources.

    Args:
        width (int): The width of the world grid.
        height (int): The height of the world grid.

    Returns:
        np.ndarray: A 2D array representing the world.
    """
    # Define the data type (dtype) for a single cell in our world.
    # This is a 'structured array', which is like a super-efficient spreadsheet.
    # Each cell has a 'resources' field (a 32-bit float) and a
    # 'terrain' field (an 8-bit integer).
    world_dtype = np.dtype([
        ('resources', np.float32),
        ('terrain', np.int8)
    ])

    # Create the 2D grid, initialized with zeros.
    world = np.zeros((height, width), dtype=world_dtype)

    # --- Initial World Generation ---
    # This is where we can get creative. For now, we will create a simple
    # world: mostly plains, with some randomly scattered forests.
    # We will leave more advanced generation (like Perlin noise for continents)
    # for a future step.

    # Initialize all terrain to PLAINS
    world['terrain'] = TERRAIN_PLAINS

    # Create some random patches of FOREST
    # Generate a random float for each cell, and if it's below a threshold,
    # turn that cell into a forest.
    forest_patches = np.random.rand(height, width) < 0.15 # 15% chance of forest
    world['terrain'][forest_patches] = TERRAIN_FOREST

    # Populate resources based on terrain type
    # Plains have a moderate amount of resources.
    world['resources'][world['terrain'] == TERRAIN_PLAINS] = np.random.uniform(10, 30, np.sum(world['terrain'] == TERRAIN_PLAINS))
    
    # Forests are rich in resources.
    world['resources'][world['terrain'] == TERRAIN_FOREST] = np.random.uniform(40, 80, np.sum(world['terrain'] == TERRAIN_FOREST))

    print(f"World created with size {width}x{height}.")
    print(f"Total cells: {width*height}")
    print(f"Forest cells: {np.sum(forest_patches)}")
    
    return world