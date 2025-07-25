# FILE_NAME: tests/test_world.py
# CODE_BLOCK_ID: SapiensSim-v0.2-test_world.py

import numpy as np
import pytest

# We need to import the function and constants we want to test
from sapiens_sim.core.world import create_world, TERRAIN_PLAINS, TERRAIN_FOREST, TERRAIN_WATER, TERRAIN_MOUNTAIN

def test_create_world_properties():
    """
    Tests the fundamental properties of a newly created world.
    """
    width, height = 100, 80

    # 1. Call the function we are testing
    world = create_world(width, height)

    # 2. Assert the outcomes are what we expect
    
    # Check if the shape is correct
    assert world.shape == (height, width), "World dimensions are incorrect."

    # Check if the dtype contains the correct fields
    assert 'resources' in world.dtype.names, "World dtype is missing 'resources' field."
    assert 'terrain' in world.dtype.names, "World dtype is missing 'terrain' field."

    # Check if the data types of the fields are correct
    assert world.dtype['resources'] == np.float32, "'resources' field has wrong numpy type."
    assert world.dtype['terrain'] == np.int8, "'terrain' field has wrong numpy type."
    
    # Check that terrain values are valid
    # This ensures we don't have any unexpected terrain types.
    allowed_terrains = [TERRAIN_PLAINS, TERRAIN_FOREST, TERRAIN_WATER, TERRAIN_MOUNTAIN]
    assert np.all(np.isin(world['terrain'], allowed_terrains)), "World contains invalid terrain values."

    # Check that resources have been initialized
    assert np.sum(world['resources']) > 0, "Resources were not initialized."

