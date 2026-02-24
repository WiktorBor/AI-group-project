"""Core functions for terrain height analysis."""

import numpy as np


def get_terrain_height(editor, x, z):
    """Find Y coordinate of solid ground at (x, z)."""
    non_solid_blocks = [
        "minecraft:air",
        "minecraft:cave_air", 
        "minecraft:water",
        "minecraft:lava",
        "minecraft:tall_grass",
        "minecraft:grass",
        "minecraft:fern",
        "minecraft:snow",
        "minecraft:dead_bush"
    ]
    
    for y in range(319, -64, -1):
        block = editor.getBlock((x, y, z))
        if block.id not in non_solid_blocks:
            return y
    
    return -64


def create_heightmap(editor, x_start, z_start, width, depth, progress_callback=None):
    """
    Create 2D array of terrain heights.
    
    EFFICIENT METHOD: Uses loadWorldSlice to get entire heightmap at once!
    
    Args:
        editor: GDPC Editor instance
        x_start, z_start: Starting world coordinates (not used with worldSlice)
        width, depth: Size of area to map (not used with worldSlice)
        progress_callback: Optional function to call with progress updates
        
    Returns:
        numpy array of shape (width, depth) with height values
    """
    print("  Loading world slice (efficient method)...")
    
    try:
        editor.loadWorldSlice(cache=True)
        
        if editor.worldSlice is None:
            raise ValueError("World slice failed to load")
        
        if "MOTION_BLOCKING_NO_LEAVES" not in editor.worldSlice.heightmaps:
            available = list(editor.worldSlice.heightmaps.keys())
            if available:
                print(f"  Using heightmap: {available[0]}")
                heightmap = editor.worldSlice.heightmaps[available[0]]
            else:
                raise ValueError("No heightmaps available in world slice")
        else:
            heightmap = editor.worldSlice.heightmaps["MOTION_BLOCKING_NO_LEAVES"]
        
        if progress_callback:
            progress_callback(100)
        
        print(f"  ✓ Heightmap loaded: {heightmap.shape}")
        
        return heightmap
        
    except Exception as e:
        print(f"  ✗ Error loading world slice: {e}")
        print(f"  Falling back to slow method (block-by-block)...")
        return create_heightmap_slow(editor, x_start, z_start, width, depth, progress_callback)


def create_heightmap_slow(editor, x_start, z_start, width, depth, progress_callback=None):
    """Fallback: Create heightmap block-by-block (slow but reliable)."""
    heightmap = np.zeros((width, depth), dtype=int)
    total = width * depth
    
    for x in range(width):
        for z in range(depth):
            world_x = x_start + x
            world_z = z_start + z
            heightmap[x, z] = get_terrain_height(editor, world_x, world_z)
            
            if progress_callback and (x * depth + z) % 100 == 0:
                progress = ((x * depth + z) / total) * 100
                progress_callback(progress)
    
    return heightmap


def calculate_slope(heightmap, x, z):
    """Calculate slope at (x, z) as max height difference to neighbors."""
    if x <= 0 or x >= heightmap.shape[0] - 1:
        return 0
    if z <= 0 or z >= heightmap.shape[1] - 1:
        return 0
    
    center_height = heightmap[x, z]
    neighbors = [
        heightmap[x+1, z],
        heightmap[x-1, z],
        heightmap[x, z+1],
        heightmap[x, z-1]
    ]
    
    max_diff = max(abs(center_height - n) for n in neighbors)
    return max_diff


def calculate_slope_map(heightmap):
    """Create map showing slope at every position."""
    height, width = heightmap.shape
    slope_map = np.zeros((height, width))
    
    for x in range(1, height - 1):
        for z in range(1, width - 1):
            slope_map[x, z] = calculate_slope(heightmap, x, z)
    
    return slope_map


def get_patch_stats(heightmap, x, z, size):
    """
    Get statistics for a square patch of terrain.
    
    Returns dict with:
        - min_height
        - max_height
        - avg_height
        - variation
    """
    if x + size > heightmap.shape[0] or z + size > heightmap.shape[1]:
        return None
    
    patch = heightmap[x:x+size, z:z+size]
    
    return {
        'min_height': float(patch.min()),
        'max_height': float(patch.max()),
        'avg_height': float(patch.mean()),
        'variation': float(patch.max() - patch.min())
    }
