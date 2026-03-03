"""Helper functions for block placement and material selection."""

from gdpc import Block


def get_biome_palette(biome_type='plains'):
    """Get appropriate block palette for biome."""
    palettes = {
        'plains': {
            'wall': 'minecraft:oak_planks',
            'roof': 'minecraft:dark_oak_stairs',
            'floor': 'minecraft:oak_planks',
            'foundation': 'minecraft:cobblestone',
            'path': 'minecraft:dirt_path',
            'accent': 'minecraft:oak_log'
        },
        'desert': {
            'wall': 'minecraft:sandstone',
            'roof': 'minecraft:smooth_sandstone_stairs',
            'floor': 'minecraft:sandstone',
            'foundation': 'minecraft:sandstone',
            'path': 'minecraft:sand',
            'accent': 'minecraft:cut_sandstone'
        },
        'taiga': {
            'wall': 'minecraft:spruce_planks',
            'roof': 'minecraft:spruce_stairs',
            'floor': 'minecraft:spruce_planks',
            'foundation': 'minecraft:stone',
            'path': 'minecraft:gravel',
            'accent': 'minecraft:spruce_log'
        },
        'mountain': {
            'wall': 'minecraft:stone_bricks',
            'roof': 'minecraft:stone_brick_stairs',
            'floor': 'minecraft:stone_bricks',
            'foundation': 'minecraft:cobblestone',
            'path': 'minecraft:cobblestone',
            'accent': 'minecraft:stone'
        }
    }
    
    return palettes.get(biome_type, palettes['plains'])

def place_cube(editor, x, y, z, width, height, depth, material, hollow=False):
    """
    Place a rectangular solid or hollow box.
    
    Args:
        editor: GDPC Editor
        x, y, z: Starting position
        width, height, depth: Dimensions
        material: Block ID string
        hollow: If True, only place outer shell
    """
    for dx in range(width):
        for dy in range(height):
            for dz in range(depth):
                # If hollow, only place on edges
                if hollow:
                    is_edge = (dx == 0 or dx == width - 1 or 
                              dy == 0 or dy == height - 1 or 
                              dz == 0 or dz == depth - 1)
                    if not is_edge:
                        continue
                
                editor.placeBlock((x + dx, y + dy, z + dz), Block(material))
