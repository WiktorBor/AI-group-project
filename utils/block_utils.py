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


def detect_biome_from_terrain(editor, heightmap, x_start, z_start, sample_size=20):
    """Detect biome by analyzing surface blocks."""
    block_counts = {}
    
    for x in range(min(sample_size, heightmap.shape[0])):
        for z in range(min(sample_size, heightmap.shape[1])):
            world_x = x_start + x
            world_z = z_start + z
            y = heightmap[x, z]
            
            block = editor.getBlock((world_x, y, world_z))
            block_id = block.id
            block_counts[block_id] = block_counts.get(block_id, 0) + 1
    
    most_common = max(block_counts.items(), key=lambda x: x[1])[0]
    
    if 'sand' in most_common:
        return 'desert'
    elif 'snow' in most_common or 'ice' in most_common:
        return 'taiga'
    elif 'stone' in most_common:
        return 'mountain'
    else:
        return 'plains'


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
