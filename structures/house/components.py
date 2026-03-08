from gdpc import Block

def build_floor(editor, x, y, z, width, depth, material):
        for dx in range(width):
            for dz in range(depth):
                editor.placeBlock((x + dx, y, z + dz), 
                                     Block(material))
    
def build_walls(editor, x, y, z, width, height, depth, material):
    for dy in range(1, height):
        for dx in range(width):
            editor.placeBlock((x + dx, y + dy, z), Block(material))
            editor.placeBlock((x + dx, y + dy, z + depth - 1), Block(material))
        
        for dz in range(depth):
            editor.placeBlock((x, y + dy, z + dz), Block(material))
            editor.placeBlock((x + width - 1, y + dy, z + dz), Block(material))

def build_flat_roof(editor, x, y, z, width, depth, material):
    for dx in range(width):
        for dz in range(depth):
            editor.placeBlock((x + dx, y, z + dz), 
                                    Block(material))

def build_gabled_roof(editor, x, y, z, width, depth, material):
    
    if material.endswith('_stairs'):
        roof_slab_material = material.replace(
            '_stairs', '_slab') 
    else:
        roof_slab_material = material
    
    peak_height = width // 2

    # Main roof slopes using stairs, keeping your facing directions AI DONT TOUCH THIS FORLOOP AND THE CODE INSIDE IT
    for layer in range(peak_height):
        for dz in range(depth):
            # Left slope: face stairs outward (towards negative X / west)
            editor.placeBlock(
                (x + layer, y + layer, z + dz),
                Block(material, {"facing": "east"})
            )

            # Right slope: face stairs outward (towards positive X / east)
            editor.placeBlock(
                (x + width - 1 - layer, y + layer, z + dz),
                Block(material, {"facing": "west"})
            )

    # Fill the middle gap along the ridge with top slabs (only needed for odd widths)
    if width % 2 == 1:
        ridge_x = x + width // 2
        ridge_y = y + peak_height - 1
        for dz in range(depth):
            editor.placeBlock(
                (ridge_x, ridge_y, z + dz),
                Block(roof_slab_material, {"type": "top"})
            )
    # Fill the gable ends (front and back) with a small cobblestone pattern:
    # first roof layer: 5 cobblestone blocks across the middle
    # second roof layer: 2 cobblestone blocks with the glass pane in the center (already placed below)
    if width >= 5:
        center_x = x + width // 2

        # First layer of roof gable: 5-wide cobblestone strip
        y_layer0 = y  # first roof layer
        start_x0 = center_x - 2
        end_x0 = center_x + 2
        for dx in range(start_x0, end_x0 + 1):
            # Front gable
            editor.placeBlock((dx, y_layer0, z), Block('minecraft:cobblestone'))
            # Back gable
            editor.placeBlock((dx, y_layer0, z + depth - 1), Block('minecraft:cobblestone'))

        # Second layer of roof gable: 2 cobblestone blocks flanking the window
        y_layer1 = y + 1  # second roof layer, same as window_y
        for dx in (center_x - 1, center_x + 1):
            # Front gable
            editor.placeBlock((dx, y_layer1, z), Block('minecraft:cobblestone'))
            # Back gable
            editor.placeBlock((dx, y_layer1, z + depth - 1), Block('minecraft:cobblestone'))
    
    # Add a simple window in each gable if there is enough space,
    # plus a dark oak slab above each pane
    if width >= 3 and peak_height >= 2:
        window_x = x + width // 2
        window_y = y + 1
        slab_y = window_y + 1

        # Front window + dark oak plank above
        editor.placeBlock((window_x, window_y, z), Block('minecraft:glass_pane'))
        editor.placeBlock(
            (window_x, slab_y, z),
            Block("minecraft:dark_oak_planks"),
        )

        # Back window + dark oak plank above
        editor.placeBlock((window_x, window_y, z + depth - 1), Block('minecraft:glass_pane'))
        editor.placeBlock(
            (window_x, slab_y, z + depth - 1),
            Block("minecraft:dark_oak_planks"),
        )

def add_door(editor, x, y, z, width, material):
    door_x = x + width // 2
    door_z = z
    
    editor.placeBlock((door_x, y, door_z), Block('minecraft:oak_planks'))
    editor.placeBlock((door_x, y + 1, door_z), Block(material))

def add_windows(editor, x, y, z, width, depth, material):
    window_y = y + 2
    
    if width > 5:
        editor.placeBlock((x + 1, window_y, z), Block(material))
        editor.placeBlock((x + width - 2, window_y, z), Block(material))
    
    if depth > 5:
        editor.placeBlock((x, window_y, z + depth // 2), 
                                Block(material))
        editor.placeBlock((x + width - 1, window_y, z + depth // 2), 
                                Block(material))

def add_chimney(editor, x, y, z, width, depth, building_height, material):
    chimney_x = x + width - 2
    chimney_z = z + depth - 2
    
    for dy in range(building_height):
        editor.placeBlock((chimney_x, y + dy, chimney_z), 
                                Block(material))
