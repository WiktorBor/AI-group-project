import math
from gdpc import Block

def clear_area(editor, analysis, building_area, buffer=3):

    center_x = (building_area.x_from + building_area.x_to) // 2
    center_z = (building_area.z_from + building_area.z_to) // 2

    width = building_area.x_to - building_area.x_from + 1
    depth = building_area.z_to - building_area.z_from + 1

    radius = max(width, depth) // 2 + buffer

    for x in range(center_x - radius, center_x + radius + 1):
        for z in range(center_z - radius, center_z + radius + 1):

            distance = math.sqrt((x - center_x)**2 + (z - center_z)**2)

            if distance > radius:
                continue

            gx = x - analysis.build_area.x_from
            gz = z - analysis.build_area.z_from

            ground = int(analysis.heightmap_ground[gx, gz])
            surface = int(analysis.heightmap_surface[gx, gz])

            for y in range(ground, surface + 1):
                block = editor.getBlock((x, y, z))
                block_id = block.id.lower()

                if not any(t in block_id for t in ["stone", "dirt", "grass_block", "sand"]):
                    editor.placeBlock((x, y, z), Block("minecraft:air"))