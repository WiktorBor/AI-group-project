"""Generates varied residential buildings."""

import random
from gdpc import Block


class HouseBuilder:
    
    def __init__(self, editor, palette=None):
        self.editor = editor
        self.palette = palette or self._default_palette()
    
    def _default_palette(self):
        return {
            'wall': 'minecraft:oak_planks',
            'roof': 'minecraft:dark_oak_stairs',
            'floor': 'minecraft:oak_planks',
            'door': 'minecraft:oak_door',
            'window': 'minecraft:glass_pane'
        }
    
    def build(self, site):
        """
        Generate a house at given site.
        
        Args:
            site: Dictionary with x, z, width, depth, height
            
        Returns:
            Dictionary with building data
        """
        x = site['x']
        y = site['height']
        z = site['z']
        width = site['width']
        depth = site['depth']
        
        wall_material = random.choice([
            'minecraft:oak_planks',
            'minecraft:spruce_planks',
            'minecraft:stone_bricks',
            'minecraft:cobblestone'
        ])
        
        roof_style = 'gabled'
        has_chimney = random.random() < 0.3
        building_height = random.randint(4, 6)
        
        self._build_floor(x, y, z, width, depth)
        self._build_walls(x, y, z, width, building_height, depth, wall_material)
        
        if roof_style == 'flat':
            self._build_flat_roof(x, y + building_height, z, width, depth)
        else:
            self._build_gabled_roof(x, y + building_height, z, width, depth)
        
        self._add_door(x, y, z, width, depth)
        self._add_windows(x, y, z, width, building_height, depth)
        
        if has_chimney:
            self._add_chimney(x, y, z, width, depth, building_height)
        
        return {
            'type': 'house',
            'position': (x, y, z),
            'size': (width, building_height, depth),
            'material': wall_material,
            'roof_style': roof_style,
            'has_chimney': has_chimney
        }
    
    def _build_floor(self, x, y, z, width, depth):
        for dx in range(width):
            for dz in range(depth):
                self.editor.placeBlock((x + dx, y, z + dz), 
                                     Block(self.palette['floor']))
    
    def _build_walls(self, x, y, z, width, height, depth, material):
        for dy in range(1, height):
            for dx in range(width):
                self.editor.placeBlock((x + dx, y + dy, z), Block(material))
                self.editor.placeBlock((x + dx, y + dy, z + depth - 1), Block(material))
            
            for dz in range(depth):
                self.editor.placeBlock((x, y + dy, z + dz), Block(material))
                self.editor.placeBlock((x + width - 1, y + dy, z + dz), Block(material))
    
    def _build_flat_roof(self, x, y, z, width, depth):
        for dx in range(width):
            for dz in range(depth):
                self.editor.placeBlock((x + dx, y, z + dz), 
                                     Block(self.palette['floor']))
    
    def _build_gabled_roof(self, x, y, z, width, depth):
        roof_material = self.palette['roof']
        peak_height = width // 2
        
        for layer in range(peak_height):
            for dz in range(depth):
                self.editor.placeBlock((x + layer, y + layer, z + dz), 
                                     Block(roof_material))
                self.editor.placeBlock((x + width - 1 - layer, y + layer, z + dz), 
                                     Block(roof_material))
    
    def _add_door(self, x, y, z, width, depth):
        door_x = x + width // 2
        door_z = z
        
        self.editor.placeBlock((door_x, y, door_z), Block(self.palette['floor']))
        self.editor.placeBlock((door_x, y + 1, door_z), Block(self.palette['door']))
    
    def _add_windows(self, x, y, z, width, height, depth):
        window_y = y + 2
        
        if width > 5:
            self.editor.placeBlock((x + 1, window_y, z), Block(self.palette['window']))
            self.editor.placeBlock((x + width - 2, window_y, z), Block(self.palette['window']))
        
        if depth > 5:
            self.editor.placeBlock((x, window_y, z + depth // 2), 
                                 Block(self.palette['window']))
            self.editor.placeBlock((x + width - 1, window_y, z + depth // 2), 
                                 Block(self.palette['window']))
    
    def _add_chimney(self, x, y, z, width, depth, building_height):
        chimney_x = x + width - 2
        chimney_z = z + depth - 2
        
        for dy in range(building_height):
            self.editor.placeBlock((chimney_x, y + dy, chimney_z), 
                                 Block('minecraft:cobblestone'))
