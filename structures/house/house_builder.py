"""Generates varied residential buildings."""

import random
from gdpc import Block

from utils.terrain_utils import clear_area
from data.build_area import BuildArea
from .components import *


class HouseBuilder:
    
    def __init__(self, editor, world, palette=None):
        self.editor = editor
        self.world = world
        self.palette = palette or self._default_palette()
    
    def _default_palette(self):
        return {
            'wall': 'minecraft:oak_planks',
            'roof': 'minecraft:dark_oak_stairs',
            'floor': 'minecraft:oak_planks',
            'door': 'minecraft:oak_door',
            'window': 'minecraft:glass_pane',
            'chimney': 'minecraft:bricks',
        }
    
    def build_house(self, site: BuildArea, decisions):

        x, z = site.x_from, site.z_from
        y = site.y_from
        w, d = site.width, site.depth

        wall_height = 5

        clear_area(editor=self.editor, analysis=self.world, building_area=site)

        build_floor(self.editor, x, y, z, w, d, self.palette['floor'])
        build_walls(self.editor, x, y, z, w, 5, d, self.palette['wall'] )

        # Roof
        roof_y = y + wall_height

        if decisions['roof_type'] == "gabled":
            build_gabled_roof(self.editor, x, roof_y, z, w, d, self.palette['roof'])
        else:
            build_flat_roof(self.editor, x, roof_y, z, w, d, self.palette['roof'])  
        
        # Windows and door
        add_windows(self.editor, x, y, z, w, d, self.palette['window'])
        add_door(self.editor, x, y, z, w, self.palette['door'])
        
        # Chimney
        if decisions.get('chimney', False):
            add_chimney(self.editor, x, y+1, z, w, d, wall_height + 3, self.palette['chimney'])
