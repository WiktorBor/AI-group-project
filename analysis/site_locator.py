"""Finds optimal building sites within terrain."""

import numpy as np
from gdpc import Block


class SiteLocator:
    
    def __init__(self, terrain_analyzer):
        self.terrain = terrain_analyzer
        self.build_area = terrain_analyzer.build_area
        self.sites = []
        self.editor = terrain_analyzer.editor
    
    def find_sites(self, building_size=(7, 7), max_sites=5):
        """
        Find best building sites.
        
        Args:
            building_size: (width, depth) for buildings
            max_sites: Maximum number of sites to return
            
        Returns:
            List of site dictionaries
        """
        print("\n=== SITE LOCATION ===")
        print(f"  Searching for {max_sites} sites ({building_size[0]}x{building_size[1]})...")
        
        sites = []
        occupied = np.zeros(self.terrain.buildability_map.shape, dtype=bool)
        
        width, depth = building_size
        margin = 3
        
        candidates = []
        
        for i in range(margin, self.terrain.buildability_map.shape[0] - width - margin):
            for j in range(margin, self.terrain.buildability_map.shape[1] - depth - margin):
                if occupied[i:i+width+margin*2, j:j+depth+margin*2].any():
                    continue
                
                area_score = self.terrain.buildability_map[i:i+width, j:j+depth].mean()
                
                if area_score > 0.4:
                    candidates.append({
                        'local_x': i,
                        'local_z': j,
                        'score': area_score
                    })
        
        candidates.sort(key=lambda c: c['score'], reverse=True)
        
        for candidate in candidates[:max_sites]:
            i, j = candidate['local_x'], candidate['local_z']
            
            world_x = self.build_area.offset.x + i
            world_z = self.build_area.offset.z + j
            terrain_height = int(self.terrain.heightmap[i, j])
            
            site = {
                'x': world_x,
                'z': world_z,
                'height': terrain_height,
                'width': width,
                'depth': depth,
                'score': candidate['score']
            }
            
            sites.append(site)
            occupied[i:i+width+margin*2, j:j+depth+margin*2] = True
        
        self.sites = sites
        print(f"  ✓ Found {len(sites)} sites")
        
        for idx, site in enumerate(sites, 1):
            print(f"    Site {idx}: ({site['x']}, {site['z']}) score={site['score']:.2f}")
        
        return sites
    
    def visualize_sites(self):
        print("\n  Placing site markers...")
        
        for site in self.sites:
            x, z = site['x'], site['z']
            y = site['height']
            
            self.editor.placeBlock((x, y - 1, z), Block('minecraft:gold_block'))
            self.editor.placeBlock((x, y, z), Block('minecraft:beacon'))
        
        self.editor.flushBuffer()
        print(f"  ✓ Placed {len(self.sites)} markers")
