"""Analyzes terrain for buildability and characteristics."""

import numpy as np
from utils.heightmap import create_heightmap, calculate_slope_map


class TerrainAnalyzer:
    
    def __init__(self, editor, build_area):
        self.editor = editor
        self.build_area = build_area
        self.heightmap = None
        self.slope_map = None
        self.buildability_map = None
    
    def analyze(self):
        print("\n=== TERRAIN ANALYSIS ===")
        
        print("\n[1/3] Creating heightmap...")
        self.heightmap = self._create_heightmap_with_progress()
        
        print("\n[2/3] Calculating slope map...")
        self.slope_map = calculate_slope_map(self.heightmap)
        print(f"  ✓ Slope range: {self.slope_map.min():.2f} to {self.slope_map.max():.2f}")
        
        print("\n[3/3] Computing buildability scores...")
        self.buildability_map = self._calculate_buildability()
        buildable_percent = (self.buildability_map > 0.5).sum() / self.buildability_map.size * 100
        print(f"  ✓ {buildable_percent:.1f}% of area is buildable")
        
        print("\n✓ Terrain analysis complete!\n")
        
        return {
            'heightmap': self.heightmap,
            'slope_map': self.slope_map,
            'buildability_map': self.buildability_map
        }
    
    def _create_heightmap_with_progress(self):
        x_start = self.build_area.offset.x
        z_start = self.build_area.offset.z
        width = min(self.build_area.size.x, 100)
        depth = min(self.build_area.size.z, 100)
        
        def progress(percent):
            if int(percent) % 10 == 0:
                print(f"  Progress: {percent:.0f}%")
        
        return create_heightmap(self.editor, x_start, z_start, width, depth, progress)
    
    def _calculate_buildability(self):
        """
        Calculate buildability score for each position.
        
        Score based on:
        - Flatness (low slope = higher score)
        - Height variation in local area
        - Accessibility
        """
        buildability = np.zeros_like(self.slope_map)
        
        for i in range(1, self.slope_map.shape[0] - 1):
            for j in range(1, self.slope_map.shape[1] - 1):
                local_slope = self.slope_map[i-1:i+2, j-1:j+2].mean()
                
                flatness_score = max(0, 1.0 - local_slope / 3.0)
                
                local_height_var = self.heightmap[i-1:i+2, j-1:j+2].std()
                smoothness_score = max(0, 1.0 - local_height_var / 5.0)
                
                buildability[i, j] = (flatness_score * 0.7 + smoothness_score * 0.3)
        
        return buildability
    
    def get_height_at(self, x, z):
        local_x = x - self.build_area.offset.x
        local_z = z - self.build_area.offset.z
        
        if (0 <= local_x < self.heightmap.shape[0] and 
            0 <= local_z < self.heightmap.shape[1]):
            return self.heightmap[local_x, local_z]
        
        return None
    
    def get_buildability_at(self, x, z):
        local_x = x - self.build_area.offset.x
        local_z = z - self.build_area.offset.z
        
        if (0 <= local_x < self.buildability_map.shape[0] and 
            0 <= local_z < self.buildability_map.shape[1]):
            return self.buildability_map[local_x, local_z]
        
        return 0.0
