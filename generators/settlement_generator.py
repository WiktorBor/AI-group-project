"""Main orchestrator for settlement generation."""

from analysis.terrain_analyzer import TerrainAnalyzer
from analysis.site_locator import SiteLocator
from structures.house_builder import HouseBuilder


class SettlementGenerator:
    
    def __init__(self, editor):
        self.editor = editor
        self.build_area = None
        self.terrain_analyzer = None
        self.site_locator = None
        self.house_builder = None
        self.buildings = []
    
    def generate(self, num_buildings=3, visualize=False):
        """
        Generate complete settlement.
        
        Args:
            num_buildings: Number of buildings to generate
            visualize: Show debug visualizations in Minecraft
        """
        print("\n" + "="*50)
        print("SETTLEMENT GENERATOR")
        print("="*50)
        
        # Phase 1: Setup
        self._setup()
        
        # Phase 2: Analyze Terrain
        self._analyze_terrain(visualize)
        
        # Phase 3: Find Building Sites
        self._locate_sites(num_buildings, visualize)
        
        # Phase 4: Generate Buildings
        self._generate_buildings()
        
        # Phase 5: Finalize
        self._finalize()
        
        print("\n" + "="*50)
        print("✓ SETTLEMENT GENERATION COMPLETE")
        print("="*50)
        print(f"\nGenerated {len(self.buildings)} buildings")
        print(f"Build area: {self.build_area.size.x}x{self.build_area.size.z}")
        print("\nCheck Minecraft to see your settlement!")
    
    def _setup(self):
        print("\n[Phase 1] Setup")
        
        self.build_area = self.editor.getBuildArea()
        print(f"  ✓ Build area: {self.build_area.size.x}x{self.build_area.size.z}")
        
        if self.build_area.size.x < 20 or self.build_area.size.z < 20:
            raise ValueError("Build area too small! Use at least 50x50")
    
    def _analyze_terrain(self, visualize):
        print("\n[Phase 2] Terrain Analysis")
        
        self.terrain_analyzer = TerrainAnalyzer(self.editor, self.build_area)
        self.terrain_analyzer.analyze()
        
        if visualize:
            self._visualize_terrain()
    
    def _locate_sites(self, num_buildings, visualize):
        print("\n[Phase 3] Site Location")
        
        self.site_locator = SiteLocator(self.terrain_analyzer)
        self.site_locator.find_sites(building_size=(7, 7), max_sites=num_buildings)
        
        if visualize:
            self.site_locator.visualize_sites()
    
    def _generate_buildings(self):
        print("\n[Phase 4] Building Generation")
        
        self.house_builder = HouseBuilder(self.editor)
        
        for idx, site in enumerate(self.site_locator.sites, 1):
            print(f"  Building {idx}/{len(self.site_locator.sites)}...")
            building_data = self.house_builder.build(site)
            self.buildings.append(building_data)
        
        print(f"  ✓ Generated {len(self.buildings)} buildings")
    
    def _finalize(self):
        print("\n[Phase 5] Finalizing...")
        print("  Flushing blocks to Minecraft...")
        
        self.editor.flushBuffer()
        
        print("  ✓ All blocks placed!")
    
    def _visualize_terrain(self):
        print("\n  Creating terrain visualization...")
        
        heightmap = self.terrain_analyzer.heightmap
        buildability = self.terrain_analyzer.buildability_map
        
        step = max(1, min(heightmap.shape) // 20)
        
        for i in range(0, heightmap.shape[0], step):
            for j in range(0, heightmap.shape[1], step):
                world_x = self.build_area.offset.x + i
                world_z = self.build_area.offset.z + j
                y = int(heightmap[i, j])
                score = buildability[i, j]
                
                if score > 0.7:
                    color = 'minecraft:lime_concrete'
                elif score > 0.4:
                    color = 'minecraft:yellow_concrete'
                else:
                    color = 'minecraft:red_concrete'
                
                from gdpc import Block
                self.editor.placeBlock((world_x, y, world_z), Block(color))
        
        self.editor.flushBuffer()
        print("  ✓ Terrain visualization placed")
