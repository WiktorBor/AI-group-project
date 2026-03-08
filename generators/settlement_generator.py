"""Main orchestrator for settlement generation."""
from planning.site_locator import SiteLocator
from data.analysis.world_analysis import WorldAnalyser
from structures.registry import STRUCTURES
from structures.house.house_builder import HouseBuilder


class SettlementGenerator:
    
    def __init__(self, editor, client):
        self.editor = editor
        self.client = client
        self.buildings = []
    
    def generate(self, num_buildings=3):
        """
        Generate complete settlement.
        
        Args:
            num_buildings: Number of buildings to generate
            visualize: Show debug visualizations in Minecraft
        """
        print("\n" + "="*50)
        print("SETTLEMENT GENERATOR")
        print("="*50)
        
        # Phase 1: Analyse world and find best building area
        analysis = self._analyse_world()
        
        # Phase 2: Locate building sites
        sites = self._locate_sites(analysis, num_buildings)
        
        # Phase 3: Generate buildings
        self._generate_buildings(analysis, sites)
        
        # Phase 4: Finalize
        self._finalize()
        
        print("\n" + "="*50)
        print("✓ SETTLEMENT GENERATION COMPLETE")
        print("="*50)
        print(f"\nGenerated {len(self.buildings)} buildings")
        print("\nCheck Minecraft to see your settlement!")
    
    # Private methods for each phase
    def _analyse_world(self):
        print("\n[Phase 1] Setup")

        # Initialise WorldAnaliser
        analyser = WorldAnalyser(self.client)
        results = analyser.prepare()

        return results
    
    def _locate_sites(self,analysis, num_buildings):
        print("\n[Phase 2] Locating building sites")

        site_locator = SiteLocator(analysis, self.editor)
        sites = site_locator.find_sites(max_sites=num_buildings)
        print(f" {len(sites)} building sites determined.")
        return sites

    def _generate_buildings(self, analysis, sites):
        print("\n[Phase 3] Building Generation")
        
        for idx, site in enumerate(sites, 1):
            structure_type = "house"
            structure_class = STRUCTURES[structure_type]

            structure = structure_class(self.editor, analysis)
            print(f"  Building {idx}/{len(sites)} at {site} ({structure_type})")

            structure.build(site["area"])
            self.buildings.append({
                "type": structure_type,
                "site": site
            })
        
        print(f"  ✓ Generated {len(self.buildings)} buildings")
    
    def _finalize(self):
        print("\n[Phase 4] Finalizing...")
        print("  Flushing blocks to Minecraft...")
        
        self.editor.flushBuffer()
        
        print("  ✓ All blocks placed!")
