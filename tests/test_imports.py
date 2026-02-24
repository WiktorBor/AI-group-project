"""Test that all imports work."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Testing imports...")

try:
    from generators.settlement_generator import SettlementGenerator
    print("  OK generators.settlement_generator")
    
    from structures.house_builder import HouseBuilder
    print("  OK structures.house_builder")
    
    from analysis.terrain_analyzer import TerrainAnalyzer
    print("  OK analysis.terrain_analyzer")
    
    from analysis.site_locator import SiteLocator
    print("  OK analysis.site_locator")
    
    from utils.heightmap import create_heightmap
    print("  OK utils.heightmap")
    
    from utils.block_utils import get_biome_palette
    print("  OK utils.block_utils")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"\nImport failed: {e}")
    import traceback
    traceback.print_exc()
