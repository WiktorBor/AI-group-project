"""Test terrain analysis functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gdpc import Editor
from analysis.terrain_analyzer import TerrainAnalyzer


def test_terrain_analysis():
    """Test terrain analysis pipeline"""
    print("\n" + "="*50)
    print("TERRAIN ANALYSIS TEST")
    print("="*50)
    
    try:
        print("\nConnecting to Minecraft...")
        editor = Editor(buffering=True)
        build_area = editor.getBuildArea()
        print(f"✓ Connected! Build area: {build_area.size.x}x{build_area.size.z}")
        
        analyzer = TerrainAnalyzer(editor, build_area)
        results = analyzer.analyze()
        
        print("\n=== RESULTS ===")
        print(f"Heightmap shape: {results['heightmap'].shape}")
        print(f"Height range: {results['heightmap'].min():.0f} to {results['heightmap'].max():.0f}")
        print(f"Slope range: {results['slope_map'].min():.2f} to {results['slope_map'].max():.2f}")
        print(f"Buildability range: {results['buildability_map'].min():.2f} to {results['buildability_map'].max():.2f}")
        
        print("\n✓ Test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_terrain_analysis()
