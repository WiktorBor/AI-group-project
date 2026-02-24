"""Test house generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gdpc import Editor
from structures.house_builder import HouseBuilder


def test_house():
    print("\n" + "="*50)
    print("HOUSE BUILDER TEST")
    print("="*50)
    
    try:
        print("\nConnecting to Minecraft...")
        editor = Editor(buffering=True)
        build_area = editor.getBuildArea()
        print(f"✓ Connected! Build area: {build_area.size.x}x{build_area.size.z}")
        
        center_x = build_area.offset.x + build_area.size.x // 2
        center_z = build_area.offset.z + build_area.size.z // 2
        
        print("\nFinding terrain height...")
        test_y = 64
        for y in range(100, 40, -1):
            block = editor.getBlock((center_x, y, center_z))
            if block and block.id != 'minecraft:air':
                test_y = y + 1
                break
        
        print(f"✓ Terrain height: {test_y}")
        
        site = {
            'x': center_x,
            'z': center_z,
            'height': test_y,
            'width': 7,
            'depth': 7
        }
        
        print("\nBuilding house...")
        builder = HouseBuilder(editor)
        building_data = builder.build(site)
        
        print("\nFlushing to Minecraft...")
        editor.flushBuffer()
        
        print("\n✓ Test passed!")
        print(f"✓ Built {building_data['type']} at ({building_data['position']})")
        print(f"  Material: {building_data['material']}")
        print(f"  Roof: {building_data['roof_style']}")
        print(f"  Chimney: {building_data['has_chimney']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_house()
