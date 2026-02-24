"""Test GDMC HTTP interface connection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gdpc import Editor

def test_connection():
    print("Testing connection to Minecraft...")
    print("Make sure:")
    print("  1. Minecraft is running")
    print("  2. GDMC HTTP Interface mod is installed")
    print("  3. You're in a world\n")
    
    try:
        editor = Editor()
        build_area = editor.getBuildArea()
        
        print(f"✓ Connection successful!")
        print(f"✓ Build area: {build_area.size.x}x{build_area.size.y}x{build_area.size.z}")
        print(f"✓ Position: ({build_area.offset.x}, {build_area.offset.y}, {build_area.offset.z})")
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  - Restart Minecraft")
        print("  - Check mods are installed")
        print("  - Verify build area is set: /buildarea set ~ ~ ~ ~50 ~ ~50")
        return False

if __name__ == '__main__':
    test_connection()
