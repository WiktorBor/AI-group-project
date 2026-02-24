"""Main entry point for settlement generation."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gdpc import Editor
from generators.settlement_generator import SettlementGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate Minecraft settlements')
    parser.add_argument('--buildings', type=int, default=3,
                       help='Number of buildings to generate (default: 3)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show debug visualizations in Minecraft')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("GDMC SETTLEMENT GENERATOR")
    print("="*60)
    print("\nInitializing...")
    
    try:
        editor = Editor(buffering=True)
        generator = SettlementGenerator(editor)
        generator.generate(num_buildings=args.buildings, visualize=args.visualize)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Generation cancelled by user")
        return 1
        
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Is Minecraft running?")
        print("  2. Is GDMC HTTP Interface mod installed?")
        print("  3. Are you in a world?")
        print("  4. Is build area set? /buildarea set ~ ~ ~ ~50 ~ ~50")
        print("\nRun: python tests/test_connection.py")
        
        import traceback
        traceback.print_exc()
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
