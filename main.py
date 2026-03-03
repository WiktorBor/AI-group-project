"""Main entry point: World analysis and settlement generation."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gdpc import Editor
from generators.settlement_generator import SettlementGenerator
from utils.http_client import GDMCClient


def main():
    
    parser = argparse.ArgumentParser(
        description='Analyse world and Generate Minecraft settlements')
    parser.add_argument('--buildings',type=int, default=3,
                       help='Number of buildings to generate (default: 3)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show debug visualizations in Minecraft')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("GDMC WORLD ANALYSIS AND SETTLEMENT GENERATOR")
    print("="*60)
    print("\nInitializing...")

    client = GDMCClient()

    if not client.check_connection():
        print("\n Could not connect to GDMC HTTP Interface.\nIs Minecraft running?")
        sys.exit(1)
    
    print("\n Connection OK. Using your current build area.")

    # Generate settlement
    print("\n Starting settlement generation...")

    try:
        editor = Editor(buffering=True)
        generator = SettlementGenerator(
            editor, client
        )  

        generator.generate(num_buildings=args.buildings)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Generation cancelled by user")
        return 1
        
    print("\n Settlement generation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
