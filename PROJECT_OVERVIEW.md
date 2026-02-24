# Project Overview

Clean, organized GDMC settlement generator for team collaboration.

## Quick Navigation

- **Start working**: Open `TEAM_ASSIGNMENTS.md` and claim a module
- **Test changes**: Run `python tests/test_<module>.py`
- **Documentation**: Check `docs/` folder

## Folder Structure

```
generators/     - Main orchestration logic
structures/     - Building generation (houses, towers, etc.)
analysis/       - Terrain analysis & site finding
utils/          - Shared utility functions
tests/          - All test files
docs/           - All documentation
learning/       - Learning materials (gitignored)
```

## Main Commands

```bash
# Test connection
python tests/test_connection.py

# Test your module
python tests/test_terrain.py
python tests/test_house.py

# Run full generator
python run_generator.py --visualize
```

## Team Workflow

1. Pick a module from `TEAM_ASSIGNMENTS.md`
2. Work on your module in your folder
3. Test independently: `python tests/test_<your_module>.py`
4. Test integration: `python run_generator.py`
5. Commit only your changes

## Key Files

- `run_generator.py` - Main entry point
- `generators/settlement_generator.py` - Orchestrates everything
- `structures/house_builder.py` - Builds houses
- `analysis/terrain_analyzer.py` - Analyzes terrain
- `analysis/site_locator.py` - Finds building sites
- `utils/heightmap.py` - Terrain height functions
- `utils/block_utils.py` - Block/material utilities

## Documentation

- `README.md` - Main project readme
- `docs/PROJECT_STRUCTURE.md` - Detailed structure guide
- `docs/MODULE_GUIDE.md` - How modules connect
- `docs/TESTING_GUIDE.md` - Testing workflows
- `TEAM_ASSIGNMENTS.md` - Work assignments

## Next Steps

1. Test current setup works: `python run_generator.py --visualize`
2. Assign team members in `TEAM_ASSIGNMENTS.md`
3. Pick features from `docs/DEVELOPMENT_ROADMAP.md`
4. Start coding!
