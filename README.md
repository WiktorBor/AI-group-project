# GDMC Settlement Generator

Procedural settlement generation for Minecraft using GDPC.
Designed for the Generative Design in Minecraft Competition (GDMC).

## Project Structure

```
AI-group-project/
│
├── generators/              # Main generation orchestrators
│   └── settlement_generator.py
│
├── structures/              # Building generation
│   └── house_builder.py     # Residential buildings
│
├── analysis/                # Terrain & site analysis
│   ├── terrain_analyzer.py  # Heightmaps, slopes, buildability
│   └── site_locator.py      # Find optimal building sites
│
├── utils/                   # Shared utilities
│   ├── heightmap.py         # Terrain height utilities
│   └── block_utils.py       # Block/material utilities
│
├── tests/                   # Test files
│   ├── test_connection.py   # Connection verification
│   ├── test_terrain.py      # Terrain analysis test
│   └── test_house.py        # House generation test
│
├── docs/                    # Documentation
│   ├── QUICK_START.md
│   ├── DEVELOPMENT_ROADMAP.md
│   └── TUTORIAL.md
│
├── learning/                # Learning materials (gitignored)
│   ├── world_analysis_examples.py
│   ├── evaluation_metrics.py
│   └── GDPC_REFERENCE.md
│
├── run_generator.py         # Main entry point
├── Requirements.txt         # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install packages
pip install -r Requirements.txt
```

### 2. Setup Minecraft

- Install Fabric Loader (client)
- Install GDMC HTTP Interface mod
- Launch Minecraft and create/load a world
- Set build area: `/buildarea set ~ ~ ~ ~50 ~ ~50`

### 3. Test Connection

```bash
python tests/test_connection.py
```

### 4. Generate Settlement

```bash
# Basic generation (3 buildings)
python run_generator.py

# Custom number of buildings
python run_generator.py --buildings 5

# With debug visualization
python run_generator.py --visualize
```

## Development Workflow

### For Team Members

Each team member can work on separate modules:

**Structures Team**: Work in `structures/`
- `house_builder.py` - Houses
- Add: `tower_builder.py`, `farm_builder.py`, etc.

**Analysis Team**: Work in `analysis/`
- `terrain_analyzer.py` - Terrain analysis
- `site_locator.py` - Site selection
- Add: `pathfinder.py`, `biome_analyzer.py`, etc.

**Generators Team**: Work in `generators/`
- `settlement_generator.py` - Main orchestrator
- Add: `road_generator.py`, `decoration_generator.py`, etc.

**Utils Team**: Work in `utils/`
- Shared utilities used by all modules

### Testing Your Changes

```bash
# Test specific module
python tests/test_connection.py
python tests/test_terrain.py
python tests/test_house.py

# Test full pipeline
python run_generator.py --visualize
```

### Adding New Features

1. Create your module in appropriate folder
2. Create a test file in `tests/`
3. Test independently
4. Integrate with `settlement_generator.py`

## Evaluation Criteria

### Reachability (25%)
- All buildings connected via paths
- A* pathfinding between all structures
- **Implementation**: TODO - `generators/road_generator.py`

### Topographic Compliance (25%)
- Buildings adapt to terrain (low Delta-Height)
- Foundations follow slopes
- **Implementation**: In progress - `analysis/terrain_analyzer.py`

### Structural Diversity (25%)
- Varied building designs (Shannon Entropy)
- Multiple building types
- Material variety
- **Implementation**: Basic - needs grammar system

### Human-Similarity (25%)
- Natural-looking layouts
- Organic patterns
- Context-appropriate designs
- **Implementation**: Basic - needs refinement

## Next Steps

See `docs/DEVELOPMENT_ROADMAP.md` for detailed roadmap.

**Priority tasks:**
1. Road generation with pathfinding
2. Building diversity (grammar system)
3. Terrain adaptation (foundations)
4. Evaluation metrics implementation

## Resources

- [GDPC Documentation](https://github.com/avdstaaij/gdpc)
- [GDMC Competition](http://gendesignmc.engineering.nyu.edu/)
- Tutorial: `docs/TUTORIAL.md`
- Quick Reference: `docs/QUICK_START.md`
