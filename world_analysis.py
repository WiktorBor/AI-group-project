from utils.http_client import GDMCClient
from world.build_area import BuildArea
from scipy.ndimage import distance_transform_edt
import numpy as np

class WorldAnalyser:
    """
    Analyse a Minecraft world build area and compute the best location
    for building based on terrain, water, flatness and biome.
    Only 'prepare()' should be called from outside, which runs the full analysis and sets 'best_area'.
    """
    def __init__(self, client: GDMCClient):
        self.client = client
        self.build_area: BuildArea | None = None
        self.best_area: BuildArea | None = None
        self.heightmap = None
        self.heightmap_surface = None
        self.heightmap_ground = None
        self.slope_map = None
        self.plant_thickness = None
        self.surface_blocks = {}
        self.water_mask = None
        self.water_distances = None
        self.biomes = None
        self.scores = None

    # Public Interface
    def prepare(self, rect = 200, stride = 20):
        """Main methods to prepare analysis and find best building area."""
        self._fetch_build_area()
        self._fetch_heightmaps()
        self._fetch_surface_blocks()
        self._fetch_biomes()
        self._build_water_mask()
        self._get_best_location(rect, stride)
        self.compute_slope_map()
    
    # HTTP FETCH FUNCTIONS
    def _fetch_build_area(self):
        data = self.client.get("/buildarea")

        self.build_area = BuildArea(
            x_from = data["xFrom"],
            y_from = data["yFrom"],
            z_from = data["zFrom"],
            x_to = data["xTo"],
            y_to = data["yTo"],
            z_to = data["zTo"],
        )
    
    def _fetch_heightmaps(self):
        params={
            "x": self.build_area.x_from,
            "z": self.build_area.z_from,
            "width": self.build_area.width,
            "depth": self.build_area.depth,
            }
        
        surface = self.client.get("/heightmap", {
            **params,
            "type": "MOTION_BLOCKING"
        })

        ground = self.client.get("/heightmap", {
            **params,
            "type": "MOTION_BLOCKING_NO_PLANTS"
        })
    
        self.heightmap_surface = np.array(surface)
        self.heightmap_ground = np.array(ground)

        self.plant_thickness = (
            self.heightmap_surface - self.heightmap_ground)
        self.heightmap = self.heightmap_ground

    def _fetch_surface_blocks(self, depth=20):
        Chunk = 16

        for x in range(0, self.build_area.width, Chunk):
            for z in range(0, self.build_area.depth, Chunk):
                dx = min(Chunk, self.build_area.width - x)
                dz = min(Chunk, self.build_area.depth - z)

                world_x = self.build_area.x_from + x
                world_z = self.build_area.z_from + z

                chunk_heights = self.heightmap[x:x+dx, z:z+dz]

                y_top = int(np.max(chunk_heights))

                blocks = self.client.get("/blocks", params={
                    "x": world_x,
                    "z": world_z,
                    "y": y_top - depth,
                    "dx": dx,
                    "dz": dz,
                    "dy": depth,
                    "withinBuildArea": "true"
                })

                columns = {}
                for block in blocks:
                    lx = block["x"] - self.build_area.x_from
                    lz = block["z"] - self.build_area.z_from
                    key = (lx, lz)

                    columns.setdefault(key, []).append(block)
                
                for key, column_blocks in columns.items():
                    column_blocks.sort(key=lambda b: b["y"], reverse=True)

                    for block in column_blocks:
                        block_id = block["id"]

                        if not any(k in block_id for k in ("air", "_leaves", "_log",
                        "_wood", "grass", "flower")):
                            self.surface_blocks[key] = (block["y"], block_id)
                            break

    def _fetch_biomes(self):
        data = self.client.get("/biomes", params={
                "x": self.build_area.x_from,
                "z": self.build_area.z_from,
                "width": self.build_area.width,
                "depth": self.build_area.depth,
            })

        arr = np.array([list(row) for row in data])

        # If only 1D, make 2D
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))

        # Repeat array to match heightmap size
        reps_x = self.build_area.width  // arr.shape[0] + 1
        reps_z = self.build_area.depth  // arr.shape[1] + 1
        arr = np.tile(arr, (reps_x, reps_z))

        # Trim to exact dimensions
        arr = arr[:self.build_area.width, :self.build_area.depth]

        self.biomes = arr

    # Helper functions for scoring
    def _compute_forest_penalty(self, x, z):
        thickness = self.plant_thickness[x][z]
        return min(max(thickness, 0) / 5.0, 1.0)

    def _compute_flatness(self, x, z, radius=5):
        x_min = max(0, x - radius)
        x_max = min(self.build_area.width, x + radius + 1)
        z_min = max(0, z - radius)
        z_max = min(self.build_area.depth, z + radius + 1)

        area = self.heightmap[x_min:x_max, z_min:z_max]

        std = np.std(area)
        return 1 / (1 + std)

    def _compute_accessibility(self, x, z):
        center_height = self.heightmap[x, z]
        walkable = 0

        for dx, dz in [(-1,0), (1,0),(0,-1),(0,1)]:
            nx, nz = x + dx, z + dz
            if 0 <= nx < self.build_area.width and 0 <= nz < self.build_area.depth:
                if abs(center_height - self.heightmap[nx, nz]) <= 1:
                    walkable += 1

        return walkable / 4
    
    def compute_slope_map(self):
        gx, gz = np.gradient(self.heightmap)
        self.slope_map = np.sqrt(gx**2 + gz**2)

    def _build_water_mask(self):
        self.water_mask = np.zeros((self.build_area.width, self.build_area.depth), dtype=bool)

        for (x, z), (_, block_id) in self.surface_blocks.items():
            if "water" in block_id:
                self.water_mask[x][z] = True
        self.water_distances = distance_transform_edt(~self.water_mask)

    def _compute_water_proximity(self, x, z, max_scan=16):
        return (max_scan - min(self.water_distances[x, z], max_scan)) / max_scan

    def _compute_elevation(self, x, z):
        return self.heightmap[x, z]

    def _compute_expansion(self, x, z, radius=5):
        x_min = max(0, x - radius)
        x_max = min(self.build_area.width, x + radius + 1)
        z_min = max(0, z - radius)
        z_max = min(self.build_area.depth, z + radius + 1)

        area = self.heightmap[x_min:x_max, z_min:z_max]
        base_height = self.heightmap[x, z]

        flat = np.abs(area - base_height) <= 1
        return np.sum(flat) / flat.size

    def _compute_biome_score(self, x, z):
        biome = self.biomes[x, z]

        biome_weights = {
            "minecraft:plains": 1.0,
            "minecraft:forest": 0.8,
            "minecraft:savanna": 0.8,
            "minecraft:desert": 0.5,
            "minecraft:swamp": 0.2,
            "minecraft:ocean": 0.0
        }

        return biome_weights.get(biome, 0.5)

    # WORLD ANALYSER
    def _analyse(self):
        if self.build_area is None:
            raise ValueError("Build area not fetched. Call fetch_build_area() first.")
        
        """Compute scores for all positions in the build area."""
        scores = np.zeros((self.build_area.width, self.build_area.depth))
        max_height = np.max(self.heightmap)

        for x in range(self.build_area.width):
            for z in range(self.build_area.depth):

                flatness = self._compute_flatness(x, z)
                access = self._compute_accessibility(x, z)
                water = self._compute_water_proximity(x, z)
                elevation = self._compute_elevation(x, z) / max_height
                expansion = self._compute_expansion(x, z)
                biome = self._compute_biome_score(x, z)
                forest_penalty = self._compute_forest_penalty(x, z)

                final_score = (
                    1.5 * flatness +
                    2.0 * access +
                    2.0 * expansion +
                    0.8 * water +
                    0.8 * elevation +
                    0.5 * biome -
                    2.0 * forest_penalty
                )

                scores[x, z] = final_score

        self.scores = scores

    # GET BEST LOCATION
    def _get_best_location(self, rect_size, stride):
        """Find the best rectangular area in the build area."""
        self._analyse()
        width = self.build_area.width
        depth = self.build_area.depth

        # Ensure the rectangle fits inside the build area
        rect_size = min(rect_size, width, depth)
        if rect_size <= 0:
            return None
        
        if rect_size < stride:
            return None

        best_score = -np.inf
        best_rect = None

        # Allow using the full area (inclusive upper bound)
        for x in range(0, width - rect_size + 1, stride):
            for z in range(0, depth - rect_size + 1, stride):

                area = self.scores[x:x + rect_size, z:z + rect_size]
                avg_score = np.mean(area)

                if avg_score > best_score:
                    best_score = avg_score
                    best_rect = (x, z)

        if best_rect is None:
            return None

        x_idx, z_idx = best_rect

        x_from = x_idx + self.build_area.x_from
        z_from = z_idx + self.build_area.z_from
        x_to = x_from + rect_size - 1
        z_to = z_from + rect_size - 1

        area_heights = self.heightmap[
            x_idx:x_idx+rect_size,
            z_idx:z_idx+rect_size
        ]

        min_y = int(np.min(area_heights))
        max_y = int(np.max(area_heights))

        self.best_area = BuildArea(x_from, min_y, z_from, x_to, max_y, z_to)