from utils.http_client import GDMCClient
from world.build_area import BuildArea
from scipy.ndimage import distance_transform_edt, label, find_objects
import numpy as np

class WorldAnalyser:
    """
    Analyse a Minecraft world build area and compute the best location
    for building a settlement based on terrain, water, flatness and biome.
    The best area is dinamically sized to fit the largest high-scoring patch.
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
    def prepare(self):
        """Main methods to prepare analysis and find best building area."""
        self._fetch_build_area()
        self._fetch_heightmaps()
        self._fetch_surface_blocks()
        self._fetch_biomes()
        self._build_water_mask()
        self.compute_slope_map()
        self._get_best_location()
    
    # HTTP FETCH FUNCTIONS
    def _fetch_build_area(self):
        """
        Determine build area around the first player, 
        fallback to server build area if needed.
        """
        if not self.client.check_build_area():
            print("\n No build area set. Set it in-game first, e.g.:")
            print("   /buildarea set ~ ~ ~ ~199 ~ ~199   (200x200 from your position)")
            print("   Or: /buildarea set x1 y1 z1 x2 y2 z2")
            raise SystemExit(1)

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
        """
        Fetch heightmaps for the full build area in chunks.
        """
        params={
            "x": self.build_area.x_from,
            "z": self.build_area.z_from,
            "width": self.build_area.width,
            "depth": self.build_area.depth,
            }
        
        # Fetch surface
        surface = self.client.get("/heightmap", {
            **params,
            "type": "MOTION_BLOCKING"
        })

        self.heightmap_surface = np.array(surface)

        # Fetch ground
        ground = self.client.get("/heightmap", {
            **params,
            "type": "MOTION_BLOCKING_NO_PLANTS"
        })
        self.heightmap_ground = np.array(ground)

        # Compute plant thickness and final heightmap
        self.plant_thickness = self.heightmap_surface - self.heightmap_ground
        self.heightmap = self.heightmap_ground

    def _fetch_surface_blocks(self, depth=20, Chunk = 32):
        """
        Fetch top surface blocks for the entire build area.
        """
        h_w, h_d = self.heightmap.shape[0], self.heightmap.shape[1]
        fetched_columns = set()

        for x in range(0, h_w, Chunk):
            for z in range(0, h_d, Chunk):
                dx = min(Chunk, h_w - x)
                dz = min(Chunk, h_d - z)

                world_x = self.build_area.x_from + x
                world_z = self.build_area.z_from + z

                chunk_heights = self.heightmap[x:x+dx, z:z+dz]

                if chunk_heights.size == 0:
                    continue

                y_top = int(np.max(chunk_heights))
                if y_top < 0:
                    continue

                blocks = self.client.get("/blocks", params={
                    "x": world_x,
                    "z": world_z,
                    "y": max(0, y_top - depth),
                    "dx": dx,
                    "dz": dz,
                    "dy": depth,
                })

                if not blocks:
                    continue

                columns = {}
                for block in blocks:
                    lx = block["x"] - self.build_area.x_from
                    lz = block["z"] - self.build_area.z_from
                    if 0 <= lx < h_w and 0 <= lz < h_d:
                        key = (lx, lz)
                        columns.setdefault(key, []).append(block)
                
                for (lx, lz), column_blocks in columns.items():
                    if (lx, lz) in fetched_columns:
                        continue

                    column_blocks.sort(key=lambda b: b["y"], reverse=True)

                    for block in column_blocks:
                        block_id = block.get("id", "air")

                        if not any(k in block_id for k in ("air", "_leaves", "_log",
                        "_wood", "grass", "flower")):
                            self.surface_blocks[(lx, lz)] = (block["y"], block_id)
                            fetched_columns.add((lx, lz))
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
        h_w, h_d = self.heightmap.shape[0], self.heightmap.shape[1]
        x_min = max(0, x - radius)
        x_max = min(h_w, x + radius + 1)
        z_min = max(0, z - radius)
        z_max = min(h_d, z + radius + 1)

        area = self.heightmap[x_min:x_max, z_min:z_max]

        std = np.std(area)
        return 1 / (1 + std)

    def _compute_accessibility(self, x, z):
        center_height = self.heightmap[x, z]
        walkable = 0
        h_height, h_depth = self.heightmap.shape[0], self.heightmap.shape[1]

        for dx, dz in [(-1,0), (1,0),(0,-1),(0,1)]:
            nx, nz = x + dx, z + dz
            if 0 <= nx < h_height and 0 <= nz < h_depth:
                if abs(center_height - self.heightmap[nx, nz]) <= 1:
                    walkable += 1

        return walkable / 4
    
    def compute_slope_map(self):
        gx, gz = np.gradient(self.heightmap)
        self.slope_map = np.sqrt(gx**2 + gz**2)

    def _build_water_mask(self):
        h_w, h_d = self.heightmap.shape[0], self.heightmap.shape[1]
        self.water_mask = np.zeros((h_w, h_d), dtype=bool)

        for (x, z), (_, block_id) in self.surface_blocks.items():
            if 0 <= x < h_w and 0 <= z < h_d and "water" in block_id:
                self.water_mask[x][z] = True
        self.water_distances = distance_transform_edt(~self.water_mask)

    def _compute_water_proximity(self, x, z, max_scan=16):
        return (max_scan - min(self.water_distances[x, z], max_scan)) / max_scan

    def _compute_elevation(self, x, z):
        return self.heightmap[x, z]

    def _compute_expansion(self, x, z, radius=5):
        h_w, h_d = self.heightmap.shape[0], self.heightmap.shape[1]
        x_min = max(0, x - radius)
        x_max = min(h_w, x + radius + 1)
        z_min = max(0, z - radius)
        z_max = min(h_d, z + radius + 1)

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
        # Use heightmap shape so we never index out of bounds (API may return smaller grid)
        h_width, h_depth = self.heightmap.shape[0], self.heightmap.shape[1]
        scores = np.zeros((h_width, h_depth))
        max_height = np.max(self.heightmap)

        for x in range(h_width):
            for z in range(h_depth):

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
    def _get_best_location(self):
        """
        Pick the larges contiguous high-scoring patch of terrain.
        Iteratively lower the threshold if patch too small.
        """
        self._analyse()
        flat_mask = self.slope_map <= 0.5
        threshold = np.percentile(self.scores, 75)
        high_score_mask = self.scores >= threshold
        mask = flat_mask & high_score_mask

        labeled, num_features = label(mask)
        best_total_score = -np.inf
        best_zone = None
        
        for i in range(1, num_features + 1): 
            coords = np.argwhere(labeled == i) 
            total_score = self.scores[labeled == i].sum() 
            if total_score > best_total_score: 
                best_total_score = total_score 
                best_zone = coords
        
        x_min, z_min = best_zone.min(axis=0)
        x_max, z_max = best_zone.max(axis=0)
        y_min = int(np.min(self.heightmap[x_min:x_max+1, z_min:z_max+1]))
        y_max = int(np.max(self.heightmap[x_min:x_max+1, z_min:z_max+1]))

        self.best_area = BuildArea(
            x_min + self.build_area.x_from,
            y_min,
            z_min + self.build_area.z_from,
            x_max + self.build_area.x_from,
            y_max,
            z_max + self.build_area.z_from
        )