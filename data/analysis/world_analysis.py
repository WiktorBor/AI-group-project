from utils.http_client import GDMCClient
from data.build_area import BuildArea
from data.analysis_results import WorldAnalysisResult
from scipy.ndimage import distance_transform_edt, label, uniform_filter
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

    def prepare(self) -> WorldAnalysisResult:
        self._fetch_build_area()
        self._fetch_heightmaps()
        self._fetch_surface_blocks()
        self._fetch_biomes()
        self.compute_slope_map()

        self._get_best_location()

        return WorldAnalysisResult(
            build_area=self.build_area,
            best_area=self.best_area,
            heightmap_ground=self.heightmap_ground,
            heightmap_surface=self.heightmap_surface,
            plant_thickness=self.plant_thickness,
            slope_map=self.slope_map,
            water_distances=self.water_distances,
            water_proximity=self.water_proximity,
            surface_blocks=self.surface_blocks,
            biomes=self.biomes,
            scores=self.scores
        )

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

    def _fetch_surface_blocks(self, depth=20, Chunk = 32):
        """
        Fetch top surface blocks for the entire build area.
        """
        self.surface_blocks = {}

        h_w, h_d = self.heightmap_ground.shape[0], self.heightmap_ground.shape[1]
        fetched_columns = set()

        for x in range(0, h_w, Chunk):
            for z in range(0, h_d, Chunk):
                dx = min(Chunk, h_w - x)
                dz = min(Chunk, h_d - z)

                world_x = self.build_area.x_from + x
                world_z = self.build_area.z_from + z

                chunk_heights = self.heightmap_ground[x:x+dx, z:z+dz]

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

    def compute_slope_map(self):
        gx, gz = np.gradient(self.heightmap_ground)
        self.slope_map = np.sqrt(gx**2 + gz**2)
    
    def _build_water_mask(self):
        h_w, h_d = self.heightmap_ground.shape[0], self.heightmap_ground.shape[1]
        self.water_mask = np.zeros((h_w, h_d), dtype=bool)

        for (x, z), (_, block_id) in self.surface_blocks.items():
            if 0 <= x < h_w and 0 <= z < h_d and "water" in block_id:
                self.water_mask[x][z] = True
        self.water_distances = distance_transform_edt(~self.water_mask)

    # Helper functions for scoring     
    def _compute_flatness(self, radius=5):
        size = 2 * radius + 1
        mean = uniform_filter(self.heightmap_ground, size=size)
        mean_sq = uniform_filter(self.heightmap_ground**2, size=size)

        variance = mean_sq - mean**2
        std = np.sqrt(np.maximum(variance, 0))

        return 1 / (1 + std)

    def _compute_accessibility(self,):
        h = self.heightmap_ground

        up = np.abs(h - np.roll(h, -1, axis=0)) <= 1
        down = np.abs(h - np.roll(h, 1, axis=0)) <= 1
        left = np.abs(h - np.roll(h, -1, axis=1)) <= 1
        right = np.abs(h - np.roll(h, 1, axis=1)) <= 1

        return (up + down + left + right) / 4.0

    def _compute_expansion(self, radius=5):
        size = 2 * radius + 1
        base = self.heightmap_ground

        local_mean = uniform_filter(base, size=size)
        flat = np.abs(base - local_mean) <= 1

        return uniform_filter(flat.astype(float), size=size)

    def _compute_biome_score(self):

        biome_weights = {
            "minecraft:plains": 1.0,
            "minecraft:forest": 0.8,
            "minecraft:savanna": 0.8,
            "minecraft:desert": 0.5,
            "minecraft:swamp": 0.2,
            "minecraft:ocean": 0.0
        }

        lookup = np.vectorize(lambda b: biome_weights.get(b, 0.5))
        return lookup(self.biomes)

    # WORLD ANALYSER
    def _analyse(self):
        """Compute scores for all positions in the build area."""

        flatness = self._compute_flatness()
        access = self._compute_accessibility()
        self._build_water_mask()
        self.water_proximity = (16 - np.minimum(self.water_distances, 16)) / 16
        expansion = self._compute_expansion()
        biome = self._compute_biome_score()
        forest_penalty = np.clip(self.plant_thickness / 5.0, 0, 1)
        max_h = np.max(self.heightmap_ground)
        elevation = self.heightmap_ground / max_h if max_h > 0 else np.zeros_like(self.heightmap)
        self.scores = (
            1.5 * flatness +
            2.0 * access +
            2.0 * expansion +
            0.8 * self.water_proximity +
            0.8 * elevation +
            0.5 * biome -
            2.0 * forest_penalty
        )

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
        
        if best_zone is None:
            raise ValueError("No suitable build location found.")

        x_min, z_min = best_zone.min(axis=0)
        x_max, z_max = best_zone.max(axis=0)
        y_min = int(np.min(self.heightmap_ground[x_min:x_max+1, z_min:z_max+1]))
        y_max = int(np.max(self.heightmap_ground[x_min:x_max+1, z_min:z_max+1]))

        self.best_area = BuildArea(
            x_min + self.build_area.x_from,
            y_min,
            z_min + self.build_area.z_from,
            x_max + self.build_area.x_from,
            y_max,
            z_max + self.build_area.z_from
        )