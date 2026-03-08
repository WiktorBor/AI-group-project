from dataclasses import dataclass
import numpy as np
from .build_area import BuildArea

@dataclass
class WorldAnalysisResult:
    build_area: BuildArea
    best_area: BuildArea
    heightmap_ground: np.ndarray
    heightmap_surface: np.ndarray
    plant_thickness: np.ndarray
    slope_map: np.ndarray
    water_distances: np.ndarray
    water_proximity: np.ndarray
    surface_blocks: dict
    biomes: np.ndarray
    scores: np.ndarray