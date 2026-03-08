from abc import ABC, abstractmethod
from data.build_area import BuildArea
import numpy as np


class StructureAgent(ABC):
    """
    Base class for terrain-aware structure agents.
    Responsible for analysing terrain and deciding structure parameters.
    """

    def __init__(self, world):
        self.world = world

    # ----------------------------
    # Terrain utilities
    # ----------------------------

    def extract_patch(self, area: BuildArea, padding=2):
        """
        Extract a terrain heightmap patch around the build site.
        """

        heightmap = self.world.heightmap_ground
        x0 = max(0, area.x_from - padding)
        z0 = max(0, area.z_from - padding)
        x1 = min(heightmap.shape[0], area.x_to + padding + 1)
        z1 = min(heightmap.shape[1], area.z_to + padding + 1)

        patch = heightmap[x0:x1, z0:z1]
        return patch

    def compute_slope(self, patch):
        """
        Simple terrain slope measurement.
        """
        return patch.max() - patch.min()

    def is_flat(self, patch, tolerance=1):
        """
        Check if terrain is flat enough for building.
        """
        return self.compute_slope(patch) <= tolerance

    # ----------------------------
    # Decision interface
    # ----------------------------

    @abstractmethod
    def decide(self, area: BuildArea):
        """
        Analyse terrain and return building decisions.
        """
        pass