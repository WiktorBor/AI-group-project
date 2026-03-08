"""Finds optimal building sites with village-style spacing."""

import numpy as np
from gdpc import Block
from data.build_area import BuildArea

def _center_distance(area_a: BuildArea, area_b: BuildArea):
    """Rough center-to-center distance between two building footprints."""
    acx = (area_a.x_from + area_a.x_to) / 2
    acz = (area_a.z_from + area_a.z_to) / 2
    bcx = (area_b.x_from + area_b.x_to) / 2
    bcz = (area_b.z_from + area_b.z_to) / 2
    return np.sqrt((acx - bcx)**2 + (acz - bcz)**2)

class SiteLocator:
    
    def __init__(self, world, editor):
        """
        World: WorldAnalysisResult
        Editor: GDPC Editor
        """
        self.world = world
        self.settlement_area: BuildArea = world.best_area
        self.editor = editor
        self.sites = []
    
    def find_sites(
        self,
        max_sites=5,
        building_size=(7, 7),
        building_height=10,
        min_gap=5,
        min_building_dist=10,
        max_building_dist=28,
    ):
        """
        Find building sites with natural village-like spacing.

        Buildings are kept at least min_gap blocks apart (path space) and
        preferred to sit within min_building_dist–max_building_dist of each
        other so the settlement clusters like a Minecraft village.

        Args:
            max_sites: Maximum number of sites to return
            building_size: (width, depth) for each building
            min_gap: Minimum clear blocks between building edges (path space)
            min_building_dist: Minimum center-to-center distance between buildings
            max_building_dist: Prefer sites within this distance of existing buildings (cluster)

        Returns:
            List of site dictionaries
        """
        print("\n=== SITE LOCATION (village spacing) ===")
        print(f"  Sites: {max_sites}, size: {building_size[0]}x{building_size[1]}, gap: {min_gap}, cluster: {min_building_dist}-{max_building_dist}m")
        print("Best area:", self.settlement_area)

        width, depth = building_size
        margin = max(3, min_gap)

        # Catch world data locally
        global_heightmap = self.world.heightmap_ground
        global_slope_map = self.world.slope_map
        global_water_distances = self.world.water_distances
        global_water_proximity = self.world.water_proximity
        global_area = self.world.build_area

        offset_x = self.settlement_area.x_from - global_area.x_from
        offset_z = self.settlement_area.z_from - global_area.z_from

        area_width = self.settlement_area.width
        area_depth = self.settlement_area.depth

        occupied = np.zeros((area_width, area_depth), dtype=bool)
        candidates = []
        
        for i in range(margin, area_width - width - margin):
            for j in range(margin, area_depth - depth - margin):

                if occupied[
                    i - margin : i + width + margin,
                    j - margin : j + depth + margin,
                ].any():
                    continue
                
                gx = offset_x + i
                gz = offset_z + j

                if gx + width > global_heightmap.shape[0] or \
                    gz + depth > global_heightmap.shape[1]:
                    continue

                h_slice = global_heightmap[gx:gx+width, gz:gz+depth]
                slope_slice = global_slope_map[gx:gx+width, gz:gz+depth]
                water_slice = global_water_distances[gx:gx+width, gz:gz+depth]

                min_water_dist = water_slice.min()
                if min_water_dist < 4:
                    continue

                height_range = h_slice.max() - h_slice.min()
                if height_range > 2:
                    continue

                mean_slope = slope_slice.mean()

                flatness_score = max(0, 1 - mean_slope / 2)
                level_score = max(0, 1 - height_range / 3)
                water_score = global_water_proximity[gx, gz]

                area_score = (
                    flatness_score * 0.5 + 
                    level_score * 0.3 + 
                    water_score * 0.2
                )
                
                if area_score > 0.4:
                    candidates.append({
                        'local_x': i,
                        'local_z': j,
                        'score': area_score,
                    })
        
        candidates.sort(key=lambda c: c['score'], reverse=True)

        sites = []
        # Block radius for "occupied" so we enforce min_gap between building edges
        block_radius = margin  # same as margin so gap between footprints is at least min_gap

        for _ in range(max_sites):
            best = None
            best_cluster_score = -1
            best_score = -1

            for c in candidates:
                i = c['local_x']
                j = c['local_z']

                if occupied[
                    max(0, i - block_radius) : min(area_width, i + width + block_radius),
                    max(0, j - block_radius) : min(area_depth, j + depth + block_radius),
                ].any():
                    continue

                world_x = self.settlement_area.x_from + i
                world_z = self.settlement_area.z_from + j
                base_height = int(global_heightmap[world_x - global_area.x_from : world_x - global_area.x_from + width,
                                                 world_z - global_area.z_from : world_z - global_area.z_from + depth].max())
                
                candidate_area = BuildArea(
                    x_from=world_x,
                    y_from=base_height,
                    z_from=world_z,
                    x_to=world_x + width - 1,
                    y_to=base_height + building_height - 1,
                    z_to=world_z + depth - 1)

                ctr_dist_to_placed = float('inf')
                if sites:
                    for s in sites:
                        d = _center_distance(candidate_area, s['area'])                       
                        ctr_dist_to_placed = min(ctr_dist_to_placed, d)

                    if ctr_dist_to_placed < min_building_dist:
                        continue  # too close to an existing building

                # Prefer sites that are within cluster range of an existing building
                in_cluster = (
                    1 if (sites and min_building_dist <= ctr_dist_to_placed <= max_building_dist) 
                    else 0
                )

                if (in_cluster, c['score']) > (best_cluster_score, best_score):
                    best_cluster_score = in_cluster
                    best_score = c['score']
                    best = candidate_area, c["score"]

            if best is None:
                break

            area, score = best
            sites.append({
                "area": area,
                "score": score
            })

            i_off = area.x_from - self.settlement_area.x_from
            j_off = area.z_from - self.settlement_area.z_from

            occupied[
                max(0, i_off - block_radius) : min(area_width, i_off + width + block_radius),
                max(0, j_off - block_radius) : min(area_depth, j_off + depth + block_radius),
            ] = True

            # Remove this candidate so we don't reconsider it
            candidates = [
                c for c in candidates 
                if (c['local_x'], c['local_z']) != (i_off, j_off)
            ]
        
        self.sites = sites
        print(f"  ✓ Found {len(sites)} sites")
        for idx, site in enumerate(sites, 1):
            a = site['area']
            print(f"    Site {idx}: ({a.x_from}, {a.z_from}) score={site['score']:.2f}")

        return sites

    def visualize_sites(self):
        print("\n  Placing site markers...")
        
        for site in self.sites:
            area: BuildArea = site['area']
            x = (area.x_from + area.x_to) // 2
            z = (area.z_from + area.z_to) // 2
            y = area.y_from
            
            self.editor.placeBlock((x, y - 1, z), Block('minecraft:gold_block'))
            self.editor.placeBlock((x, y, z), Block('minecraft:beacon'))
        
        self.editor.flushBuffer()
        print(f"  ✓ Placed {len(self.sites)} markers")
