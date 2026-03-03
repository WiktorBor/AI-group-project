"""Finds optimal building sites with village-style spacing."""

import numpy as np
from gdpc import Block


def _center_distance(ax, az, aw, ad, bx, bz, bw, bd):
    """Rough center-to-center distance between two building footprints."""
    acx = ax + aw / 2
    acz = az + ad / 2
    bcx = bx + bw / 2
    bcz = bz + bd / 2
    return np.sqrt((acx - bcx) ** 2 + (acz - bcz) ** 2)


class SiteLocator:
    
    def __init__(self, world_analyser, settlement_area, editor):
        self.world = world_analyser
        self.settlement_area = settlement_area
        self.sites = []
        self.editor = editor
    
    def find_sites(
        self,
        max_sites=5,
        building_size=(7, 7),
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
        
        width, depth = building_size
        margin = max(3, min_gap)

        global_heightmap = self.world.heightmap
        global_slope_map = self.world.slope_map
        global_area = self.world.build_area
        global_water_mask = self.world.water_mask
        global_water_distances = self.world.water_distances

        offset_x = self.settlement_area.x_from - global_area.x_from
        offset_z = self.settlement_area.z_from - global_area.z_from

        area_width = self.settlement_area.width
        area_depth = self.settlement_area.depth

        occupied = np.zeros((area_width, area_depth), dtype=bool)
        candidates = []
        
        for i in range(margin, area_width - width - margin):
            for j in range(margin, area_depth - depth - margin):

                if occupied[i - margin : i + width + margin, j - margin : j + depth + margin].any():
                    continue
                
                gx = offset_x + i
                gz = offset_z + j

                if gx + width > global_heightmap.shape[0] or gz + depth > global_heightmap.shape[1]:
                    continue
                if global_water_mask[gx : gx + width, gz : gz + depth].any():
                    continue
                if global_water_distances[gx : gx + width, gz : gz + depth].min() < 4:
                    continue   

                area_heights = global_heightmap[gx : gx + width, gz : gz + depth]
                area_slopes = global_slope_map[gx : gx + width, gz : gz + depth]

                height_range = area_heights.max() - area_heights.min()
                mean_slope = area_slopes.mean()

                if height_range > 2:
                    continue

                flatness_score = max(0, 1 - mean_slope / 2)
                level_score = max(0, 1 - height_range / 3)
                water_proximity = self.world.water_proximity[gx, gz]
                area_score = flatness_score * 0.5 + level_score * 0.3 + water_proximity * 0.2
                
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
                i, j = c['local_x'], c['local_z']
                if occupied[
                    max(0, i - block_radius) : min(area_width, i + width + block_radius),
                    max(0, j - block_radius) : min(area_depth, j + depth + block_radius),
                ].any():
                    continue

                world_x = self.settlement_area.x_from + i
                world_z = self.settlement_area.z_from + j
                ctr_dist_to_placed = None
                if sites:
                    for s in sites:
                        d = _center_distance(
                            world_x, world_z, width, depth,
                            s['x'], s['z'], s['width'], s['depth'],
                        )
                        if ctr_dist_to_placed is None or d < ctr_dist_to_placed:
                            ctr_dist_to_placed = d
                    if ctr_dist_to_placed < min_building_dist:
                        continue  # too close to an existing building
                else:
                    ctr_dist_to_placed = float('inf')

                # Prefer sites that are within cluster range of an existing building
                in_cluster = 1 if (sites and min_building_dist <= ctr_dist_to_placed <= max_building_dist) else 0
                if (in_cluster, c['score']) > (best_cluster_score, best_score):
                    best_cluster_score = in_cluster
                    best_score = c['score']
                    best = (i, j, c['score'], world_x, world_z)

            if best is None:
                break

            i, j, score, world_x, world_z = best
            gx = offset_x + i
            gz = offset_z + j
            base_height = int(global_heightmap[gx:gx+width, gz:gz+depth].mean())
            
            site = {
                'x': world_x,
                'z': world_z,
                'height': base_height,
                'width': width,
                'depth': depth,
                'score': score,
            }
            sites.append(site)
            occupied[
                max(0, i - block_radius) : min(area_width, i + width + block_radius),
                max(0, j - block_radius) : min(area_depth, j + depth + block_radius),
            ] = True

            # Remove this candidate so we don't reconsider it
            candidates = [c for c in candidates if (c['local_x'], c['local_z']) != (i, j)]
        
        self.sites = sites
        print(f"  ✓ Found {len(sites)} sites")
        for idx, site in enumerate(sites, 1):
            print(f"    Site {idx}: ({site['x']}, {site['z']}) score={site['score']:.2f}")
        return sites

    def visualize_sites(self):
        print("\n  Placing site markers...")
        
        for site in self.sites:
            x, z = site['x'], site['z']
            y = site['height']
            
            self.editor.placeBlock((x, y - 1, z), Block('minecraft:gold_block'))
            self.editor.placeBlock((x, y, z), Block('minecraft:beacon'))
        
        self.editor.flushBuffer()
        print(f"  ✓ Placed {len(self.sites)} markers")
