from structures.base.structure_agent import StructureAgent
from data.build_area import BuildArea

class HouseAgent(StructureAgent):
    """
    Agent responsible for deciding where and how to build houses.
    Analyzes terrain and returns building parameters.
    """

    def decide(self, area: BuildArea) -> dict:
        """
        Analyze the site and return building decisions.
        
        Args:
            site: Dictionary with x, z, width, depth
            
        Returns:
            Dictionary with building parameters
        """
        patch = self.extract_patch(area)

        # Simple decision logic based on terrain features
        slope = self.compute_slope(patch)

        roof_type = "gabled"
        floors = 1
        
        if slope > 2:
            roof_type = "steep"
        
        if area.width > 10:
            floors = 2

        return {
            "roof_type": roof_type,
            "floors": floors,
            "chimney": True
        }