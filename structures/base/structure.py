from abc import ABC, abstractmethod

class Structure(ABC):
    """
    Abstract base class for structures.
    """

    def __init__(self, editor, world):
        self.editor = editor
        self.world = world

    @abstractmethod
    def build(self, site):
        """
        Build the structure at the given site.
        
        Args:
            site:
                {
                    'x': int,
                    'z': int,
                    'width': int,
                    'depth': int,
                    'height': int
                }
        """
        pass