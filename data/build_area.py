from dataclasses import dataclass

@dataclass
class BuildArea:
    x_from: int
    y_from: int
    z_from: int
    x_to: int
    y_to: int
    z_to: int

    @property
    def width(self):
        return self.x_to - self.x_from + 1

    @property
    def depth(self):
        return self.z_to - self.z_from + 1

    @property
    def height(self):
        return self.y_to - self.y_from + 1
    
    def contains(self, x, y, z):
        return (
            self.x_from <= x <= self.x_to and
            self.y_from <= y <= self.y_to and
            self.z_from <= z <= self.z_to
        )