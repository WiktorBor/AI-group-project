from structures.base.structure import Structure
from .house_agent import HouseAgent
from .house_builder import HouseBuilder

class House(Structure):

    def __init__(self, editor, world):
        super().__init__(editor, world)

        self.agent = HouseAgent(world)
        self.builder = HouseBuilder(editor, world)

    def build(self, site):

        decisions = self.agent.decide(site)

        self.builder.build_house(site, decisions)