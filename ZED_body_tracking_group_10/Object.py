class Object:
    def __init__(self, type, pos):
        self.type = type
        self.name = "Name"
        self.relations = []
        self.pos = pos

    def set_new_relation(self, relative_position):
        self.relations = relative_position