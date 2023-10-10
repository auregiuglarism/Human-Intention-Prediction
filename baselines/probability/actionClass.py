class Action:

    def __init__(self, name, probability):
        self.name = name
        self.probability = probability
    
    def getActionName(self):
        return self.name
    
    def getActionProbability(self):
        return self.probability
    
    def setActionProbability(self, probability):
        self.probability = probability
        return self.probability