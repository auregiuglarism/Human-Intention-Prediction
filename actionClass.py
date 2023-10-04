class Action:

    def __init__(self, actionID, name, probability):
        self.actionID = actionID
        self.name = name
        self.probability = probability
        
    def getActionID(self, actionID):
        return self.actionID
    
    def getActionName(self, name):
        return self.name
    
    def getActionProbability(self, probability):
        return self.probability
    
    def setActionProbability(self, probability):
        self.probability = probability
        return self.probability