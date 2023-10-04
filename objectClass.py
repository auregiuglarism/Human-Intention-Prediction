class Object:

    def __init__(self, objectID, name, probability):
        self.probability = probability
        self.objectID = objectID
        self.name = name
        pass

    def getObjectID(self, objectID):
        return self.objectID
    
    def getObjectName(self, name):
        return self.name
    
    def getObjectProbability(self, probability):
        return self.probability
    
    def setObjectProbability(self, probability):
        self.probability = probability
        return self.probability