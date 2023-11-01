class Worker:
    
    def __init__(self, name, probability, state):
        self.name = name  
        self.probability = probability  
        self.state = state
        pass

    def getWorkerName(self):
        return self.name
    
    def getWorkerProbability(self):
        return self.probability
    
    def setWorkerProbability(self, probability):
        self.probability = probability
        return self.probability
    
    def getWorkerState(self):
        return self.state
    
    def setWorkerState(self, state):
        self.state = state
        return self.state
    