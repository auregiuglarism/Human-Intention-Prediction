class Worker:
    
    def __init__(self, workerID, name, probability, state):
        self.workerID = workerID
        self.name = name  
        self.probability = probability  
        self.state = state
        pass

    def getWorkerID(self, workerId):
        return self.workerID

    def getWorkerName(self, name):
        return self.name
    
    def getWorkerProbability(self, probability):
        return self.probability
    
    def setWorkerProbability(self, probability):
        self.probability = probability
        return self.probability
    
    def getWorkerState(self, state):
        return self.state
    
    def setWorkerState(self, state):
        self.state = state
        return self.state
    