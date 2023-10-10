from actionClass import Action
from objectClass import Object
from workerClass import Worker

# Initial Assumptions 
# We know the active workers at any time
# We know the object(s) being interacted with through computer vision

# TODO : come up with a way to determine initial probability for each object based on its occurence to be more accurate
# TODO : Come up with some initial knowledge base to make sure some actions will never be assigned to some objects
# no matter their probability (eg : you cannot "drink" a "crate"), will make the predict function more efficient (you can think of it as pruning)
# TODO : Make tests to optimize threshold 
# TODO : Create a function or code to receive output detected object from computer vision
# if never encountered, create new object and add it in the list ?

TESTING = True

### Define Objects, Workers, and Actions ###
# Objects
obj1 = Object("bottle", 0.3) 
obj2 = Object("crate", 0.8)

all_objects = [obj1, obj2] #init

def addObject(object):
    all_objects.append(object)
    return all_objects

# Workers
worker1 = Worker("Nicolas", 0.5, True)
worker2 = Worker("Vitaly", 0.7, False)
worker3 = Worker("Simeon", 0.9, False)

all_workers = [worker1, worker2, worker3] #init

def addWorker(worker):
    all_workers.append(worker)
    return all_workers

# Actions
action1 = Action("pick up", 0.2)
action2 = Action("drink", 0.3)

all_actions = [action1, action2] # init

def addAction(action):
    all_actions.append(action)
    return action

# Default action
default_action = ("default action", 0) # default if prob < THRESHOLD

def defaultaction():
    # Do something default
    (print("default action"))
    return default_action # Corresponds to actionID of default action

## Baseline Model ##

THRESHOLD = 0.3

def matchObjectWorkerPairToAction(object_prob, worker_prob, action_prob):
    # Using Bayes' theorem to calculate the probability of the object-worker pair to perform the action 
    pair_prob = object_prob*worker_prob

    numerator = pair_prob*action_prob
    denominator = numerator + (1-pair_prob)*(1-action_prob)

    bayes_prob = numerator/denominator

    # Testing
    if TESTING:
        print(bayes_prob)

    if bayes_prob < THRESHOLD:
        defaultaction()
    else:
        return bayes_prob

def predict():
    # Build pairs of all possible object-worker pairs with Active workers
    active_workers = [] # ID list of all active worker
    for worker in all_workers:
        if worker.getWorkerState() == True:
            active_workers.append(worker)
    
    object_worker_pairs = []
    for object in all_objects:
        for active_worker in active_workers:
            pair = [Object (object), Worker(active_worker)]
            object_worker_pairs.append(pair)

    # For each pair, calculate the probability of the pair to perform the action, return the highest probability
    # In a greedy manner :
    highest_prob = 0 # init
    highest_action = defaultaction() # init

    for pair in object_worker_pairs:
        object = pair[0]

        print("object", object.getObjectProbability())
        print(type(object)) 

        worker = pair[1] 
        print(type(worker))

        # error ? 
        for action in all_actions:
            action_prob = matchObjectWorkerPairToAction(object.getObjectProbability(), worker.getWorkerProbability(), action.getActionProbability())
            if action_prob > highest_prob:
                highest_prob = action_prob
                highest_action = action

    return highest_action

### Testing ###
# Testing matchObjectWorkerPairToAction() 
obj_prob = 0.7
worker_prob = 0.8
action_prob = 0.6

# matchObjectWorkerPairToAction(obj_prob, worker_prob, action_prob) # is working properly
# with settings at : 0.3, 0.5, 0.7, returns 0.2916666666666667, underneath threshold so default action
# with settings at : 0.7, 0.8, 0.6, returns 0.6562499999999999, above threshold so returns bayes_prob

# Testing predict() TODO : Test it
predict() 






