from actionClass import Action
from objectClass import Object 
from workerClass import Worker

TESTING = True

### Define Objects, Workers, and Actions ###

# TODO : come up with a way to determine initial probability for each object based on its occurence to be more accurate
obj1 = Object(1, "bottle", 0.3) 
obj2 = Object(2, "crate", 0.8)

worker1 = Worker(1, "Nicolas", 0.5, True)
worker2 = Worker(2, "Vitaly", 0.7, False)
worker3 = Worker(3, "Simeon", 0.9, False)

# TODO : Come up with some initial knowledge base to make sure some actions will never be assigned to some objects
# no matter their probability (eg : you cannot "drink" a "crate"), will make the predict function more efficient (you can think of it as pruning)
action1 = Action(1, "pick up", 0.2)
action2 = Action(2, "drink", 0.3)
default_action = (0, "default action", 0)

### Match Object-Worker Pair to Action ###

THRESHOLD = 0.3

def defaultaction():
    # Do something default
    (print("default action"))
    return 0 # Corresponds to actionID of default action

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

def predict(objectIDLists, workerIDLists, actionIDLists):
    # Build pairs of all possible object-worker pairs with Active workers
    active_workers = []
    for worker in workerIDLists:
        if worker.getWorkerState() == True:
            active_workers.append(worker)
    
    object_worker_pairs = []
    for object in objectIDLists:
        for worker in active_workers:
            object_worker_pairs.append((object, worker))

    # For each pair, calculate the probability of the pair to perform the action, return the highest probability
    # In a greedy manner :
    max = 0
    actionID_max = 0

    for pair in object_worker_pairs:
        object = pair[0]
        worker = pair[1]
        for action in actionIDLists:
            action_prob = matchObjectWorkerPairToAction(object.getObjectProbability(), worker.getWorkerProbability(), action.getActionProbability())
            if action_prob > max:
                max = action_prob
                actionID_max = action.getActionID()
        
    if actionID_max == 0:
        defaultaction()
    else:    
        return actionID_max

### Testing ###

# Testing matchObjectWorkerPairToAction() 
obj_prob = 0.7
worker_prob = 0.8
action_prob = 0.6

matchObjectWorkerPairToAction(obj_prob, worker_prob, action_prob) # is working properly
# with settings at : 0.3, 0.5, 0.7, returns 0.2916666666666667, underneath threshold so default action
# with settings at : 0.7, 0.8, 0.6, returns 0.6562499999999999, above threshold so returns actionID of action

# Testing predict() TODO : Test it
objectIDLists = [obj1, obj2]
workerIDLists = [worker1, worker2, worker3]
actionIDLists = [action1, action2]

predict(objectIDLists, workerIDLists, actionIDLists) 






