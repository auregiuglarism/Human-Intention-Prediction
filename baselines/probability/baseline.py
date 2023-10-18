import time
from actionClass import Action
from objectClass import Object
from workerClass import Worker
import conf
import cv2 as cv
from ultralytics import YOLO
import math

# Initial Assumptions 
# We know the active workers at any time
# We know the object(s) being interacted with through computer vision

# TODO : Come up with some initial knowledge base to make sure some actions will never be assigned to some objects
# no matter their probability (eg : you cannot "drink" a "crate"), will make the predict function more efficient (you can think of it as pruning)
# TODO : Make tests to optimize threshold 

TESTING = True

### Define Objects, Workers, and Actions ###

# Update and Define Real-time Objects and their probabilities
all_objects = [] #init

def streamProb():
    """Calculate probabilities from stream"""
    # Load a model
    model = YOLO('Models/yolov8n.pt')  # pretrained YOLOv8n model

    vidcap = cv.VideoCapture(0)

    previous_time = time.time()
    timestep = 1 # seconds

    while True:
        current_time = time.time()
        elapsed_time = current_time - previous_time

        if elapsed_time > timestep:

            ret, frame = vidcap.read()
            cv.imshow("Stream", frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
            if not ret: 
                break

            results = model.track(frame, persist=True)
            typeProbMatch = conf.getProbabilities(results) # prints detected probs
            # print(typeProbMatch)

            for key in typeProbMatch:
                exist = False # Detect if objects already exist in the list
                name = key
                probability = typeProbMatch[key]

                for object in all_objects:
                    if name == object.getObjectName(): # If object already exists, update probability
                        object.setObjectProbability(probability)
                        exist = True
                        break # Break out of loop we don't need to check the rest of the list
        
                if exist == False: # If object does not exist, create new object
                    object = Object(name, probability)
                    all_objects.append(object)

            # printing in real-time
            for o in all_objects:
                print(o)
    

# Workers
worker1 = Worker("Nicolas", 0.3, True)
worker2 = Worker("Vitaly", 0.7, False)
worker3 = Worker("Simeon", 0.9, False)

all_workers = [worker1, worker2, worker3] #init

def addWorker(worker):
    all_workers.append(worker)
    return all_workers

# Actions
action1 = Action("pick up", 0.3)
action2 = Action("drink", 0.1)

all_actions = [action1, action2] # init

def addAction(action):
    all_actions.append(action)
    return action

def defaultaction():
    default_action = ("default action", -1)
    return default_action 

## Baseline Model ##

THRESHOLD = 0.5

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
        return -1 # default action
    else:
        return bayes_prob

def predict(all_workers, all_objects, all_actions):
    # Build pairs of all possible object-worker pairs with Active workers
    active_workers = [] # ID list of all active worker
    for worker in all_workers:
        if worker.getWorkerState() == True:
            active_workers.append(worker)
    
    object_worker_pairs = []
    for object in all_objects:
        for active_worker in active_workers:
            pair = [object, active_worker]
            object_worker_pairs.append(pair)

    # For each pair, calculate the probability of the pair to perform the action, return the highest probability
    # In a greedy manner :
    highest_prob = -1 # init
    highest_action = defaultaction() # init

    for pair in object_worker_pairs: # number of total iterations = len(object_worker_pairs)*len(all_actions)
        object = pair[0]
        worker = pair[1] 
        for action in all_actions:
            action_prob = matchObjectWorkerPairToAction(object.getObjectProbability(), worker.getWorkerProbability(), action.getActionProbability())
            if action_prob > highest_prob:
                highest_prob = action_prob
                highest_action = action

    if highest_prob == -1:
        return defaultaction()
    return [highest_action.getActionName(), highest_action.getActionProbability()]


# Program Loop
if TESTING:
    streamProb()

cv.destroyAllWindows()

