## Import section
import cv2 as cv
from ultralytics import YOLO
import math

## Methods section
def inferImage(path):
    """Run YOLO inference model for detection and return result object"""
    # Load a model
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

    # Run inference on an images
    results = model(path)  # return a list of results
    return results

# TODO: current problem with the method and approach in general is that if there are multiple 
#   instances of thesame object lying around 
#   Current work around is to just give out the highest probability encounterred
def getProbabilities(results):
    """Aurelien's baseline model uses probabilties which I assume are coming from YOLO model.
    Returns dictionary"""
    typeProbMatch = dict()

    detectedBoxes = results[0].boxes.cpu().numpy() # only one detected box at [0]. Convert to 
                                                    # numpy instead of tensor
    detectedClasses = detectedBoxes.cls # all class indexes
    detectedProb = detectedBoxes.conf # all confidence levels for relevant class indexes

    # zip into pair of tuples
    for pair in zip(detectedClasses, detectedProb):
        print(pair)
        className = results[0].names[int(pair[0])]
        if (className in typeProbMatch) and (typeProbMatch.get(className) > pair[1]): continue

        typeProbMatch[className] = pair[1] 
    return typeProbMatch

## Executable section
# results = inferImage('C:/Users/aured/Desktop/Learning Material/Bsc - DSAI UM/Year 3/Semester 1 Period 1/Project 3-1 Phase 1/Code/Project3-1/baselines/probability/bus.jpg')
# typeProbMatch = getProbabilities(results)
# print(typeProbMatch)