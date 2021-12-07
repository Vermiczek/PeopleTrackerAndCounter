import numpy as np
import cv2
import time
import threading
import centroidtracker
import counter
from datetime import datetime
from datetime import datetime
import os



# system parameters
CONFIDENCE_FILTER=0.5
THRESHOLD= 0.45
SAVING_TIME=4
inputFrameSize= (256,256)



(W, H) = (None, None)

timetest=[]

# read neural network structure, weights and biases
yolo = cv2.dnn.readNetFromDarknet("yolov4.cfg",
                                  "yolov4.weights")
outputlayers = yolo.getUnconnectedOutLayersNames()

# read labels
with open("coco.names", 'r') as f:
    LABELS = f.read().splitlines()

# create rgb colors for every label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

xSize = 1000
ySize = 600
counter = counter.Counter(119, 0, 350, 220)
previousCentroids = []
currentCentroids = []


cap = cv2.VideoCapture('12.mp4')
savingTime = time.time()

def checkId(id1, id2):
    if id1 == id2:
        return True
    else:
        return False


tracker = centroidtracker.CentroidTracker()

while (True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (600, 608))

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # create input matrix from frame, apply transformations and pass it to the first layer of ANN
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, inputFrameSize, swapRB=True, crop=False)
    yolo.setInput(blob)

    # make forward pass and calculate its time
    start = time.time()
    layerOutputs = yolo.forward(outputlayers)
    end = time.time()

    cv2.rectangle(frame, (counter.xCoord, counter.yCoord), (counter.xCoord+counter.xSize, counter.yCoord+counter.ySize), (0, 0, 255), 5)


    # initialize our lists of detected bounding boxes, confidences and class IDs for every grabbed frame
    boxes = []
    confidences = []
    classIDs = []

    # use output of ANN
    for output in layerOutputs:
        for detection in output:

            # calculate highest score and get it`s confidence number
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            # if confidence is higher than selected value of CONFIDENCE_FILTER create bounding box for every detection
            if confidence > CONFIDENCE_FILTER:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # get left corner coordinates of bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(class_id)

        # apply non-maxima suppression to overlapping bounding boxes with low confidence
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_FILTER,
                                THRESHOLD)
        rects = []
        # check if any bounding box exists
        if len(idxs) > 0:


            # plot bounding boxes
            for i in idxs.flatten():

                # get the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle, label and confidence on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
               # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                rects.append(boxes[i])
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                           confidences[i])
                #cv2.putText(frame, text, (x, y-5),
                 #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            bboxes = np.array(rects)
            #print(bboxes)
            bboxes = bboxes.astype(int)
            objects = tracker.update(rects)
            #print(objects.items())
            for (objectid, box1) in objects.items():
                x1, y1 = box1
                #print(box)
                x1 = int(x1)
                y1 = int(y1)
                #y2 = int(y2)
                #x2 = int(x2)
                cv2.rectangle(frame, (x1,y1), (x1,y1), (0, 0, 255), 10)
                text= "Object ID: {}".format(objectid)
                cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
                centroid = [x1, y1, objectid]
                currentCentroids.append(centroid)

            # saving frame
            if (time.time() - savingTime >= SAVING_TIME and classIDs.count(1)):

                print("saving")
                savingTime = time.time()

                try:
                   # saveThread = threading.Thread(target=save_frame, args=(frame,dir))
                    saveThread.start()
                except:
                    print("error while saving")

            for (x1, y1, id1) in currentCentroids:
               found = False
               for (x2, y2, id2) in previousCentroids:
                   if id1==id2:
                    counter.RegisterAction(x1,y1,x2,y2)
                    found = True
                    break
               if found is not True:
                   counter.RegisterAction(x1, y1, x1, y1)

            if counter.peopleInsideTheBuilding>=0:
                temp=" + "
            else:
                temp=" "

            Text = "People inside the building = N" + temp + str(counter.peopleInsideTheBuilding)
            cv2.putText(frame, Text, (15, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            previousCentroids = currentCentroids
            currentCentroids = []

            print(counter.peopleInsideTheBuilding)


    cv2.imshow("frame", frame)


    pressedKey = cv2.waitKey(1) & 0xFF


print(sum(timetest)/len(timetest))
cap.release()
cv2.destroyAllWindows()