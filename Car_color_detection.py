import cv2
import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from imutils.video import FPS
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


CONFIDENCE = 0.1
SCORE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.3

whT =320



config_path = "yolov3.cfg"
weights_path = "yolov3.weights"

# loading all the class labels (objects)
labels = open("coco.names").read().strip().split("\n")
# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

def getColorName(R ,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname
vehicles =  ['car', 'motorbike', 'bus', 'truck', 'aeroplane']


def getcars(image):
    # print("It's working ...")
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    layer_outputs = net.forward(ln)
    
    font_scale = 1
    thickness = 1
    boxes, confidences, class_ids = [], [], []
    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # perform the non maximum suppression given the scores defined before
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    clt = KMeans(n_clusters = 4)



    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            if i in idxs:
                if labels[class_ids[i]] in vehicles:
                    copied_image = image.copy()
                    x,y,w,h = boxes[i]
                    scaleTop = int(h * 0.30) # scaling the top with the % 
                    scaleBottom = int(h * 0.15)  # scaling the bottom with the % 
                    x1 = x
                    y1 = y + scaleTop
                    x2 = x + w
                    y2 = (y + h) - scaleBottom
                    #print("x: {}, y: {}, w: {}, h: {}, scaleTop: {}, scaleBottom: {}".format(x,y, w, h, scaleTop, scaleBottom))
                    #print("scaleTop:y + scaleBottom, x:x + w")
                    crop_img = copied_image[y1:y2, x1:x2]
                    #print(crop_img.shape)
                    
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    crop_img = cv2.fastNlMeansDenoisingColored(crop_img,None,10,10,7,21)
                    
                    #cv2.imwrite(filename +  str(i) + "_Object." + ext, crop_img)
                    pixels = crop_img.reshape((crop_img.shape[0] * crop_img.shape[1], 3))
                    labelsinvehicle = clt.fit_predict(pixels)
                    label_counts = Counter(labelsinvehicle)
                    #subset out most popular centroid
                    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
                    r, g, b = dominant_color
                    color_present = getColorName(b, g, r)
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            if labels[class_ids[i]] in vehicles:
                text = f"{color_present} { labels[class_ids[i]]}: {confidences[i]:.2f} "
            else:
                text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            
            #overlay = image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            # cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # add opacity (transparency to the box)
            
            # image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            
            # now put the text (label: confidence %)
            cv2.putText(image,text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)


cap = cv2.VideoCapture('input.mp4')


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
result = cv2.VideoWriter('Final_Output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)


while True:
    # Save our image and tell us if it was done succesfully
    success,img = cap.read()
    getcars(img)
    result.write(img)
    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Add the delay
result.release
cv2.waitKey(0)
