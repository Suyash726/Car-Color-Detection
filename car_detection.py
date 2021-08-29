import numpy as np
import cv2

# Get coco class names
classes = 'coco.names'
classnames = []

with open(classes,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

print(classnames)

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

whT =320
confidenceThresh = 0.5
nmsThreshold =0.3

def findObjects(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confidence_val = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThresh:
                w,h = int(detect[2]*wT),int(detect[3]*hT)
                x,y = int((detect[0]*wT) - w/2),int((detect[1])*hT-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confidence_val.append(float(confidence))
    print(len(bbox))

    indices = cv2.dnn.NMSBoxes(bbox,confidence_val,confidenceThresh,nmsThreshold)

    

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        cv2.putText(img,f'{classnames[classIds[i]].upper()}',(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6,(0,0,255),1)






# Get the frame of our webcam
cap = cv2.VideoCapture('input.mp4')
while True:
    success,img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)    
    
    cv2.imshow('Image',img)
    cv2.waitKey(1)





