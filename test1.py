# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:52:43 2023

@author: Natália França dos Reis and Vitor Hugo Miranda Mourão
"""

# Import necessary libraries
import cv2 # OpenCV library for computer vision tasks
import numpy as np
import os
import matplotlib.pyplot as plt

# Define file paths
nat_names = "C:/Users/usuario/Desktop/DL_project_1/"
vit_names = "C:/Users/vitor/Downloads/Pos grad/Doutorado/MT862 - Deep Learning/projeto 1/"

path_weights = os.path.join(vit_names, "weights/yolov3.weights")
path_cfg = os.path.join(vit_names, "cfg/yolov3.cfg")
path_names = os.path.join(vit_names, "data/coco.names")
path_image = os.path.join(vit_names, "images/food.jpg")

# Load the YOLO network with pre-trained weights and configuration
net = cv2.dnn.readNet(path_weights, path_cfg)

# Load the object classes
with open(path_names, 'r') as f:
    classes = f.read().strip().split('\n')
    
np.random.seed(99)
# Define a list of unique colors for each class
class_colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the image
image = cv2.imread(path_image)

# Get the dimensions of the image
height, width = image.shape[:2]

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set the input for the network
net.setInput(blob)

# Get the network outputs (detections)
outs = net.forward(net.getUnconnectedOutLayersNames())

# Initialize lists for bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

# Loop through the outputs
for out in outs:
    for detection in out:
        # Get confidence scores for each class
        scores = detection[5:]
        # Identify the class with the highest confidence score
        class_id = np.argmax(scores)
        # Get the confidence associated with the identified class
        confidence = scores[class_id]

        # Check if confidence is greater than 0.5 (a confidence filter)
        if confidence > 0.5:
            # Scale back the bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the coordinates of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Store bounding box information, confidence, and class
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


# Apply non-maximum suppression to eliminate overlapping detections
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 1-0.5)

# Draw bounding boxes and labels on the remaining detections
for i in indices:
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    confidence = confidences[i]

    # Draw the bounding box
    color = class_colors[class_ids[i]]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Write the label and confidence
    cv2.putText(image,f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.75, color,1)
    
# Display the image with detections using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.title("YOLO Detections")
plt.axis('off')  # Turn off axis labels and ticks
plt.show()