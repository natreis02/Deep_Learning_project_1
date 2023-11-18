# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:52:43 2023

@author: Natália França dos Reis and Vitor Hugo Miranda Mourão
"""

import cv2
import numpy as np

path_weights = "C:/Users/usuario/Desktop/DL_project_1/weights/yolov3.weights"
path_cfg = "C:/Users/usuario/Desktop/DL_project_1/cfg/yolov3.cfg"
path_names = "C:/Users/usuario/Desktop/DL_project_1/data/coco.names"
path_image = "C:/Users/usuario/Desktop/DL_project_1/images/pessoas.jpg"

# Carregue a rede YOLO com os pesos pré-treinados e configuração
net = cv2.dnn.readNet(path_weights, path_cfg)

# Carregue as classes de objetos
with open(path_names, 'r') as f:
    classes = f.read().strip().split('\n')

# Carregue a imagem
image = cv2.imread(path_image)

# Obtenha as dimensões da imagem
height, width = image.shape[:2]

# Crie um blob a partir da imagem
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Defina as entradas da rede
net.setInput(blob)

# Obtenha as saídas da rede (detecções)
outs = net.forward(net.getUnconnectedOutLayersNames())

# Inicialize listas para caixas delimitadoras, confiança e IDs de classe
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


# Aplicar supressão não máxima para eliminar detecções sobrepostas
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Desenhe as caixas delimitadoras e rótulos nas detecções restantes
for i in indices:
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    confidence = confidences[i]

    # Desenhe a caixa delimitadora
    color = (0, 255, 0)  # Cor verde
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Escreva o rótulo e a confiança
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Exiba a imagem com as detecções
cv2.imshow("Detecções YOLO", image)
cv2.waitKey(0)
cv2.destroyAllWindows()