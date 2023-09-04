# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 20:03:43 2023

@author: usuario
"""

import cv2
import numpy as np

# Carregue a rede YOLO com os pesos pré-treinados e configuração
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Carregue as classes de objetos
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Carregue a imagem
image = cv2.imread('image.jpg')

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

# Loop pelas saídas
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # Filtro de confiança
            # Escala de volta as coordenadas da caixa delimitadora
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Coordenadas da caixa delimitadora
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Aplicar supressão não máxima para eliminar detecções sobrepostas
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Desenhe as caixas delimitadoras e rótulos nas detecções restantes
for i in indices:
    i = i[0]
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