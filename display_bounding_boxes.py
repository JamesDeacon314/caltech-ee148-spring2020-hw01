import os
import json
from PIL import Image
import numpy as np
import cv2
import sys

data_path = 'data/RedLights2011_Medium'
preds_path = 'data/hw01_preds'
images_path = 'data/hw1/images'
os.makedirs(images_path,exist_ok=True) # create directory if needed

if (len(sys.argv) == 1):
    print("Saving images to: " + images_path + ".  If you put in any command line arguments they will be displayed")

with open(os.path.join(preds_path,'preds.json'),'r') as f:
    boxes = json.load(f)


for img, box_set in boxes.items():
    I = Image.open(os.path.join(data_path,img))
    I = np.asarray(I)

    image = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)

    # Draw the boxes
    for i in range(len(box_set)):
        assert len(box_set[i]) == 4
        for i in range(len(box_set)):
            cv2.rectangle(image,(tuple(box_set[i][:2])),(tuple(box_set[i][2:])),(255, 255, 0),2)
    # Display results if there is any argument
    if (len(sys.argv) == 2):
        cv2.imshow("Red Lights", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    I = Image.fromarray(image)
    I.save(os.path.join(images_path,img))
