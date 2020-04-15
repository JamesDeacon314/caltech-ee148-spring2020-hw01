import os
import numpy as np
import json
from PIL import Image
import cv2
import math

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''


    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below.

    # Format the image
    image = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Hue thresholds
    min_sat = min(90, int(cv2.mean(hsv)[2]))
    lower_red1 = np.array([0, min_sat, min_sat])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, min_sat, min_sat])
    upper_red2 = np.array([180, 255, 255])
    lower_not_red = np.array([30, min_sat, min_sat])
    upper_not_red = np.array([150, 255, 255])

    # Mask generation
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskr = cv2.add(mask1, mask2)

    maskbg = cv2.bitwise_not(cv2.inRange(hsv, lower_not_red, upper_not_red))
    maskr = cv2.bitwise_and(maskr, maskbg)

    # Mask filtering
    kernele = np.ones((2,2),np.uint8)
    kernel = np.ones((1,1), np.uint8)
    kerneld = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
    maskr = cv2.erode(maskr,kernel,iterations=1)
    maskr = cv2.morphologyEx(maskr, cv2.MORPH_CLOSE, kerneld, iterations=1)
    maskr = cv2.dilate(cv2.erode(maskr,kernele,iterations=1),kernele,iterations=1)

    # get contours
    contours, hierarchy = cv2.findContours(maskr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contour is a circle
    for con in contours:
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if 10 > area or area > 250:
            continue
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        if 0.8 < circularity < 1.15:
            mask = np.zeros(maskr.shape, np.uint8)
            cv2.drawContours(mask, con, -1, 255, -1)
            if cv2.mean(image, mask=mask)[2] >= 100 * min_sat / 90:
                mean_val = cv2.mean(image, mask=mask)
                if (mean_val[2] / (mean_val[1] + mean_val[0])) > 0.8:
                    bbox = cv2.boundingRect(con)
                    bounding_boxes.append([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])

    return bounding_boxes

# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = 'data/hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

preds = {}
for i in range(len(file_names)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
