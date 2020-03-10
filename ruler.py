import cv2
import os
import numpy as np

def extract_lines(image, x=17, y=17):
    kernel = np.ones((1, 5), np.uint8)
    lines1 = np.copy(image)
    lines1 = cv2.dilate(lines1, kernel, iterations=x)
    lines1 = cv2.erode(lines1, kernel, iterations=x)
    
    kernel = np.ones((5, 1), np.uint8)
    lines2 = np.copy(image)
    lines2 = cv2.dilate(lines2, kernel, iterations=y)
    lines2 = cv2.erode(lines2, kernel, iterations=y)

    lines2, lines1 = np.uint8(np.clip(np.int16(lines2) - np.int16(lines1) + 255, 0, 255)), \
                     np.uint8(np.clip(np.int16(lines1) - np.int16(lines2) + 255, 0, 255))
    lines = np.uint8(np.clip((255 - np.int16(lines1)) + (255 - np.int16(lines2)), 0, 255))
    return lines

_images_dir = "images_tab/"
images = os.listdir(_images_dir)

for image_name in images[:15]:
    image_path = _images_dir + image_name
    print(image_path)
    image = cv2.imread(image_path)
    # ret,thresh = cv2.threshold(image,220,255,cv2.THRESH_BINARY)
    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tmp = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
    lines = extract_lines(tmp)
    lines = lines[5:-5, 5:-5]

    ret, labels = cv2.connectedComponents(lines)
    print(ret)
    if ret > 1:
        cv2.rectangle(image, (0, image.shape[0]-50), (250, image.shape[0]), (0,255,255),-1)
        cv2.putText(image, "RULING LINES DETECTED",(5,image.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(116,90,53),1,cv2.LINE_AA)

    cv2.imshow("img", image)
    cv2.imshow("lines", lines)
    cv2.waitKey(0)