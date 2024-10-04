import os
import cv2
import numpy as np

count= 700
for name in os.listdir("empty/"):
    img= cv2.imread("empty/"+ name)
    container= np.zeros_like(img, dtype=np.uint8)
    char= np.zeros_like(img, dtype=np.uint8)
    cv2.imwrite("images/"+ str(count)+ ".jpg", img)
    cv2.imwrite("container/"+ str(count)+ ".jpg", container)
    cv2.imwrite("char/"+ str(count)+ ".jpg", char)
    count+=1