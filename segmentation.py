import os
import tensorflow as tf
import cv2
import numpy as np
import math
from model import encoder, head, decoder
import warnings
warnings.filterwarnings("ignore")

# tf.keras.backend.clear_session()
encoder, head, decoder= encoder(weights="encoder.h5"), head(weights="head.h5"), decoder(weights="decoder.h5")

def vol_help_create(vol_help):
    vol_help= np.argmax(np.array(vol_help))
    if vol_help==0:
        return 600
    elif vol_help==1:
        return 700
    elif vol_help==2:
        return 750
    elif vol_help==3:
        return 800
    elif vol_help==4:
        return 825
    elif vol_help==5:
        return 850
    elif vol_help==6:
        return 875
    elif vol_help==7:
        return 900
    elif vol_help==8:
        return 925
    elif vol_help==9:
        return 950
    elif vol_help==10:
        return 975
    return 1000

def calc_volume(semi_major_axis, semi_minor_axis, depth_minor_axis):

    reference_distance_mm = 1500
    scaling_factor = reference_distance_mm /(2* semi_major_axis)

    Y = semi_minor_axis * scaling_factor   
    d = depth_minor_axis * scaling_factor

    projection_angle = math.acos(Y / 750)
    projection_angle_degrees = math.degrees(projection_angle)
    p = projection_angle_degrees
    depth_D = 930 - ((math.cos(math.radians(20)) * d) / (math.cos(math.radians(70 - p))))

    diameter_base = 823 /1000  # Diameter of the base in m
    height = depth_D /1000  # height of the fulcrum in m
    slope_angle = 20  # Slope angle in degrees

    radius_top_m = diameter_base / 2
    radius_bottom_m = radius_top_m + math.tan(math.radians(slope_angle)) * height                          

    # Calculate the volume of the fulcrum
    volume = (1/3)*1000 * math.pi * height * (radius_top_m**2 + radius_bottom_m**2 + radius_top_m * radius_bottom_m)
    return volume

def volume(img, big_contour, small_contour):

    result1, result2, result3= np.zeros_like(img, dtype=np.uint8), np.zeros_like(img, dtype=np.uint8), np.zeros_like(img, dtype=np.uint8)
    # result1, result2, result3= img.copy(), img.copy(), img.copy() 
    ellipse = cv2.fitEllipse(big_contour)
    (xc,yc),(d1,d2),angle = ellipse
    rmajor = max(d1,d2)/2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    x1 = xc + math.cos(math.radians(angle))*rmajor
    y1 = yc + math.sin(math.radians(angle))*rmajor
    x2 = xc + math.cos(math.radians(angle+180))*rmajor
    y2 = yc + math.sin(math.radians(angle+180))*rmajor
    result1= cv2.line(result1, (int(x1),int(y1)), (int(x2),int(y2)), (255, 255, 255), 1)
    result1= cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)

    rminor = min(d1,d2)/2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    x1 = xc + math.cos(math.radians(angle))*rminor
    y1 = yc + math.sin(math.radians(angle))*rminor
    x2 = xc + math.cos(math.radians(angle+180))*rminor
    y2 = yc + math.sin(math.radians(angle+180))*rminor
    result2= cv2.line(result2, (int(x1),int(y1)), (int(x2),int(y2)), (255, 255, 255), 1)
    result2= cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)

    result3= cv2.drawContours(result3, small_contour, -1, (255, 255, 255), 1)
    result3= cv2.cvtColor(result3, cv2.COLOR_BGR2GRAY)
    overlap = np.where(result3 * result2)
    overlap= list(zip(*overlap))
    intersect_y, intersect= math.inf, (0, 0)
    for points in overlap:
        if points[0]<intersect_y:
            intersect_y= points[0]
            intersect= points
    depth_minor_axis= math.sqrt((intersect[1]- xc)**2 + (intersect[0]- yc)**2)
    depth_minor_axis= max(0.0, rminor-depth_minor_axis)
    return calc_volume(rmajor, rminor, depth_minor_axis)


def ret_annot(img1, img2, img):

    contour1, hierarchy = cv2.findContours(img1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # outline detection
    contour1 = sorted(contour1, key=cv2.contourArea)
    contour1= contour1[-1]

    contour2, hierarchy = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour2 = sorted(contour2, key=cv2.contourArea)
    contour2= contour2[-1]

    img= cv2.drawContours(img, [contour1], -1, (255,0,0), 3)
    img= cv2.drawContours(img, [contour2], -1, (0,255,0), 3)
    result= volume(img, contour1, contour2)
    return img, img1, img2, result

def preprocess(img):
    img= cv2.resize(img, (512, 512), interpolation = cv2.INTER_CUBIC)
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gauss= cv2.GaussianBlur(gray, (5, 5),0)
    median= cv2.medianBlur(gauss, 3)
    img= np.asarray(median, dtype=np.float32)/255.0
    return np.asarray(img, dtype=np.float16).reshape(1, 512, 512, 3)


def segmentation(img):
    img_input= preprocess(img)
    out, _, _, _= decoder(encoder(img_input, training=False), training=False)
    out= np.asarray(out)*256.0
    out= np.asarray(np.clip(out, 0, 255), dtype=np.uint8)
    char, container= out[0, :, :, 0], out[0, :, :, 1]

    char= cv2.resize(char, (1024, 576), interpolation = cv2.INTER_CUBIC)
    container= cv2.resize(container, (1024, 576), interpolation = cv2.INTER_CUBIC)
    _,char = cv2.threshold(char,64,255,cv2.THRESH_BINARY)
    _,container = cv2.threshold(container,64,255,cv2.THRESH_BINARY)

    # kernel = np.ones((5, 5), np.uint8)
    # char = cv2.dilate(char, kernel, iterations=2)
    # char = cv2.erode(char, kernel, iterations=2)
    # container = cv2.dilate(container, kernel, iterations=2)
    # container = cv2.erode(container, kernel, iterations=2)

    # kernel = np.ones((3, 3), np.uint8)
    # char = cv2.erode(char, kernel, iterations=1)
    # char = cv2.dilate(char, kernel, iterations=1)
    # container = cv2.erode(container, kernel, iterations=1)
    # container = cv2.dilate(container, kernel, iterations=1)

    img, container, char, vol= ret_annot(container, char, img)
    return img, container, char, vol

def volume_estimation(img):
    img_input= preprocess(img)
    _, _, _, _, latent= encoder(img_input, training=False)
    vol, vol_help= head(latent, training=False)
    vol, vol_help= np.array(vol[0])*1000.0, vol_help_create(vol_help[0])
    return round(vol[0], 3), vol_help


img= cv2.imread("images/133.jpg") # BGR image
img1, container, char, vol= segmentation(img)
volume, vol_help= volume_estimation(img)
print(f"Volume is {round(min(1000.0, vol), 3)} liters")
print(f"Volume is {min(1000.0, volume)} liters and is less than {vol_help}")
# cv2.imshow("char",char)
# cv2.imshow("container",container)
cv2.imshow("Image",img1)
# cv2.imwrite("Predicted.jpg", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()