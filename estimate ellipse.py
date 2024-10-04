import cv2
import numpy as np
from PIL import Image, ImageFilter

R, r, H= 1500/2, 823/2, 930
H_hypothetical= R*H/(R-r)

def find_volume(r_container, r_ash):
    scale= R/r_container
    r_container= scale* r_container
    r_ash= scale* r_ash
    print(r_ash)
    h= H_hypothetical*r_ash/R
    h_extra= H_hypothetical*r/R

    print(f"Container volume: {np.pi*((r_container**2)*h - (r**2)*h_extra)/3}")
    volume= np.pi*((r_ash**2)*h - (r**2)*h_extra)/3
    return volume

label= cv2.imread("labels/0.jpg")
image= cv2.imread("images/0.jpg")
label= np.asarray(label, dtype= np.uint8)
label= cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(label, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea)

contour_container= contours[-1]
contour_char= contours[-2]
ellipse_container = cv2.fitEllipse(contour_container)
ellipse_char = cv2.fitEllipse(contour_char)

image= cv2.drawContours(image, [contour_container], -1, (255,0,0), 3)

(xc,yc),(d1,d2),angle = ellipse_container
(xc_char,yc_char),(d1_char,d2_char),angle_char = ellipse_char

volume= round(find_volume(max(d1,d2)/2, max(d1_char,d2_char)/2), 2)
string= f"Volume of char is {volume} cubic units"
print(string)

image= cv2.ellipse(image,ellipse_container, (0,0,255), 3)
image= cv2.ellipse(image,ellipse_char, (0,255, 0), 3)
image= cv2.putText(image, string, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0,0,0), 2, cv2.LINE_AA)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()