# #####
# Tutorial: Object detection using HSV Color space – OpenCV 3.4 with python 3 Tutorial 9
# URL: https://pysource.com/2018/01/31/object-detection-using-hsv-color-space-opencv-3-4-with-python-3-tutorial-9/
# ###
import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    frame = cv2.imread('ball2_resized.png') # default is BGR (0 -255)
    # frame = cv2.resize(frame, ())
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_colour = np.array([l_h, l_s, l_v])
    upper_colour = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_colour, upper_colour)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    key = cv2.waitKey(0)
    if key == 27 or key == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break