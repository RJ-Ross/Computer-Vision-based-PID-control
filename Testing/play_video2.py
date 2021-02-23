# https://stackoverflow.com/a/44640275
import numpy as np
import cv2

cap = cv2.VideoCapture('02.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.namedWindow("res", cv2.WINDOW_NORMAL)
        cv2.imshow('res', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()