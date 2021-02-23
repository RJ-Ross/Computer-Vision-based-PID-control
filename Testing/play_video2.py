# Ryan Ross
# 23/2/2021
# Testing opencv playing video files - framerates, resolution etc
#
# https://stackoverflow.com/a/44640275
# Framerate fix https://stackoverflow.com/a/25184326
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/
# https://stackoverflow.com/questions/39953263/get-video-dimension-in-python-opencv
import numpy as np
import cv2
import imutils

cap = cv2.VideoCapture('C:/Users/Ryan/Documents/SharedUbuntu16.4LTS/BallPID/test_files/02.avi')

fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
print("Width: {0}".format(width))
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
print("height: {0}".format(height))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while(cap.isOpened()):
    ret, frame = cap.read()
    # frame = imutils.resize(frame, width=640, height=480)
    if ret:
        cv2.namedWindow("WINDOW_NORMAL", cv2.WINDOW_NORMAL)
        cv2.namedWindow("WND_PROP_FULLSCREEN", cv2.WND_PROP_FULLSCREEN)
        cv2.namedWindow("WINDOW_AUTOSIZE", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('WINDOW_NORMAL', frame)
        cv2.imshow('WND_PROP_FULLSCREEN', frame)
        cv2.imshow('WINDOW_AUTOSIZE', frame)
        if (cv2.waitKey(33) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()