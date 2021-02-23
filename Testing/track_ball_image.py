# Ryan Ross
# 23/2/2021
# Testing opencv playing video files - framerates, resolution etc

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
ballLower = (7, 88, 164)
ballUpper = (179, 255, 255)
pts = deque(maxlen=5)


def main():
    # cap = open_video()
    frame = cv2.imread('C:/Users/Ryan/Documents/SharedUbuntu16.4LTS/BallPID/test_files/ball2_resized.png')
    while(True):
            cv2.namedWindow("OG", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('OG', frame)

            # blur frame, and convert it to the HSV color space
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            cv2.namedWindow("blurred", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('blurred', blurred)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, ballLower, ballUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('mask', mask)

            result = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('result', result)








            if (cv2.waitKey(33) & 0xFF) == ord('q'): # Hit `q` to exit
                print("break here")
                # Release everything if job is finished
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main()