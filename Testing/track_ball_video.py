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
# ballLower = (7, 88, 164)
ballLower = (7, 84, 172)
ballUpper = (179, 255, 255)
queue_size = 10
pts = deque(maxlen=queue_size) # quicker append and pop from both ends than lists

def morph_operations(frame):
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

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        print("none")
    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(queue_size / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)



def main():
    # frame = cv2.imread('C:/Users/Ryan/Documents/SharedUbuntu16.4LTS/BallPID/test_files/ball2_resized.png')
    cap = cv2.VideoCapture('C:/Users/Ryan/Documents/SharedUbuntu16.4LTS/BallPID/test_files/01.avi')
    # cap = cv2.VideoCapture('C:/Users/Ryan/Documents/SharedUbuntu16.4LTS/BallPID/test_files/02.avi')

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            start_func = time.clock()
            # morph_operations(frame)

            # -----
            cv2.namedWindow("OG", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('OG', frame)

            # blur frame, and convert it to the HSV color space
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            # cv2.namedWindow("blurred", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('blurred', blurred)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, ballLower, ballUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            # cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('mask', mask)

            result = cv2.bitwise_and(frame, frame, mask=mask)
            # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('result', result)

            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None
            # only proceed if at least one contour was found
            if len(cnts) > 0:    #source: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1) # neg size = fill circle
            else:
                print("none")
            # update the points queue
            pts.appendleft(center)

            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(5 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

                cv2.namedWindow("tracked", cv2.WINDOW_AUTOSIZE)
                cv2.imshow('tracked', frame)

            # -----
            end_func = time.clock()
            print("Func time: {:.4f} ms".format(float(1000*(end_func- start_func))))
            start_func = time.clock()

            if (cv2.waitKey(33) & 0xFF) == ord('q'): # Hit `q` to exit
                break
        else:
            break

    # Release everything if job is finished
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()