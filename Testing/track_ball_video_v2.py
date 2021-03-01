# Ryan Ross
# 24/2/2021

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import cv2
import imutils
import time
import csv

# define the lower and upper boundaries of the ball in the HSV color space,
# then initialize the list of tracked points
# ball_HSV_Lower = (7, 84, 172)
# ball_HSV_Upper = (179, 255, 255)
ball_HSV_Lower = (12, 72, 157)
ball_HSV_Upper = (30, 153, 255)
queue_size = 10# length of tail of tracked ball
pts = deque(maxlen=queue_size) # quicker append and pop from both ends than lists

# class Window:
#
#     def __init__(self):


class Ball:
    # class to store ball pixel parameters
    max_x = 0
    min_x = 1080
    mid_x = 0
#     csv_file = Nul
    # def __init__(self):
    #     csv =

# class PID:
#     pid_p, pid_i, pid_d = 0
#     kp = 0
#     ki = 0
#     kd = 0
#
#     SP = 0
#     error = 0
#     time_now, time_previous, elapsed_time = 0
#
#     def calculate(self, setpoint, cX):
#         # pos = const + kp * (SP - cX) + ki * integral[ SP - cX ]dt   +  kd * cX/dt
#
#         self.SP = setpoint
#         self.error = setpoint - cX  # e(t)
#
#         # proportional
#         self.pid_p = self.error * self.kp
#
#         # Integral
#         self.pid_i = self.pid_i + (self.ki * self.error)
#         # SP = T1 if t < 50 else 50
#
#         # Derivative
#         self.time_previous = self.time_now
#         self.time_now = time.time()
#         self.elapsedTime = time_now - time_previous
#         pid_d = kd * ((self.error - self.previous_error) / self.elapsed_time)
#         previous_error = error
#
#         PID = pid_p + pid_i + pid_d



def open_video_source():
    cap = cv2.VideoCapture('C:/Users/Ryan/Documents/SharedUbuntu16.4LTS/BallPID/test_files/01.avi')
    # cap = cv2.VideoCapture('C:/Users/Ryan/Documents/SharedUbuntu16.4LTS/BallPID/test_files/02.avi')
    return cap

def main():
    ball = Ball()
    cap = open_video_source()
    fps = 0
    func_time = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            start_func = time.clock()
            # morph_operations(frame)
            OG = frame.copy()
            # -----
            # blur frame, and convert it to the HSV color space
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            # cv2.namedWindow("blurred", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('blurred', blurred)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, ball_HSV_Lower, ball_HSV_Upper)
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
            center = None #
            # only proceed if at least one contour was found
            if len(cnts) > 0:    #source: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw line @ midpoint
                if cX > ball.max_x:
                    ball.max_x = cX
                elif cX < ball.min_x:
                    ball.min_x = cX
                ball.mid_x = int((ball.max_x + ball.min_x)/2)
                # cv2.line(frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2), 0), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (0, 0, 255), 2)
                cv2.line(frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)-25, 0), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)-25, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (255, 229, 128), 1)
                cv2.line(frame, (ball.mid_x, 0), (ball.mid_x, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (255, 0, 0), 1)


                # crosshairs
                cv2.line(OG, (cX, 0), (cX, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (0, 0, 255), 1)
                cv2.line(OG, (0, cY), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), cY), (0, 0, 255), 1)



                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    # cv2.circle(frame, center, 5, (0, 0, 255), -1) # neg size = fill circle
                    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1) # neg size = fill circle
                    text = "cX: " + str(cX) + ", cY:" + str(cY)
                    cv2.putText(frame, text, (cX -65, cY - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            else:
                print("none")
            # update the points queue
            pts.appendleft(center) #ppends the centroid to the pts  list

            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line andq
                # draw the connecting lines
                thickness = int(np.sqrt(5 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

            cv2.putText(frame, "fps "+ str(int(fps)), (20,20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, "{:.4f} ms".format(float(func_time)), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.namedWindow("Tracked", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Tracked', frame)

            cv2.namedWindow("OG", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('OG', OG)

            # -----
            # end_func = time.clock()
            # func_time = end_func- start_func
            # fps = 1/(func_time)
            # print("Func time: {:.4f} ms".format(float(1000*(func_time))))

            if (cv2.waitKey(20) & 0xFF) == ord('q'): # Hit `q` to exit
                break

            end_func = time.clock()
            func_time = end_func - start_func
            fps = 1 / (func_time)
            print("Func time: {:.4f} ms".format(float(1000 * (func_time))))
        else:
            # break
            cap = cv2.VideoCapture('C:/Users/Ryan/Documents/SharedUbuntu16.4LTS/BallPID/test_files/01.avi')

    # Release everything if job is finished
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()