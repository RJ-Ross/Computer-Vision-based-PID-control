# import the necessary packages
from collections import deque #list of the past N (x, y)-locations of the ball in our video stream
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size") #he maximum size of our deque , which maintains a list of the previous (x, y)-coordinates of the ball we are tracking. This deque  allows us to draw the “contrail” of the ball, detailing its past locations.
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
i = 1
# keep looping
while(vs.isOpened()):
	# grab the current frame
	ret, frame = cap.read()
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
	i += 1
	print(i)
	print(str(frame.size))
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		print("frame none ")
		break

	cv2.imshow("frame", frame)
	key = cv2.waitKey(0)

	if key == 27 or key == ord('q'):
		cv2.destroyAllWindows()
		vs.release()
		break