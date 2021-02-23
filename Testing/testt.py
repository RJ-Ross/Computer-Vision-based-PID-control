# https://stackoverflow.com/a/64082506
import cv2

camera_capture = cv2.VideoCapture(0)

fps = 30
size = int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),\
        int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

vidwrite = cv2.VideoWriter('testvideo.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps,
           size,True)


# Write and show recording
while camera_capture.isOpened():

        _, frame =  camera_capture.read()
        vidwrite.write(frame)

        cv2.imshow("cweBCAM fUCING teST", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break
# Closes all the frames
cv2.destroyAllWindows()