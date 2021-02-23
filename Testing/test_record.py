import cv2
cap = cv2.VideoCapture(0)
cap.set(8,100)
out = cv2.VideoWriter('/home/pi/Work/01.mp4',cv2.CV_FOURCC(*'DIVX'),20.0,(640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(10) == 27: 
            break
cap.release()
out.release()
cv2.destroyAllWindows()