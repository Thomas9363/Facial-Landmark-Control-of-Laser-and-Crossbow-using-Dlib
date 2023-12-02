# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import RPi.GPIO as GPIO
from time import sleep
from adafruit_servokit import ServoKit

def ratio(point):
    A = dist.euclidean(point[1], point[5])
    B = dist.euclidean(point[2], point[4])
    C = dist.euclidean(point[1], point[2])
    D = dist.euclidean(point[4], point[5])
    E = dist.euclidean(point[1], point[4])
    ar = (A + B) / (C+D)
    S1 = (C+B+E)/2
    S2 = (A+D+E)/2
    total_area= ((S1*(S1-C)*(S1-B)*(S1-E))**0.5)+((S2*(S2-A)*(S2-D)*(S2-E))**0.5)
    return ar, total_area # return the eye aspect ratio

laser = 12
LED=21
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(laser, GPIO.OUT)
GPIO.setup(LED, GPIO.OUT) 
GPIO.output(laser, False)
GPIO.output(LED, False)

kit = ServoKit(channels=16,address=0x40) # Initialize servo controller
PAN_CHANNEL,LASER_CHANNEL  = 0, 14 # Channel 0 for servo control
pan_initial_angle=110.0 # Set initial angles for servos
kit.servo[PAN_CHANNEL].angle = pan_initial_angle # Move servos to initial angles
kit._pca.channels[LASER_CHANNEL].duty_cycle = 0 # initial laser off
pan_angle = pan_initial_angle # servo initial position in degree

EYE_AR_THRESH = 0.7
MOUTH_AR_THRESH = 3.0

print("[INFO] loading facial landmark predictor...")
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# grab the indexes of the facial landmarks for eyes and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

vs = VideoStream(src="EmilyBlink.mp4").start()
#vs = VideoStream(src=0).start()
time.sleep(1.0)

pTime = 0.0 #time tracking for FPS
cTime = 0.0
fps=0

while True:
     
    frame = vs.read()
    if frame is None:
        break
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=480)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0) # detect faces in the grayscale frame

    for rect in rects: # loop over the face detections
        shape = predictor(gray, rect) # determine the facial landmarks for the face region
        shape = face_utils.shape_to_np(shape) # convert the facial landmark (x, y)-coordinates to a NumPy array

        leftEye = shape[lStart:lEnd] # extract eye coordinates
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd] # extract mouth coordinates and slice to only outline
        indices_to_extract = [48, 50, 52, 54, 56, 58]
        mouth_outline=shape[indices_to_extract]

        leftAR, leftArea = ratio(leftEye )
        rightAR, rightArea = ratio(rightEye)
        mouthAR, mouthArea = ratio(mouth_outline)
        area_ratio=leftArea/rightArea
#         print(f"{leftAR:.2f}",',',f"{rightAR:.2f}" )
#         print(f"{leftArea:.2f}",',' f"{rightArea:.2f}" )
#        print(f"{leftAR:.2f}",',',f"{rightAR:.2f}",',',f"{leftArea:.2f}",',' f"{rightArea:.2f}"  )
       
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull= cv2.convexHull(mouth_outline)
#         for point in leftEyeHull:
#             x, y = point[0]
#             cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
#         for point in rightEyeHull:
#             x, y = point[0]
#             cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)         
#         for point in mouthHull:
#             x, y = point[0]
#             cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)         
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
#         cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)

        if leftAR < EYE_AR_THRESH and area_ratio<1:
            pan_angle += 1.0 
            kit.servo[0].angle = pan_angle
            print('right closed ------ left open', pan_angle)
            for point in leftEyeHull:
                x, y = point[0]
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
            
        if rightAR < EYE_AR_THRESH and area_ratio>1:            
            pan_angle -= 1.0 #0.3
            kit.servo[0].angle = pan_angle
            print('left closed ------ lright open', pan_angle )
            for point in rightEyeHull:
                x, y = point[0]
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
                
        if mouthAR < MOUTH_AR_THRESH:
            GPIO.output(laser, False) # turn on laser and led on 
            GPIO.output(LED, False)
            kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
        if mouthAR >= MOUTH_AR_THRESH :
            GPIO.output(laser, True) # turn on laser and led on 
            GPIO.output(LED, True)
#            kit._pca.channels[LASER_CHANNEL].duty_cycle = 65535
#            cv2.putText(frame, "Fire", (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#             for point in mouthHull:
#                 x, y = point[0]
#                 cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)         
#             cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)

    cTime = time.time() # calculate and display FPS
    fps = 0.9*fps+0.1*(1/(cTime-pTime))
    pTime = cTime
    cv2.putText(frame, f"FPS : {int(fps)}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)     

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break
kit.servo[0].angle = pan_initial_angle 
kit._pca.channels[LASER_CHANNEL].duty_cycle = 0

cv2.destroyAllWindows()
vs.stop()




