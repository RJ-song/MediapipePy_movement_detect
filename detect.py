import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import csv


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
fieldnames = ['x', 'y', 'z', 'visibility']

cap = cv2.VideoCapture("videos/pushups-1.mp4")


def export_landmarks(results, action) :
        try:
           keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
           keypoints.insert(0,action)
           
           with open('output.csv', mode='a',newline='') as f:
               csv_writer=csv.writer(f,delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
               csv_writer.writerow(keypoints)
        except Exception as e:
                pass 

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)
        # print(results.pose_landmarks)
        
        landmarks = ['class']
        for val in range(1,33+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val),]

        #write first row
        # with open('output.csv', mode = 'w' ,newline='') as f:
        #     csv_writer=csv.writer(f,delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
        #     csv_writer.writerow(landmarks)
        
        
        k = cv2.waitKey(10)
        if k == 117 :
            export_landmarks(results, 'up')
        if k == 100 :
            export_landmarks(results, 'down')
        
        
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    # export_landmarks(results,'up')   
    
            
    cap.release()
    cv2.destroyAllWindows()
    
   
  
 