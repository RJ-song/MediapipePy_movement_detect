from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import csv

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with open('models\pickle\pushup_front.pkl', 'rb') as f:
    model = pickle.load(f)
    
landmarks = ['class']
for val in range(1,33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val),]
# X = pd.DataFrame([row], columns = landmarks[1:])

cap = cv2.VideoCapture(0) #"videos/pushups-front1.mp4"
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)
counter = 0
current_stage = ' '
#output videos

# out = cv2.VideoWriter('output_videos\pushup.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (int(height),int(width)))
# cv2.VideoWriter_fourcc('P','I','M','1')
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # frame = cv2.flip(frame,0)

        # write the flipped frame
        
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2) 
                                 )               
        
        
    
        
        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks[1:])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class,body_language_prob)
            
            if body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] >= .8:
                current_stage = 'down'
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(255, 208, 0), thickness=2, circle_radius=2) 
                                 )
            elif current_stage == 'down' and body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] >= .7:
                current_stage = 'up'
                counter +=1
                print(current_stage)
        
            #get status box    
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            #display class
            cv2.putText(image, "CLASS", (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1,  cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (95,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2,  cv2.LINE_AA)
            
            #display probability
            cv2.putText(image, "PROB", (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1,  cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            #display conter
            cv2.putText(image, "COUNT", (180,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1,  cv2.LINE_AA)
            cv2.putText(image, str(counter), (175,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2,  cv2.LINE_AA)
            cv2.imshow('model test conut', image)
    
        except Exception as e:
            pass
        
        
        try:
            cv2.imshow('model test conut', image)
            #  out.write(image)
        except Exception as e:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        

    
   
    cap.release()
    # out.release()
    cv2.destroyAllWindows()