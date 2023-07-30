import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
fieldnames = ['x', 'y', 'z', 'visibility']

cap = cv2.VideoCapture("videos\squat2.mp4")

# Define the output video path
output_path = os.path.join("dotDisplay", "output_video_dot.mp4")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mjpg')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render detections with only individual circles at each landmark point
        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility > 0.5:
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(image, (cx, cy), radius=2, color=(245, 117, 66), thickness=2)
        
        # Draw the connections between the landmarks (this will connect adjacent landmarks)
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        cv2.imshow('detect', image)
        
        # Write the frame to the output video
        out.write(image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
