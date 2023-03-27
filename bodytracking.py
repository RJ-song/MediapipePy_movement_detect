import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils         
mp_drawing_styles = mp.solutions.drawing_styles 
mp_holistic = mp.solutions.holistic 

cap = cv2.VideoCapture('videos/situps.mp4')

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    if not cap.isOpened():
        print("Cannot activate camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(img,(700,400))
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
        results = holistic.process(img2)              
     
        mp_drawing.draw_landmarks(
            img,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
    
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())

        cv2.imshow('detection', img)
        if cv2.waitKey(1) == ord('q'):
            break   
cap.release()
cv2.destroyAllWindows()