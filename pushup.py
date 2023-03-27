import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils         
mp_drawing_styles = mp.solutions.drawing_styles 
mp_pose = mp.solutions.pose

count = 0
position = None

cap = cv2.VideoCapture('videos/push-ups-1.mp4')

with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success,image=cap.read()
        if not success:
            print("Cannot activate camera")
            exit()
        
        image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
        result=pose.process(image)
        
        imList=[]
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            for id,im in enumerate(result.pose_landmarks.landmark):
                h,w,_=image.shape
                x,y=int(im.x*w),int(im.y*h)
                imList.append([id,x,y])
        if len(imList) != 0:
            if ((imList[12][2] - imList[14][2])>=15 and (imList[11][2] - imList[13][2])>=15):
                position = "down"
            if ((imList[12][2] - imList[14][2])<=5 and (imList[11][2] - imList[13][2])<=5) and position == "down":
                position = "up"
            count +=1 
            print(count)
            
        cv2.imshow("Pushup counter", cv2.flip(image,1))
        if cv2.waitKey(1) == ord('q'):
            break   
cap.release()