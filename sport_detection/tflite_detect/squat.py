import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.(model_path="models/tflite/squat.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load your video
cap = cv2.VideoCapture("videos/squat.mp4")
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
counter = 0
current_stage = ' '

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame to fit the input of the TFLite model
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = input_frame / 255.0  # Normalize the input
    input_frame = np.expand_dims(input_frame, axis=0)

    # Set the input tensor to the TFLite model
    interpreter.set_tensor(input_details[0]['index'], input_frame)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process the output data to determine the pose
    # You'll need to implement your own logic here based on the model's output

    # Display the frame with pose information
    cv2.imshow('Pose Estimation', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
