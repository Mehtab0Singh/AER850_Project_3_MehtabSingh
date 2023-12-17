# AER-850 Intro to Machine Learning: Project 3 (Mehtab Singh 500960754)
from Project3_Step1_MehtabSingh import image_masking

# Step 1 demonstration of code output
image_path = 'Motherboard Image/motherboard_image.JPEG'

# Call the process_image function
image_masking(image_path)

# Step 2: YOLOv8 Training
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(data='C:/Users/mehta/OneDrive/Documents/AER850 Intro to Machine Learning/AER850 Project 3- Mehtab Singh/Project 3 Data/data/data.yaml', epochs=100, imgsz=900, batch=3, name='AER850_Project3_Trained_Model')

# Step 3: YOLOv8 Evaluation
# Paths to the images you want to evaluate
image_paths = [
    'Project 3 Data/data/evaluation/ardmega.jpg',
    'Project 3 Data/data/evaluation/arduno.jpg',
    'Project 3 Data/data/evaluation/rasppi.jpg',
]

# Loop through each image and perform prediction
for image_path in image_paths:
    # Perform prediction on model
    results = model.predict(image_path)
    
    annotated_frame = results[0].plot() # Plot the frame for the image
    
    # Resize the annotated frame while maintaining the aspect ratio
    desired_width = 800  # Scale the Width accordningly
    aspect_ratio = annotated_frame.shape[1] / annotated_frame.shape[0] # Scale height accordingly to keep aspect ratio
    desired_height = int(desired_width / aspect_ratio)
    resized_frame = cv2.resize(annotated_frame, (desired_width, desired_height))

    # Display the resized frame
    cv2.imshow("YOLOv8 Inference", resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows() # Close any window before opening a new one