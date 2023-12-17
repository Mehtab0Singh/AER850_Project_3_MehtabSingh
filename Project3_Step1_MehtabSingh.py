# AER-850 Intro to Machine Learning: Project 3 (Mehtab Singh 500960754)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Object Masking
# Making a code that can be called when doing Epoch
def image_masking(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # First Threshold: Inverted Threshold
    _, threshold1 = cv2.threshold(grey_image, 85, 255, cv2.THRESH_BINARY)
    # Use cv2.bitwise_not for inverse extraction
    inverse_threshold = cv2.bitwise_not(threshold1)
    # Find contours in the edge-detected image
    contours,_  = cv2.findContours(inverse_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out small contours based on area
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]
    # Create a mask for the filtered contours
    mask = np.zeros_like(image)
    cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # Use cv2.bitwise_and to extract the Motherboard from the background
    masked_image = cv2.bitwise_and(image, mask)
    
    # Second Threshold
    _, threshold2 = cv2.threshold(grey_image, 90, 255, cv2.THRESH_BINARY)
    # Find contours in the edge-detected image
    contours,_  = cv2.findContours(threshold2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out small contours based on area
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]
    # Create a mask for the filtered contours
    mask = np.zeros_like(image)
    cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # Use cv2.bitwise_and to extract the Motherboard from the background
    masked_image = cv2.bitwise_and(masked_image, mask)
    
    # Make a window layout to plot all images
    plt.figure(figsize=(128, 48))
    
    plt.subplot(141)
    plt.title('Original Image', fontsize=100)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(142)
    plt.title('Thresholded Image', fontsize=100)
    plt.imshow(threshold2, cmap='gray')
    plt.axis('off')
    
    plt.subplot(143)
    plt.title('Thresholded Inverted Image', fontsize=100)
    plt.imshow(inverse_threshold, cmap='gray')
    plt.axis('off')
    
    plt.subplot(144)
    plt.title('Masked Image', fontsize=100)
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()