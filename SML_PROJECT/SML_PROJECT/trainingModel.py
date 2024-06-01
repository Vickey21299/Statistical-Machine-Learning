import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Function to load and preprocess images
def load_images(directory, label):
    images = []
    labels = []
    
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (150, 150))  # Resize images to 150x150
        img = img.astype('float32') / 255.0  # Normalize pixel values
        images.append(img)
        labels.append(label)
        
    return images, labels

# Load images from TRAIN/O and TRAIN/R directories
o_images, o_labels = load_images('DATASET/TRAIN/O', 1)  # Degradable (1)
r_images, r_labels = load_images('DATASET/TRAIN/R', 0)  # Non-Biodegradable (0)

# Combine images and labels
images = o_images + r_images
labels = o_labels + r_labels

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

print(train_labels[1:10])