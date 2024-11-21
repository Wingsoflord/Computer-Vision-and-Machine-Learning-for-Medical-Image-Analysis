# Computer-Vision-and-Machine-Learning-for-Medical-Image-Analysis


## Overview
This repository demonstrates the use of computer vision and machine learning techniques to enhance medical image analysis. The aim is to explore various approaches, from basic edge detection to advanced deep learning models for tasks like tumor segmentation, anomaly detection, and diagnostic prediction.

## Goals
- Explore edge-detection techniques in medical imaging.
- Demonstrate the role of machine learning in automating diagnostic tasks.
- Provide scripts and resources to build an understanding of computer vision concepts.
- Encourage contributions and research in this critical field.

## Initial Commit
The initial code demonstrates edge detection techniques (Canny, Sobel, Laplacian) applied to a sample image, highlighting the basic preprocessing methods for medical images.

## Requirements
Install the following Python packages:
- `opencv-python`
- `numpy`

```bash
pip install opencv-python numpy

Usage
Replace C:/Users/CASH/OneDrive/Pictures/133743798416011033.jpg with your own image path.
Run the script to visualize different edge-detection methods.
Next Steps
Add public datasets for experimentation (e.g., Chest X-ray Images (Pneumonia)).
Implement machine learning models for image classification and segmentation.
Explore the use of neural networks for advanced analysis.
Contributing
Feel free to fork this repository and contribute by:

Adding more preprocessing techniques.
Integrating machine learning models.
Enhancing documentation.

HERE IS THE CODE

---


```python
import cv2
import numpy as np

# Load and resize the image
img = cv2.imread("C:/Users/CASH/OneDrive/Pictures/133743798416011033.jpg")
img = cv2.resize(img, (400, 300))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges_canny = cv2.Canny(gray, 100, 200)
edges_sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
edges_laplacian = cv2.Laplacian(gray, cv2.CV_8U)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Canny Edges', edges_canny)
cv2.imshow('Sobel Edges', edges_sobel)
cv2.imshow('Laplacian Edges', edges_laplacian)

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()



