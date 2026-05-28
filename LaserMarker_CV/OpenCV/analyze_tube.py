import cv2
import numpy as np

# Load and display the reference image
img = cv2.imread('single_white_tube.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Image shape: {img.shape}")
print(f"Grayscale range: {gray.min()} - {gray.max()}")

# Display both
cv2.imshow('Original Tube', img)
cv2.imshow('Grayscale Tube', gray)

# Show histogram
print("\nAnalyzing tube characteristics...")
print(f"Mean brightness: {gray.mean():.2f}")
print(f"Std deviation: {gray.std():.2f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
