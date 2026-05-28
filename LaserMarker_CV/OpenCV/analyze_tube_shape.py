import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the reference tube image
img = cv2.imread('single_white_tube.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Image shape: {img.shape}")
print(f"Grayscale range: {gray.min()} - {gray.max()}")
print(f"Mean brightness: {gray.mean():.1f}")

# Extract only the bright central tube
# Use a higher threshold to isolate just the white tube, not background noise
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Apply morphological operations to clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours found: {len(contours)}")

# Find the largest contour (the tube itself - the white central shape)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    print(f"\n" + "=" * 60)
    print("TUBE SHAPE CHARACTERISTICS")
    print("=" * 60)
    print(f"Contour area: {area:.0f} pixels")
    print(f"Contour perimeter: {perimeter:.0f} pixels")
    
    # Circularity metric: 4*pi*area / perimeter^2 (1.0 = perfect circle)
    circularity = 4 * np.pi * area / (perimeter ** 2)
    print(f"Circularity (1.0=perfect circle): {circularity:.4f}")
    print(f"  → Interpretation: {0.1 if circularity < 0.5 else 'Elongated shape'}")
    
    # Fit ellipse if possible
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (cx, cy), (major, minor), angle = ellipse
        aspect_ratio = major / minor
        print(f"\nEllipse approximation:")
        print(f"  Center: ({cx:.0f}, {cy:.0f})")
        print(f"  Major axis (long): {major:.1f} pixels")
        print(f"  Minor axis (short): {minor:.1f} pixels")
        print(f"  Aspect ratio (major/minor): {aspect_ratio:.4f}")
        print(f"  Orientation angle: {angle:.1f}°")
    
    # Approximate polygon
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    print(f"\nPolygon approximation:")
    print(f"  Number of vertices: {len(approx)}")
    print(f"  Shape type: {'Rectangular/Square' if len(approx) <= 4 else 'Complex polygon'}")
    
    # Get bounding rect
    x, y, w, h = cv2.boundingRect(largest_contour)
    print(f"\nBounding box:")
    print(f"  Position: ({x}, {y})")
    print(f"  Width x Height: {w} x {h}")
    print(f"  Aspect ratio (W/H): {w/h:.4f}")
    
    # Solidity (how filled is the contour)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    print(f"\nSolidity (fill ratio): {solidity:.4f}")
    print(f"  → {'Solid shape' if solidity > 0.9 else 'Has indentations/concave parts'}")

# Display the contours
img_display = img.copy()
cv2.drawContours(img_display, [largest_contour], 0, (0, 255, 0), 3)
cv2.drawContours(img_display, [hull], 0, (255, 0, 0), 2)

if len(largest_contour) >= 5:
    ellipse = cv2.fitEllipse(largest_contour)
    cv2.ellipse(img_display, ellipse, (0, 0, 255), 2)

cv2.imshow('Original Single Tube', img)
cv2.imshow('Binary Threshold (Isolated Tube)', binary)
cv2.imshow('Detected Tube Shape - Green:Contour, Blue:Hull, Red:Ellipse', img_display)

print("\n" + "=" * 60)
print("Visualization:")
print("  Green circle: Actual tube contour")
print("  Blue outline: Convex hull")
print("  Red ellipse: Best-fit ellipse approximation")
print("=" * 60)
print("\nClose the windows to continue.")

cv2.waitKey(0)
cv2.destroyAllWindows()
