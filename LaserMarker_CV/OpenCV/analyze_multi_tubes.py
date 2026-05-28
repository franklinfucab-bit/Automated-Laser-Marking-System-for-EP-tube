import cv2
import numpy as np

# Load the reference multi-tube image
img = cv2.imread('Multi_white_tube.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Image shape: {img.shape}")
print(f"Grayscale range: {gray.min()} - {gray.max()}")
print(f"Mean brightness: {gray.mean():.1f}")

# Extract bright regions
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Apply morphological operations to clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\n" + "=" * 70)
print(f"MULTI-TUBE ANALYSIS")
print("=" * 70)
print(f"Total contours found: {len(contours)}\n")

# Analyze each contour
tube_properties = []
for idx, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    
    # Skip very small contours
    if area < 100:
        continue
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        continue
    
    # Circularity
    circularity = 4 * np.pi * area / (perimeter ** 2)
    
    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        continue
    solidity = area / hull_area
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    
    # Center
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2
    
    tube_properties.append({
        'idx': idx,
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'solidity': solidity,
        'aspect_ratio': aspect_ratio,
        'center': (cx, cy),
        'bbox': (x, y, w, h),
        'contour': contour
    })

# Sort by area for display
tube_properties.sort(key=lambda t: t['area'], reverse=True)

print(f"Tubes detected (area > 100): {len(tube_properties)}\n")

# Print detailed statistics
for i, tube in enumerate(tube_properties):
    print(f"Tube {i+1}:")
    print(f"  Area: {tube['area']:.0f} pixels")
    print(f"  Perimeter: {tube['perimeter']:.0f} pixels")
    print(f"  Circularity: {tube['circularity']:.4f}")
    print(f"  Solidity: {tube['solidity']:.4f}")
    print(f"  Aspect Ratio: {tube['aspect_ratio']:.4f}")
    print(f"  Center: {tube['center']}")
    print(f"  Bounding Box: {tube['bbox']}")
    print()

# Summary statistics
if tube_properties:
    print("-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    
    areas = [t['area'] for t in tube_properties]
    circularities = [t['circularity'] for t in tube_properties]
    solidities = [t['solidity'] for t in tube_properties]
    aspect_ratios = [t['aspect_ratio'] for t in tube_properties]
    
    print(f"\nAREA:")
    print(f"  Range: {min(areas):.0f} - {max(areas):.0f} pixels")
    print(f"  Average: {np.mean(areas):.0f}")
    print(f"  Std Dev: {np.std(areas):.0f}")
    
    print(f"\nCIRCULARITY (1.0=circle, lower=irregular):")
    print(f"  Range: {min(circularities):.4f} - {max(circularities):.4f}")
    print(f"  Average: {np.mean(circularities):.4f}")
    
    print(f"\nSOLIDITY (how filled, 1.0=solid):")
    print(f"  Range: {min(solidities):.4f} - {max(solidities):.4f}")
    print(f"  Average: {np.mean(solidities):.4f}")
    
    print(f"\nASPECT RATIO (width/height):")
    print(f"  Range: {min(aspect_ratios):.4f} - {max(aspect_ratios):.4f}")
    print(f"  Average: {np.mean(aspect_ratios):.4f}")
    
    print("\n" + "-" * 70)
    print("RECOMMENDED DETECTION PARAMETERS")
    print("-" * 70)
    
    min_area = int(np.mean(areas) * 0.6)
    max_area = int(np.mean(areas) * 1.5)
    min_circ = max(0.3, np.min(circularities) - 0.1)
    max_circ = min(1.0, np.max(circularities) + 0.1)
    min_solid = max(0.6, np.min(solidities) - 0.1)
    
    print(f"\nThreshold: 150 (for bright tubes)")
    print(f"Min Area: {min_area} pixels")
    print(f"Max Area: {max_area} pixels")
    print(f"Min Circularity: {min_circ:.3f}")
    print(f"Max Circularity: {max_circ:.3f}")
    print(f"Min Solidity: {min_solid:.3f}")
    
    print("\n" + "=" * 70)

# Display with contours drawn
img_display = img.copy()
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
          (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]

for i, tube in enumerate(tube_properties):
    color = colors[i % len(colors)]
    cv2.drawContours(img_display, [tube['contour']], 0, color, 2)
    cv2.circle(img_display, tube['center'], 5, color, -1)
    cv2.putText(img_display, str(i+1), (tube['center'][0]-10, tube['center'][1]-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.imshow('Original Multi-Tube Image', img)
cv2.imshow('Binary Threshold', binary)
cv2.imshow('Detected Tubes (Contours & Centers)', img_display)

print("Close the windows to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
