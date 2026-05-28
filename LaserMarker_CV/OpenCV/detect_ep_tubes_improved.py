import cv2
import numpy as np
import os

def resize_image_by_height(image, target_height=800):
    """Resize image to target height while maintaining aspect ratio."""
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / height
    new_width = int(width * scale)
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    return resized

def nothing(x):
    """Dummy callback function for trackbars."""
    pass

def detect_tubes_improved(image, gray, threshold_val, min_area, max_area, 
                         min_solidity, min_circularity, max_aspect_ratio):
    """
    Improved tube detection using contour analysis.
    Filters for actual tubes with good characteristics.
    """
    # Apply threshold to isolate bright regions (tubes)
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find tubes
    detected_tubes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Calculate solidity (how filled is the shape)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        
        # Filter by solidity - tubes should be solid
        if solidity < min_solidity:
            continue
        
        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Filter by circularity - avoid very irregular shapes
        if circularity < min_circularity:
            continue
        
        # Calculate aspect ratio (bounding box)
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        aspect_ratio = max(w, h) / min(w, h)
        
        # Filter by aspect ratio
        if aspect_ratio > max_aspect_ratio:
            continue
        
        # Get center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # Average of width and height as approximate size
        approx_radius = (w + h) // 4
        
        detected_tubes.append({
            'contour': contour,
            'center': (cx, cy),
            'radius': approx_radius,
            'area': area,
            'solidity': solidity,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'bbox': (x, y, w, h)
        })
    
    return detected_tubes, binary

def main():
    # Load the image
    image_path = "tubes.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found in the current directory.")
        return
    
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Resize image to reasonable size (height ~800)
    image = resize_image_by_height(original_image, target_height=800)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create window and trackbars
    window_name = "EP Tube Detection - Improved"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create trackbars for detection parameters
    cv2.createTrackbar("Threshold", window_name, 150, 255, nothing)
    cv2.createTrackbar("Min Area", window_name, 1000, 10000, nothing)
    cv2.createTrackbar("Max Area", window_name, 5000, 20000, nothing)
    cv2.createTrackbar("Min Solidity", window_name, 65, 100, nothing)  # 0-100 as percent
    cv2.createTrackbar("Min Circularity", window_name, 25, 100, nothing)  # 0-100 as percent
    cv2.createTrackbar("Max Aspect", window_name, 30, 50, nothing)  # 1.0-5.0
    
    print("=" * 70)
    print("EP TUBE DETECTION - IMPROVED CONTOUR-BASED")
    print("=" * 70)
    print("\nBased on multi-tube analysis:")
    print("  - Threshold: brightness cutoff for white tubes")
    print("  - Min Area: filter out noise (small fragments)")
    print("  - Max Area: filter out background artifacts")
    print("  - Min Solidity: > 0.65 for solid tubes")
    print("  - Min Circularity: > 0.25 to avoid very irregular shapes")
    print("  - Max Aspect: <= 3.0 to avoid elongated artifacts")
    print("\nPress 'q' to quit and print final parameters.")
    print("-" * 70)
    
    # Main loop for real-time detection
    while True:
        # Read trackbar values
        threshold = cv2.getTrackbarPos("Threshold", window_name)
        min_area = cv2.getTrackbarPos("Min Area", window_name)
        max_area = cv2.getTrackbarPos("Max Area", window_name)
        min_solidity = cv2.getTrackbarPos("Min Solidity", window_name) / 100.0
        min_circularity = cv2.getTrackbarPos("Min Circularity", window_name) / 100.0
        max_aspect = 1.0 + cv2.getTrackbarPos("Max Aspect", window_name) / 10.0
        
        # Ensure valid ranges
        min_area = max(1, min_area)
        max_area = max(min_area + 1, max_area)
        min_solidity = max(0.0, min(1.0, min_solidity))
        min_circularity = max(0.0, min(1.0, min_circularity))
        max_aspect = max(1.0, max_aspect)
        
        # Detect tubes
        tubes, binary = detect_tubes_improved(
            image, gray, threshold, min_area, max_area,
            min_solidity, min_circularity, max_aspect
        )
        
        # Create a copy for drawing
        display_image = image.copy()
        
        # Draw detected tubes
        if tubes:
            for tube in tubes:
                # Draw contour in green
                cv2.drawContours(display_image, [tube['contour']], 0, (0, 255, 0), 2)
                # Draw center point in red
                cv2.circle(display_image, tube['center'], 4, (0, 0, 255), -1)
            
            # Add text showing number of tubes detected
            text = f"Tubes detected: {len(tubes)}"
            cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
        else:
            # Add text showing no tubes detected
            text = "Tubes detected: 0"
            cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
        
        # Add parameter info on display
        info_text = (f"Thresh:{threshold} Area:{min_area}-{max_area} "
                    f"Solid:{min_solidity:.2f} Circ:{min_circularity:.2f} Asp:{max_aspect:.1f}")
        cv2.putText(display_image, info_text, (10, display_image.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the image
        cv2.imshow(window_name, display_image)
        
        # Check for key press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("\n" + "=" * 70)
            print("FINAL PARAMETERS")
            print("=" * 70)
            print(f"Threshold:        {threshold}")
            print(f"Min Area:         {min_area} pixels")
            print(f"Max Area:         {max_area} pixels")
            print(f"Min Solidity:     {min_solidity:.3f} (0.0-1.0)")
            print(f"Min Circularity:  {min_circularity:.3f} (0.0-1.0)")
            print(f"Max Aspect Ratio: {max_aspect:.2f}")
            
            if tubes:
                print(f"\nTubes detected: {len(tubes)}")
                print("\nDetected tube statistics:")
                areas = [t['area'] for t in tubes]
                solids = [t['solidity'] for t in tubes]
                circs = [t['circularity'] for t in tubes]
                aspects = [t['aspect_ratio'] for t in tubes]
                print(f"  Area range: {min(areas):.0f} - {max(areas):.0f} pixels")
                print(f"  Solidity range: {min(solids):.3f} - {max(solids):.3f}")
                print(f"  Circularity range: {min(circs):.3f} - {max(circs):.3f}")
                print(f"  Aspect ratio range: {min(aspects):.2f} - {max(aspects):.2f}")
            else:
                print("\nNo tubes detected with final parameters.")
            
            print("=" * 70)
            break
    
    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
