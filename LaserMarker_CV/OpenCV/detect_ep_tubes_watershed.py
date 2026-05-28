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

def detect_tubes_watershed(image, gray, threshold_val, erosion_iter, dilation_iter,
                          min_area, max_area):
    """
    Detect tubes using Watershed algorithm to separate touching tubes.
    Much better for multi-tube detection.
    """
    # Apply threshold to isolate bright regions (tubes)
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Distance Transform - creates peaks for each tube
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Normalize distance transform for visualization
    dist_norm = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX).astype('uint8')
    
    # Find local maxima (tube centers) using threshold on distance transform
    # If a pixel has distance value higher than threshold, it's a sure foreground
    _, sure_fg = cv2.threshold(dist_norm, 150, 255, cv2.THRESH_BINARY)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_ERODE, kernel, iterations=erosion_iter)
    
    # Find sure background (areas definitely not tubes)
    sure_bg = cv2.dilate(binary, kernel, iterations=dilation_iter)
    
    # Find unknown region (neither sure foreground nor sure background)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label connected components in sure foreground
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Mark the unknown region as 0
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 
                           else cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)
    
    # Extract individual tubes from markers
    detected_tubes = []
    
    # Get all unique markers (skip background/borders which are -1 and 1)
    unique_markers = np.unique(markers)
    
    for marker_id in unique_markers:
        if marker_id <= 1:  # Skip background (0) and outer border (-1)
            continue
        
        # Create mask for this marker
        mask = (markers == marker_id).astype('uint8') * 255
        
        # Find contour for this tube
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Get center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
        
        # Calculate properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        detected_tubes.append({
            'marker_id': marker_id,
            'contour': contour,
            'center': (cx, cy),
            'area': area,
            'solidity': solidity,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'bbox': (x, y, w, h)
        })
    
    return detected_tubes, dist_norm, sure_fg, markers

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
    window_name = "EP Tube Detection - Watershed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create trackbars for detection parameters
    cv2.createTrackbar("Threshold", window_name, 150, 255, nothing)
    cv2.createTrackbar("Erosion", window_name, 2, 10, nothing)
    cv2.createTrackbar("Dilation", window_name, 3, 10, nothing)
    cv2.createTrackbar("Min Area", window_name, 800, 5000, nothing)
    cv2.createTrackbar("Max Area", window_name, 5000, 20000, nothing)
    
    print("=" * 70)
    print("EP TUBE DETECTION - WATERSHED ALGORITHM")
    print("=" * 70)
    print("\nWatershed algorithm separates TOUCHING tubes!")
    print("\nParameters:")
    print("  - Threshold: brightness cutoff")
    print("  - Erosion: shrink foreground (find definite tube centers)")
    print("  - Dilation: expand background (define tube boundaries)")
    print("  - Min/Max Area: filter by size")
    print("\nPress 'q' to quit and print final parameters.")
    print("-" * 70)
    
    # Main loop for real-time detection
    while True:
        # Read trackbar values
        threshold = cv2.getTrackbarPos("Threshold", window_name)
        erosion = cv2.getTrackbarPos("Erosion", window_name)
        dilation = cv2.getTrackbarPos("Dilation", window_name)
        min_area = cv2.getTrackbarPos("Min Area", window_name)
        max_area = cv2.getTrackbarPos("Max Area", window_name)
        
        # Ensure valid ranges
        erosion = max(1, erosion)
        dilation = max(1, dilation)
        min_area = max(1, min_area)
        max_area = max(min_area + 1, max_area)
        
        # Detect tubes using watershed
        tubes, dist_norm, sure_fg, markers = detect_tubes_watershed(
            image, gray, threshold, erosion, dilation, min_area, max_area
        )
        
        # Create a copy for drawing
        display_image = image.copy()
        
        # Draw detected tubes
        if tubes:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
            
            for i, tube in enumerate(tubes):
                color = colors[i % len(colors)]
                # Draw contour
                cv2.drawContours(display_image, [tube['contour']], 0, color, 2)
                # Draw center point
                cv2.circle(display_image, tube['center'], 4, (0, 0, 255), -1)
                # Draw number
                cv2.putText(display_image, str(i+1), 
                           (tube['center'][0]-8, tube['center'][1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
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
        info_text = (f"Thresh:{threshold} Ero:{erosion} Dil:{dilation} "
                    f"Area:{min_area}-{max_area}")
        cv2.putText(display_image, info_text, (10, display_image.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the image
        cv2.imshow(window_name, display_image)
        
        # Check for key press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("\n" + "=" * 70)
            print("FINAL PARAMETERS (Watershed)")
            print("=" * 70)
            print(f"Threshold:   {threshold}")
            print(f"Erosion:     {erosion} iterations")
            print(f"Dilation:    {dilation} iterations")
            print(f"Min Area:    {min_area} pixels")
            print(f"Max Area:    {max_area} pixels")
            
            if tubes:
                print(f"\nTubes detected: {len(tubes)}")
                print("\nDetected tube statistics:")
                areas = [t['area'] for t in tubes]
                circs = [t['circularity'] for t in tubes]
                solids = [t['solidity'] for t in tubes]
                aspects = [t['aspect_ratio'] for t in tubes]
                
                print(f"  Area range: {min(areas):.0f} - {max(areas):.0f} pixels")
                print(f"  Circularity range: {min(circs):.3f} - {max(circs):.3f}")
                print(f"  Solidity range: {min(solids):.3f} - {max(solids):.3f}")
                print(f"  Aspect ratio range: {min(aspects):.2f} - {max(aspects):.2f}")
                
                for i, tube in enumerate(tubes):
                    print(f"\n  Tube {i+1}:")
                    print(f"    Area: {tube['area']:.0f}, Circularity: {tube['circularity']:.3f}")
            else:
                print("\nNo tubes detected with final parameters.")
            
            print("=" * 70)
            break
    
    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
