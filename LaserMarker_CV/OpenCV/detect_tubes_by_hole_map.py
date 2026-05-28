
import cv2
import numpy as np
import os
from scipy import ndimage

# ⚠️ CRITICAL REMINDERS:
# 1. img.shape is TUPLE (height, width, channels) - use img.shape[0] for height, img.shape[1] for width
# 2. OpenCV uses BGR order - cv2.split() returns (Blue, Green, Red)

def resize_image_by_height(image, target_height=800):
    """Resize image to target height while maintaining aspect ratio."""
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / height
    new_width = int(width * scale)
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    return resized

def detect_board(image, gray):
    """
    STEP 1: Detect the tube holding board - a filled rectangular shape.
    Returns the board mask, bounding box, and contour.
    """
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    image_area = image.shape[0] * image.shape[1]
    if area < image_area * 0.1:
        return None, None, None, None
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    board_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(board_mask, [largest_contour], 0, 255, -1)
    
    return board_mask, (x, y, w, h), largest_contour, gray[y:y+h, x:x+w]

def detect_holes(board_roi_gray, board_bbox):
    """
    STEP 2: Detect dark orange holes by analyzing brightness.
    Maps hole positions in a grid pattern.
    """
    x_offset, y_offset, _, _ = board_bbox
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(board_roi_gray, cv2.MORPH_CLOSE, kernel)
    
    # Invert threshold to get dark holes
    _, holes_binary = cv2.threshold(closed, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Label connected components
    labeled_array, num_features = ndimage.label(holes_binary)
    
    hole_list = []
    
    # Find centroid of each hole
    for i in range(1, num_features + 1):
        component_mask = (labeled_array == i)
        area = np.sum(component_mask)
        
        # Filter by area
        if area < 50 or area > 3000:
            continue
        
        coords = np.where(component_mask)
        if len(coords[0]) == 0:
            continue
        
        cy = int(np.mean(coords[0]))
        cx = int(np.mean(coords[1]))
        
        # Convert to original coordinates
        orig_x = x_offset + cx
        orig_y = y_offset + cy
        
        hole_list.append({
            'x': orig_x,
            'y': orig_y,
            'area': area,
            'roi_x': cx,
            'roi_y': cy,
            'occupied': False
        })
    
    return hole_list, holes_binary

def detect_tubes_in_holes(board_roi, board_roi_gray, hole_list, board_bbox):
    """
    STEP 3: Detect which holes are covered by tubes.
    Direct hole-based detection: check each hole for tube presence
    by analyzing brightness and color around the hole position.
    """
    x_offset, y_offset, _, _ = board_bbox
    b_channel, g_channel, r_channel = cv2.split(board_roi)
    blurred_b = cv2.medianBlur(b_channel, 5)
    blurred_r = cv2.medianBlur(r_channel, 5)
    blurred_g = cv2.medianBlur(g_channel, 5)
    
    occupied_tubes = []
    
    # For each detected hole, check if a tube is present
    for hole_idx, hole in enumerate(hole_list):
        roi_x = hole['roi_x']
        roi_y = hole['roi_y']
        
        # Define analysis region around hole (larger area to detect tube)
        analysis_radius = 35
        y_start = max(0, roi_y - analysis_radius)
        y_end = min(board_roi.shape[0], roi_y + analysis_radius)
        x_start = max(0, roi_x - analysis_radius)
        x_end = min(board_roi.shape[1], roi_x + analysis_radius)
        
        # Get pixels in analysis region
        roi_b = blurred_b[y_start:y_end, x_start:x_end]
        roi_r = blurred_r[y_start:y_end, x_start:x_end]
        roi_g = blurred_g[y_start:y_end, x_start:x_end]
        
        # Calculate average brightness and color in region
        avg_brightness = np.mean(roi_b)
        avg_red = np.mean(roi_r)
        avg_green = np.mean(roi_g)
        
        # Check for high variance in brightness (tubes have visible edges/gradient)
        brightness_std = np.std(roi_b)
        
        # Tube detection criteria:
        # 1. Region should have higher than average brightness (tubes are lighter than board)
        # 2. Should have good red component (tubes appear reddish)
        # 3. Should have brightness variance (not uniform hole)
        
        has_tube = False
        tube_confidence = 0.0
        
        # Criterion 1: Brightness significantly higher than dark hole area
        if avg_brightness > 100:  # Tubes are much brighter than dark orange board
            has_tube = True
            tube_confidence += avg_brightness / 255.0
        
        # Criterion 2: Red channel dominance (tubes have red/pink tint)
        if avg_red > 110:
            red_dominance = (avg_red - avg_green) / 255.0
            if red_dominance > 0.05:
                has_tube = True
                tube_confidence += abs(red_dominance)
        
        # Criterion 3: Variance indicates visible structure (not uniform dark hole)
        if brightness_std > 15:
            has_tube = True
            tube_confidence += min(brightness_std / 50.0, 1.0)
        
        # Criterion 4: Check for lighter spots (tubes appear lighter)
        bright_pixels = np.sum(roi_b > 130)
        bright_ratio = bright_pixels / roi_b.size if roi_b.size > 0 else 0
        
        if bright_ratio > 0.15:  # More than 15% of area is bright
            has_tube = True
            tube_confidence += bright_ratio
        
        # Mark hole as occupied if confidence is high enough
        if has_tube and tube_confidence > 0.5:
            hole_list[hole_idx]['occupied'] = True
            
            # Calculate orientation angle from brightness gradient
            # Get larger region for gradient calculation
            gradient_radius = int(analysis_radius * 1.2)
            gy_start = max(0, roi_y - gradient_radius)
            gy_end = min(board_roi.shape[0], roi_y + gradient_radius)
            gx_start = max(0, roi_x - gradient_radius)
            gx_end = min(board_roi.shape[1], roi_x + gradient_radius)
            
            grad_region = blurred_b[gy_start:gy_end, gx_start:gx_end]
            
            # Calculate gradients
            gy, gx = np.gradient(grad_region.astype(float))
            angle = np.arctan2(np.mean(gy), np.mean(gx)) * 180 / np.pi
            
            occupied_tubes.append({
                'x': hole['x'],
                'y': hole['y'],
                'radius': 20,  # Standard tube radius for visualization
                'brightness': avg_brightness,
                'color_diff': avg_red - avg_green,
                'angle': angle,
                'hole_idx': hole_idx,
                'confidence': tube_confidence
            })
    
    return occupied_tubes, hole_list

def main():
    image_path = "tubes_on_board_shape.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Could not load image")
        return
    
    image = resize_image_by_height(original_image, target_height=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("\n" + "=" * 70)
    print("TUBE DETECTION BY HOLE MAPPING")
    print("=" * 70)
    
    # STEP 1
    print("\nSTEP 1: Detecting board contour...")
    board_mask, board_bbox, board_contour, board_roi_gray = detect_board(image, gray)
    
    if board_mask is None:
        print("Could not detect board!")
        return
    
    x, y, w, h = board_bbox
    print(f"Board found at ({x}, {y}) with size {w}x{h}")
    
    # STEP 2
    print("\nSTEP 2: Mapping dark orange holes...")
    hole_list, holes_binary = detect_holes(board_roi_gray, board_bbox)
    print(f"Found {len(hole_list)} holes")
    
    # STEP 3
    print("\nSTEP 3: Detecting tubes covering holes...")
    board_roi = image[y:y+h, x:x+w]
    tubes, updated_holes = detect_tubes_in_holes(board_roi, board_roi_gray, hole_list, board_bbox)
    
    occupied = sum(1 for h in updated_holes if h['occupied'])
    print(f"Found {len(tubes)} tubes in {occupied} holes\n")
    
    # Results
    print("-" * 70)
    print("HOLE MAP:")
    for i, hole in enumerate(updated_holes, 1):
        status = "🔴 OCCUPIED" if hole['occupied'] else "⚪ EMPTY"
        print(f"Hole {i:2d}: ({hole['x']:4d}, {hole['y']:4d}) {status}")
    
    print("\n" + "-" * 70)
    print("DETECTED TUBES:")
    for i, tube in enumerate(tubes, 1):
        angle_display = f"{tube['angle']:.1f}°" if tube['angle'] != 0 else "0.0°"
        print(f"T{i}: Hole {tube['hole_idx']+1} | Pos: ({tube['x']}, {tube['y']}) | Angle: {angle_display} | Brightness: {tube['brightness']:.1f} | Conf: {tube['confidence']:.2f}")
    print("-" * 70)
    
    # Visualization
    display = image.copy()
    if board_contour is not None:
        cv2.drawContours(display, [board_contour], 0, (0, 255, 255), 3)
    
    # Draw holes with labels
    for hole_idx, hole in enumerate(updated_holes, 1):
        color = (0, 165, 255) if hole['occupied'] else (128, 128, 128)
        cv2.circle(display, (hole['x'], hole['y']), 15, color, 2)
        
        # Add hole label
        label = f"H{hole_idx}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = hole['x'] - text_size[0] // 2
        text_y = hole['y'] + text_size[1] // 2
        cv2.putText(display, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1)
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (255, 127, 0), (127, 255, 0)]
    
    # Draw tubes with labels and orientation
    for i, tube in enumerate(tubes, 1):
        color = colors[(i - 1) % len(colors)]
        cv2.circle(display, (tube['x'], tube['y']), tube['radius'], color, 2)
        cv2.circle(display, (tube['x'], tube['y']), 4, (0, 0, 255), -1)
        
        # Draw orientation line
        angle_rad = tube['angle'] * np.pi / 180
        line_len = int(tube['radius'] * 1.3)
        end_x = int(tube['x'] + line_len * np.cos(angle_rad))
        end_y = int(tube['y'] + line_len * np.sin(angle_rad))
        cv2.line(display, (tube['x'], tube['y']), (end_x, end_y), color, 2)
        
        # Add tube label
        label = f"T{i}"
        label_x = tube['x'] + 25
        label_y = tube['y'] - 10
        cv2.putText(display, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2)
    
    # Summary at top
    cv2.putText(display, f"Tubes Found: {len(tubes)} | Holes: {len(hole_list)} | Occupied: {occupied}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    
    cv2.imshow("Hole Mapping Detection", display)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite("detected_tubes_by_hole_map.jpg", display)
    print(f"Saved to detected_tubes_by_hole_map.jpg\n" + "=" * 70)

if __name__ == "__main__":
    main()