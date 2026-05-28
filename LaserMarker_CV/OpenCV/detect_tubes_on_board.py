import cv2
import numpy as np
import os

# ⚠️ CRITICAL REMINDER FOR FUTURE IMPLEMENTATIONS:
# OpenCV uses BGR (Blue-Green-Red) color order, NOT RGB!
# When using cv2.imread(), image[:,:,0]=Blue, image[:,:,1]=Green, image[:,:,2]=Red
# When using cv2.split(), returns (Blue, Green, Red) - NOT (Red, Green, Blue)
# This is a VERY common source of bugs in color-based detection!
# Always test with print statements: print(f"R value: {r_channel.mean()}, B value: {b_channel.mean()}")

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
    Detect the tube holding board - a filled rectangular shape with rounded corners.
    Returns the board mask and bounding box.
    """
    # Find contours - the board should be a large filled shape
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
    
    # Find the largest contour (should be the board)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Board should be reasonably large (more than 10% of image)
    image_area = image.shape[0] * image.shape[1]
    if area < image_area * 0.1:
        return None, None, None
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Create mask for the board
    board_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(board_mask, [largest_contour], 0, 255, -1)
    
    return board_mask, (x, y, w, h), largest_contour

def detect_tubes_on_board(image, board_mask, board_bbox):
    """
    Detect tubes within the board region using inner circles and color analysis.
    """
    if board_mask is None:
        return []
    
    # Extract board ROI
    x, y, w, h = board_bbox
    board_roi = image[y:y+h, x:x+w]
    board_roi_mask = board_mask[y:y+h, x:x+w]
    
    # Extract channels from ROI
    # ⚠️ IMPORTANT: OpenCV uses BGR order, NOT RGB!
    # cv2.split() returns (Blue, Green, Red) - not (Red, Green, Blue)
    b_channel, g_channel, r_channel = cv2.split(board_roi)
    
    # Apply median blur to reduce noise
    blurred_b = cv2.medianBlur(b_channel, 7)
    
    # Detect circles (inner circles of tube caps)
    circles = cv2.HoughCircles(
        blurred_b,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=50,
        param2=20,
        minRadius=20,
        maxRadius=60
    )
    
    detected_tubes = []
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (cX, cY, r) in circles:
            # Check if circle is within board bounds
            if cY - 5 < 0 or cY + 5 >= board_roi.shape[0]:
                continue
            if cX - 5 < 0 or cX + 5 >= board_roi.shape[1]:
                continue
            
            # Check brightness at center (should be relatively bright for tube caps)
            y_start = max(0, cY - 5)
            y_end = min(board_roi.shape[0], cY + 5)
            x_start = max(0, cX - 5)
            x_end = min(board_roi.shape[1], cX + 5)
            
            center_brightness = np.mean(blurred_b[y_start:y_end, x_start:x_end])
            
            if center_brightness < 100:  # Skip dark areas
                continue
            
            # Color spectrum test: R - B difference
            r_center = int(np.mean(r_channel[y_start:y_end, x_start:x_end]))
            b_center = int(np.mean(blurred_b[y_start:y_end, x_start:x_end]))
            color_diff = r_center - b_center
            
            # Real tubes should have positive R-B difference
            if color_diff < 10:
                continue
            
            # Calculate angle via local centroid offset
            roi_size = int(r * 2.5)
            ry1 = max(0, cY - roi_size)
            ry2 = min(board_roi.shape[0], cY + roi_size)
            rx1 = max(0, cX - roi_size)
            rx2 = min(board_roi.shape[1], cX + roi_size)
            
            roi_b = blurred_b[ry1:ry2, rx1:rx2]
            _, roi_thresh = cv2.threshold(roi_b, 130, 255, cv2.THRESH_BINARY)
            
            M = cv2.moments(roi_thresh)
            angle = 0.0
            if M["m00"] != 0:
                local_cX = int(M["m10"] / M["m00"])
                local_cY = int(M["m01"] / M["m00"])
                roi_cX = roi_size
                roi_cY = roi_size
                angle = np.arctan2(local_cY - roi_cY, local_cX - roi_cX) * 180 / np.pi
            
            # Convert back to original image coordinates
            orig_x = x + cX
            orig_y = y + cY
            
            detected_tubes.append({
                'x': orig_x,
                'y': orig_y,
                'radius': r,
                'brightness': center_brightness,
                'color_diff': color_diff,
                'angle': angle,
                'roi_x': cX,
                'roi_y': cY
            })
    
    return detected_tubes

def main():
    image_path = "tubes_on_board_shape.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} not found!")
        return
    
    # Load and resize image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ Could not load image from {image_path}")
        return
    
    image = resize_image_by_height(original_image, target_height=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("\n" + "=" * 70)
    print("🎯 TUBE DETECTION ON BOARD")
    print("=" * 70)
    
    # Step 1: Detect board
    print("\n📍 Step 1: Detecting tube holding board...")
    board_mask, board_bbox, board_contour = detect_board(image, gray)
    
    if board_mask is None:
        print("❌ Could not detect board!")
        return
    
    x, y, w, h = board_bbox
    print(f"✅ Board found at ({x}, {y}) with size {w}x{h}")
    
    # Step 2: Detect tubes on board
    print("\n📍 Step 2: Detecting tubes on the board...")
    tubes = detect_tubes_on_board(image, board_mask, board_bbox)
    
    print(f"✅ Found {len(tubes)} tubes\n")
    
    if tubes:
        print("-" * 70)
        for i, tube in enumerate(tubes, 1):
            print(f"Tube {i}:")
            print(f"  Coordinates: ({tube['x']}, {tube['y']})")
            print(f"  Radius: {tube['radius']} pixels")
            print(f"  Brightness: {tube['brightness']:.1f}")
            print(f"  Color Diff (R-B): {tube['color_diff']:.1f}")
            print(f"  Rotation Angle: {tube['angle']:.1f}°")
        print("-" * 70)
    
    # Visualization
    display = image.copy()
    
    # Draw board outline
    if board_contour is not None:
        cv2.drawContours(display, [board_contour], 0, (0, 255, 255), 3)
    
    # Draw detected tubes
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
    ]
    
    for i, tube in enumerate(tubes):
        color = colors[i % len(colors)]
        x, y, r = tube['x'], tube['y'], tube['radius']
        
        # Draw circle
        cv2.circle(display, (x, y), r, color, 2)
        # Draw center
        cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
        # Draw number
        cv2.putText(display, str(i+1), (x-10, y-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # Draw angle indicator
        angle_rad = tube['angle'] * np.pi / 180
        end_x = int(x + r * np.cos(angle_rad))
        end_y = int(y + r * np.sin(angle_rad))
        cv2.line(display, (x, y), (end_x, end_y), color, 2)
    
    # Add title
    cv2.putText(display, f"Tubes Detected: {len(tubes)}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    # Display
    cv2.imshow("Tube Detection on Board", display)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = "detected_tubes_on_board.jpg"
    cv2.imwrite(output_path, display)
    print(f"✅ Result saved to {output_path}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()