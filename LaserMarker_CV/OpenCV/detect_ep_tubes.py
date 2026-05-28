python detect_ep_tubes_watershed.pyimport cv2
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
    
    # Apply median blur for noise reduction
    blurred = cv2.medianBlur(gray, 11)
    
    # Create a bright regions mask to isolate tubes (they are white/bright)
    # This helps filter out dark holes in the board
    bright_threshold = 100
    bright_mask = cv2.threshold(blurred, bright_threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Create window and trackbars
    window_name = "EP Tube Tuning"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create trackbars for HoughCircles parameters
    cv2.createTrackbar("minDist", window_name, 30, 150, nothing)
    cv2.createTrackbar("param1", window_name, 50, 200, nothing)
    cv2.createTrackbar("param2", window_name, 30, 100, nothing)
    cv2.createTrackbar("minRadius", window_name, 10, 100, nothing)
    cv2.createTrackbar("maxRadius", window_name, 50, 150, nothing)
    
    print("=" * 60)
    print("EP TUBE DETECTION - Parameter Tuning")
    print("=" * 60)
    print("\nTrackbars created. Adjust parameters in real-time.")
    print("Press 'q' to quit and print final parameters.")
    print("\nInitial parameter ranges:")
    print("  - minDist: 10-150")
    print("  - param1: 10-200")
    print("  - param2: 10-100")
    print("  - minRadius: 5-100")
    print("  - maxRadius: 10-150")
    print("-" * 60)
    
    # Main loop for real-time detection
    while True:
        # Read trackbar values
        minDist = cv2.getTrackbarPos("minDist", window_name)
        param1 = cv2.getTrackbarPos("param1", window_name)
        param2 = cv2.getTrackbarPos("param2", window_name)
        minRadius = cv2.getTrackbarPos("minRadius", window_name)
        maxRadius = cv2.getTrackbarPos("maxRadius", window_name)
        
        # Ensure minDist is at least 1
        minDist = max(1, minDist)
        param1 = max(1, param1)
        param2 = max(1, param2)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )
        
        # Filter circles: only keep those in bright regions (tubes, not holes)
        filtered_circles = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            valid_circles = []
            
            for i in circles[0, :]:
                x, y, r = i[0], i[1], i[2]
                # Check if circle is mostly in bright region
                circle_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(circle_mask, (x, y), r, 255, -1)
                
                # Calculate mean brightness inside the circle
                circle_pixels = blurred[circle_mask == 255]
                if len(circle_pixels) > 0:
                    mean_brightness = np.mean(circle_pixels)
                    # Only keep circles with high brightness (tubes, not dark holes)
                    if mean_brightness > 120:  # Threshold for tube brightness
                        valid_circles.append(i)
            
            if len(valid_circles) > 0:
                filtered_circles = np.array([valid_circles])
        
        # Create a copy for drawing
        display_image = image.copy()
        
        # Draw circles if detected
        if filtered_circles is not None:
            circles_uint = np.uint16(np.around(filtered_circles))
            num_circles = circles_uint.shape[2]
            
            for i in circles_uint[0, :]:
                # Draw outer circle in green
                cv2.circle(display_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw center point in red
                cv2.circle(display_image, (i[0], i[1]), 2, (0, 0, 255), -1)
            
            # Add text showing number of circles detected
            text = f"Tubes detected: {num_circles}"
            cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
        else:
            # Add text showing no circles detected
            text = "Tubes detected: 0"
            cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
        
        # Add parameter info on display
        info_text = (f"minDist:{minDist} param1:{param1} param2:{param2} "
                    f"minR:{minRadius} maxR:{maxRadius}")
        cv2.putText(display_image, info_text, (10, display_image.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the image
        cv2.imshow(window_name, display_image)
        
        # Check for key press
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("\n" + "=" * 60)
            print("FINAL PARAMETERS")
            print("=" * 60)
            print(f"minDist:   {minDist}")
            print(f"param1:    {param1}")
            print(f"param2:    {param2}")
            print(f"minRadius: {minRadius}")
            print(f"maxRadius: {maxRadius}")
            
            if filtered_circles is not None:
                print(f"\nTubes detected: {filtered_circles.shape[2]}")
            else:
                print("\nNo tubes detected with final parameters.")
            
            print("=" * 60)
            break
    
    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
