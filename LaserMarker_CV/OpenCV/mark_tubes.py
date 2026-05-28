import cv2
import numpy as np
import os

marked_tubes = []
image_copy = None
gray = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to mark tubes by clicking on them."""
    global marked_tubes, image_copy, gray
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add marked point
        marked_tubes.append((x, y))
        
        # Draw a circle at the click point (temporary large radius for visibility)
        cv2.circle(image_copy, (x, y), 5, (0, 255, 255), 2)  # Yellow temporarily
        cv2.imshow("Mark Tubes - Click on tube centers (Right-click to undo, 'q' to analyze)", image_copy)
        
        print(f"Marked point {len(marked_tubes)}: ({x}, {y})")
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Undo last mark
        if marked_tubes:
            marked_tubes.pop()
            image_copy = param.copy()
            # Redraw all marked points
            for i, (px, py) in enumerate(marked_tubes):
                cv2.circle(image_copy, (px, py), 5, (0, 255, 255), 2)
            cv2.imshow("Mark Tubes - Click on tube centers (Right-click to undo, 'q' to analyze)", image_copy)
            print(f"Undone! {len(marked_tubes)} points remaining")

def analyze_marked_tubes(image, gray, marked_tubes):
    """Analyze the characteristics of marked tubes."""
    if len(marked_tubes) == 0:
        print("No tubes marked!")
        return
    
    print("\n" + "=" * 70)
    print("ANALYZING MARKED TUBES")
    print("=" * 70)
    
    tube_properties = []
    
    for idx, (x, y) in enumerate(marked_tubes):
        # Extract circular regions around marks and analyze them
        min_dist_to_edge = min(x, y, image.shape[1] - x, image.shape[0] - y)
        
        # Try different radii to find tube boundaries
        radii_to_try = range(10, min(100, min_dist_to_edge - 5), 5)
        
        best_radius = None
        best_contrast = 0
        
        for r in radii_to_try:
            # Create mask for circular region
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Analyze brightness inside and outside
            inside = gray[mask == 255]
            
            if len(inside) > 0:
                mean_inside = np.mean(inside)
                
                # Check a ring just outside for contrast
                ring_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(ring_mask, (x, y), r + 10, 255, -1)
                cv2.circle(ring_mask, (x, y), r, 0, -1)
                outside = gray[ring_mask == 255]
                
                if len(outside) > 0:
                    mean_outside = np.mean(outside)
                    contrast = abs(mean_inside - mean_outside)
                    
                    if contrast > best_contrast:
                        best_contrast = contrast
                        best_radius = r
        
        # Analyze at the best radius found
        if best_radius:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), best_radius, 255, -1)
            inside_pixels = gray[mask == 255]
            
            mean_brightness = np.mean(inside_pixels)
            std_brightness = np.std(inside_pixels)
            
            tube_properties.append({
                'idx': idx + 1,
                'x': x,
                'y': y,
                'radius': best_radius,
                'mean_brightness': mean_brightness,
                'std_brightness': std_brightness,
                'contrast': best_contrast
            })
            
            print(f"\nTube {idx + 1} at ({x}, {y}):")
            print(f"  Estimated Radius: {best_radius}")
            print(f"  Mean Brightness: {mean_brightness:.2f}")
            print(f"  Brightness Std Dev: {std_brightness:.2f}")
            print(f"  Contrast with surroundings: {best_contrast:.2f}")
    
    # Summary statistics
    if tube_properties:
        print("\n" + "-" * 70)
        print("SUMMARY STATISTICS")
        print("-" * 70)
        
        radii = [p['radius'] for p in tube_properties]
        brightnesses = [p['mean_brightness'] for p in tube_properties]
        
        print(f"Number of tubes marked: {len(tube_properties)}")
        print(f"\nRadius range: {min(radii)} - {max(radii)} pixels")
        print(f"  Avg radius: {np.mean(radii):.1f}")
        print(f"\nBrightness range: {min(brightnesses):.1f} - {max(brightnesses):.1f}")
        print(f"  Avg brightness: {np.mean(brightnesses):.1f}")
        print(f"  Min brightness: {min(brightnesses):.1f}")
        
        # Recommended HoughCircles parameters
        print("\n" + "-" * 70)
        print("RECOMMENDED HOUGHCIRCLES PARAMETERS")
        print("-" * 70)
        min_r = max(1, int(min(radii) * 0.8))
        max_r = int(max(radii) * 1.2)
        min_bright = int(min(brightnesses) - 10)
        
        print(f"minRadius: {min_r}")
        print(f"maxRadius: {max_r}")
        print(f"minDist: {int(max(radii) * 2.5)} (avoid overlaps)")
        print(f"param1 (edge threshold): 50-100")
        print(f"param2 (center threshold): 20-40")
        print(f"\nBrightness filter: > {max(min_bright, 50)} for tube detection")
        print("=" * 70)

def main():
    global image_copy, gray, marked_tubes
    
    # Load the image
    image_path = "tubes.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found in the current directory.")
        return
    
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Resize image to reasonable size
    height, width = original_image.shape[:2]
    if height > 800:
        scale = 800 / height
        new_width = int(width * scale)
        original_image = cv2.resize(original_image, (new_width, 800), interpolation=cv2.INTER_AREA)
    
    image_copy = original_image.copy()
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Create window and set mouse callback
    window_name = "Mark Tubes - Click on tube centers (Right-click to undo, 'q' to analyze)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, original_image)
    
    print("=" * 70)
    print("EP TUBE MARKING TOOL")
    print("=" * 70)
    print("\nInstructions:")
    print("  - LEFT-CLICK on the center of each tube to mark it")
    print("  - RIGHT-CLICK to undo the last mark")
    print("  - Press 'q' to analyze marked tubes and exit")
    print("\nMarking tubes to learn their characteristics...")
    print("=" * 70)
    
    cv2.imshow(window_name, image_copy)
    
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Analyze the marked tubes
    analyze_marked_tubes(original_image, gray, marked_tubes)

if __name__ == "__main__":
    main()
