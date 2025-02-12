import cv2
import numpy as np
import os

def detect_car_color_and_count(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    color_ranges = {
        'blue': ([100, 150, 0], [140, 255, 255]),
        'red': ([0, 120, 70], [10, 255, 255])
    }
    
    detected_cars = []
    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Find contours of the cars
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                detected_cars.append((x, y, w, h, color))
                rectangle_color = (255, 0, 0) if color == 'red' else (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_color, 2)
    
    car_count = len(detected_cars)
    print(f'Total cars detected: {car_count}')
    
    # Display result
    cv2.imshow('Car Color Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Example usage
detect_car_color_and_count('traffic_image.jpg')
