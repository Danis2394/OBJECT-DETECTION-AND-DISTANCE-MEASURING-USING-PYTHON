from cv2 import cv2
import math
import time
from cvzone import Detector

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Initialize the YOLO model
detector = Detector(detectionModel=True)

# Define object dimensions and distances for distance estimation
object_dimensions = {
    "cell phone": {"width": 15, "distance": 46},
    "bottle": {"width": 35, "distance": 46},
    "keyboard": {"width": 30.75, "distance": 46},
    "book": {"width": 14, "distance": 46}
}

# Main loop for object detection and annotation
while True:
    # Capture a frame from the webcam
    success, img = cap.read()

    # Perform object detection using YOLO model
    detections = detector.findObjects(img)

    # Process the detected objects
    for detection in detections:
        class_name, bbox, _ = detection
        x, y, w, h = bbox
        object_width = object_dimensions.get(class_name, {}).get("width", None)
        object_distance = object_dimensions.get(class_name, {}).get("distance", None)

        if object_width is not None and object_distance is not None:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            object_focal_length = (object_width * object_distance) / w
            object_apparent_width = w
            object_estimated_distance = ((object_width * object_focal_length) / object_apparent_width)
            cv2.putText(img, f'{class_name}, distance: {int(object_estimated_distance)}cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Display annotated image
    cv2.imshow("Image", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
