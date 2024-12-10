from ultralytics import YOLO
import cv2
import time
import numpy as np
import os
import json

# Load a pretrained Limonates model
model = YOLO("models/Limonates.pt")

# Predict on an image
image_path = "images/bzoj.jpeg"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

start_time = time.time()
results = model(image_path, conf=0.25)  # Perform prediction with confidence threshold
inference_time = time.time() - start_time

# Prepare for visualization
annotated_image = results[0].plot()  # Annotated image as numpy array
class_colors = {}  # To store unique colors for each class

# Gather detections
detection_data = []
detection_count = 0

for box in results[0].boxes:
    cls = int(box.cls.item())  # Class index
    conf = box.conf.item()  # Confidence score
    bbox = box.xyxy.tolist()[0]  # Bounding box coordinates [x_min, y_min, x_max, y_max]
    class_name = model.names[cls]

    # Assign a unique color for each class
    if cls not in class_colors:
        class_colors[cls] = tuple(np.random.randint(0, 255, size=3).tolist())

    # Add detection info to the list
    detection_data.append({
        "class_id": cls,
        "class_name": class_name,
        "confidence": conf,
        "bbox": bbox
    })

    detection_count += 1

# Add a colored top bar with general information
top_bar_height = 60
top_bar_color = (50, 50, 50)  # Dark gray
cv2.rectangle(annotated_image, (0, 0), (annotated_image.shape[1], top_bar_height), top_bar_color, -1)

# Add general info text
cv2.putText(annotated_image, f"Limonates Detected | Inf.: {inference_time:.2f}s | Total Limons: {detection_count}",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Add a colored bottom bar for detection metadata
bottom_bar_height = 80
bottom_bar_color = (50, 50, 50)  # Dark gray
cv2.rectangle(annotated_image, (0, annotated_image.shape[0] - bottom_bar_height),
              (annotated_image.shape[1], annotated_image.shape[0]), bottom_bar_color, -1)

# Add metadata for each detection alternately on the left and right
text_y = annotated_image.shape[0] - bottom_bar_height + 25
for i, detection in enumerate(detection_data):
    cls_name = detection["class_name"]
    conf = detection["confidence"]
    bbox = detection["bbox"]
    text = f"{cls_name} | Cf: {conf:.2f} | BBox: {bbox}"

    # Alternate between left and right for better visibility
    if i % 2 == 0:  # Left side
        text_x = 10
    else:  # Right side
        text_x = annotated_image.shape[1] // 2

    cv2.putText(annotated_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    text_y += 25

# Save annotated image
annotated_image_path = os.path.join(output_dir, "annotated_image.jpg")
cv2.imwrite(annotated_image_path, annotated_image)

# Save detection data to JSON
json_path = os.path.join(output_dir, "detection_results.json")
with open(json_path, "w") as json_file:
    json.dump(detection_data, json_file, indent=4)

# Optional: Display the annotated image
cv2.imshow("YOLO Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Annotated image saved to: {annotated_image_path}")
print(f"Detection data saved to: {json_path}")
