from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("D:\\Yolov\\ultralytics-8.2.0\\bsp\\runs\\detect\\train199\\weights\\best.pt")

# Path to the single image
image_path = "D:\\Yolov\\ultralytics-8.2.0\\bsp\\testing_result\\frame_20250306_211300_000000_aug11.png"  # Replace with the path to your image

# Load the image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

# Run YOLO inference on the image with a lower confidence threshold
results = model(image_path, conf=0.1)  # Lower the confidence threshold to 0.1 to detect more defects

# Process the result
for result in results:
    boxes = result.boxes  # Bounding box outputs

    # Draw bounding boxes and labels on the image
    for box in boxes:
        # Extract coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        class_id = 0  # Force the label to "0" as per your request

        # Draw the bounding box (red color, thinner line)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Thickness set to 1

        # Add the class label "0" at the top-left of the bounding box (smaller font)
        label = str(class_id)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # Smaller font size (0.4), thickness 1

# Display the processed image
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()

# Optionally, save the processed image to a file
# cv2.imwrite("processed_image.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))