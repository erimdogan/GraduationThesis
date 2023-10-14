from ultralytics import YOLO
import cv2

model = YOLO('C:/GT/runs/detect/train/weights/best.pt')

img = cv2.imread("C:/GT/istockphoto-1273932631-612x612.jpg")

results = model(img)
results_plot = results[0].plot()

cv2.imshow("Results", results_plot)
cv2.waitKey(0)