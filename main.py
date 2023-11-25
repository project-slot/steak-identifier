from ultralytics import YOLO
import cv2

model = YOLO("./runs/detect/train/weights/best.pt")
img = cv2.imread("./assets/steak2.jpg")

results = model(img)  # predict on an image
annotated_img = results[0].plot()

cv2.imshow("window", annotated_img)
if cv2.waitKey(0) == ord("q"):
    cv2.destroyAllWindows()
