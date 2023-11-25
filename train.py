from ultralytics import YOLO


# Load a model
model = YOLO("./runs/detect/train7/weights/best.pt").load("yolov8s.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="data.yaml", epochs=3, imgsz=416, device='mps')  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export()  # export the model to ONNX format
