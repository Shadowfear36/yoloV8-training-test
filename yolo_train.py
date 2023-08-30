from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="dataset.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("./test.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format



#How to convert the coordinates of the bounding box to YOLO format
https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h
