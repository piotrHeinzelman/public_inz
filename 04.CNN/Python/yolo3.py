from ultralytics import YOLO
import time


# Load a COCO-pretrained YOLOv3u model
#model = YOLO("yolov3u.pt")



start=time.time()
model = YOLO("yolov3-tiny.pt")
end=time.time()
TIME_createModel=end-start


# Display model information (optional)
model.info()


start=time.time()
# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
end=time.time()
TIME_training=end-start



# Run inference with the YOLOv3u model on the 'bus.jpg' image
start=time.time()
results = model("path/to/dog.jpg")
end=time.time()
TIME_detect=end-start



print("# Create Model: " , TIME_createModel)
print("# Training Model: " , TIME_training)
print("# Detect time: " , TIME_detect)




# YOLOv3(u) 	yolov3u.pt 	Object Detection 	✅ 	✅ 	✅ 	✅
# YOLOv3-Tiny(u) 	yolov3-tinyu.pt 	Object Detection 	✅ 	✅ 	✅ 	✅
# YOLOv3u-SPP(u) 	yolov3-sppu.pt
