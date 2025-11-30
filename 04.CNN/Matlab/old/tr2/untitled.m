% openExample('vision/TrainYOLOV4NetworkForVehicleDetectionExample')
openExample('vision/TrainYOLOV4NetworkForVehicleDetectionExample')


detector = yolov4ObjectDetector("tiny-yolov4-coco")

data = load("vehicleTrainingData.mat");
trainingData = data.vehicleTrainingData;