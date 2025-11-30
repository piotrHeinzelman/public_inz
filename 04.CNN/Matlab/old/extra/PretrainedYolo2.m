% Yolo v2
% Matlab, tworzenie przetrenowanego obiektu
% wymaga : Computer Vision Toolbox™
% Computer Vision Toolbox Model for YOLO v2 Object Detection



% CNN
% Train net 
% https://www.mathworks.com/help/deeplearning/ref/trainnet.html



% Yolo Matlab v2
% https://www.mathworks.com/help/vision/ug/getting-started-with-yolo-v2.html
% https://www.mathworks.com/help/vision/ref/yolov2objectdetector.html

%
%
% custom !!!!!!!!!!
%
%
%   https://www.mathworks.com/help/vision/ug/create-yolo-v2-object-detection-network.html
%
%
% export to ONNX
% https://www.mathworks.com/help/vision/ug/export-yolo-v2-object-detector-to-onnx.html
%
% detect params : 
% https://www.mathworks.com/help/vision/ref/yolov2objectdetector.detect.html
%
% training data
% https://www.mathworks.com/help/vision/ref/objectdetectortrainingdata.html
% 
% combine datastore
% https://www.mathworks.com/help/matlab/ref/matlab.io.datastore.combine.html
% !!! train data
% https://www.mathworks.com/help/vision/ref/trainacfobjectdetector.html#d126e285996


if (false)
    name = "tiny-yolov2-coco";  % "darknet19-coco" | "tiny-yolov2-coco" 
    detector = yolov2ObjectDetector( name );
end




% Yolo Matlab v3
% https://www.mathworks.com/help/vision/ref/yolov3objectdetector.html
% autorun Example: openExample('vision/DetectObjectsUsingYOLOV3DetectorExample')

if (false)
    name = "tiny-yolov3-coco";
    detector = yolov3ObjectDetector( name );
end  

% yolo 4
% https://www.mathworks.com/help/vision/ref/yolov4objectdetector.html
if (true)
    name = "tiny-yolov4-coco";
    detector = yolov4ObjectDetector( name );
end  
 
% yoloX
if (false)
    name = "nano-coco"; % "nano-coco" "tiny-coco"  "small-coco" "medium-coco"  "large-coco"  
    detector = yoloxObjectDetector( name );
end  





% Load YOLO v8 model
% https://github.com/matlab-deep-learning/Pretrained-YOLOv8-Network-For-Object-Detection?tab=readme-ov-file#object-detection-1
if (false)
    %name = "yolov8s";
    %name = "yolov8m";
    %name = "yolov8l";
    name = "yolov8x";
    detector = yolov8ObjectDetector( name );
%    analyzeNetwork(detector.Network)
%yolov8n
%yolov8s
%yolov8m
%yolov8l
%yolov8x

%                   'yolov8n'   Use this model for speed and efficiency.
%
%                   'yolov8s'   Use this model for a balance between speed
%                               and accuracy, suitable for applications
%                               requiring real-time performance with good
%                               segmentation quality.
%
%                   'yolov8m'   Use this model for higher accuracy with
%                               moderate computational demands.
%
%                   'yolov8l'   Use this model to prioritize maximum
%                               segmentation accuracy for high-end systems,
%                               at the cost of computational intensity.
%
%                   'yolov8x'   Use this model to get most accurate
%                               segmentation but requires significant
%                               computational resources, ideal for high-end
%                               systems prioritizing segmentation
%                               performance.

end



disp(detector)




% detekcja
img = imread('some_team.jpg'); 
% img = imread('dog-5519360_1280.jpg'); 
% img = imread('dog-7956828_1280.jpg'); 
%img = imread('sherlock.jpg'); 
[bboxes,scores,labels] = detect(detector,img); 

detectedImg = insertObjectAnnotation(img,"Rectangle",bboxes,labels);
figure
imshow(detectedImg)

% show weights 2, 6, 10, 14, 18, 22, 26, 32
w = detector.Network.Layers(2).Weights;
w = rescale(w);
figure
montage(w)

u = detector.Network.Layers(6).Weights(:,:,1:3,:);
u = rescale(u);
figure
montage(u)

u = detector.Network.Layers(14).Weights(:,:,1:3,:);
u = rescale(u);
figure
montage(u)

u = detector.Network.Layers(26).Weights(1,1,:,:);
%u = rescale(u);
figure
montage(u)

% Yolo Matlab v4
% https://www.mathworks.com/help/vision/ug/getting-started-with-yolo-v4.html










% https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html?searchHighlight=trainnet+load+pretrained+data&s_tid=srchtitle_support_results_2_trainnet+load+pretrained+data
%Pretrained Object Detection Network Name Arguments	Object Detection Model	Required Support Package
%    "darknet19-coco"
%    "tiny-yolov2-coco"
%
%YOLO v2 – yolov2ObjectDetector (Computer Vision Toolbox)
%	Computer Vision Toolbox Model for YOLO v2 Object Detection
%    "darknet53-coco"
%    "tiny-yolov3-coco"
%
%YOLO v3 – yolov3ObjectDetector (Computer Vision Toolbox)
%	Computer Vision Toolbox Model for YOLO v3 Object Detection
%    "csp-darknet53-coco"
%    "tiny-yolov4-coco"
%
%YOLO v4 – yolov4ObjectDetector (Computer Vision Toolbox)
%	Computer Vision Toolbox Model for YOLO v4 Object Detection
%    "nano-coco"
%    "tiny-coco"
%    "small-coco"
%    "medium-coco"
%    "large-coco"
%
%YOLOX – yoloxObjectDetector (Computer Vision Toolbox)
%	
%Automated Visual Inspection Library for Computer Vision Toolbox
%
%    "tiny-network-coco"
%    "small-network-coco"
%    "medium-network-coco"
%    "large-network-coco"
%
%





% Hardware and Acceleration
% ExecutionEnvironment — Hardware resource for training neural network
% "auto" (default) | "cpu" | "gpu" | "multi-gpu" | "parallel-auto" | "parallel-cpu" | "parallel-gpu"

% PreprocessingEnvironment — Environment for fetching and preprocessing data
% "serial" (default) | "background" | "parallel"

% Acceleration — Performance optimization
% "auto" (default) | "none"






% https://www.mathworks.com/help/vision/ug/object-detection-using-deep-learning.html