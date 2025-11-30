 

%unzip("DigitsData.zip")

% imds = imageDatastore("DigitsData", ...
% imds = imageDatastore("D:\\INZ\\SAS_and_NoSAS_train_Data\\", IncludeSubfolders=true, LabelSource="foldernames");
imds = imageDatastore("D:\\INZ\\SAS_and_NoSAS_train_Data_240\\", IncludeSubfolders=true, LabelSource="foldernames");
	
	numTrainFiles = 800;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainFiles,"randomized");

inputSize = [240 240 3];
classNames = categories(imds.Labels);
numClasses = numel(classNames);

 

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(11,20)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer];
	
options = trainingOptions("sgdm", ...
    MaxEpochs=30, ...
    Verbose=false, ...
    Plots="training-progress", ...
    ExecutionEnvironment="parallel", ... % https://www.mathworks.com/help/releases/R2025a/deeplearning/ref/trainingoptions.html?searchHighlight=trainingOptions&s_tid=doc_srchtitle
    Metrics="accuracy");
	
if (true) 
    load(  'D:\\INZ\\SAS_and_NoSAS_train_Data_240\\mainNet.mat', 'net' ); 
end    

net = trainnet(imdsTrain,layers,"crossentropy",options); 
 
save( append('D:\\INZ\\SAS_and_NoSAS_train_Data_240\\','\mainNet.mat'),'net');
   

%accuracy = testnet(net,imdsTest,"accuracy")


%scoresTest = minibatchpredict(net,imdsTest);
%YTest = scores2label(scoresTest,classNames);

%confusionchart(imdsTest.Labels,YTest)

%save(filename)
%Then
%load('filename.mat')


% MLP
% https://www.mathworks.com/help/deeplearning/ref/trainnet.html#mw_56bfbbd7-51a0-449d-8db7-32c9a0070293

% ReTrain net
% https://www.mathworks.com/help/deeplearning/ug/retrain-neural-network-to-classify-new-images.html

% Freeze Network
% https://www.mathworks.com/help/deeplearning/ug/retrain-neural-network-to-classify-new-images.html

% Converter Tensorflow <-> ONXX <-PyTorch->
% https://www.mathworks.com/help/deeplearning/networks-from-external-platforms.html

% https://www.mathworks.com/help/deeplearning/ug/experiment-with-pretrained-networks.html?searchHighlight=u%C5%BCycie+wcze%C5%9Bniejszych+przetrenowanych+wag+w+matlab&s_tid=srchtitle_support_results_5_u%25C5%25BCycie+wcze%25C5%259Bniejszych+przetrenowanych+wag+w+matlab

% https://www.mathworks.com/help/deeplearning/ug/experiment-with-weight-initializers.html?searchHighlight=transfer+weight+trainnet&s_tid=srchtitle_support_results_1_transfer+weight+trainnet


% custom layer
% https://www.mathworks.com/help/deeplearning/ug/define-custom-layer-with-multiple-inputs.html?searchHighlight=load+custom+net+&s_tid=srchtitle_support_results_6_load+custom+net+
% https://www.mathworks.com/help/deeplearning/ug/train-network-using-custom-training-loop.html?searchHighlight=load+custom+net+&s_tid=srchtitle_support_results_1_load+custom+net+



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