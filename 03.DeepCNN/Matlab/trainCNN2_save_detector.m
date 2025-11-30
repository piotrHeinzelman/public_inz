% https://www.mathworks.com/help/vision/ref/trainyolov4objectdetector.html#d126e346295
% Subnetworks to freeze during training, specified as one of these values:
%    "none" — Do not freeze subnetworks
%    "backbone" — Freeze the feature extraction subnetwork
%    "backboneAndNeck" — Freeze both the feature extraction and the path aggregation subnetworks

% Number of filters in the output convolutional layer must be 18 for 3 anchor boxes and 1 classes.
% Number of filters in the output convolutional layer must be 21 for 3 anchor boxes and 2 classes.


path="../../imagesAndRegions/sas/";
load("trainData.mat");
load("netCNN_SAS.mat");
%load("dlnet.mat");
%net=dlnet;
%trainingData=toolsTrainingData;

%disp(net.Layers);

dataDir = fullfile(path );
trainData.imageFilename = fullfile(path,trainData.imageFilename);


imds = imageDatastore(trainData.imageFilename);
blds = boxLabelDatastore(trainData(:,2:end));
ds = combine(imds,blds);

inputSize = [240 240 3];
trainingDataForEstimation = transform(ds,@(data)preprocessData(data,inputSize));

%numAnchors = 6;
%[anchors,meanIoU] = estimateAnchorBoxes( blds, numAnchors);
%area = anchors(:,1).*anchors(:,2);
%[~,idx] = sort(area,"descend");
%anchors = anchors(idx,:);
%anchorBoxes = {anchors(1:3,:);anchors(4:6,:)};
%aboxes=anchorBoxes;

classes = ["sas" ];
anchorBoxes = {[122,177;223,84;80,94] };


%detector = yolov4ObjectDetector(net,classes,aboxes,'DetectionNetworkSource',layer );







%netUpdated = removeLayers( net , ['softmax'] );
%net2 = removeLayers( netUpdated , ['fc'] );

%imageSize = net.Layers(1).InputSize;
%layerName = net.Layers(1).Name;
%newInputLayer = imageInputLayer(imageSize,Normalization="none",Name=layerName);
%Replace the image input layer in the base network with the new input layer.
%net = replaceLayer(net,layerName,newInputLayer);
%Specify the names of the feature extraction layers in the base network to use as the detection heads.
featureExtractionLayers = ["Lay" ]; % ["activation_22_relu","activation_40_relu"];


net=net.removeLayers("softmax");
net=net.removeLayers("flatten");
net=net.removeLayers("relu_7");
net=net.removeLayers("batchnorm_9");
net=net.removeLayers("maxpool_7");
net=net.removeLayers("relu_6");
net=net.removeLayers("batchnorm_8");
net=net.removeLayers("conv_8");
net=net.removeLayers("maxpool_6");
net=net.removeLayers("relu_5");
net=net.removeLayers("batchnorm_7");
net=net.removeLayers("conv_7");
net=net.removeLayers("maxpool_5");
net=net.removeLayers("conv_9");
net = initialize(net);

detector = yolov4ObjectDetector(net,classes,anchorBoxes );
save('detector','detector');

