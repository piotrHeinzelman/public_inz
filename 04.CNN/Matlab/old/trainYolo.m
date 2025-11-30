% https://www.mathworks.com/help/vision/ref/trainyolov4objectdetector.html#d126e346295
% Subnetworks to freeze during training, specified as one of these values:

%    "none" — Do not freeze subnetworks

%    "backbone" — Freeze the feature extraction subnetwork

%    "backboneAndNeck" — Freeze both the feature extraction and the path aggregation subnetworks


path="/home/john/inz_DATA/GROUPS/jpg/";


load("/home/john/inz/MixedProj/03.R-CNN/Matlab/trainingData.mat");

%trainingData = Data.Data.vehicleTrainingData;

%save('trainingData', 'trainingData');

%trainingData1.imageFilename(1) = "/home/john/inz_DATA/Groups_small/group_1074.jpga";
%trainingData1.vehicle{1} = [6.16259501406154,12.0616543130181,34.2760466778522,36.2677829970486];


%dataDir = fullfile(toolboxdir("/home/john/inz_DATA/GROUPS/jpg/"),"visiondata");
%trainingData.imageFilename = fullfile(dataDir,trainingData.imageFilename);
 
detector = yolov4ObjectDetector("tiny-yolov4-coco")
detector.Network

imds = imageDatastore(trainingData.imageFilename);
blds = boxLabelDatastore(trainingData(:,2:end));
ds = combine(imds,blds);

inputSize = [224 224 3];
trainingDataForEstimation = transform(ds,@(data)preprocessData(data,inputSize));

numAnchors = 6;
[anchors,meanIoU] = estimateAnchorBoxes(trainingData,numAnchors);
area = anchors(:,1).*anchors(:,2);
[~,idx] = sort(area,"descend");
anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:);anchors(4:6,:)};


classes = ["vehicle"];
detector = yolov4ObjectDetector("tiny-yolov4-coco",classes,anchorBoxes,InputSize=inputSize);
