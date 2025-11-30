% https://www.mathworks.com/help/deeplearning/ref/dlnetwork.initialize.html
%
gpuDevice();
epoch=333;
load("trainData.mat");
load('detector');


dataDir = "../../imagesAndRegions/sas/";
trainData.imageFilename = fullfile(dataDir,trainData.imageFilename);
imds = imageDatastore(trainData.imageFilename);
blds = boxLabelDatastore(trainData(:,2:end));
ds = combine(imds,blds);

inputSize = [240 240 3];
trainingDataForEstimation = transform(ds,@(data)preprocessData(data,inputSize));

%numAnchors = 6;
%[anchors, meanIoU] = estimateAnchorBoxes( trainingDataForEstimation.UnderlyingDatastores{1,1}, numAnchors );
%area = anchors(:,1).*anchors(:,2);
%[~,idx] = sort(area,"descend");
%anchors = anchors(idx,:);
%anchorBoxes = {anchors(1:3,:);anchors(4:6,:)};

classes = ["sas"];
anchorBoxes = {[70,110;70,110;70,120] };

TS = datetime('now');
detector = yolov4ObjectDetector(detector.Network, classes,anchorBoxes,InputSize=inputSize);
T = datetime('now');
TIME_NetworkPrepare = seconds( duration( T-TS ));

options = trainingOptions("sgdm", ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=16,...
    MaxEpochs=epoch, ...
    ResetInputNormalization=true,...
    VerboseFrequency=30);

TS = datetime('now');
detector = trainYOLOv4ObjectDetector(ds,detector,options );
T = datetime('now');
TIME_detector = seconds(duration(T-TS));

I = imread("dedra_www2.jpg");

[bboxes, scores, labels] = detect(detector,I,Threshold=0.48);
detectedImg = insertObjectAnnotation(I,"Rectangle",bboxes,labels);


fprintf ('# prepare Net: %f \n'    ,TIME_NetworkPrepare );
fprintf ('# train detector: %f \n' ,TIME_detector );
fprintf ('# epoch: %d \n', epoch );
fprintf ('# images: %d \n', size(imds) );

if (false)
figure;
imshow(detectedImg);
end



