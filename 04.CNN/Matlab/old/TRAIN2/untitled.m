% https://www.mathworks.com/help/vision/ug/multiclass-object-detection-using-yolo-v2-deep-learning.html
% openExample('deeplearning_shared/MulticlassObjectDetectionUsingDeepLearningExample')

tempdir="D:\INZ\inz\MixedProj\04.R-CNN\Matlab\TRAIN2\TEMPDIR";
pretrainedFolder = fullfile(tempdir,"pretrainedNetwork");

if (false)
    % load pretrained detector 
    pretrainedURL = "https://www.mathworks.com/supportfiles/vision/data/yolov2IndoorObjectDetector23b.zip";
    
    pretrainedNetworkZip = fullfile(pretrainedFolder,"yolov2IndoorObjectDetector23b.zip"); 
    
    if ~exist(pretrainedNetworkZip,"file")
        mkdir(pretrainedFolder)
        disp("Downloading pretrained network (6 MB)...")
        websave(pretrainedNetworkZip,pretrainedURL)
    end
    
    unzip(pretrainedNetworkZip,pretrainedFolder)
end

trainedNetwork = fullfile(pretrainedFolder,"yolov2IndoorObjectDetector.mat");
trainedNetwork = load(trainedNetwork);
trainedDetector = trainedNetwork.detector;


%Detect Multiple Indoor Objects
%Read a test image that contains objects of the target classes, run the object detector on the image, and display an image annotated with the detection results.

I = imread("indoorTest.jpg");
[bbox,score,label] = detect(trainedDetector,I);

LabelScoreStr = compose("%s-%.2f",label,score); 
annotatedImage = insertObjectAnnotation(I,"rectangle",bbox,LabelScoreStr,LineWidth=4,FontSize=24);
figure
imshow(annotatedImage)



%Load Data for Training
%This example uses the Indoor Object Detection Dataset created by Bishwo Adhikari [1]. The data set consists of 2213 labeled images collected from indoor scenes and contains 7 classes: fire extinguisher, chair, clock, trash bin, screen, and printer.  Each image contains one or more labeled instances of these classes. Check whether the data set has already been downloaded and, if it is not, use websave to download it.

outputFolder = fullfile(tempdir,"indoorObjectDetection");
if (false)
    dsURL = "https://zenodo.org/record/2654485/files/Indoor%20Object%20Detection%20Dataset.zip?download=1"; 
     
    imagesZip = fullfile(outputFolder,"indoor.zip");
    
    if ~exist(imagesZip,"file")   
        mkdir(outputFolder)       
        disp("Downloading 401 MB Indoor Objects Dataset images...") 
        websave(imagesZip,dsURL)
        unzip(imagesZip,fullfile(outputFolder))  
    end
end

% Create an imageDatastore object to store the images from the data set.

datapath = fullfile(outputFolder,"Indoor Object Detection Dataset");
imds = imageDatastore(datapath,IncludeSubfolders=true, FileExtensions=".jpg");

% The annotationsIndoor.mat file contains annotations for each of the images in the data, as well as vectors that specify the indices of the data set images to use for the training, validation, and test sets. Load the file into the workspace, and extract annotations and the indices corresponding to the training, validation, and test sets from the data variable. The indices specify 2207 images in total, instead of 2213 images, as 6 images have no labels associated with them. Use the indices of the images that contain labels to remove these 6 images from the image and annotations datastores.
data = load("annotationsIndoor.mat");
blds = data.BBstore;
trainingIdx = data.trainingIdx;
validationIdx = data.validationIdx;
testIdx = data.testIdx;
cleanIdx = data.idxs;

% Remove the 6 images with no labels.
imds = subset(imds,cleanIdx);
blds = subset(blds,cleanIdx);


% Analyze Training Data 
% Analyze the distribution of object class labels and sizes to understand the data better. This analysis is critical because it helps you determine how to prepare the training data and how to configure an object detector for this specific data set.
% Analyze Class Distribution
% Measure the distribution of bounding box class labels in the data set by using the countEachLabel function. 
tbl = countEachLabel(blds)

% Visualize the counts by class.
bar(tbl.Label,tbl.Count)
ylabel("Frequency")


% Analyze Object Sizes and Choose Object Detector
% Read all the bounding boxes and labels within the data set, and calculate the diagonal length of the bounding box. 
data = readall(blds);
bboxes = vertcat(data{:,1});
labels = vertcat(data{:,2});
diagonalLength = hypot(bboxes(:,3),bboxes(:,4));

% Group the object lengths by class.
%G = findgroups(labels);
%groupedDiagonalLength = splitapply(@(x){x},diagonalLength,G);
groupedDiagonalLength = splitapply(@(x){x},diagonalLength,"SAS");

%Visualize the distribution of object lengths for each class. 
figure
classes = "SAS"; %tbl.Label;
numClasses = numel(classes);
for i = 1:numClasses
    len = groupedDiagonalLength{i};
    x = repelem(i,numel(len),1);
    plot(x,len,"o")
    hold on
end
hold off
ylabel("Object extent (pixels)")

xticks(1:numClasses)
xticklabels(classes)

% exit();
% *****************

pretrainedDetector = yolov2ObjectDetector("tiny-yolov2-coco");

% Determine the input size of the pretrained Tiny YOLO v2 network.
pretrainedDetector.Network.Layers(1).InputSize

inputSize = [720 720 3];

% Combine the image and bounding box datastores.
blds = bboxes;
ds = combine(imds,blds);

%Use transform to apply a preprocessing function that resizes images and their corresponding bounding boxes. The function also sanitizes the bounding boxes to convert them to a valid shape. 
preprocessedData = transform(ds,@(data)resizeImageAndLabel(data,inputSize));

% Display one of the preprocessed images and its bounding box labels to verify that the objects in the resized images still have visible features.  
data = preview(preprocessedData);
I = data{1};
bbox = data{2};
label = data{3};
imshow(I)
showShape("rectangle",bbox,Label=label)

%For this data set, specify the "leaky_relu_5" layer of the Tiny YOLO v2 network, which outputs feature maps downsampled by 16x. This amount of downsampling is a good trade-off between spatial resolution and the strength of the extracted features, as features extracted further down the network encode stronger image features at the cost of spatial resolution. 
featureLayer = "leaky_relu_5";

numAnchors = 5;
aboxes = estimateAnchorBoxes(preprocessedData,numAnchors);

pretrainedNet = pretrainedDetector.Network;
classes = {'exit','fireextinguisher','chair','clock','trashbin','screen','printer'};

detector = yolov2ObjectDetector(pretrainedNet,classes,aboxes, ...
    DetectionNetworkSource=featureLayer,InputSize= inputSize);

rng(0);
preprocessedData = shuffle(preprocessedData);

% Split the data set into training, test, and validation subsets using the subset function.
dsTrain = subset(preprocessedData,trainingIdx);
dsVal = subset(preprocessedData,validationIdx);
dsTest = subset(preprocessedData,testIdx);

augmentedTrainingData = transform(dsTrain,@augmentData);

% Display one of the training images and box labels.
data = read(augmentedTrainingData);
I = data{1};
bbox = data{2};
label = data{3};
imshow(I)
showShape("rectangle",bbox,Label=label)

%
% Train YOLOv2 Object Detector
%

% Specify the network training options using the trainingOptions function. 
opts = trainingOptions("rmsprop", ...
        InitialLearnRate=0.001, ...
        MiniBatchSize=8, ...
        MaxEpochs=10, ...
        LearnRateSchedule="piecewise", ...
        LearnRateDropPeriod=5, ...
        VerboseFrequency=30, ...
        L2Regularization=0.001, ...
        ValidationData=dsVal, ...
        ValidationFrequency=50, ...
        OutputNetwork="best-validation-loss");


% These training options have been selected using Experiment Manager. For more information on using Experiment Manager for hyperparameter tuning, see Train Object Detectors in Experiment Manager.
% To use the trainYOLOv2ObjectDetector function to train a YOLO v2 object detector, set doTraining is set to true. 

doTraining = false;
if doTraining
    [detector,info] = trainYOLOv2ObjectDetector(augmentedTrainingData,detector,opts);
else
    detector = trainedDetector;
end


% Evaluate Object Detector
% Evaluate the trained object detector on test images to measure the detector performance. The Computer Vision Toolbox™ provides an object detector evaluation function (evaluateObjectDetection) to measure common metrics such as average precision and precision recall, with an option to specify the overlap, or intersection-over-union (IoU), thresholds at which to compute the metrics.
% Run the detector on the test data set using the detect object function. To evaluate the detector precision across the full range of recall values, set the detection threshold to a low value to detect as many objects as possible. 
detectionThreshold = 0.01;
results = detect(detector,dsTest,MiniBatchSize=8,Threshold=detectionThreshold);

iouThresholds = [0.5 0.75 0.9];
metrics = evaluateObjectDetection(results,dsTest,iouThresholds);

% Evaluate Object Detection Metrics Summary
% Evaluate the summarized detector performance at the overall dataset level and at the individual class level using the summarize object function. 
[datasetSummary,classSummary] = summarize(metrics)

%Compute Average Precision
%Compute the AP at each of the specified overlap thresholds for all classes using the averagePrecision object function. To visualize how the AP value at the specified thresholds values varies across all the classes in the data set, plot a bar plot.
figure
classAP = averagePrecision(metrics);
bar(classAP)
xticklabels(metrics.ClassNames)
ylabel("AP")
legend(string(iouThresholds))


%Compute Precision and Recall Metrics
%Compute the precision and recall metrics using the precisionRecall object function. Plot the precision-recall (PR) curve and the detection confidence scores side-by-side. The PR curve highlights how precise a detector is at varying levels of recall for each class. By plotting the detector scores next to the PR curve, you can choose a detection threshold that achieves the precision and recall you require for your application.
%Precision and Recall for a Single Class
%Select a class, extract the precision and recall metrics for the class at the specified overlap thresholds, and plot the PR curves.
classes = metrics.ClassNames;
class = "chair";

% Extract precision and recall values.
[precision,recall,scores] = precisionRecall(metrics,ClassName=class);

% Plot precision-recall curves.
figure
tiledlayout(1,3)
nexttile
plot(cat(1,recall{:})',cat(1,precision{:})')
ylim([0 1])
xlim([0 1])
xlabel("Recall")
ylabel("Precision")
grid on
axis square
title(class + " Precision/Recall")
legend(string(iouThresholds) + " IoU",Location="southoutside")

% Plot the confidence scores for precision and recall at the specified overlap thresholds, to the right of the PR curve.
nexttile
plot(scores{:},cat(1,recall{:})')
ylim([0 1])
xlim([0 1])
ylabel("Recall")
xlabel("Score")
grid on
axis square
title(class + " Recall/Scores")
legend(string(iouThresholds) + " IoU",Location="southoutside")

nexttile
plot(scores{:},cat(1,precision{:})')
ylim([0 1])
xlim([0 1])
ylabel("Precision")
xlabel("Score")
grid on
axis square
title(class + " Precision/Scores")
legend(string(iouThresholds) + " IoU",Location="southoutside")