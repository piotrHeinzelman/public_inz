data = load('rcnnStopSigns.mat', 'stopSigns', 'fastRCNNLayers');
stopSigns = data.stopSigns;
fastRCNNLayers = data.fastRCNNLayers;

stopSigns.imageFilename = fullfile(toolboxdir('vision'),'visiondata', ...
    stopSigns.imageFilename);

rng(0);
shuffledIdx = randperm(height(stopSigns));
stopSigns = stopSigns(shuffledIdx,:);

imds = imageDatastore(stopSigns.imageFilename);

blds = boxLabelDatastore(stopSigns(:,2:end));

ds = combine(imds, blds);
ds = transform(ds,@(data)preprocessData(data,[920 968 3]));

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 10, ...
    'CheckpointPath', tempdir);

frcnn = trainFastRCNNObjectDetector(ds, fastRCNNLayers , options, ...
    'NegativeOverlapRange', [0 0.1], ...
    'PositiveOverlapRange', [0.7 1]);


img = imread('stopSignTest.jpg');

[bbox, score, label] = detect(frcnn, img);

detectedImg = insertObjectAnnotation(img,'rectangle',bbox,score);
figure
imshow(detectedImg)


function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
bboxes = round(data{2});
data{2} = bboxresize(bboxes,scale);
end