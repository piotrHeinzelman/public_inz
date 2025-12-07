% https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
% unzip("DigitsData.zip"); % MNIST

epoch=50; %50
GPU=true;
TIME_START=datetime('now');

dataFolder = "../../../data/240pix2classSAS/";
imds = imageDatastore(dataFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
%{
figure
tiledlayout("flow");
perm = randperm(1764,20);
for i = 1:20
    nexttile
    imshow(imds.Files{perm(i)});
end
%}

classNames = categories(imds.Labels);
labelCount = countEachLabel(imds);
numberOfClass=2;

img = readimage(imds,1);
size(img);


numTrainFiles = 240; % 800;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,"randomize");


TIME_LOAD_AND_PREPARE_DATA = datetime('now');




% layers std
%layers = [
%    imageInputLayer([240 240 3])
%    convolution2dLayer(3,8,Padding="same")
%    batchNormalizationLayer
%    reluLayer

%    maxPooling2dLayer(2,Stride=2)

%    convolution2dLayer(3,16,Padding="same")
%    batchNormalizationLayer
%    reluLayer

%    maxPooling2dLayer(2,Stride=2)

%    convolution2dLayer(3,32,Padding="same")
%    batchNormalizationLayer
%    reluLayer

%    fullyConnectedLayer( numberOfClass )
%    softmaxLayer];


%layers a`la yolo4
layers = [
    imageInputLayer([240 240 3], "Min",0, "Max",1, NormalizationDimension="auto", Normalization="rescale-zero-one", SplitComplexInputs=0 )

    convolution2dLayer(7,32,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer(5,64,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)


    convolution2dLayer(3,128,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer(3,256,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer(1,256,Padding="same")
    batchNormalizationLayer

    convolution2dLayer(1,18,Padding="same", Stride=2)
    batchNormalizationLayer
    reluLayer(Name="Lay")
    maxPooling2dLayer(2)

    convolution2dLayer(1,8,Padding="same", Stride=2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2)

    convolution2dLayer(1,6,Padding="same", Stride=2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2)

    convolution2dLayer(1,2,Padding="same" )
    batchNormalizationLayer
    reluLayer

    %maxPooling2dLayer(2, Stride=2)
    % fullyConnectedLayer( numberOfClass )
    flattenLayer
    softmaxLayer
    %flattenLayer

    ];



options = trainingOptions("sgdm", ...
    InitialLearnRate=0.01, ...
    MaxEpochs=epoch, ... 
    ValidationData=imdsValidation, ... 
    Metrics="accuracy", ... 
    Verbose=true);


TIME_CREATE_MODEL = datetime('now');

net = trainnet(imdsTrain,layers,"crossentropy",options);
 %analyzeNetwork(net);

TIME_TRAIN_TIME = datetime('now');

scores = minibatchpredict(net,imdsValidation);
YValidation = scores2label(scores,classNames);

TValidation = imdsValidation.Labels;
accuracy = mean(YValidation == TValidation);

TIME_ACCURACY_TIME = datetime('now');

if (false) % No generate on CPU !
   save('netCNN_SAS','net');
end




%%%%%  REPORT %%%%%

timeLoadData   = seconds( duration( TIME_LOAD_AND_PREPARE_DATA-TIME_START ));
timeMakeLayers = seconds( duration( TIME_CREATE_MODEL-TIME_LOAD_AND_PREPARE_DATA));
timeTrainModel = seconds( duration( TIME_TRAIN_TIME-TIME_CREATE_MODEL));
timeAccuracy   = seconds( duration( TIME_ACCURACY_TIME-TIME_TRAIN_TIME));


fprintf ('## Epoch:%d, GPU:%s, loadData:%f, createLayers:%f, trainTime:%f, timeAccuracy:%f, accuracy: %f\n', epoch, GPU, timeLoadData, timeMakeLayers, timeTrainModel, timeAccuracy, accuracy );
fprintf ('library[0]="Matlab GPU"\n'  );
fprintf ('d0[0]=%f\n', timeLoadData   );
fprintf ('d1[0]=%f\n', timeMakeLayers );
fprintf ('d2[0]=%f/%d\n', timeTrainModel, epoch );
fprintf ('d3[0]=%f\n', timeAccuracy   );
%fprintf ('%s2[%d]=%f \n');

