% https://www.mathworks.com/help/deeplearning/gs/create-simple-deep-learning-classification-network.html
% C:\Users\PiotrH\Desktop\IMG_SAS\0and1

%unzip("DigitsData.zip")
imds = imageDatastore("D:\INZ\inz\MixedProj\04.R-CNN\MatlabClassificationMLP\240pix2classSAS", ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

numTrainFiles = 800;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,"randomized");

classNames = categories(imdsTrain.Labels)


% Define Network Architecture
% inputSize = [28 28 1];
inputSize = [240 240 3];
% numClasses = 10;
numClasses = 2;

%layers = [
%    imageInputLayer(inputSize)
%    convolution2dLayer(5,20)
%    batchNormalizationLayer
%    reluLayer
%    fullyConnectedLayer(numClasses)
%    softmaxLayer];

% https://www.mathworks.com/help/deeplearning/ug/list-of-deep-learning-layers.html
% Yolo2 
layers = [  
imageInputLayer(inputSize)
convolution2dLayer(3,16)
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2)

convolution2dLayer(3,32)
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2)

convolution2dLayer(3,64)
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2)

%convolution2dLayer(3,128)
%batchNormalizationLayer
%reluLayer
%maxPooling2dLayer(2)

%convolution2dLayer(3,256)
%batchNormalizationLayer
%reluLayer
%maxPooling2dLayer(2)

%convolution2dLayer(3,512)
%batchNormalizationLayer
%reluLayer
%maxPooling2dLayer(2)

%convolution2dLayer(3,1024)
%batchNormalizationLayer
%reluLayer

%convolution2dLayer(3,512)
%batchNormalizationLayer
%leakyReluLayer(.1)

%convolution2dLayer(3,425)

%% Yolo enter here !!!

%batchNormalizationLayer
%reluLayer
fullyConnectedLayer(numClasses)
softmaxLayer];

% "sgdm" — Stochastic gradient descent with momentum (SGDM). SGDM is a stochastic solver. For additional training options, see Stochastic Solver Options. For more information, see Stochastic Gradient Descent with Momentum.
% "rmsprop" — Root mean square propagation (RMSProp). RMSProp is a stochastic solver. For additional training options, see Stochastic Solver Options. For more information, see Root Mean Square Propagation.
% "adam" — Adaptive moment estimation (Adam). Adam is a stochastic solver. For additional training options, see Stochastic Solver Options. For more information, see Adaptive Moment Estimation.
% "lbfgs" (since R2023b) — Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS). L-BFGS is a batch solver. Use the L-BFGS algorithm for small networks and data sets that you can process in a single batch. For additional training options, see Batch Solver Options. For more information, see Limited-Memory BFGS.
% "lm" (since R2024b) — Levenberg–Marquardt (LM). LM is a batch solver. Use the LM algorithm for regression networks with small numbers of learnable parameters, where you can process the data set in a single batch. If solverName is "lm", then the lossFcn argument of the trainnet function must be "mse" or "l2loss". For additional training options, see Batch Solver Options. For more information, see Levenberg–Marquardt.

options = trainingOptions("adam", ...
    MaxEpochs=1, ...
    ValidationData=imdsValidation, ...
    ValidationFrequency=30, ...  
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false); % MiniBatchSize=20, ...

net = trainnet(imdsTrain,layers,"crossentropy",options);


doTraining = true;

%if doTraining    
    % Train a network.
%    cifar10Net = trainNetwork(imdsTrain, layers, opts);
    save( 'net.mat','net');
%else
    % Load pre-trained detector for the example.
%    load( append(path , 'cifar10Net.mat'), 'cifar10Net' );
%end

accuracy = testnet(net,imdsValidation,"accuracy")



