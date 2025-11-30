% S.Osowski, R.Szmurlo : modele matematyczne uczenia maszynowego 
% Alexnet - tryb Transfer Learning
% https://www.mathworks.com/help/deeplearning/ug/introduction-to-convolutional-neural-networks.html

    epochs=20;
    percent=80;


input = imageInputLayer([28 28 1]);  % 28x28px 1 channel
conv = convolution2dLayer(5, 20); % 20 filter, 5x5
relu = reluLayer;                    %reLU
max = maxPooling2dLayer(2,Stride=2);
fc = fullyConnectedLayer(10);
sm = softmaxLayer;
co = classificationLayer;


layers = [ input
    conv
    relu
    max
    fc
    sm
    co];



function accuracy = accuracyCheck( first, second )
    goals=0;
    s=size(first);
    s=s(2);
    for i=(1:s)
        val1=first(i);
        val2=second(i);
        if (val1==val2)
            goals=goals+1;
        end
    end
    accuracy = goals/s;
end


function index = indexOfMaxInVector( vec ) %
    val=vec(1);
    index=1;
    s=size(vec);
    s=s(1);
    for i = (2:s)
        if (val<vec(i))
            index=i;
            val=vec(i);
        end
    end
end


function aryOfInt = aryOfVectorToAryOfInt( aryOfVec )
    s = size( aryOfVec );
    h=s(1);
    s=s(2);
    aryOfInt=zeros(1,s);
    for i=1:s
      vec=aryOfVec(1:h,i);
      index = indexOfMaxInVector(vec);
      if index==10
          index=0;
      end
      aryOfInt(i)=index;
    end
end


function showx( arrayx , i )
    img0=arrayx(1:784,i);
    img0=img0*256;
    image(img0);

    img=zeros(28,28);
        for i=(1:28)
            row=img0((i-1)*28+1:(i)*28);
           img(i,1:28)=row;
        end
    image(img)
end

if ( 1==1 )
    ST = datetime('now');

    fileIMG=fopen( '../../data/train-labels-idx1-ubyte','r');
    fileData=fread( fileIMG, 'uint8' );
    fclose(fileIMG);
    ytmp=fileData(9:8+percent*600)';
    ysize=percent*600;
    yyy=zeros(1,ysize);
    for i=(1:ysize)
        d=ytmp(i);
        d=d+1;
        yyy(i)=d ;
    end
    ytrain=categorical(yyy);

    fileData=1;

    fileIMG=fopen( '../../data/t10k-labels-idx1-ubyte','r');
    fileData=fread( fileIMG, 'uint8' );
    fclose(fileIMG);

    ytmp=fileData(9:8+percent*100)';
    ysize=percent*100;
    yyy=zeros(1,ysize);
    for i=(1:ysize)
        d=ytmp(i);
        d=d+1;
        yyy(i)=d ;
    end
    ytest=categorical(yyy);

    fileIMG=fopen( '../../data/train-images-idx3-ubyte','r');
    fileData=fread( fileIMG, 'uint8' );
    fclose(fileIMG);
    tmp=fileData(17:16+percent*784*600);

    for i=1:percent*600
        col=tmp(1+(i-1)*784:i*784);
        row=col';
        ary=zeros(28,28);
        for j=1:28
            for k=1:28
                val=row(k+((j-1)*28));
                xtrain(j,k,1,i)=val; %/255

            end
        end
    end
    fileData=1;

    fileIMG=fopen( '../../data/t10k-images-idx3-ubyte','r');
    fileData=fread( fileIMG, 'uint8' );
    fclose(fileIMG);
    tmp=fileData(17:16+percent*784*100);

    for i=1:percent*100
        col=tmp(1+(i-1)*784:i*784);
        row=col';
        ary=zeros(28,28);
        for j=1:28
            for k=1:28
                val=row(k+((j-1)*28));
                xtest(j,k,1,i)=val; %/255

            end
        end
    end
    fileData=1;
end



ED = datetime('now'); loadDataTime = duration( ED-ST );



options=trainingOptions('adam', 'MaxEpochs',epochs, 'ExecutionEnvironment','gpu' , 'Verbose', 1 , 'MiniBatchSize',  1024, 'InitialLearnRate',0.01 );

ST = datetime('now');

	netTransfer = trainNetwork( xtrain, ytrain, layers, options);

ED = datetime('now');
trainTime = duration( ED-ST );




weights_first=netTransfer.Layers(2,1).Weights(:,:,1,1);
%image(weights_first*255)


ST = datetime('now');
 predictedLabels = classify(netTransfer, xtest);
 accuracy = accuracyCheck( predictedLabels', ytest );
ED = datetime('now');
accuracyTime = duration( ED-ST );

    fprintf('# CNN: 5x5 * 20 *  50 cycles: (Linux GPU, batch mode)\n' );
    fprintf( '# accuracy: a:%f\n\n' , accuracy );
    fprintf ('loadDataTime: %f\n' , seconds(loadDataTime)  );
    fprintf ('accuracyTime: %f\n' , seconds(accuracyTime)  );

    fprintf ('Train time: %f\n', seconds(trainTime)  );


