

 

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



timeLoadDataStart = datetime('now');

if ( true )


    fileIMG=fopen( '../../data/train-labels-idx1-ubyte','r');
    fileData=fread( fileIMG, 'uint8' );
    fclose(fileIMG);
    ytmp=fileData(9:8+(percent*600))';
    ysize=size(ytmp);
    ysize=ysize(2);
    ytrain=zeros(10,ysize);
    for i=(1:ysize)
        d=ytmp(i);
        if (d==0)
            d=10;
        end
        ytrain(d,i)=1;
    end

    fileData=1;

    fileIMG=fopen( '../../data/t10k-labels-idx1-ubyte','r');
    fileData=fread( fileIMG, 'uint8' );
    fclose(fileIMG);

    ytmp=fileData(9:8+(percent*100))';
    ysize=size(ytmp);
    ysize=ysize(2);
    ytest=zeros(10,ysize);
    for i=(1:ysize)
        d=ytmp(i);
        if (d==0)
            d=10;
        end
        ytest(d,i)=1;
    end


    fileIMG=fopen( '../../data/train-images-idx3-ubyte','r');
    fileData=fread( fileIMG, 'uint8' );
    fclose(fileIMG);
    tmp=fileData(17:16+(percent*784*600));

    for i=1:percent*600
        col=tmp(1+(i-1)*784:i*784);
        xtrain(1:784,i)=col';
    end
    xtrain=xtrain/255;
    fileData=1;

    fileIMG=fopen( '../../data/t10k-images-idx3-ubyte','r');
    fileData=fread( fileIMG, 'uint8' );
    fclose(fileIMG);
    tmp=fileData(17:16+(percent*784*100));

    for i=1:percent*100
        col=tmp(1+(i-1)*784:i*784);
        xtest(1:784,i)=col';
    end
    xtest=xtest/255;
    fileData=1;
end
timeLoadDataEnd = datetime('now');



load("net");
X=xtrain(:,26 );
showx( xtrain , 26 );
Z = net( X )
indexOfMaxInVector( Z )

 
   

