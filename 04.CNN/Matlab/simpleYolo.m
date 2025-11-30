% https://www.mathworks.com/help/vision/ug/object-detection-using-deep-learning.html


taskNames(1)="dog1";
taskNames(2)="dog2";
taskNames(3)="team";

seriesName(1)="d";
seriesName(2)="e";
seriesName(3)="f";


for t=[1:3]
for i=[1:4]

    TIME_START = datetime('now');
    switch i
        case 1
            % required: Computer Vision Toolbox
            % Computer Vision Toolbox Model for YOLO v2 Object Detection
            name="tiny-yolov2-coco";
            detector = yolov2ObjectDetector( name );
        case 2
            name="tiny-yolov3-coco";
            detector = yolov3ObjectDetector( name );
        case 3
            name="tiny-yolov4-coco";
            detector = yolov4ObjectDetector( name );
        case 4
            name = "tiny-coco";
            detector = yoloxObjectDetector( name );
    end


%disp(name);
TIME_GenerateDetector = datetime('now');
%disp(detector);

% detekcja
imgName = strcat(taskNames(t),'.jpg');
img = imread( imgName );
% img = imread('team.jpg');
% img = imread('dog1.jpg');
% img = imread('dog2.jpg');
% img = imread('sherlock.jpg');

TIME_ReadImage = datetime('now');
[bboxes,scores,labels] = detect(detector,img);
TIME_DetectionTime = datetime('now');
detectedImg = insertObjectAnnotation(img,"Rectangle",bboxes,labels);

%   figure;
%   imshow(detectedImg);

targetName = strcat(  name , "_" , taskNames(t) , ".png" );
imwrite(detectedImg, targetName );

prepareDetectorTime = duration( TIME_GenerateDetector-TIME_START );
readImageTime = duration( TIME_ReadImage-TIME_GenerateDetector );
detectTime = duration( TIME_DetectionTime-TIME_ReadImage );

sc=0;
lb='?';
if ( size(scores) > 0)
    sc = scores(1);
end
if ( size(labels) > 0)
    lb = labels(1);
end

fprintf ('# net: %s; task:%s; class: %s score: %f;  prepare detectortime: :%f; \n# load image time: %f; detect object time: %f \n', name, taskNames(t), lb, sc,  seconds( prepareDetectorTime ), seconds( readImageTime ), seconds(detectTime)  );

% if (t==1)
    fprintf ('library[ %d ]=\" %s score:%f, find(%i) \" \n', i-1, name, sc, (0+size(labels)));
% end

% prepare[0]
% load image[1]
% detect [2]
% score [3]
% detections length [4]
fprintf ('%s0[%d]=%f \n' ,seriesName(t), i-1, seconds( prepareDetectorTime ));
fprintf ('%s1[%d]=%f \n' ,seriesName(t), i-1, seconds( readImageTime       ));
fprintf ('%s2[%d]=%f \n' ,seriesName(t), i-1, seconds( detectTime          ));



end



end
