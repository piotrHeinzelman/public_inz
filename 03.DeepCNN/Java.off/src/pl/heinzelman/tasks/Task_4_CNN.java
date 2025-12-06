package pl.heinzelman.tasks;

import pl.heinzelman.LayerDeep.LayerConv;
import pl.heinzelman.LayerDeep.LayerFlatten;
import pl.heinzelman.LayerDeep.LayerPoolingMax;
import pl.heinzelman.LayerDeep.LayerReLU;
import pl.heinzelman.neu.LayerSoftmaxMultiClass;
import pl.heinzelman.tools.Tools2;


import java.util.Arrays;
import java.util.Random;

public class Task_4_CNN implements Task{

    private Tools2 tools = new Tools2();
    private float[][][][] testX;
    private float[][] testY;
    private float[][][][] trainX;
    private float[][] trainY;
    private int[][] errors = new int [2][2];


    private LayerConv conv1 = new LayerConv( 7 , 32, null, null  );
    private LayerPoolingMax poolMax1 = new LayerPoolingMax(2,2);
    private LayerReLU      relu1 = new LayerReLU();


    private LayerConv conv2 = new LayerConv( 5 , 64, null, null  );
    private LayerPoolingMax poolMax2 = new LayerPoolingMax(2,2);
    private LayerReLU      relu2 = new LayerReLU();


    private LayerConv conv3 = new LayerConv( 3 , 128, null, null  );
    private LayerPoolingMax poolMax3 = new LayerPoolingMax(2,2);
    private LayerReLU      relu3 = new LayerReLU();

    private LayerConv conv4 = new LayerConv( 3 , 256, null, null  );
    private LayerPoolingMax poolMax4 = new LayerPoolingMax(2,2);
    private LayerReLU      relu4 = new LayerReLU();


    private LayerConv conv5 = new LayerConv( 1 , 256, null, null  );
    private LayerPoolingMax poolMax5 = new LayerPoolingMax(2,2);
    private LayerReLU      relu5 = new LayerReLU();


    private LayerConv conv6 = new LayerConv( 1 , 18, null, null  );
    private LayerPoolingMax poolMax6 = new LayerPoolingMax(2,2);
    private LayerReLU      relu6 = new LayerReLU();



    private LayerConv conv7 = new LayerConv( 1 , 8, null, null  );
    private LayerPoolingMax poolMax7 = new LayerPoolingMax(1,1);
    private LayerReLU      relu7 = new LayerReLU();



    private LayerConv conv8 = new LayerConv( 1 , 6, null, null  );
    private LayerPoolingMax poolMax8 = new LayerPoolingMax(1,1);
    private LayerReLU      relu8 = new LayerReLU();


    private LayerConv conv9 = new LayerConv( 1 , 2, null, null  );
    private LayerPoolingMax poolMax9 = new LayerPoolingMax(1,1);
    private LayerReLU      relu9 = new LayerReLU();



    private LayerFlatten flatten = new LayerFlatten();
    private LayerSoftmaxMultiClass softmax = new LayerSoftmaxMultiClass( 2, 2 );


    public void prepare() {
        int dataSize=30;
        tools.prepareData3C( dataSize );

        testX = tools.getTestX();
        testY = tools.getTestY();
        trainX = tools.getTrainX();
        trainY = tools.getTrainY();

        conv1.setUpByX( 3,240 );
        conv2.setUpByX(32,117);
        conv3.setUpByX(64,56);
        conv4.setUpByX(128,27);
        conv5.setUpByX(256,12);
        conv6.setUpByX(256,6);
        conv7.setUpByX(18,3);
        conv8.setUpByX(8,1);
        conv9.setUpByX(6,1);
    }


    public float[] forward_( float[][][] X ){
        float[][][]t1=relu1.Forward ( poolMax1.Forward( conv1.Forward( X )));
        float[][][]t2=relu2.Forward ( poolMax2.Forward( conv2.Forward ( t1  )));
        float[][][]t3=relu3.Forward ( poolMax3.Forward( conv3.Forward ( t2  )));
        float[][][]t4=relu4.Forward ( poolMax4.Forward( conv4.Forward ( t3  )));
        float[][][]t5=relu5.Forward ( poolMax5.Forward( conv5.Forward ( t4  )));
        float[][][]t6=relu6.Forward ( poolMax6.Forward( conv6.Forward ( t5  )));
        float[][][]t7=relu7.Forward ( poolMax7.Forward( conv7.Forward ( t6  )));
        float[][][]t8=relu8.Forward ( poolMax8.Forward( conv8.Forward ( t7  )));
        float[][][]t9=relu9.Forward ( poolMax9.Forward( conv9.Forward ( t8  )));
        return softmax.nForward( flatten.Forward( t9 ));
    }

    public float[][][] backward_( float[] gradient ){
     //   System.out.println(Arrays.toString( gradient ) );
        float[][][] t9 =flatten.Backward(  softmax.nBackward( gradient ));
        float[][][] t8 = conv9.Backward ( poolMax9.Backward  ( relu9.Backward   ( t9 )));
        float[][][] t7 = conv8.Backward ( poolMax8.Backward  ( relu8.Backward   ( t8 )));
        float[][][] t6 = conv7.Backward ( poolMax7.Backward  ( relu7.Backward   ( t7 )));
        float[][][] t5 = conv6.Backward ( poolMax6.Backward  ( relu6.Backward   ( t6 )));
        float[][][] t4 = conv5.Backward ( poolMax5.Backward  ( relu5.Backward   ( t5 )));
        float[][][] t3 = conv4.Backward ( poolMax4.Backward  ( relu4.Backward   ( t4 )));
        float[][][] t2 = conv3.Backward ( poolMax3.Backward  ( relu3.Backward   ( t3 )));
        float[][][] t1 = conv2.Backward ( poolMax2.Backward  ( relu2.Backward   ( t2 )));
        float[][][] t0 = conv1.Backward ( poolMax1.Backward  ( relu1.Backward   ( t1 )));

        return conv1.Backward(  poolMax1.Backward   ( relu1.Backward ( t0 )));
    }

    @Override
    public void run() {
        prepare();

            int epochs = 50; //50;
            for (int j = 0; j < epochs; j++) {
                train(240);
                System.out.println(  "Epo: "+j );
            }
            test(240);

    }

    public void train( int test_size ){
        int accuracy=0;
        float loss = 0.0f;
        float ce_loss = 0.0f;
        int sum=0;
        Random rand = new Random();

        for (int i = 0; i < test_size; i++) {
            //FORWARD PROPAGATION

            int ind_ex =  (int) ( rand.nextFloat()*test_size );

            float[][][] X = trainX[ind_ex];//; tools.convertToSquare240x240( trainX[ind_ex] );
            int correct_label = tools.getIndexMaxFloat(trainY[ind_ex]);

            float[] Z = forward_(X);
            loss += softmax.delta_Loss( correct_label, Z );
System.out.println( "loss:" +loss );
System.out.println( "Z[0]:" +Z[0]+", Z[1]:"+Z[1]+", ... correct:"+correct_label );
            int findClass = 0;
                 if (Z[1]>Z[0]) { findClass=1; }//tools.getIndexMaxFloat(Z);
            if ( correct_label==findClass ){ accuracy++; }
//System.out.println( "findClass:" +correct_label );
//System.out.println( "Z:" + Arrays.toString( Z ));
            float[] gradient =  tools.gradientCNN(Z, correct_label);// softmax.gradientCNN( Z, correct_label );

//System.out.println( "gradient:" + Arrays.toString( gradient ) );
            backward_( gradient );
        }
        System.out.println( "Acc: " + ((100.0f*accuracy)/ test_size) + ", Loss: " + loss /*+ ", of: " + test_size*/ );
        loss=0.0f;
    }




    public  void test ( int test_size  )   {
        Random rand = new Random();
        int[][] errors = new int[2][2];
        int error = 0;

        int label_counter = 0;
        int accuracy=0;
        int sum=0;

        float[] out_l = new float[10];
        for (int i = 0; i < test_size; i++) {
            int ind_ex =  (int) ( rand.nextFloat()*test_size );
            label_counter++;
            //FORWARD PROPAGATION

            int correct_label=tools.getIndexMaxFloat( testY[i] );
            float[][][] pxl = testX[i];

            out_l = forward_( pxl );

            int findClass = tools.getIndexMaxFloat(out_l);
            if ( correct_label!=findClass ){
                errors[correct_label][findClass]++;
                error++;
            } else { accuracy++;  }
            sum ++;
        }
        System.out.println("\n***********\n** TEST ** errors "+ ( error ) + " .. " + ( 100 * accuracy / test_size )  + "]%\n" );
    }

}


