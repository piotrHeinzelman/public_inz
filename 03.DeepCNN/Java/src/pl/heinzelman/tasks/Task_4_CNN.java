package pl.heinzelman.tasks;

import pl.heinzelman.LayerDeep.LayerConv;
import pl.heinzelman.LayerDeep.LayerFlatten;
import pl.heinzelman.LayerDeep.LayerPoolingMax;
import pl.heinzelman.LayerDeep.LayerReLU;
import pl.heinzelman.neu.LayerSoftmaxMultiClass;
import pl.heinzelman.tools.Tools2;

import java.sql.Timestamp;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.Random;

public class Task_4_CNN implements Task{

    private int percent=1;
    private Tools2 tools = new Tools2();
    private float[][][][] testX;
    private float[][] testY;
    private float[][][][] trainX;
    private float[][] trainY;
    private int[][] errors = new int [2][2];

/*
    private LayerConv conv = new LayerConv( 5 , 20, null, null  );
    private LayerPoolingMax poolMax = new LayerPoolingMax(2,2);
    private LayerFlatten flatten = new LayerFlatten();
    private LayerSoftmaxMultiClass softmax = new LayerSoftmaxMultiClass( 118*118*20, 2 );
*/

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


    private LayerConv conv9 = new LayerConv( 3 , 2, null, null  );
    private LayerPoolingMax poolMax9 = new LayerPoolingMax(1,1);
    private LayerReLU      relu9 = new LayerReLU();



    private LayerFlatten flatten = new LayerFlatten();
    private LayerSoftmaxMultiClass softmax = new LayerSoftmaxMultiClass( 2, 2 );



    public void prepare( int percent ) {
        this.percent=percent;
        tools.prepareData3C( percent );

        testX = tools.getTestX();
        testY = tools.getTestY();
        trainX = tools.getTrainX();
        trainY = tools.getTrainY();

        if ( false ) {
        // check data
        tools.saveXasJPG( testX[0] );
        System.out.println( "CLASS:" + testY[0][0] );
        }

        conv1.setUpByX( 3,240 );
        conv2.setUpByX(32,117);
        conv3.setUpByX(64,56);
        conv4.setUpByX(128,27);
        conv5.setUpByX(256,12);
        conv6.setUpByX(256,6);
        conv7.setUpByX(18,3);
        conv8.setUpByX(8,3);
        conv9.setUpByX(6,3);


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
        float[] flat = flatten.Forward(t9);
        float[] soft = softmax.nForward(flat);


        //Instant end = Instant.now();
        //System.out.println( "\r\n\r\n------ PREDICTION TIME --------");
        //long gap = ChronoUnit.MILLIS.between(start, end);
        //System.out.println( " -GAP - : " + gap/1000.0 + " [sek.]" );

        return soft;
    }

    public float[][][] backward_( float[] gradient ){
        float[][][] t9 =  flatten.Backward( softmax.nBackward( gradient ));
        float[][][] t90 = relu9.Backward(t9);
        float[][][] t91 = poolMax9.Backward(t90);
        float[][][] t8 = conv9.Backward ( t91 );
        float[][][] t7 = conv8.Backward ( poolMax8.Backward  ( relu8.Backward   ( t8 )));
        float[][][] t6 = conv7.Backward ( poolMax7.Backward  ( relu7.Backward   ( t7 )));
        float[][][] t5 = conv6.Backward ( poolMax6.Backward  ( relu6.Backward   ( t6 )));
        float[][][] t4 = conv5.Backward ( poolMax5.Backward  ( relu5.Backward   ( t5 )));
        float[][][] t3 = conv4.Backward ( poolMax4.Backward  ( relu4.Backward   ( t4 )));
        float[][][] t2 = conv3.Backward ( poolMax3.Backward  ( relu3.Backward   ( t3 )));
        float[][][] t1 = conv2.Backward ( poolMax2.Backward  ( relu2.Backward   ( t2 )));
        float[][][] t0 = conv1.Backward ( poolMax1.Backward  ( relu1.Backward   ( t1 )));
        return t0;
    }




// *********************

    @Override
    public void run() {
        prepare( percent );
        for ( int i=0;i<50;i++) {
            System.out.println( "epoch: " + i );
            train(percent*8 );
       }
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

            float[][][] X = trainX[ i ]; //tools.convertToSquare240x240( trainX[ ind_ex ]);
            int correct_label = tools.getIndexMaxFloat(trainY[i]);
            float[] Z = forward_(X);

            loss += softmax.delta_Loss( correct_label );

            float[] gradient = softmax.gradientCNN( Z, correct_label );
                if (true) System.out.println("I:" + i + "correct_label:" + correct_label + "   Z:" + Arrays.toString( Z ) + ", grad"+Arrays.toString(gradient) );
            backward_(gradient);
        }
	System.out.println("LOSS:"+loss);

        // TEST
        for (int i = 0; i < test_size/5; i++) {
            //FORWARD PROPAGATION
            int ind_ex =  (int) ( rand.nextFloat()*test_size );

            int correct_label = tools.getIndexMaxFloat(trainY[i]);
            float[] Z = forward_(trainX[ i ]);
            int findClass = tools.getIndexMaxFloat(Z);
            if ( correct_label==findClass ){ accuracy++; }
            float[] gradient = softmax.gradientCNN( Z, correct_label );
            backward_(gradient);
            System.out.println( "test: " + correct_label + " ? " + findClass );
        }
        System.out.println( "ACCURACY: " + 1f*accuracy / (test_size/5)  );
        loss=0.0f;
    }

}


