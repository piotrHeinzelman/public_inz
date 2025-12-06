package pl.heinzelman.tasks;

import pl.heinzelman.LayerDeep.LayerConv;
import pl.heinzelman.LayerDeep.LayerFlatten;
import pl.heinzelman.LayerDeep.LayerPoolingMax;
import pl.heinzelman.neu.LayerSoftmaxMultiClass;
import pl.heinzelman.tools.Tools2;

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


    private LayerConv conv = new LayerConv( 5 , 20, null, null  );
    private LayerPoolingMax poolMax = new LayerPoolingMax(2,2);
    private LayerFlatten flatten = new LayerFlatten();
    private LayerSoftmaxMultiClass softmax = new LayerSoftmaxMultiClass( 118*118*20, 2 );


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
        conv.setUpByX( 3,240);
    }


    public float[] forward_( float[][][] X ){
        float[][][]  oneX = X;
        return softmax.nForward( flatten.Forward( poolMax.Forward( conv.Forward( oneX ))));
    }

    public float[][][] backward_( float[] gradient ){
        return conv.Backward(  poolMax.Backward( flatten.Backward(  softmax.nBackward( gradient ))));
    }


// *********************

    @Override
    public void run() {
        prepare( percent );
        for ( int i=0;i<20;i++) {
            train(percent*8 );
            System.out.println( "epoch: " + i );
       }
        test(percent*1 );
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

            float[][][] X = trainX[ ind_ex ]; //tools.convertToSquare240x240( trainX[ ind_ex ]);
            int correct_label = tools.getIndexMaxFloat(trainY[ind_ex]);
            float[] Z = forward_(X);
                     if (i==0) System.out.println("I:" + i + "correct_label:" + correct_label + "   Z:" + Arrays.toString( Z ));

            loss += softmax.delta_Loss( correct_label );

            int findClass = tools.getIndexMaxFloat(Z);
            if ( correct_label==findClass ){ accuracy++; }

            float[] gradient = softmax.gradientCNN( Z, correct_label );
            backward_( gradient );
        }
        System.out.println( loss );
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
            float[][][] pxl = testX[i]; //tools.convertToSquare240x240(new float[][]{testX[i]});

            out_l = forward_( pxl );
            int findClass = tools.getIndexMaxFloat(out_l);
            if ( correct_label!=findClass ){
                errors[correct_label][findClass]++;
                error++;
            } else { accuracy++;  }
            sum ++;
        }
        System.out.println("\n*************\n** TEST ** errors "+ ( error ) + " : acc["+(accuracy*100) / test_size+ "]%\n" );
    }

}


