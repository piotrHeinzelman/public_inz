package pl.heinzelman.tasks;

import pl.heinzelman.LayerDeep.LayerConv;
import pl.heinzelman.LayerDeep.LayerFlatten;
import pl.heinzelman.LayerDeep.LayerPoolingMax;
import pl.heinzelman.neu.LayerSoftmaxMultiClass;
import pl.heinzelman.tools.Tools2;
import java.time.Instant;
import java.time.temporal.ChronoUnit;


import java.util.Random;

public class Task_4_CNN implements Task{

    private Tools2 tools = new Tools2();
    private float[][] testX;
    private float[][] testY;
    private float[][] trainX;
    private float[][] trainY;
    private int[][] errors = new int [10][10];


    private LayerConv conv = new LayerConv( 5 , 20, null, null  );
    private LayerPoolingMax poolMax = new LayerPoolingMax(2,2);
    private LayerFlatten flatten = new LayerFlatten();
    private LayerSoftmaxMultiClass softmax = new LayerSoftmaxMultiClass( 12*12*20, 10 );


    public void prepare() {
        int dataSize=80;
        tools.prepareData( dataSize );

        testX = tools.getTestX();
        testY = tools.getTestY();
        trainX = tools.getTrainX();
        trainY = tools.getTrainY();

        float[][][] oneX = new float[1][28][28];
        oneX[0] = tools.convertToSquare28x28( trainX[0] );
        conv.setUpByX( oneX );
    }


    public float[] forward_( float[][] X ){
        float[][][] oneX = new float[1][][];
        oneX[0]=X;
        return softmax.nForward( flatten.Forward( poolMax.Forward( conv.Forward( oneX ))));
    }

    public float[][][] backward_( float[] gradient ){
        return conv.Backward(  poolMax.Backward( flatten.Backward(  softmax.nBackward( gradient ))));
    }


// *********************

    @Override
    public void run() {
        prepare();
        for ( int i=0;i<20;i++) {
            train(48000);
            System.out.println( "epoch: " + i );
       }
//        Instant Fstart=Instant.now();
        test(8000);
//	Instant Fend=Instant.now();
//	Double time=( ChronoUnit.MILLIS.between(Fstart,Fend))/1000.0;
//        System.out.println("Accuracy Time:" +  time );
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
            // System.out.println( ind_ex );

            float[][] X = tools.convertToSquare28x28( trainX[ind_ex] );
            int correct_label = tools.getIndexMaxFloat(trainY[ind_ex]);

            float[] Z = forward_(X);
            loss += softmax.delta_Loss( correct_label );
            int findClass = tools.getIndexMaxFloat(Z);
            if ( correct_label==findClass ){ accuracy++; }

            float[] gradient = softmax.gradientCNN( Z, correct_label );
            backward_( gradient );
        }
        //System.out.println( "Acc: " + ((100.0f*accuracy)/ test_size) + ", Loss: " + loss + ", of: " + test_size );
        loss=0.0f;
    }




    public  void test ( int test_size  )   {
        Random rand = new Random();
        int[][] errors = new int[10][10];
        int error = 0;

        int label_counter = 0;
        int accuracy=0;
        int sum=0;

        float[] out_l = new float[10];
        for (int i = 0; i < test_size; i++) {
            int ind_ex =  (int) ( rand.nextFloat()*test_size );
            label_counter++;
            //FORWARD PROPAGATION

            // importImage
            int correct_label=tools.getIndexMaxFloat( testY[i] );
            float[][] pxl = tools.convertToSquare28x28( testX[i] );

            // perform convolution 28*28 --> 8x26x26
            out_l = forward_( pxl );

            // compute cross-entropy loss
            int findClass = tools.getIndexMaxFloat(out_l);//  ()int) Mat.v_argmax(out_l);
            if ( correct_label!=findClass ){
                errors[correct_label][findClass]++;
                error++;
            } else { accuracy++;  }
            //accuracy += correct_label == Mat.v_argmax(out_l) ? 1 : 0;
            sum ++;
        }
        System.out.println("\n***************************************\n** TEST ** errors "+ ( error ) + " : acc["+(accuracy*100) / test_size+ "]%\n" );
        // Tools2.printTable2( errors );
    }

}


