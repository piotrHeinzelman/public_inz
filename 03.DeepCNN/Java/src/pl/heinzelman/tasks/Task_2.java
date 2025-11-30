package pl.heinzelman.tasks;

import pl.heinzelman.neu.LayerSigmoidFullConn;
import pl.heinzelman.neu.LayerSoftmaxMultiClassONLYFORWARD;
import pl.heinzelman.tools.Tools;

public class Task_2 implements Task{

    private float[][] testX;
    private float[][] testY;
    private float[][] trainX;
    private float[][] trainY;


    private float[][][] testXX;
    private float[][][] trainXX;


    private LayerSigmoidFullConn layer1FC;
    private LayerSigmoidFullConn layer2FC;
    private LayerSoftmaxMultiClassONLYFORWARD layer3SoftmaxMulticlass;
    private Tools tools = new Tools();

    int numOfEpoch=50;
    int cyclesOfEpoch=20;
    float[] CSBin_data=new float[numOfEpoch*cyclesOfEpoch];


    @Override
    public void prepare() {

        int dataSize = 10;
        //tools.prepareDataAsFlatArray( dataSize );
        tools.prepareData( dataSize );

        testX = tools.getTestX();
        testY = tools.getTestY();
        trainX = tools.getTrainX();
        trainY = tools.getTrainY();

        //System.out.println( testX[0].length );

        layer1FC = new LayerSigmoidFullConn( 784, 64 ); layer1FC.setName("Layer1"); // n neurons
        layer2FC = new LayerSigmoidFullConn( 64, 10 ); layer2FC.setName("Layer2"); // n neurons
        layer3SoftmaxMulticlass = new LayerSoftmaxMultiClassONLYFORWARD( 10 ); layer3SoftmaxMulticlass.setName("Layer3"); // n neurons

        // ****************************


    }

    public float[] forward_( float[] X ){
        float[] Z1 = layer1FC.nForward(X);
        float[] Z2 = layer2FC.nForward( Z1 );
        float[] Z3 = layer3SoftmaxMulticlass.nForward( Z2 );
        return Z3;
    }

    public float[] backward_( float[] eIN ){
        return layer1FC.nBackward( layer2FC.nBackward( layer3SoftmaxMulticlass.nBackward( eIN )));
    }


    @Override
    public void run() {
        prepare();

        int epochNum=0;
        for (int cycle=0;cycle<cyclesOfEpoch;cycle++) {
            float Loss=0.0f;
            int step=1;

            for (int epoch = 0; epoch < numOfEpoch; epoch++) {
                step++; epochNum++; Loss=0.0f;
                for ( int index = 0; index < trainX.length; index++ ) {

                    // ONE CYCLE
                    int ind_ex = /*index; //*/ (index * step) % trainX.length;

                    float[] X = trainX[ind_ex];
                    float[] trueZ = trainY[ind_ex];

                    float[] outZ = forward_( X );

                    float[][] ZZ = new float[1][10];  ZZ[0]=outZ;
                    float[] Z_S = layer3SoftmaxMulticlass.compute_gradient( ZZ, tools.getIndexMaxFloat(trueZ) )[0];
                    //float[] Z_S = tools.vectorSubstZsubS( outZ, trueZ );

                                   backward_(Z_S);
                    //Loss += Tools.meanSquareError( outZ, trueZ );
                    //Loss += Tools.crossEntropyMulticlassError( outZ );
                    Loss += layer3SoftmaxMulticlass.delta_Loss( tools.getIndexMaxFloat( trueZ ));

                }
                CSBin_data[epoch]=Loss/trainX.length;
            }
            System.out.println("Loss: " + Loss );

            int acc = 0;
            int sam = 0;
            for (int i=0;i<testX.length;i++){
                float[] Z = forward_(testX[i]);
                float[] trueZ = testY[i];
                if ( tools.getIndexMaxFloat( Z ) == tools.getIndexMaxFloat( trueZ ) ) { acc++; }
                sam++;
            }
            System.out.println("test accuracy: " + 100.0f * acc / sam + "%     ("+epochNum+")");
        }
    }
}



    /*
Loss: 138339.98
test accuracy: 25.7%     (50)
Loss: 138373.53
test accuracy: 59.7%     (100)
Loss: 138473.97
test accuracy: 66.7%     (150)
Loss: 138564.06
test accuracy: 72.2%     (200)
Loss: 138617.77
test accuracy: 75.6%     (250)
Loss: 138650.27
test accuracy: 76.8%     (300)
Loss: 138665.27
test accuracy: 77.5%     (350)
Loss: 138663.52
test accuracy: 78.3%     (400)
Loss: 138669.45
test accuracy: 78.9%     (450)
Loss: 138677.48
test accuracy: 79.5%     (500)
Loss: 138684.42
test accuracy: 79.8%     (550)
Loss: 138690.8
test accuracy: 80.2%     (600)
Loss: 138696.44
test accuracy: 80.4%     (650)
Loss: 138700.02
test accuracy: 80.6%     (700)
Loss: 138701.28
test accuracy: 80.8%     (750)
Loss: 138705.19
test accuracy: 80.8%     (800)
Loss: 138708.58
test accuracy: 81.1%     (850)
Loss: 138711.75
test accuracy: 81.2%     (900)
Loss: 138714.28
test accuracy: 81.4%     (950)
Loss: 138716.81
test accuracy: 81.7%     (1000)
     */
