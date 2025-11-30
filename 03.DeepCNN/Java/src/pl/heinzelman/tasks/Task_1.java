package pl.heinzelman.tasks;

import pl.heinzelman.neu.LayerSigmoidFullConn;
import pl.heinzelman.tools.Tools;

public class Task_1 implements Task{

    private float[][] testX;
    private float[][] testY;
    private float[][] trainX;
    private float[][] trainY;


    private float[][][] testXX;
    private float[][][] trainXX;


    private LayerSigmoidFullConn layer1FC;
    private LayerSigmoidFullConn layer2FC;
    private LayerSigmoidFullConn layer3FC;
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

        System.out.println( testX[0].length );

        layer1FC =new LayerSigmoidFullConn( 784, 64 ); layer1FC.setName("Layer1"); // n neurons
        layer2FC =new LayerSigmoidFullConn( 64, 10 ); layer2FC.setName("Layer2"); // n neurons
        layer3FC =new LayerSigmoidFullConn( 10, 10 ); layer3FC.setName("Layer3"); // n neurons

        // ****************************


    }

    public float[] forward_( float[] X ){
        float[] Z1 = layer1FC.nForward(X);
        float[] Z2 = layer2FC.nForward( Z1 );
        float[] Z3 = layer3FC.nForward( Z2 );
        return Z3;
    }

    public float[] backward_( float[] eIN ){
        return layer1FC.nBackward( layer2FC.nBackward( layer3FC.nBackward( eIN )));
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

                    float[] Z_S = tools.vectorSubstZsubS( outZ, trueZ );
                    //float[] Z_S = Tools.gradientSoftMax( outZ, trueZ  );

                                   backward_(Z_S);
                    Loss += Tools.meanSquareError( outZ, trueZ );
                    //Loss += Tools.crossEntropyMulticlassError( outZ );

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
        tools.saveVectorAsImg(  CSBin_data, "csbin_task1" );
    }
}


/*

Loss: 5383.8667
test accuracy: 29.9%     (50)
Loss: 5260.1445
test accuracy: 40.9%     (100)
Loss: 5079.2637
test accuracy: 57.4%     (150)
Loss: 4927.308
test accuracy: 69.7%     (200)
Loss: 4812.505
test accuracy: 75.0%     (250)
Loss: 4725.671
test accuracy: 78.0%     (300)
Loss: 4661.4614
test accuracy: 79.5%     (350)
Loss: 4616.4824
test accuracy: 80.1%     (400)
Loss: 4587.25
test accuracy: 81.2%     (450)
Loss: 4568.7515
test accuracy: 81.2%     (500)
Loss: 4555.9395
test accuracy: 81.5%     (550)
Loss: 4546.6504
test accuracy: 81.9%     (600)
Loss: 4539.971
test accuracy: 81.8%     (650)
Loss: 4535.0845
test accuracy: 81.9%     (700)
Loss: 4531.2827
test accuracy: 82.2%     (750)
Loss: 4528.162
test accuracy: 82.1%     (800)
Loss: 4525.3174
test accuracy: 82.5%     (850)
Loss: 4522.3755
test accuracy: 82.8%     (900)
Loss: 4520.3135
test accuracy: 83.1%     (950)
Loss: 4519.029
test accuracy: 83.2%     (1000)

 */