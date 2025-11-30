package pl.heinzelman.tasks;

import pl.heinzelman.neu.LayerSigmoidFullConn;
import pl.heinzelman.neu.LayerSoftmaxMultiClassONLYFORWARD;
import pl.heinzelman.tools.Tools;

import java.util.Arrays;

public class Task_64_64_simple_backward implements Task{

    private float[][] testX;
    private float[][] testY;
    private float[][] trainX;
    private float[][] trainY;

    private LayerSigmoidFullConn layer1;
    private LayerSigmoidFullConn layer2;
    private LayerSoftmaxMultiClassONLYFORWARD layer3;

    private Tools tools = new Tools();

    int numOfEpoch=130;
    float[] CSBin_data=new float[numOfEpoch];

    @Override
    public void prepare() {
        tools.prepareData(100 );

        testX = tools.getTestX();
        testY = tools.getTestY();
        trainX = tools.getTrainX();
        trainY = tools.getTrainY();

        layer1=new LayerSigmoidFullConn( 784, 64 ); layer1.setName("Layer1"); // n neurons

        layer2=new LayerSigmoidFullConn( 64 ,10 ); layer2.setName("Layer2"); // n neurons

        layer3=new LayerSoftmaxMultiClassONLYFORWARD( 10 ); layer3.setName("Layer3"); // n neurons

    }

    @Override
    public void run() {

        for (int cycle=0;cycle<10;cycle++) {

            float Loss = 0.0f;
            int step=1;
            for (int epoch = 0; epoch < numOfEpoch; epoch++) {
                step++;
                for ( int index = 0; index < trainX.length; index++ ) {

                    // ONE CYCLE
                    int ind_ex = /*index; //*/ (index*step) % trainX.length;


                    layer1.nForward(trainX[ ind_ex ]);
                    layer2.nForward(layer1.getZ());
                    layer3.nForward(layer2.getZ());

                    //System.out.println( "trainX[ ind_ex ]:" + Arrays.toString(  trainX[ ind_ex ] ));
                    //System.out.println( "layer3.getZ()" + Arrays.toString(  layer3.getZ() ));

                    float[] Z_S = tools.vectorSubstZsubS( layer3.getZ(), trainY[ ind_ex ]);
                    Loss += Tools.crossEntropyMulticlassError( layer3.getZ() );

                    //float[] gradientSM = tools.gradientSoftMax( trainY[ ind_ex ], layer3.getZ() );

                    //System.out.println( " Z_S: "+Arrays.toString( Z_S ) );
                    //System.out.println( " gradientSM: "+Arrays.toString( gradientSM ) );
                    layer3.nBackward(Z_S);
                    layer2.nBackward( layer3.getEout() );
                    // if  ( ind_ex==5 && epoch%100==0 ) System.out.println( " layer3.getEout() "+Arrays.toString( layer3.getEout() ) );
                    layer1.nBackward( layer2.getEout() );
                }
                CSBin_data[epoch]=Loss/trainX.length;

            }
            System.out.println( Arrays.toString( layer1.getNeuronWeight(0)));
            System.out.println( Arrays.toString( layer3.getX()));
            System.out.println( Arrays.toString( layer3.getZ()));

            // check accuracy
            int len = testX.length;
            int accuracy = 0;
            for (int i = 0; i < len; i++) {
                layer1.nForward(testX[i]);
                layer2.nForward(layer1.getZ());
                layer3.nForward(layer2.getZ());

                int netClassId = tools.getIndexMaxFloat( layer3.getZ() );
                int fileClassId = tools.getIndexMaxFloat( testY[i] );
                if (fileClassId == netClassId) {
                    accuracy++;
                }
            }
            System.out.println(100.0f * accuracy / len + "%");
        }
    }
}
