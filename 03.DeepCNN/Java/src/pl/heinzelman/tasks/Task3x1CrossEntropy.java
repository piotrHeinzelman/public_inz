package pl.heinzelman.tasks;

import pl.heinzelman.neu.LType;
import pl.heinzelman.neu.Layer;
import pl.heinzelman.neu.LayerSigmoidFullConn;
import pl.heinzelman.tools.Tools;

import java.util.Arrays;

public class Task3x1CrossEntropy implements Task{


    @Override
    public void prepare() {
        // NOP
    }

    @Override
    public void run() {


        //Layer layer1=new Layer( LType.sigmod , 3 ,2 );
        LayerSigmoidFullConn layer1=new LayerSigmoidFullConn( 2, 3 );
        layer1.setName( "Layer: 1" );

        // first neu
        layer1.setWmn( 0, 0, 1 );
        layer1.setWmn( 0, 1, -1 );

        // secont neu
        layer1.setWmn( 1, 0, 1 );
        layer1.setWmn( 1, 1, 1 );

        // third neu
        layer1.setWmn( 2, 0, -1 );
        layer1.setWmn( 2, 1, 1 );

        float[] firstX = new float[]{1,2};
        layer1.nForward( firstX );

        float[] XforL2 = layer1.getZ();

        // 3*neu / 2*weight
        Layer layer2 = new Layer( LType.sigmod_CrossEntropy_Binary, 1 ,3  );
        layer2.setName( "Layer: 2" );
        // first neu
        layer2.setWmn( 0, 0,  1 );
        layer2.setWmn( 0, 1, -1 );
        layer2.setWmn( 0, 2,  1 );


        layer2.setX( XforL2 );
System.out.println( "XforL2"+XforL2 );
        layer2.nForward();

        float[] s = new float[]{1,0};

        float[] Z_S = Tools.vectorSubstZsubS(layer2.getZ(), s);

        layer2.nBackward( Z_S );
System.out.println( "s:" + Arrays.toString( Z_S ));

        float[] eOut=layer2.getEout();
        layer1.nBackward( eOut );
        System.out.println( "eOut (L2):" + Arrays.toString( eOut ));
        System.out.println( layer1 );
        System.out.println( layer2 );
    }
}
