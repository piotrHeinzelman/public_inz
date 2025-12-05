package pl.heinzelman.neu;

import java.util.Arrays;
import java.util.Random;

// Forward
// y = Neu[n]*X[]; /y is scalar, a number !
// z = F( y ) ;    /z is scalar, a number !

// Backward
// eIn = s-z; // last layer
// Eout[i]  = Ein * dF(Z) * [W^T]
// <- Sum(Eout)

//Weight
//dF(Z)*E

public class LayerSigmoidFullConn implements LayerParent {
    private String name;
    private  Neuron[] neurons;
    private  float X[];
    private  float Y[];
    private  float Z[];
    private  float dFofZ[];
    private  float Eout[]; // S-Z for last

    public LayerSigmoidFullConn( int m, int n ) { // m - number of inputs  = input  size X[m]
                                                  // n - number of neurons & output size Y[n], Z[n]
        this.neurons = new Neuron[n];
        for (int i=0; i<n; i++){
            this.neurons[i]=new Neuron( m, this );
        }
        X = new float[m];
        Y = new float[n];
        Z = new float[n];
        dFofZ = new float[n];
        Eout = new float[m];
        rnd();
    }


    private void rnd(){
        Random random=new Random();
        float normalization=X.length;
        for ( Neuron neu : neurons ) {
            for ( int m=0; m<X.length; m++ ) {
                neu.setWm( m , random.nextFloat() / normalization );
            }
        }
    }


    public float[] nForward( float[] _x ) {
        for (int m=0;m<X.length;m++){ X[m]=_x[m]; Eout[m]=0; }
        for (int n = 0; n < neurons.length; n++) {
            Y[n] = neurons[n].Forward(X);
            Z[n] = F(Y[n]);
            dFofZ[n] = dF(Z[n]);
        }
        return Z;
    }

    public float[] nBackward( float[] Ein ){ // S-Z or Ein
        for ( int m=0;m<Eout.length;m++ ){ Eout[m]=0.0f;}
        for ( int n=0; n<neurons.length; n++ ){
            neurons[n].Backward( Ein[n] * dFofZ[n], Ein[n] );
        }
        return Eout;
    }

    private float F ( float y ){
        float z;
        z = (float) (1.0f/(1.0f + Math.exp( -y )));
        return z;
    }

    private float dF ( float z ){
        float df;
        df = z*(1.0f-z);
        return df;
    }


    // getters / setters
    public float[] getZ() { return Z; }
    public float[] getX() { return X; }
    public float[] getEout() { return Eout; }


    @Override
    public String toString() {
        return "\nLayer{" + name + " : "+
                "\nneurons=" + Arrays.toString(neurons) +
                "\nX=" + Arrays.toString(X) +
                "\nY=" + Arrays.toString(Y) +
                "\nZ=" + Arrays.toString(Z) +
                "\ndZ=" + Arrays.toString(dFofZ) +
                '}';
    }

    public void setName(String name) { this.name = name; }
    public void setWmn( int n, int m, float wji ){
        neurons[n].setWm( m, wji );
    }


    //@Deprecated
    public float[] getY() { return Y; }

    //@Deprecated
    public float[] getNeuronWeight( int i ){
        return neurons[i].getMyWeight();
    }

    public Neuron getNeuron(int i) {
        return neurons[i];
    }
    public void setAllWeight( float[][] w ){
        for (int i=0;i<neurons.length;i++){
            neurons[i].setWeights( w[i] );
        }
    }

    public float[] getdZ(  float [] target ){
        // https://www.youtube.com/watch?v=vbUozbkMhI0
        // dZ3 = (A3-Y)
        //
        float[] dZ = new float[Z.length];
        for ( int i=0; i<Z.length; i++ ){
            dZ[i] = ( Z[i] - target[i] );
        }
        return dZ;
    }

    public float[] getdZ(  int targetClass ){
        float[] Vtarget = new float[Z.length];
        for (int i=0;i<Z.length; i++){
            Vtarget[i]=0.0f;
        }
        Vtarget[targetClass]=1.0f;
        return getdZ( Vtarget );
    }

}
