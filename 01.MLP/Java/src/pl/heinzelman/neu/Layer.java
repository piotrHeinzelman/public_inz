package pl.heinzelman.neu;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static java.awt.image.BufferedImage.TYPE_INT_RGB;

// Forward
// y = Neu[n]*X[]; /y is scalar, a number !
// z = F( y ) ;    /z is scalar, a number !

// Backward
// eIn = s-z; // last layer
// Eout[i]  = Ein * dF(Z) * [W^T]
// <- Sum(Eout)

//Weight
//dF(Z)*E

public class Layer {
    private String name;
    private  LType lType;
    private  Neuron[] neurons;
    private  float X[];
    private  float Y[];
    private  float Z[];
    private  float dFofZ[];
    //private  float Ein[]; // S-Z for last
    private  float Eout[]; // S-Z for last

    public Layer( LType lType, int n, int m ) { // n - number of neurons & output size Y[n], Z[n]
        this.lType=lType;                        // m - number of inputs  = input  size X[m]
        this.neurons = new Neuron[n];
        for (int i=0; i<n; i++){
            this.neurons[i]=new Neuron(m, this);
        }
        X = new float[m];
        Y = new float[n];
        Z = new float[n];
        dFofZ = new float[n];
        //Ein = new float[n];
        Eout = new float[m];
    }


    public void setWmn( int n, int m, float wji ){
        neurons[n].setWm( m, wji );
    }


    public void rnd(){
        Random random=new Random();
        float normalization=1;//this.X.length/5.0f;
        for ( Neuron neu : neurons ) {
            for ( int m=0; m<X.length; m++ ) {
                neu.setWm( m , (float)( -1.0f+2.0f*random.nextFloat() / normalization )  );
            }
            //System.out.println( neu );
        }
    }


    public float[] nForward() {
        switch (lType) {
            case sigmod:
            case sigmod_CrossEntropy_Binary:{
                for (int n = 0; n < neurons.length; n++) {
                    Y[n] = neurons[n].Forward(X);
                    Z[n] = F(Y[n]);
                    dFofZ[n] = dF(Z[n]);

                }
                for (int m=0;m<Eout.length;m++){ Eout[m]=0; }
                return Z;
            }


            case softmaxMultiClass: {
                int len = neurons.length;
                float sum = 0.0f;
                float max = Y[0] = neurons[0].Forward(X);
                for (int i=1;i<len;i++){
                    dFofZ[i] = 1;
                    Y[i]=neurons[i].Forward(X);
                    if (Y[i]>max) { max=Y[i]; }
                }
                for (int i = 0; i < len; i++) {
                    Y[i] = (float) Math.exp( Y[i]-max );
                    sum += Y[i];
                }
                for (int i = 0; i < len; i++) {
                    Z[i] = Y[i] / sum;
                }
                return Z;
            }

            default: {
                return Z;
            }
        }
    }


    public void nBackward( float[] Ein ){ // S-Z or Ein
        for ( int i=0;i<neurons.length;i++ ){ Eout[i]=0.0f; }
        for ( int n=0; n< neurons.length; n++ ){
            neurons[n].Backward( Ein[n] * dFofZ[n] );
        }
    }

    private float F ( float y ){
        float z;
        switch (this.lType) {
            case sigmod:
            case sigmod_CrossEntropy_Binary:
                { z = (float) (1.0f/(1.0f + Math.exp( -y ))); break; }
            case linear:
                default: { z=y; break; }
        }
        return z;
    }

    private float dF ( float z ){
        float df;
        switch (lType) {
            case sigmod:
                { df = z*(1-z); break; }
            case linear:
            case sigmod_CrossEntropy_Binary:
            default: { df=1; break; }
        }
        return df;
    }



    // getters / setters
    public void setX( float[] _x ) {
        for ( int m=0; m<X.length; m++ ){
            this.X[m] = _x[m];
        }
    }

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

    //@Deprecated
    public float[] getY() { return Y; }

    //@Deprecated
    public float[] getNeuronWeight( int i ){
        return neurons[i].getMyWeight();
    }

    //@Deprecated
    public void saveAllWeightAsImg( String nameSuffix ){
        int width = Y.length; // size of Y, number of neurons
        int height = X.length; // size of X , number of input signal
        BufferedImage image = new BufferedImage( width , height , TYPE_INT_RGB );

        float k=-255;
        File file = new File("weights_"+nameSuffix+".png");
        for ( int i=0; i<width; i++){

            // my Neuros is...
            float[] neuronWeights = getNeuronWeight(i);
            for (int j=0;j<height;j++){
                float aDouble = neuronWeights[j];
                //    k=aDouble*255;
                //image.setRGB( i, j, (int) aDouble );

                //if ( k<256 ) { k++; }
                k = aDouble*90;
                k=k*255;
                k=k%255;
                k=127-k;
                float c=0; float R=0; float G=0; float B=0;
                if ( k < 0 ) { // RED FF0000, 50% 804040, 0% 000000   R F->12->8->4->0, G,B 0->2->4->2->0
                     R=-k;
                     G=0;
                     B=0;
                } else {
                     R=0;  G=k;  B=k;
                }
                c=R*256*256+G*255+B;
                image.setRGB( i, j,  (int) c );
                //System.out.println( k );
            }
        }
        try {
            ImageIO.write(image ,  "png", file );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public float BinaryCrossEntropy( float[] s, float[] z ){ // s(yi) - target class : z(p) - reply of net : , yi = true label (0 or 1)
        float sum=0.0f;  // BCE = -1/n * SUM ( yi*log(pi) + (1-yi)*log(1-pi)   )
        for ( int i=0;i<s.length;i++ ){
            sum += s[i] * Math.log ( z[i] ) + ( 1-s[i]) *Math.log (1-z[i]);
        }
        return sum/( s.length );
    }

}
