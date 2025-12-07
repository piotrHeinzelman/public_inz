package pl.heinzelman.neu;
import java.util.Random;

public class LayerSoftmaxMultiClass implements LayerParent {
    private  Neuron[] neurons;
    private  float X[];
    private  float Y[];
    private  float sum=0.0f;
    private  float Z[];
    private  float dFofZ[];
    private  float Eout[]; // S-Z for last

    public LayerSoftmaxMultiClass( int m, int n ) { // m - number of inputs  = input  size X[m]
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
                neu.setWm( m , random.nextFloat() / 5f /* normalization*/ );
            }
        }
    }

    public float[] nForward( float[] _x ) {
        for (int m=0;m<X.length;m++){ X[m]=_x[m]; Eout[m]=0; }
        for (int n = 0; n < neurons.length; n++) {
            Y[n] = neurons[n].Forward( X );
        }
        // Softmax
        int len=Y.length;
        sum = 0.0f;
        float max = 0.0f;
        for ( int i=0;i<len;i++ ){
            if (Y[i]>max) { max=Y[i]; }
        }
        for ( int i=0; i<len; i++ ) {
            Y[i] = (float) Math.exp( Y[i]);
            sum += Y[i];
        }
        for ( int i = 0; i < len; i++ ) {
            Z[i] = Y[i] / sum;
        }
        return Z;
    }

    public float[] nBackward( float[] Ein ){ // S-Z or Ein
        for ( int m=0;m<Eout.length;m++ ){ Eout[m]=0.0f;}

        int i=0;
        for ( ;i<Ein.length;i++ ){ if (Ein[i]!=0) break; }
        float EinTrue =Ein[i];

        int len=Z.length;
        for ( int j=0;j<len;j++ ){
            dFofZ[j] = -1.0f*(Z[i]*Z[j]);
        }
            dFofZ[i] = Z[i]*(1.0f-Z[i]);

        float[] dFofZ_x_EinI_True_I = new float[dFofZ.length];
        for (int j=0;j<dFofZ.length;j++){ dFofZ_x_EinI_True_I[j] = dFofZ[j]*EinTrue; }

        for ( int n=0; n<neurons.length; n++ ){
            neurons[n].Backward( dFofZ_x_EinI_True_I[n]  , Ein[n] ); // EN[i] * dFofZ
            neurons[n].BackwardBias( dFofZ_x_EinI_True_I[n] );
        }
        return Eout;
    }

    public float[] getX() { return X; }
    public float[] getEout() { return Eout; }


    @Override  public String toString() { return "";}

    public float delta_Loss( int correct_label ) {
        float value_correctLabel = Z[correct_label];
        return  (float) -Math.log( value_correctLabel );
    }

    public float[] gradientCNN( float[] out_l, int correct_label ){
        float[] gradient=new float[out_l.length]; //Mat.v_zeros(10);
        for (int i=0;i< out_l.length;i++){ gradient[i]=-out_l[correct_label]*out_l[i]; }
        //gradient[correct_label]=-1/out_l[correct_label];
        gradient[correct_label]=1-out_l[correct_label];
        return gradient;
    }
}
