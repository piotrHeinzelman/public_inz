package pl.heinzelman.neu;

public class Neuron {
    private float bias=0.0f;
    private final float[] W;
    private final LayerParent parent;
    private final static float mu=0.0001f;

    public Neuron( int m, LayerParent parent ) {
        this.parent=parent;
        this.W = new float[m];
    }


    public void setWm( int m, float wji ){ W[m] = wji; }

    public float Forward( float[] X ) {
        float res=bias;
        for ( int m=0; m<W.length; m++ ) {
            res+= X[m]*W[m];
        }
        return res;
    }

    public void Backward( float en_x_dFIznI , float eIn ) {
        float[] X = parent.getX();
        for ( int m=0; m<W.length; m++ ) {
            parent.getEout()[m] += ( W[m] * en_x_dFIznI );
            W[m] = W[m] - ( mu * en_x_dFIznI * X[m] );
        }
    }

    public void BackwardBias( float en_x_dFIznI ) {
        bias = bias - ( mu * en_x_dFIznI );
    }

    @Override
    public String toString() {
        return "";
    }
}
