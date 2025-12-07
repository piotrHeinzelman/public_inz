package pl.heinzelman.LayerDeep;
import java.util.Random;

public class Neuron2D {
private int m;
private float[][] W;

private final LayerConv parent;
protected final static float mu=0.01f;

public Neuron2D( int m, LayerConv parent ) {
        this.parent=parent;
        this.m=m;
        this.W = new float[m][m];
        for ( int i=0; i<m ; i++ ) {
            this.W[i] = new float[m];
        }
    }

    public void trainW(float [][] dLdF ){
        int m=W.length;
        for ( int i=0;i<m; i++ ){
            for ( int j=0;j<m; j++ ){
                W[i][j]= W[i][j] - (mu * dLdF[i][j]);
            }
        }
    }

    @Override public String toString() {  return "";  }
    public float[][] getMyWeight() { return W; }

    public void rnd( Random rand , float max ){
        for ( int i=0;i< W.length; i++){
            for (int j=0;j<W[0].length; j++){
                W[i][j]= -max +  2*max*(rand.nextFloat()%1f);
            }
        }
    }

    public float[][] getRot180(){
        float[][] Rot180 = new float[ m ][ m ];
            for (int i=0;i<m;i++){
                for (int j=0;j<m;j++){
                    Rot180[m-i-1][m-j-1] = W[i][j];
            }
        }
    return Rot180;
    }
}
