package pl.heinzelman.LayerDeep;

public class LayerReLU {

    protected String name;


    protected float[][][] dX;
    protected int c;
    protected int h;
    protected int w;


    public LayerReLU() {}







    public void setName( String name ) { this.name = name; }
/*
    public float[][][] Forward ( ) {
        float[][][] Z = new float[channels][xsize][xsize];
            for (int c=0;c<channels;c++){
                Z[c] = forwardChannel( c );
            }
        return Z;
    }
*/


    public float[][][] Forward( float[][][] _x  ) {
        c = _x.length;
        h = _x[0].length;
        w = _x[0][0].length;

        float[][][] X = new float[c][h][w];
        float[][][] dX = new float[c][h][w];
        for (int n = 0; n < c; n++) {
            for (int x = 0; x < h; x++) {
                for (int y = 0; y < w; y++) {
                    if ( _x[n][x][y] > 0 ) {  X[n][x][y] = _x[n][x][y]; ; dX[n][x][y]=1.0f;  }
                                      else {  X[n][x][y] = 0f; ; dX[n][x][y]=0f;   };
                }
            }
        }
        return X;
    }








    public float[][][] Backward( float[][][] delta ){ // delta = (s-z)*d....
        float[][][] OUT = new float[c][h][w];
        for (int c=0;c<c;c++ ){
            // every channel
            for (int i=0;i<h;i++){
                for (int j=0;j<w;j++) {

                    OUT[c][i][j] = delta[c][i][j] * dX[c][i][j];

                }
            }
        }
        return OUT;
    }
}
