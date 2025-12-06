package pl.heinzelman.LayerDeep;

public class LayerReLU {

    protected String name;

    protected float[][][] dX;
    protected int C=0;
    protected int H=0;
    protected int W=0;


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
        C = _x.length;
        H = _x[0].length;
        W = _x[0][0].length;

        float[][][] X = new float[C][H][W];
        dX = new float[C][H][W];

        for (int n = 0; n < C; n++) {
            for (int x = 0; x < H; x++) {
                for (int y = 0; y < W; y++) {
                    if ( _x[n][x][y] > 0 ) {  X[n][x][y] = _x[n][x][y]; dX[n][x][y]=1.0f;  }
                                      else {  X[n][x][y] = 0f;   dX[n][x][y]=0f;   };
                }
            }
        }
        return X;
    }


    public float[][][] Backward( float[][][] delta ){ // delta = (s-z)*d....
        int delta_size=delta[0].length;
        float[][][] OUT = new float[ C ][ W ][ H ];
        for (int c=0;c<C;c++ ){
            for (int i=0;i<H;i++){
                for (int j=0;j<W;j++) {
                     if ( W <= delta_size )  OUT[c][i][j] = delta[c][i][j] * dX[c][i][j];
                     else {
                         OUT[c][i][j] = dX[c][i][j];
                         System.out.println( "c:" + c + ", i:" + i + ", j:" + j + ", C:" + H + ", W:" + dX[0][0].length + ", deltaW:" + delta[0][0].length );
                       }
                }
            }
        }
        return OUT;
    }
}
