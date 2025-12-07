package pl.heinzelman.LayerDeep;

public class LayerReLU {
    protected float[][][] dX;
    protected float[][][] X;
    protected int C=0;
    protected int H=0;
    protected int W=0;

    public LayerReLU() {}

    public float[][][] Forward( float[][][] _x  ) {
        C = _x.length;
        H = _x[0].length;
        W = _x[0][0].length;

         X = new float[C][H][W];
        dX = new float[C][H][W];

        for (int n = 0; n < C; n++) {
            for (int x = 0; x < H; x++) {
                for (int y = 0; y < W; y++) {
                    if ( _x[n][x][y] > 0f ) {  X[n][x][y] = _x[n][x][y]; dX[n][x][y]=1.0f;  }
                                      else {   X[n][x][y] = 0f;          dX[n][x][y]=0f;   };
                }
            }
        }
        return X;
    }

    public float[][][] Backward( float[][][] delta ){ return delta; }
}
