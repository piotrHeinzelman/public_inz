package pl.heinzelman.LayerDeep;

public class LayerPoolingMax {
    protected float[][][] X;
    protected float[][][] dX;
    protected float[][][] Y;
    protected int filterSize;
    protected int stride;

    protected int channels;
    protected int xsize;
    protected int ysize;

    public LayerPoolingMax(int filterSize, Integer stride ) {
        this.filterSize = filterSize;
        this.stride = filterSize ;
    }

    public int getYSize(){
        return ( xsize/filterSize );
    }

    protected void initAry(){
        X  = new float[ channels ][ xsize ][ xsize ];
        dX = new float[ channels ][ xsize ][ xsize ];
    }

    private void setX(float[][][] _x ) {
        this.channels= _x.length;
        this.xsize=_x[0].length;
        this.ysize = getYSize();
        initAry();

        for (int n = 0; n < channels; n++) {
            for (int i = 0; i < xsize; i++) {
                for (int j = 0; j < xsize; j++) {
                    X[n][i][j] = _x[n][i][j];
                    dX[n][i][j] = 0f;
                }
            }
        }
    }

    public float[][][] Forward ( float[][][] _x ) {
        setX( _x );
        float max=0f; int maxX=0; int maxY=0;
        float[][][] Z = new float[channels][ysize][ysize];
            for (int c=0;c<channels;c++){
                for (int i=0; i<ysize; i++){
                    for (int j=0;j<ysize;j++){
                        max=_x[c][i][j]; maxX=0; maxY=0;

                        // *** startPix
                        for ( int x=0;x<filterSize;x++){
                            for (int y=0;y<filterSize;y++){
                                if ( _x[c][i][j]>max ){ maxX=x; maxY=y; max=_x[c][i][j]; }
                            }
                        }
                        Z[c][i][j]=max;
                        dX[c][i*filterSize +maxX][j*filterSize +maxY]=1f;
                    }
                }
            }
        return Z;
    }

    public float[][][] Backward( float[][][] delta ){
        float[][][] OUT = new float[channels][xsize][xsize];
        for (int c=0;c<channels;c++ ){
            for (int i=0;i<ysize;i++){
                for (int j=0;j<ysize;j++) {

                    for (int x=0;x<filterSize;x++){
                        for (int y=0;y<filterSize;y++){
                            OUT[c][i*filterSize +x][j*filterSize +y] = delta[c][i][j] * dX[c][i*filterSize  +x][j*filterSize +y];
                        }
                    }

                }
            }
        }
        return OUT;
    }
}
