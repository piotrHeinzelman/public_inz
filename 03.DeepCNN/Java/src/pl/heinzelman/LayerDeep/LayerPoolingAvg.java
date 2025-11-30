package pl.heinzelman.LayerDeep;

public class LayerPoolingAvg {

    protected String name;

    protected float[][][] X;
    protected float dX;
    protected float[][][] Y;
    protected int filterSize;
    protected int stride;

    protected int channels;
    protected int xsize;
    protected int ysize;

    protected float oneOverFilterSize2;


    public LayerPoolingAvg(int filterSize, Integer stride ) {
        this.filterSize = filterSize;
        this.stride = (stride==null) ? 1 : stride;
        oneOverFilterSize2 = 1.0f/(filterSize*filterSize);
    }

    public int getYSize(){
        return 1+(( xsize-filterSize )/stride);
    }

    protected void initAry(){
        X  = new float[ channels ][ xsize ][ xsize ];
        dX = oneOverFilterSize2;
    }

    public void setX(float[][][] _x ) {
        this.channels= _x.length;
        this.xsize=_x[0].length;
        this.ysize = getYSize();
        initAry();

        for (int n = 0; n < channels; n++) {
            for (int i = 0; i < xsize; i++) {
                for (int j = 0; j < xsize; j++) {
                    X[n][i][j] = _x[n][i][j];
                }
            }
        }
    }

    public void setName( String name ) { this.name = name; }

    public float[][][] Forward () {
        float[][][] Z = new float[channels][ysize][ysize];
            for (int c=0;c<channels;c++){
                Z[c] = forwardChannel( c );
            }
        return Z;
    }

    public float[][] forwardChannel ( int channel ) {
        float[][] Z = new float[ysize][ysize];
            for (int i=0;i<ysize;i++){
                for (int j=0;j<ysize;j++) {

                    // MAX
                    // --- max --- X[i][j] : X[i+size][j+size]
                    float sum = 0f;
                    for (int x=0;x<filterSize;x++){
                        for (int y=0;y<filterSize;y++) {
                            sum+=X[channel][i*filterSize+x][j*filterSize+y];
                        }
                    }
                    Z[i][j]=sum*oneOverFilterSize2;
                }
            }
        return Z;
    }



    public float[][][] Backward( float[][][] delta ){ // delta = (s-z)*d....
        float[][][] OUT = new float[channels][xsize][xsize];
        for (int c=0;c<channels;c++ ){
            // every channel

            for (int i=0;i<ysize;i++){
                for (int j=0;j<ysize;j++) {
                    //System.out.println( "i: " + i+", j:" + j );
                    // delta[i][i]
                    // every returned error value
                    // * Filter
                    for (int x=0;x<filterSize;x++){
                        for (int y=0;y<filterSize;y++){
                            OUT[c][i*(filterSize) +x][j*(filterSize) +y] = delta[c][i][j] * dX;
                        }
                    }
                }
            }
        }
        return OUT;
    }
}
