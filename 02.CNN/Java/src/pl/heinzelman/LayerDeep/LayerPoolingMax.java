package pl.heinzelman.LayerDeep;

import java.sql.SQLOutput;
import java.util.Random;

public class LayerPoolingMax {

    protected String name;

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
        this.stride = (stride==null) ? 1 : stride;
    }

    public int getYSize(){
        return 1+(( xsize-filterSize )/stride);
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
                }
            }
        }
    }

    public void setName( String name ) { this.name = name; }

    public float[][][] Forward ( float[][][] _x ) {
        setX( _x );
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
                    float max = X[channel][i*filterSize][j*filterSize];
                    int maxx = 0;
                    int maxy = 0;
                    for (int x=0;x<filterSize;x++){
                        for (int y=0;y<filterSize;y++) {
                            dX[channel][i*filterSize][j*filterSize]=0.0f;
                            if ( max<X[channel][i*filterSize+x][j*filterSize+y] ) { max=X[channel][i*filterSize+x][j*filterSize+y]; maxx=x; maxy=y; }
                        }
                    }
                    dX[channel][i*filterSize+maxx][j*filterSize+maxy]=1.0f;
                    Z[i][j]=max;
                    // ***  ENC
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
                            //System.out.println("delta["+c+"]["+i+"]["+j+"]" + delta[c][i][j] + "dX["+c+"]["+i+"*"+filterSize+"+"+x+"][j*filterSize+y]:" + dX[c][i*filterSize+x][j*filterSize+y] );
                            OUT[c][i*(filterSize) +x][j*(filterSize) +y] = delta[c][i][j] * dX[c][i*filterSize+x][j*filterSize+y];
                        }
                    }
                }
            }
        }
        return OUT;
    }
}
