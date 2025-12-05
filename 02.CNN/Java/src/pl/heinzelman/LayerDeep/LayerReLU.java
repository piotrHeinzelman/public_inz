package pl.heinzelman.LayerDeep;

public class LayerReLU {

    protected String name;

    protected float[][][] X;
    protected float[][][] dX;
    protected float[][][] Y;
    protected int channels;
    protected int xsize;



    public LayerReLU() {}

    public int getYSize(){
        return  xsize;
    }

    protected void initAry(){
        X  = new float[ channels ][ xsize ][ xsize ];
        dX = new float[ channels ][ xsize ][ xsize ];
    }

    public void setX(float[][][] _x ) {
        this.channels= _x.length;
        this.xsize=_x[0].length;
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
        float[][][] Z = new float[channels][xsize][xsize];
            for (int c=0;c<channels;c++){
                Z[c] = forwardChannel( c );
            }
        return Z;
    }

    public float[][] forwardChannel ( int channel ) {
        float[][] Z = new float[xsize][xsize];
            for (int i=0;i<xsize;i++){
                for (int j=0;j<xsize;j++) {
                    if ( X[channel][i][j]>0 ){
                        Z[i][j]=X[channel][i][j];
                        dX[channel][i][j]=1f;
                    }
                    else {
                        Z[i][j]=0f;
                        dX[channel][i][j]=0f;
                    }
                }
            }
        return Z;
    }



    public float[][][] Backward( float[][][] delta ){ // delta = (s-z)*d....
        float[][][] OUT = new float[channels][xsize][xsize];
        for (int c=0;c<channels;c++ ){
            // every channel

            for (int i=0;i<xsize;i++){
                for (int j=0;j<xsize;j++) {

                    OUT[c][i][j] = delta[c][i][j] * dX[c][i][j];

                }
            }
        }
        return OUT;
    }
}
