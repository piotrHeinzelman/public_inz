package pl.heinzelman.LayerDeep;
import pl.heinzelman.tools.Conv;
import pl.heinzelman.tools.Tools2;
import java.util.Random;

public class LayerConv {
    protected Neuron2D[] filters;
    protected Neuron2D[] biases;

    protected float[][][] X;

    protected float[][][] Y;
    protected int filterNum;
    protected int filterForChannel; // output channel
    protected int filterSize;
    protected int padding=0;
    protected int stride;

    protected int channels; // input channel
    protected int xsize;
    protected int ysize;

    public LayerConv( int filterSize, Integer _filterForChannel, Integer padding, Integer stride ) {
        this.filterSize = filterSize;
        this.filterForChannel = _filterForChannel;
        this.padding = ( padding==null ) ? 0 : padding;
        this.stride = ( stride==null ) ? 1 : stride;
    }

    public void setUpByX( int inchannel, int inputSize ) {
        this.channels = inchannel;
        this.filterNum = filterForChannel*channels;
        this.xsize = inputSize;
        this.ysize = getYSize();
        initAry();
        initFilters();
    }

    private void initFilters(){
        this.filters = new Neuron2D[ filterNum ];
        Random rand = new Random();
        float max=getMaxRand();
        for ( int i=0;i<filterNum;i++ ){
            filters[i] = new Neuron2D( filterSize, this );
            filters[i].rnd(rand, max);
        }
        this.biases = new Neuron2D[ filterForChannel ];
        for ( int i=0;i<filterForChannel;i++ ){
            biases[i] = new Neuron2D( ysize, this );
        }
    }

    private void initAry(){
        X  = new float[ channels ][ xsize ][ xsize ];
        Y = new float[ filterForChannel ][ ysize ][ ysize ];
    }

    public void setUpByX(float[][][] _x ) {
        this.channels = _x.length;
        this.filterNum = filterForChannel*channels;
        this.xsize = _x[0].length;
        this.ysize = getYSize();
        initAry();
        initFilters();
    }


    public float[][][] Forward( float[][][] _x  ) {
        for (int n = 0; n < channels; n++) {
            for (int x = 0; x < xsize; x++) {
                for (int y = 0; y < xsize; y++) {
                    X[n][y][x] = _x[n][y][x];
                }
            }
        }


        float[][][] Y_ = new float[filterForChannel][ysize][ysize];
        float[][][] FtmpOUT = new float[filterNum][ysize][ysize];

       for ( int f=0;f<filterForChannel; f++ ) {
           for ( int c=0;c<channels; c++) {
               FtmpOUT[f*channels+c] = ConvolutionFilterTimesXc( filters[f], X[c] );
           }
       }

        for ( int f=0;f<filterForChannel;f++ ) {
            float[][] bTMP = biases[f].getMyWeight();
            int biasSize = bTMP.length;
            for (int x=0; x<biasSize; x++){
                for (int y=0; x<biasSize; x++) {
                    Y_[f][x][y] = 0.0f;
                }
            }

            for ( int c=0; c<channels; c++ ){
                for (int x = 0; x < ysize; x++) {
                    for (int y = 0; y < ysize; y++) {
                        Y_[f][x][y] += ( FtmpOUT[ f*channels +c ][x][y] );
                    }
                }
            }
        }
        return Y_;
    }

    public float[][] ConvolutionFilterTimesXc(Neuron2D filter, float[][] Xc ) {
        float[][] W = filter.getMyWeight();
        float[][] OUT = new float[ysize][ysize];

        for ( int i=0;i<ysize;i++){
            for ( int j=0;j<ysize;j++) {

                for (int x=0;x<filterSize;x++){
                    for (int y=0;y<filterSize;y++) {
                        OUT[i][j] += ( ( Xc[ i*stride + x ][ j*stride + y ]) * ( W[x][y] ) );
                    }
                }
            }
        }
        return OUT;
    }

    public float[][][] Backward( float[][][] dLdO ) {
        float[][][] dOUT = new float[ channels ][xsize][xsize];

        for (int c=0;c<channels;c++){
            float [][] dOUTc = new float[ xsize ][ xsize ];

            for (int f=0;f<filterForChannel;f++){
                float[][] OUTDeltafc = Conv.fullConv( dLdO[f], filters[ f*channels + c ].getRot180() , 1 /* stride ! */ ); // !!! ?
                dOUTc = Tools2.aryAdd( dOUTc, OUTDeltafc );

                float[][] deltaW = Conv.conv( X[ c ], dLdO[f], 0  );
                filters[ f*channels + c ].trainW( deltaW );
            }
            dOUT[c] = dOUTc;
        }

        for (int f=0;f<filterForChannel;f++){
            if (false ) biases[f].trainW( dLdO[f] );
        }
        return dOUT;
    }

    public String toString(){ if ( filters==null) { return ""; }
        StringBuffer out = new StringBuffer();
        for ( int i=0;i<filterNum;i++){
            out.append( "\n" ); out.append( filters[i].toString() );
        }
        out.append("\nbiases:");
        for ( int i=0;i<filterForChannel;i++){
            out.append( "\n" ); out.append( biases[i].toString() );
        }
        return out.toString(); }

    public int getYSize(){
        return 1+(( xsize+padding+padding-filterSize )/stride);
    }
    protected float getMaxRand(){
        //float inputChannelNum = 6; // inputs
        float inputChannelNum =2 * xsize*xsize*channels ; // inputs
        float outputChannelNum = filterNum; //
        return (float)Math.pow((filterNum / ((inputChannelNum + outputChannelNum) * (filterSize * filterSize))), .5f);
    }

}


