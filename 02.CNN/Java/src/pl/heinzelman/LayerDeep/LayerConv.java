package pl.heinzelman.LayerDeep;

import pl.heinzelman.tools.Conv;
import pl.heinzelman.tools.Tools;

import javax.tools.Tool;
import java.util.Random;

//
//  Fupdate  = F - u dL/dF ; = Conv ( X, delta )     ; // delta = dL/dO
//  deltaOut = dL/dX = FullConv ( rot180 F , delta ) ; // delta = dL/dO
//

// X [num][i][j]
// F [num][m][n]
// D [num][x][y]

//                        F11  F1c1    Fnc*Xc
//  Xc1 [ ]---------+----> [ ] -----     b1
//                 |                \   |
//                 |      F12  F1c2  ( + ) ----->
//  Xc2 [ ]---+--- | ---> [ ] ----- /
//           |     |
//           |     |
//           |     |      F21  F2channel1
//           |     +----> [ ] ------    b2
//           |     |                \   |
//           |            F22  F2c2   ( + ) ----->
//           +----------> [ ] ------/
//           |
//           |     |      F31  F3c1     b3
//           |     ---->  [ ] ----- \   |
//           |                       ( + ) ------>
//           |            F32  F3c2 /
//           ---------->  [ ] -----
//
//
//           Filter ( FilterNum*Channels + Channel )
//
//           out[n][c] = out[ n*c ] = Fnc * Xc                       separated multiply X & filter
//                       out[n]     = SUM by c ( out[n][c] )  +bn    mass add filter outputs
// bias is OUTPUT SIZE ! ( ysize )
// Filter if Any SIZE !!!

public class LayerConv {
    protected String name;
    protected Neuron2D[] filters;
    protected Neuron2D[] biases;

    protected float[][][] X;

    protected float[][][] Y;
    protected int filterNum; // 6  ( input channel * filterForChannel )  for (n) for (c) filters(  n*channels + c ) // n: filterForChannel = output size // c: channel = input size
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
        //if ( padding!=0 ) { _x = Conv.extendAry( _x, padding ); }
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

       // MASS multiply Fnc * Xc
       for ( int f=0;f<filterForChannel; f++ ) {
           for ( int c=0;c<channels; c++) {
               FtmpOUT[f*channels+c] = ConvolutionFilterTimesXc( filters[f], X[c] );
               // System.out.println( Tools.AryToString(  filters[f].getMyWeight() ));
               // System.out.println( Tools.AryToString(  X[c] ));
               //System.out.println( Tools.AryToString(  ConvolutionFilterTimesXc(filters[f], X[c]) ) );
           }
       }

       // by set of filter ( output channel )
        for ( int f=0;f<filterForChannel;f++ ) {
            // init bias // copy bias to Y[f]
            float[][] bTMP = biases[f].getMyWeight();
            int biasSize = bTMP.length;
            for (int x=0; x<biasSize; x++){
                for (int y=0; x<biasSize; x++) {
                    Y_[f][x][y] = 0.0f;//bTMP[x][y];
                }
            }

            // sum FOUT
            for ( int c=0; c<channels; c++ ){
                for (int x = 0; x < ysize; x++) {
                    for (int y = 0; y < ysize; y++) {
                        Y_[f][x][y] += ( FtmpOUT[ f*channels +c ][x][y] );
                    }
                }
            }
        }
        //System.out.println( Tools.AryToString( getNeuron(0).getMyWeight() ));
        //System.out.println( Tools.AryToString( Y ));
        //if (true) throw new RuntimeException("!");
        return Y_;
    }

    public float[][] ConvolutionFilterTimesXc(Neuron2D filter, float[][] Xc ) {
        float[][] W = filter.getMyWeight();
        float[][] OUT = new float[ysize][ysize];

        for ( int i=0;i<ysize;i++){
            for ( int j=0;j<ysize;j++) {

                for (int x=0;x<filterSize;x++){
                    for (int y=0;y<filterSize;y++) {
                        // every channel (c)
                        // target output[i][j]
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
                // every filter f on channel c

                // print('outputDelta SUM: for all channel:  SEND FORWARD ')
                // print( signal.convolve2d( delta, f, "full" ) )
                float[][] OUTDeltafc = Conv.fullConv( dLdO[f], filters[ f*channels + c ].getRot180() , 1 /* stride ! */ ); // !!! ?
                dOUTc = Tools.aryAdd( dOUTc, OUTDeltafc );

                // print('Kernel gradient SUM: every channel: UPDATE WEIGHT ')
                // print( signal.correlate2d( x, delta, "valid") )
                float[][] deltaW = Conv.conv( X[ c ], dLdO[f], 0  );
                //System.out.println( "UPDATE WEIGHTS:" );
                //System.out.println( "Filter:" + filters[ f*channels + c ].toString() + ", \n\ndeltaW: " + Tools.AryToString( deltaW  ));
                filters[ f*channels + c ].trainW( deltaW );
                //System.out.println( "Updated Filter:" + filters[ f*channels + c ].toString() );
            }
            dOUT[c] = dOUTc;
            //System.out.println( "dOUT: " + Tools.AryToString( dOUT ) );
        }

        for (int f=0;f<filterForChannel;f++){
            if (false ) biases[f].trainW( dLdO[f] );
        }

        return dOUT;
    }

    // ****************

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

    public void setName( String name ) { this.name = name; }
    public Neuron2D getNeuron(int i){ return filters[i]; }
    private float relu(float x){
        return (x>0)? x : 0;
    }
    public int getYSize(){
        return 1+(( xsize+padding+padding-filterSize )/stride);
    }
    protected float getMaxRand(){
        float inputChannelNum = 6; // inputs
        float outputChannelNum = filterNum; //
        return (float)Math.pow((filterNum / ((inputChannelNum + outputChannelNum) * (filterSize * filterSize))), .5f);
    }

}


