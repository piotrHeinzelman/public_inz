package pl.heinzelman.tools;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.sql.SQLOutput;
import java.util.Arrays;



/*


def readFileX ( fileName ):
    file=open( fileName, 'rb' )
    data=np.fromfile( fileName, np.uint8, percent*8*240*240*3, '')
    data=data.reshape(percent*8, 240*240*3)
    data=(data/255)
    file.close()
    return data

def readFileY ( fileName ):
    file=open( fileName, 'rb' )
    len=percent*8
    data=np.fromfile( fileName, np.uint8, len, '' )
    file.close()
    return data


start1=time.time()

trainX = readFileX ('../../../../inz_Hidden/SAS/out.bin' )
trainY = readFileY ('../../../../inz_Hidden/SAS/out.class' )

trainX = trainX.astype("float32")
trainY = trainY.astype("int")

trainX = trainX.reshape(percent*8, 3, 240,240).astype("float32") # / 255
#trainY = trainY.reshape(percent*8)



*/




public class Tools2 {

    private static String path="../../../data/";
    private static String testXname="output.bin";
    private static String testYname="output.class";
    private static String trainXname="output.bin";
    private static String trainYname="output.class";

    private byte[] trainYfile=null;
    private byte[] testYfile=null;
    private float[][] trainY=null;
    private float[][] testY=null;

    private float[][][][] trainX=null;
    private float[][][][] testX=null;



    public  void prepareData3C( int percent ){
        int H=240; int W=240; int C=3; String out="";

        try {
            trainYfile =  loadBin( path + trainYname,  0, percent*8*2 ); // offset=8, size=percent*600  // OK
            testYfile =  loadBin( path + testYname,   0, percent*8*2 );   // offset=8, size=percent*100 // OK

            trainY = new float[percent*8][2];
            testY  = new float[percent*8][2];
            //train Y
            for (int i=0;i<percent*8;i++){
                trainY[i][0] = 1.0f * trainYfile[i*2 + 0]  ;  // = new float[]{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                trainY[i][1] = 1.0f * trainYfile[i*2 + 1];
            }
            // test Y
            for (int i=0;i<percent*8;i++){
                trainY[i][0] = 1.0f * trainYfile[i*2 + 0]  ;
                trainY[i][1] = 1.0f * trainYfile[i*2 + 1];
            }
            //System.out.println( trainY[4][0]  + " : " +trainY[4][1]  );

            byte[] trainXfile = loadBin(path + trainXname, 0, percent*8 * 240*240*3 );// offset=16 size=percent*784*600

            trainX=new float[percent*8][3][240][240];
            for (int n=0;n<percent*8;n++) {
                for (int h=0;h<H;h++){
                    for (int w=0;w<W;w++) {
                       for (int c = 0; c < C; c++) {

                            byte val =  trainXfile[n *W*H*C + h * W*C + w*C + c];

                            //System.out.println( val );
                           trainX[n][c][h][w] = 0.01f*val;///255.0f;///256.0f; //0-1
                          // System.out.println(  trainX[n][c][h][w] );
                         //   out += " " + Byte.toUnsignedInt(val);

                      }
                    }
                   // System.out.println(out);
                }
            }

            byte[] testXfile = loadBin(path + trainXname, 0, percent*8 * 240*240*3 );// offset=16 size=percent*784*600

            testX=new float[percent*8][3][240][240];
            for (int n=0;n<percent*8;n++) {
                for (int h=0;h<H;h++){
                    for (int w=0;w<W;w++) {
                        for (int c = 0; c < C; c++) {

                            byte val =  trainXfile[n *W*H*C + h * W*C + w*C + c];

                            //System.out.println( val );
                            testX[n][c][h][w] = 0.01f*val;///255.0f;///256.0f; //0-1
                            // System.out.println(  trainX[n][c][h][w] );
                            //   out += " " + Byte.toUnsignedInt(val);

                        }
                    }
                    // System.out.println(out);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }



    public  void prepareData( int percent ){

        try {
            trainYfile =  loadBin( path + trainYname,  0, percent*8 ); // offset=8, size=percent*600  // OK
            testYfile =  loadBin( path + testYname,   0, percent*8 );   // offset=8, size=percent*100 // OK

            trainY = new float[percent*8][];
            testY  = new float[percent*8][];
            //train Y
            for (int i=0;i<percent*8;i++){
                trainY[i] = new float[]{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                trainY[i][ trainYfile[i] ]=1.0f;
            }
            // test Y
            for (int i=0;i<percent*8;i++){
                testY[i] = new float[]{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                testY[i][ testYfile[i] ]=1.0f;
            }

/*
            byte[] trainXfile = loadBin(path + trainXname, 16, percent * 240*240 * 8);// offset=16 size=percent*784*600
            trainX=new float[percent*8][240*240*3];
            for (int i=0;i<percent*8;i++) {
                int off=i*240;
                for (int j=0;j<240*3;j++){
                    trainX[i][j]=Byte.toUnsignedInt( trainXfile[off+j] )/254.0f;///256.0f; //0-1
                    //System.out.println( trainX[i][j] );
                }
            }

            byte[] testXfile =   loadBin( path + testXname,  16, percent*240*240*8 );   // offset=16, size=percent*784*100
            testX=new float[percent*8][240*240];
            for (int i=0;i<percent*8;i++) {
                int off=i*240;
                for (int j=0;j<240*3;j++){
                    testX[i][j]=Byte.toUnsignedInt( testXfile[off+j] )/254.0f;///256.0f;
                }
            }
*/
            // show data:
            //for ( int i=0;i<100;i++ ) {
            //    saveVectorAsImg( trainX[i], trainYfile[i] +"_key_is_" + i );
            //}
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    public static byte[] loadBin( String filename, int offset, int len ) throws IOException {
        byte[] bytesBuf = new byte[ len ];
        File f = new File( filename );
        FileInputStream fis = new FileInputStream( filename );
        fis.skip(offset);
        fis.read( bytesBuf, 0, len );
        return bytesBuf;
    }

    public float[][] convertToSquare240x240(float[] vector ){
        float[][] square = new float[240][240];
        for ( int y=0;y<240;y++ ){
            for ( int x=0;x<240;x++ ) {
                square[y][x]=vector[x+240*y];
            }
        }
        return square;
    }


    public int getIndexMaxFloat(float[] floats ){
        int maxI=0;
        float val=0f + floats[0];
        for ( int i=1;i<floats.length; i++ ){
            if ( 0f+floats[i] > val ) { val=floats[i]; maxI=i; }
        }
        return maxI;
    }

    public float[][] getTrainY() { return trainY; }
    public float[][] getTestY()  { return testY;  }
    public float[][][][] getTrainX() { return trainX; }
    public float[][][][] getTestX()  { return testX;  }

    public  String AryToString( float[]X ){
        return Arrays.toString( X );
    }
    public  String AryToString( float[][]X ){
        StringBuffer out = new StringBuffer();
        out.append("[");
        for (int i=0;i<X.length;i++) {
            out.append(""+Arrays.toString( X[i])+"\n");
        }
        out.append("]");
        return out.toString();
    }
    public  String AryToString( float[][][]X ){
        StringBuffer out = new StringBuffer();
        out.append("[");
        for (int i=0;i<X.length;i++) {
            out.append(i+": " + AryToString( X[i] ) + "\n");
        }
        out.append("]");
        return out.toString();
    }

    public float[][] gradientCNN( float[][] out_l, int correct_label ){

        //BACKWARD PROPAGATION --- STOCHASTIC GRADIENT DESCENT
        //gradient of the cross entropy loss

        float[][] gradient=new float[1][10]; //Mat.v_zeros(10);
        for (int i=0;i<10;i++){ gradient[0][i]=0.0f; }
        gradient[0][correct_label]=-1/out_l[0][correct_label];
        return gradient;
    }

    public float getCeLoss_CNN( float [] out_l , int correct_label ){
        //ce_loss += (float) -Math.log(out_l[0][correct_label]);
        return (float) -Math.log(out_l[correct_label]);
    }



    public  void echo ( Number n ) { System.out.println( n.toString() ); }
    public  void echo ( float[] v  ) { System.out.println( Tools.AryToString( v ) ); }
    public  void echo ( float[][] v  ) { System.out.println( Tools.AryToString( v ) ); }
    public  void echo ( float[][][] v  ) { System.out.println( Tools.AryToString( v ) ); }

    public  void echo ( String name, Number n ) { System.out.println( name + " : " + n.toString() ); }
    public  void echo ( String name, float[] v  ) { System.out.println( name + " : " + Tools.AryToString( v ) ); }
    public  void echo ( String name, float[][] v  ) { System.out.println( name + " : " + Tools.AryToString( v ) ); }
    public  void echo ( String name, float[][][] v  ) { System.out.println( name + " : " + Tools.AryToString( v ) ); }

    public static void printTable2( int[][] table ){
        System.out.println( " incorrect class  ->  [0]  |  [1]  |  [2]  |  [3]  |  [4]  |  [5]  |  [6]  |  [7]  |  [8]  |  [9]\n" );
        for (int y=0;y<table[0].length;y++){
            System.out.println("True class    ("+y+")   " + printRow3( table[y] ));
        }
    }

    public static String printRow3( int[] row ){
        String out="";
        for (int x=0;x<row.length;x++){
            out += "  "+ ((row[x])>9 ? "" : " " ) +  ( row[x]==0 ? "." : row[x] )   +"   |";
        }
        return ( out );
    }

}








