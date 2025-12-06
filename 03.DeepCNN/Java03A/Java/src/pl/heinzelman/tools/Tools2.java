package pl.heinzelman.tools;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;



public class Tools2 {

    private static String path="../../data/";
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
            byte[] trainXfile = loadBin(path + trainXname, 0, percent*8 * 240*240*3 );// offset=16 size=percent*784*600

            trainX=new float[percent*8][3][240][240];
            for (int n=0;n<percent*8;n++) {
                for (int h=0;h<H;h++){
                    for (int w=0;w<W;w++) {
                       for (int c = 0; c < C; c++) {

                            byte val =  trainXfile[n *W*H*C + h * W*C + w*C + c];

                           trainX[n][c][h][w] = 0.01f*val;
                      }
                    }
                }
            }

            byte[] testXfile = loadBin(path + trainXname, 0, percent*8 * 240*240*3 );// offset=16 size=percent*784*600

            testX=new float[percent*8][3][240][240];
            for (int n=0;n<percent*8;n++) {
                for (int h=0;h<H;h++){
                    for (int w=0;w<W;w++) {
                        for (int c = 0; c < C; c++) {

                            byte val =  trainXfile[n *W*H*C + h * W*C + w*C + c];

                            testX[n][c][h][w] = 0.01f*val;
                        }
                    }
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

    public  String AryToString( float[][]X ){
        StringBuffer out = new StringBuffer();
        out.append("[");
        for (int i=0;i<X.length;i++) {
            out.append(""+Arrays.toString( X[i])+"\n");
        }
        out.append("]");
        return out.toString();
    }



    public float[][] gradientCNN( float[][] out_l, int correct_label ){
        float[][] gradient=new float[1][out_l.length]; //Mat.v_zeros(10);
        for (int i=0;i< out_l.length;i++){ gradient[0][i]=0.0f; }
        gradient[0][correct_label]=-1/out_l[0][correct_label];
        return gradient;
    }

    public float[] gradientCNN( float[] out_l, int correct_label ){
        System.out.println( "!!" + Arrays.toString(out_l) );
        float[] gradient=new float[out_l.length];
        gradient[0]=0.00000000001f; gradient[1]=0.000000000001f;//Mat.v_zeros(10);
        //for (int i=0;i< out_l.length;i++){ gradient[i]=0.0f; }
        if (out_l[correct_label]*out_l[correct_label]<0.00000001f ) {out_l[correct_label]=0.000001f;}
        gradient[correct_label]=-1.0f/out_l[correct_label];
        return gradient;
    }

    public float getCeLoss_CNN( float [] out_l , int correct_label ){
        return (float) -Math.log(out_l[correct_label]);
    }








    public String AryToString ( float[][][] T ){
        StringBuffer out = new StringBuffer();
        if (T==null) return "";
        for ( int i=0;i<T.length;i++ ){
            out.append( AryToString( T[i] ));
        }
        return out.toString();
    }








    public static float[][] aryAdd( float[][] A, float[][] B){
        float [][] C = new float[A.length][A[0].length];
        for ( int i=0;i<A.length;i++ ){
            for ( int j=0;j<A[0].length;j++ ) {
                C[i][j] = A[i][j]+B[i][j];
            }
        }
        return C;
    }
}








