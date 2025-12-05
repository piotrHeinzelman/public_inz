package pl.heinzelman.tools;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;


public class Tools2 {

    private static String path="../../../data/";
    private static String testXname="t10k-images-idx3-ubyte";
    private static String testYname="t10k-labels-idx1-ubyte";
    private static String trainXname="train-images-idx3-ubyte";
    private static String trainYname="train-labels-idx1-ubyte";

    private byte[] trainYfile=null;
    private byte[] testYfile=null;
    private float[][] trainY=null;
    private float[][] testY=null;

    private float[][] trainX=null;
    private float[][] testX=null;


    public  void prepareData( int percent ){

        try {
            trainYfile =  loadBin( path + trainYname,  8, percent*600 ); // offset=8, size=percent*600  // OK
            testYfile =  loadBin( path + testYname,   8, percent*100 );   // offset=8, size=percent*100 // OK

            trainY = new float[percent*600][];
            testY  = new float[percent*100][];
            //train Y
            for (int i=0;i<percent*600;i++){
                trainY[i] = new float[]{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                trainY[i][ trainYfile[i] ]=1.0f;
            }
            // test Y
            for (int i=0;i<percent*100;i++){
                testY[i] = new float[]{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                testY[i][ testYfile[i] ]=1.0f;
            }


            byte[] trainXfile = loadBin(path + trainXname, 16, percent * 784 * 600);// offset=16 size=percent*784*600
            trainX=new float[percent*600][784];
            for (int i=0;i<percent*100;i++) {
                int off=i*784;
                for (int j=0;j<784;j++){
                    trainX[i][j]=Byte.toUnsignedInt( trainXfile[off+j] )/254.0f;///256.0f; //0-1
                    //System.out.println( trainX[i][j] );
                }
            }

            byte[] testXfile =   loadBin( path + testXname,  16, percent*784*100 );   // offset=16, size=percent*784*100
            testX=new float[percent*100][784];
            for (int i=0;i<percent*100;i++) {
                int off=i*784;
                for (int j=0;j<784;j++){
                    testX[i][j]=Byte.toUnsignedInt( testXfile[off+j] )/254.0f;///256.0f;
                }
            }

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

    public float[][] convertToSquare28x28( float[] vector ){
        float[][] square = new float[28][28];
        for ( int y=0;y<28;y++ ){
            for ( int x=0;x<28;x++ ) {
                square[y][x]=vector[x+28*y];
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
    public float[][] getTrainX() { return trainX; }
    public float[][] getTestX()  { return testX;  }

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








