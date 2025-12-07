package pl.heinzelman.tools;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import static java.awt.image.BufferedImage.TYPE_INT_RGB;

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
            trainYfile =  loadBin( path + trainYname,  2, percent*8*2 ); // offset=8, size=percent*600  // OK
            testYfile =  loadBin( path + testYname,   2, percent*8*2 );   // offset=8, size=percent*100 // OK

            trainY = new float[percent*8][2];
            testY  = new float[percent*8][2];
            //train Y
            for (int i=0;i<percent*8;i++){ trainY[i][0]=0f; trainY[i][1]=0f; testY[i][0]=0f; testY[i][1]=0f; }
            for (int i=0;i<percent*8;i++){
                if ( trainYfile[i*2 + 0] > .5f ) { trainY[i][0]=1f; testY[i][0]=1f;} else { trainY[i][1]=1f; testY[i][1]=1f;}
            }



            byte[] trainXfile = loadBin(path + trainXname, 240*240*3, percent*8 * 240*240*3 );// offset=16 size=percent*784*600

            trainX=new float[percent*8][C][H][W];
            testX=new float[percent*8][C][H][W];
            for (int n=0;n<percent*8;n++) {
                for (int h=0;h<H;h++){
                    for (int w=0;w<W;w++) {
                        for (int c = 0; c < C; c++) {

                            byte val =  trainXfile[n *W*H*C + h * W*C + w*C + c];
                            float fval = (1f/127f)*val;//*(1f/127f);
                            //System.out.println( fval );
                            trainX[n][c][h][w] =  fval;
                            testX[n][c][h][w] = fval;
                        }
                    }
                }
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

    public float[][] convertToSquare240x240( float[] vector ){
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


    public void saveXasJPG( float[][][] tensor ){
        int H=tensor[0].length;
        int W=tensor[0][0].length;
        // System.out.println( "H:" + H + ", W:" + W + " :: " + ( tensor[0][120][120] )  );
        BufferedImage image = new BufferedImage( W, H, TYPE_INT_RGB ); // 	TYPE_3BYTE_BGR

        File file = new File("image.png");
        for ( int i=0; i<H; i++){
            for (int j=0;j<W;j++){
                int pixelRGB = (int)( (255-tensor[0][i][j]*127)*2  *256*256 + (255-tensor[1][i][j]*127)*2 *256 + (255-tensor[2][i][j]*127)*2 );
                image.setRGB( j, i, pixelRGB );
            }
        }
        try {
            ImageIO.write(image ,  "png", file );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }








public float[][] getTrainY() { return trainY; }
    public float[][] getTestY()  { return testY;  }
    public float[][][][] getTrainX() { return trainX; }
    public float[][][][] getTestX()  { return testX;  }

    public static float[][] aryAdd( float[][] A, float[][] B){
        float [][] C = new float[A.length][A[0].length];
        for ( int i=0;i<A.length;i++ ){
            for ( int j=0;j<A[0].length;j++ ) {
                C[i][j] = A[i][j]+B[i][j];
            }
        }
        return C;
    }

    public void shape(float[][][] tensor){
        System.out.println( "size:" + tensor.length + ", " + tensor[0].length + ", "+tensor[0][0].length  );
    }

}








