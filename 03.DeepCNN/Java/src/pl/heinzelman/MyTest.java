package pl.heinzelman;

import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.Raster;
import java.io.*;

public class MyTest {


    @Test
    public void CopyDataToBinRepo240x240_240imgs() throws IOException {
        //if (true) return;
        //System.out.println("?");
        byte[] buff = new byte[480 * 240*240*3];
        byte[] buffc = new byte[480 * 2];
        int[] pixBuf =new int[3];
        int H=240;
        int W=240;
        int C=3;
        int offset=0;

        String path="/home/john/inz_DATA/INZ2/SAS_and_NoSAS_train_Data_240";

            int n=0;
            for (int b=0;b<16;b++){
                for (int c=0;c<46;c++){
                    for (int a=0;a<2;a++){ //class...


                        String name= "/"+a+"/img"+a+"0"+b+"10"+c +".jpg";

                        try ( FileInputStream fis = new FileInputStream(path + name ) ){

                            if (fis.available()>0 && n<481) {
                                n++;
                                //if (n==481) break;

                                System.out.println( " N: " + n + " = "+ path+name + ": " + fis.available());

                                // copy HOT-ONE to buff
                                if ( a==0 ) { buffc[ n*2 ]=1; buffc[ n*2 +1 ]=0;  } // class 1 SAS
                                       else { buffc[ n*2 ]=0; buffc[ n*2 +1 ]=1;  } // class 2 noSAS

                                // copy image to buff
                                 offset=n*240*240*3;

                                BufferedImage bufImage = ImageIO.read(fis);
                                Raster raster = bufImage.getData();


                                for (int h=0;h<H;h++){
                                    for (int w=0;w<W;w++) {

                                        raster.getPixel(w, h, pixBuf );
                                        buff[offset + h*W*C + w*C +0]= (byte) (127-(pixBuf[0]/2)); //R
                                        buff[offset + h*W*C + w*C +1]= (byte) (127-(pixBuf[1]/2)); //R
                                        buff[offset + h*W*C + w*C +2]= (byte) (127-(pixBuf[2]/2)); //R

                                    }
                                }
                            }
                        fis.close();
                        } catch( Throwable t ) {  continue; }
                }
            }
        }



     //   FileInputStream fis = new FileInputStream(path+"img0001001.jpg");
     //   BufferedImage bufImage = ImageIO.read(fis);
     //   int height = bufImage.getHeight();
     //   System.out.println(height);

if ( true ) {
    FileOutputStream fos = new FileOutputStream("/home/john/public_inz/data/output.bin");
    fos.write(buff);
    fos.close();

    FileOutputStream fosc = new FileOutputStream("/home/john/public_inz/data/output.class");
    fosc.write(buffc);
    fosc.close();
}

    }


    @Test
    public void showImage(  ) throws IOException {
        String fullName="/home/john/inz_DATA/INZ2/SAS_and_NoSAS_train_Data_240/0/img0001001.jpg";


        FileInputStream fis = new FileInputStream( fullName );
        BufferedImage bufImage = ImageIO.read(fis);
        int H = bufImage.getHeight();
        int W = bufImage.getWidth();
        ColorModel colorModel = bufImage.getColorModel();
        int C = colorModel.getNumColorComponents();

        Raster raster = bufImage.getData();

        int[] buf =new int[4];
        raster.getPixel(0,0, buf );
        System.out.println( Byte.toUnsignedInt((byte)buf[0])+ " : "+(byte)buf[1]+" : "+(byte)buf[2] );


        int pixel = bufImage.getRGB( 110, 110 );
        byte r = (byte) (pixel & 0xff0000);
        byte g = (byte) (pixel & 0xff00);
        byte b = (byte) (pixel & 0xff);

        System.out.println( "R:" +  	  Byte.toUnsignedInt(r) + ", G: " +  Byte.toUnsignedInt(g) + ", B: " +  Byte.toUnsignedInt(b) );

        /*
        for (int h=0;h<H/5;h+=5){
            for (int w=0;w<W/5;w+=5){

                    //System.out.println( h+W*C + w*C + c  );
                    byte pixelValue = bufImage.getRGB( w*5, h*5 );
                    //for (int c=0;c<C;c++){
                    System.out.println( pixelValue );
                    //}
            }
            System.out.println( "\n" );
        }
        */


    }




}
