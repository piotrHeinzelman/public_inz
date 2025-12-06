package pl.heinzelman.tools;

public class Conv {
    public static float[][] extendAry( float[][] X, int padding ){
        int oversize=X.length+padding+padding;
        float[][] XPadd = new float[oversize][oversize];

        for ( int i=0;i<oversize;i++){
            for (int j=0;j<oversize;j++){
                XPadd[i][j]=0f;
            }
        }
        for ( int i=0;i<X.length;i++ ){
            for ( int j=0;j<X.length;j++){
                XPadd[i+padding][j+padding]=X[i][j];
            }
        }
        return XPadd;
    }



    public static float[][] fullConv( float[][] X, float[][] F , int stride){
        int padding = F.length-1;
        return  conv (  extendAry( X , padding ) , F,  0 , stride ) ;
    }

    public static float[][] conv( float[][] X, float[][] F, float bias ){
        return conv( X, F, bias, 1 );
    }

    public static float[][] conv( float[][] X, float[][] F, float bias, int stride ){

        int outputSize= 1+(( X.length-F.length )/stride);
        int fSize=F.length;
        float [][] Y = new float[outputSize][outputSize];
        for ( int i=0;i<outputSize;i++ ){
            for ( int j=0;j<outputSize;j++ ){
                float YIJ = bias;
                {
                    for (int m=0;m<fSize;m++){
                        for (int n=0;n<fSize;n++){
                            YIJ += ( F[m][n] * X[i+m][j+n] );
                        }
                    }
                }
                Y[i][j]=( YIJ );
            }
        }
        return Y;
    }


}
