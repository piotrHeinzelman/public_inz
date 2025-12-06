package pl.heinzelman.LayerDeep;

public class LayerFlatten {
    protected int channels;
    protected int xsize;

    public LayerFlatten () {}

    public float[] Forward ( float[][][] _x ) {
        channels= _x.length;
        xsize=_x[0].length;

        float[] Z = new float[ channels*xsize*xsize ];
            for (int c=0;c<channels;c++){
                for (int i=0;i<xsize;i++){
                    for (int j=0;j<xsize;j++) {
                        Z[ c + channels*i + channels*xsize*j ] = _x[c][i][j];
                    }
                }
            }
        return Z;
    }

    public float[][][] Backward( float[] delta ){

        float[][][] OUT = new float[ channels ][ xsize ][ xsize ];
        for (int c=0;c<channels;c++){
            for (int i=0;i<xsize;i++){
                for (int j=0;j<xsize;j++) {
                    OUT[c][i][j] = delta[ c + channels*i + channels*xsize*j ];
                }
            }
        }
        return OUT;
    }
}
