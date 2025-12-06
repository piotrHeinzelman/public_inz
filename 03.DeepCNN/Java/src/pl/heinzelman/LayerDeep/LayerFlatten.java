package pl.heinzelman.LayerDeep;

public class LayerFlatten {
    protected int channels;
    protected int xsize;

    public LayerFlatten () {}

    public float[] Forward ( float[][][] _x ) {
        this.channels= _x.length;
        this.xsize=_x[0].length;
        int index=0;
        float[] Z = new float[ channels*xsize*xsize ];
            for (int c=0;c<channels;c++){
                for (int i=0;i<xsize;i++){
                    for (int j=0;j<xsize;j++) {
                        Z[ index ] = _x[c][i][j];
                        index++;
                    }
                }
            }
        return Z;
    }

    public float[][][] Backward( float[]delta ){
        int index=0;
        float[][][] OUT = new float[ channels ][ xsize ][ xsize ];
        for (int c=0;c<channels;c++){
            for (int i=0;i<xsize;i++){
                for (int j=0;j<xsize;j++) {
                    OUT[c][i][j] = delta[ index ];
                    index++;
                }
            }
        }
        return OUT;
    }
}
