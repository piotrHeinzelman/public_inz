package pl.heinzelman.tasks;

public class TaskForLoop2sek implements Task {

    @Override
    public void prepare() {
        System.out.println( "prepare...");
    }

    @Override
    public void run() {
        //int[] M = new int[100000]; +15sek
        for (int i=0;i<100000;i++){
            for ( int j=0;j<100000;j++ ) {
                for ( int k=0; k<100000;k++ ) {
                    int z = 2*(i*j*k)+1*i+2*k+3-3*i;
                    z=z+1;
                    //M[i]++; +15sek.
                }
            }
        }
    }
}
