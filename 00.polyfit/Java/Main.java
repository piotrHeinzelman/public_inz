import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.time.Instant;
import java.time.temporal.ChronoUnit;

public class Main {

    private static int size = 64000000;

    public static void main(String[] args) throws IOException {

	Instant prepareStart = Instant.now();
        double[] x = new double[size];
        double[] y = new double[size];
	for (int i=0;i<size;i++){
	   x[i]=0.1*i;
	   y[i]=0.2*i;
	// System.out.println(x[i]);
	}
	Instant prepareEnd = Instant.now();

    double w1 = 0.0;
    double w0 = 0.0;

    Instant start = Instant.now();
        // -- start
        double xsr = 0.0;
        double ysr = 0.0;
        for (int i = 0; i < size; i++) {
            xsr += x[i];
            ysr += y[i];
        }
        xsr = xsr / size;
        ysr = ysr / size;
        System.out.println( "# srX: " + xsr + "[sek.]");
        w1 = 0.0;
        w0 = 0.0;
        double sumTop = 0.0;
        double sumBottom = 0.0;
        for (int i = 0; i < size; i++) {
            sumTop += ((x[i] - xsr) * (y[i] - ysr));
            sumBottom += ((x[i] - xsr) * (x[i] - xsr));
        }
        w1 = sumTop / sumBottom;
        w0 = ysr - w1 * xsr;
        // -- end

    Instant end = Instant.now();
	Double prepareTime=ChronoUnit.MILLIS.between(prepareStart, prepareEnd)/1000.0;

    System.out.println( "# prepare Table time:" + prepareTime + "[sek.]");
    System.out.println( "# time: " + (   ChronoUnit.MILLIS.between(start, end)/1000.0  ) + " [sek.],");
    System.out.println( " result: [" + w1 + ", " + w0 + "]" );
    }
}


















