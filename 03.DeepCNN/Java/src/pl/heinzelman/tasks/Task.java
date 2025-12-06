package pl.heinzelman.tasks;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.temporal.ChronoUnit;

public interface Task {

    public void prepare( int percent );
    public void run();

    default public void doTask( int percent ){
        this.prepare( percent );

            Instant start = Instant.now();
            this.run();
            Instant end = Instant.now();

        System.out.println( "\r\n\r\n---------------------");
        System.out.println( "start: " + Timestamp.from(start) );
        System.out.println( "end  : " + Timestamp.from(end) );
        long gap = ChronoUnit.MILLIS.between(start, end);
        System.out.println( " -GAP - : " + gap/1000.0 + " [sek.]" );
    }
}
