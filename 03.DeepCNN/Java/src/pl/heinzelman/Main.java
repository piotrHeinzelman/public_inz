package pl.heinzelman;
import pl.heinzelman.tasks.Task;
import pl.heinzelman.tasks.Task_4_CNN;

public class Main {
    public static void main( String[] args ) {
        Task task;
        task = new Task_4_CNN();
        task.doTask( 5 );
        //  -GAP - : 1398.269 [sek.] - 1 epoka 30%;
    }
}
