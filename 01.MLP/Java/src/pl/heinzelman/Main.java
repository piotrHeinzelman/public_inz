package pl.heinzelman;

import pl.heinzelman.tasks.*;

public class Main {

    public static void main(String[] args) {

        Task task;
        task = new Task_64_64_simple_backward();
        task.doTask();
    }
}

