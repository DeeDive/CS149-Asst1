# asst1 prog3 writeup

QA to Part2: I chose n_tasks to 40 and achieved a speedup of 45.40x and 33.19x on view 1 and 2 respectively. The reason is that by choosing more tasks has the potential to make the task decomposition more even, despite the context switching cost. (Also n_tasks should be divided by height.)

QA to Extra Credit: (2 points) What are differences between the thread abstraction (used in Program 1) and the ISPC task abstraction? There are some obvious differences in semantics between the (create/join and (launch/sync) mechanisms, but the implications of these differences are more subtle. Here's a thought experiment to guide your answer: what happens when you launch 10,000 ISPC tasks? What happens when you launch 10,000 threads? (For this thought experiment, please discuss in the general case

In this mandelbrot task, we got speedup from scheduling tasks given our uneven task decomposition. Hence, the decalration of more independent tasks is necessary. However, we may not get much speedup when task scheduling overhead dominates the performance.

```
In general, one should launch many more tasks than there are processors in the system to ensure good load-balancing, but not so many that the overhead of scheduling and running tasks dominates the computation.
```
See the documentation of ispc: https://ispc.github.io/ispc.html

Also the discussion of the distinction between threads and tasks are discussed at http://15418.courses.cs.cmu.edu/spring2016/lecture/progabstractions/slide_007.

To quote @karima:

```
@mrrobot I think your confusion might come from not fully understanding the distinction between threads and tasks. YOU as the programmer, tell the compiler how many tasks you wish to launch. The ISPC compiler will make an intelligent decision on how many threads to spawn based how many tasks you've told it to launch and on its knowledge of the hardware it's compiling for.

So for example if you only launched one task, obviously the compiler can only spawn at most one thread. But if you launch 2 or more tasks, the compiler can then choose to spawn more than one thread. Say you chose to launch 1000 tasks on a 2 core machine. Do you think a good compiler would spawn 1000 pthreads as well?

Tasks are an abstraction that tell the compiler what independent units of work exist in your program. Each task must be fully executed on one core as each task maps to one ISPC gang, but because tasks are independent of each other, different tasks can execute on different cores, or the same core. Whichever happens ultimately depends on how the ISPC compiler decides to compiler your code.

The number of tasks you launch does not necessarily equal the number of threads the compiler decides to generate. You are correct that ISPC uses a worker pool approach in which they spawn worker threads based on the underlying hardware.

So I have some questions for you:

1) Say you wrote a program where you launched 100 tasks. If you compile your program for a hyper-threaded machine with 4 cores, how many threads do you think a reasonable compiler will decide to spawn?

2) Obviously each thread can only work on one task at a time. What happens when one of these threads finishes a task and there are still tasks left to be done?
```



The smart-thinking student's question: Hey wait! Why are there two different mechanisms (foreach and launch) for expressing independent, parallelizable work to the ISPC system? Couldn't the system just partition the many iterations of foreach across all cores and also emit the appropriate SIMD code for the cores?

Answer: Great question! And there are a lot of possible answers. Come to office hours.

(Is is possbily because that the system can not easily determine the partition factor? It needs running feedback to get the optimal number of tasks, i.e. a hyperparameter)