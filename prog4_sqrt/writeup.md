# Asst1 Prog4 Writeup

For running outputs, see `outputs` folder. Note that the algorithm only converges within the (0,3) open interval.

Q1: About 5x of speedup due to SIMD parallelization. And 10x of speedup due to multi-core parallelization.

    [sqrt serial]:          [998.640] ms
    [sqrt ispc]:            [201.182] ms
    [sqrt task ispc]:       [20.765] ms
                                (4.96x speedup from ISPC)
                                (48.09x speedup from task ISPC)

Q2: Based on the formula of speedup, the maximum speedup of the ISPC implementation over the sequential version of the code could be achieved when we maximize the time needed in each sqrt call in the sequential version of the program and we improve the effiency of parallelization, i.e., decompose the task more evenly. We initialize the all array elements to 2.99;

We achieve

    [sqrt serial]:          [1229.223] ms
    [sqrt ispc]:            [174.197] ms
    [sqrt task ispc]:       [24.739] ms
                                (7.06x speedup from ISPC)
                                (49.69x speedup from task ISPC)
The modification improves SIMD speedup by even task decomposition. But it does not improve multi-core speedup (i.e., the benefit of moving from ISPC without-tasks to ISPC with tasks) as a result of task scheduling overhead.

Q3: To minimize speedup for ISPC (without-tasks) over the sequential version of the code, we construst an extremely uneven input case with one 2.99 and seven 1 in a 8-wide instruction.

    [sqrt serial]:          [330.177] ms
    [sqrt ispc]:            [346.202] ms
    [sqrt task ispc]:       [50.229] ms
                                (0.95x speedup from ISPC)
                                (6.57x speedup from task ISPC)

The reason of inefficiency is that most of the 8-wide SIMD lanes lie idle during the running of the program (and a performance degradation from vector instruction vs. normal instruction).