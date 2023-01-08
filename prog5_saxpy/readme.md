# Asst1 Prog5 Writeup
The performance of ispc:

    [saxpy ispc]:           [21.242] ms     [14.030] GB/s   [1.883] GFLOPS
    [saxpy task ispc]:      [20.692] ms     [14.403] GB/s   [1.933] GFLOPS
                                (1.03x speedup from use of tasks)

I guess the reason of that the multi-core ISPC performance is not significant is that the time for memory loading dominiates the time for floating point operations.

TODO 


Extra Credit: (1 point) Note that the total memory bandwidth consumed computation in main.cpp is TOTAL_BYTES = 4 * N * sizeof(float);. Even though saxpy loads one element from X, one element from Y, and writes one element to result the multiplier by 4 is correct. Why is this the case? (Hint, think about how CPU caches work.)

Extra Credit: (points handled on a case-by-case basis) Improve the performance of saxpy. We're looking for a significant speedup here, not just a few percentage points. If successful, describe how you did it and what a best-possible implementation on these systems might achieve. Also, if successful, come tell the staff, we'll be interested. ;-)

TODO Yongqiang: maybe improve the memory latency?