### Describe  all  optimizations  you  tried  regardless of  whether  you  committed  to  them or  abandoned them and whether they improved or hurt performance.
I make local bins for each block in shared memory, update the shared memory, then at the very end, copy the shared memory bins to global memory. This reduces the amount of global memory atomic operations significantly. The gobal memory and shared memory reads are coalesced in the histogram kernel, I don't think it's possible to coalesce the writes as they may access any bin. the conversion_kernel is also coalesced. 
I attempted but did not succeed in speeding up the final summation of the private bins. In my current implementation, global memory is written to by each block. I attempted to have a second kernel that would read the private bins and add them in shared memory, then add them to global memory in one go. My best solution was to create as many global bins as I had blocks, which allowed me to update the block global bin without atomics, and then have another kernel add up all the block global bins into the total output bins. Speedup from removal of atomics was offset by the increased global memory reads.

### Were there any difficulties you had with completing the optimization correctly?
When I was zeroing out the bins I miscalculated the size and had to debug a seg fault which is always fun.

### Which optimizations gave the most benefit?
Using private shared memory bins instead of the global bins gave a significant speedup as all the blocks had their own memory to access and didn't have to queue to access the global memory

### For  the  histogram  kernel,  how  many  global  memory  reads  are  being  performed  by  your  kernel?
1 read per thread (if thread is inside the image bounds) so 1025.

### For  the  histogram  kernel,  how  many  global  memory  writes  are  being  performed  by  your  kernel?
4096 atomic adds per block which write to global memory per block.
But if you mean where I just directly write to it, then zero.

### For the histogram kernel, how many atomic operations are being performed by your kernel? Explain.
4096 atomic adds per block. At the end of the kernel all the private bins need to be added to the total in global memory. Each thread does  (num_threads/num_bins) atomic adds. For this example 4096 / 256 = 16 atomic operations per thread

### For the histogram kernel, what contentions would you expect if every element in the array has the same value?
The performance would degrade considerably as all the threads would have to wait to access the same bin address in memory. 

### For the histogram kernel, what contentions would you expect if every element in the input array has a random value?
The higher the entropy of the input image, the better the kernel will perform as there will be very few collisions so the threads will spend less time waiting to update the private bins.