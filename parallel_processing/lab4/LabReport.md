## 1. How many floating operations are being performed in each of your matrix multiply kernels? Explain.  
Basic Multiplication kernel loops through i to matrix with multiplying together and adding to a sum, so `2xwidth` per kernel and there are width x height kernels so `2x16x16x16 = 8192` total flops with 32 per thread.  
tiled multiplication kernel loops through i to numAcolumns with TileWidth increments and then loops from 0 to tile width multiplying two numbers and summing the products. So `2 x tilewidth x (ceil(numAcolumns / float(tilewidth))` which works out to 2x the width of the input array A so 16 flops per thread.  

## 2. How many global memory reads are being performed by each of your kernels? Explain.  
Basic - `2 x Array A width` per thread so 32 for this example. Each time it multiplies two elements of A & B, it fetches the elements from global memory.  
Tiled - `2 x (ceil(numAcolumns / float(tilewidth))` each thread will loop `ceil(numAcolumns / float(tilewidth))` times adding the elements from A and B to shared memory. Then the dot product is calculated using shared memory.  

## 3. How many global memory writes are being performed by each of your kernels? Explain.  
Basic and Tiled - `1` per thread at the end of the dot product calculation, the result is stored in the out array C.  

## 4. Describe what possible further optimizations can be implemented to each of your kernels to achieve performance speedups.  
Basic - Arrays A and B could be loaded into shared memory to speed up data access. 
Tiled - 

## 5. Name three applications of matrix multiplication.  
Scaling, translations and other manipulations of images. Graph analytics when calculating paths of length N between two nodes. Generating Adjacency matrices.

## 6.  Compare the implementation difficulty of the tiledMult.cu kernel to that of the basicMult.cu kernel. What difficulties did you have with this implementation?  
The tiled kernel is far more complex and has to calculate the x and y offset differently taking into account the tile width. The tiled kernel also takes advantage of faster shared memory so there is added the complexity of loading the input arrays into shared memory and using syncthreads to avoid trying to access uninitialized data.

## 7. Suppose you have matrices with dimensions bigger than the max thread dimensions. Sketch an algorithm that would perform matrix multiplication algorithm that would perform the multiplication in this case.  
If the matrices have dimensions bigger than the max thread dimensions then you would need to implement tiling in such a way that the every thread available is working on a given section of the matrix multiplication. Then when it finishes that section of the matrix, each thread then moves onto another tile in the matrix.

## 8. Suppose you have matrices that would not fit in global memory. Sketch an algorithm that would perform matrix multiplication algorithm that would perform the multiplication out of place.  

If the matrices do not fit into shared memory then they would have to be broken into smaller sub matrices and loaded then unloaded into memory two at a time. Breaking matrix A into rows that would fit into memory and B into colulmn blocks that would fit into memory. It would then be possible to multiply these submatrices together to calculate a portion of the final output matrix and then stitch the results together at the end. 