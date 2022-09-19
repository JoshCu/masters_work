// 1A
#DEFINE BLOCK_SIZE = 5;
__host__ void vecAdd(float *A, float *B, float *C, int n)
{
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Allocate device memory for A, B, and C
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // copy A and B to device memory
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Kernel launch code – to have the device
    dim3 numBlocks(size/BLOCK_SIZE,size/BLOCK_SIZE);
    dim3 numThreads(BLOCK_SIZE,BLOCK_SIZE);


    // Dimensions for Each row / column
    dim3 alternativeBlocks(size/BLOCK_SIZE);
    fix
   // dim3 alternativeThreads(, 1, 1);

    MatrixMulKernelPerElement<<<numBlocks, numThreads>>>(A_d, B_d, C_d, n);
    // to perform the actual vector addition

    // MatrixMulKernelPerRow<<<alternativeBlocks, alternativeThreads>>>(A_d, B_d, C_d, n);

    // MatrixMulKernelPerColumn<<<alternativeBlocks, alternativeThreads>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudafree(A_d);
    cudafree(B_d);
    cudafree(C_d);
}

// This has one thread per element of the matrix so should be faster
__global__ void MatrixMulKernelPerElement(float *A_d, float *B_d, float *C_d, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size)
    {
        for (k = 0; k < size; k++)
        {
            C_d[i][j] += A_d[i][k] * B_d[k][j];
        }
    }
}

// this has one thread per row which should be slower
__global__ void MatrixMulKernelPerRow(float *A_d, float *B_d, float *C_d, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        for (k = 0; k < size; k++)
        {
            for (j = 0; j < size; j++)
            {
                C_d[i][j] += A_d[i][k] * B_d[k][j];
            }
        }
    }
}

// this is the same as the column one but with the rows and columns swapped
__global__ void MatrixMulKernelPerColumn(float *A_d, float *B_d, float *C_d, int size)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < size)
    {
        for (k = 0; k < size; k++)
        {
            for (i = 0; i < size; i++)
            {
                C_d[i][j] += A_d[i][k] * B_d[k][j];
            }
        }
    }
}

// PROBLEM B
/*
    1. The block size should be set to 1-5 because there are 32 threads in a warp and 5*5 = 25  6*6 = 36.
        therefore the smallest square grid you can have is a 5x5  dimBlock(BLOCK_SIZE,BLOCK_SIZE)

    note: warp is a group of threads with consecutive indexes that execute together

    2. Add a synchtreads() between the write and read to clock A to ensure that all the writes have happened before the read does
       // how would this even work withouth the syncthreads()?

/*